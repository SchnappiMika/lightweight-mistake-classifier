"""A batch owns these model processes; closing it releases every process."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

from app.core.models import ImageOcrRecord, MediaReviewRecord
from app.core.preprocess.media_jobs import MediaJobError, ROOT, TIMEOUTS, _hash_file, _terminate

# 哪些媒体类型适合"一个常驻进程处理多份文件"。
#
# 图片被排除，是实测结论，不是配置偏好：同一进程内连续识别同一张 8.7KB 图片，
# 第 1 次约 2.3 秒、第 2 次约 6.0 秒、第 3 次起稳定在约 11.4 秒；而每次新起
# 进程连续跑 8 次，全部稳定在 2.3–2.5 秒。重建 RapidOCR 模型对象（同进程）
# 仍为 2.3→9.2→12.7 秒，说明是进程级状态累积，不是模型对象复用的问题；
# 日志处理器、活动线程数、可用内存（4.5GB）都已排除。
# 因此图片改回"一图一进程"，宁可每次多花约 0.2 秒进程启动，也不要退化到 11 秒。
REUSABLE_KINDS = ("audio", "formula")


def server_command(kind, directory):
    if kind == "formula":
        from app.core.preprocess.formula_runtime import formula_runtime
        from app.core.preprocess.media_recognition import _model_dir
        python, backend = formula_runtime()
        return [str(python), str(backend), "--serve", str(directory),
                "--model-dir", str(_model_dir("formula").resolve())]
    if getattr(sys, "frozen", False):
        helper = Path(sys.executable).parent / "media_worker.exe"
        if helper.is_file():
            return [str(helper), "serve", kind, str(directory)]
        return [sys.executable, "--media-session-worker", kind, str(directory)]
    return [sys.executable, "-m", "app.core.preprocess.media_server", kind, str(directory)]


@dataclass
class _Server:
    process: object
    directory: Path
    logfile: object
    index: int = 0


class MediaSession:
    """Reuse models across distinct media files, without reusing their results."""

    def __init__(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="mc_media_batch_")
        self._servers = {}
        self._closed = False
        self._lock = threading.RLock()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def _start(self, kind):
        previous = self._servers.get(kind)
        if previous is not None and previous.process.poll() is None:
            return previous
        self._remove(kind)
        # A new subdirectory avoids replaying requests from a failed process.
        directory = Path(tempfile.mkdtemp(prefix=kind + "_", dir=self._temporary.name))
        environment = os.environ.copy()
        environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1",
                           TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                           NO_ALBUMENTATIONS_UPDATE="1")
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        options = ({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
                   if os.name == "nt" else {"start_new_session": True})
        command = server_command(kind, directory)
        logfile = (directory / "worker.log").open("wb")
        try:
            process = subprocess.Popen(command, cwd=None if getattr(sys, "frozen", False) else str(ROOT),
                env=environment, stdin=subprocess.DEVNULL, stdout=logfile, stderr=logfile, **options)
        except BaseException:
            logfile.close()
            raise
        server = _Server(process, directory, logfile)
        self._servers[kind] = server
        return server

    def _remove(self, kind):
        server = self._servers.pop(kind, None)
        if server is not None:
            try:
                (server.directory / "stop").touch()
                try:
                    server.process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    _terminate(server.process)
            finally:
                server.logfile.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                for kind in list(self._servers):
                    self._remove(kind)
            finally:
                self._temporary.cleanup()

    def recognize(self, path, kind, options=None, *, timeout=None, should_cancel=None):
        with self._lock:
            if self._closed:
                raise MediaJobError("媒体批次已经结束，请重新开始处理。")
            if kind not in TIMEOUTS:
                raise MediaJobError("不支持的媒体类型：{}".format(kind))
            if kind not in REUSABLE_KINDS:
                # 该类型复用会越用越慢，改走"一次一进程"的既有路径。
                from app.core.preprocess.media_jobs import _recognize_once
                return _recognize_once(path, kind, options, timeout=timeout,
                                       should_cancel=should_cancel)
            if options is not None and (kind != "formula" or not isinstance(options, dict)):
                raise MediaJobError("公式处理选项必须是 JSON 对象，且仅用于公式图片。")
            timeout = TIMEOUTS[kind] if timeout is None else float(timeout)
            if not math.isfinite(timeout) or timeout <= 0:
                raise MediaJobError("识别超时时间必须是有限正数。")
            def check_cancelled():
                if should_cancel is not None and should_cancel():
                    raise MediaJobError("媒体识别已取消，已有内容保持不变。", cancelled=True)
            try:
                check_cancelled()
                source = Path(path).expanduser().resolve()
                if not source.is_file():
                    raise MediaJobError("媒体文件不存在：{}".format(source))
                original_hash = _hash_file(source)
                started = time.monotonic()
                server = self._start(kind)
                server.index += 1
                request = server.directory / "request-{:08d}.json".format(server.index)
                output = server.directory / "response-{:08d}.json".format(server.index)
                temporary = request.with_suffix(".tmp")
                temporary.write_text(json.dumps({"input_path": str(source), "options": options},
                    ensure_ascii=False, allow_nan=False), encoding="utf-8")
                os.replace(temporary, request)
                while not output.is_file():
                    check_cancelled()
                    if time.monotonic() - started >= timeout:
                        raise MediaJobError("媒体识别超过 {:.0f} 秒，已结束本次任务。".format(timeout), timed_out=True)
                    if server.process.poll() is not None:
                        server.logfile.flush()
                        log_path = server.directory / "worker.log"
                        with log_path.open("rb") as stream:
                            stream.seek(max(0, log_path.stat().st_size - 2000))
                            detail = stream.read().decode("utf-8", errors="replace").strip()
                        raise MediaJobError(detail or "媒体工作进程提前退出。", returncode=server.process.returncode)
                    time.sleep(0.01)
                check_cancelled()
                payload = json.loads(output.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise MediaJobError("媒体识别结果不是有效的 JSON 对象。")
                cls = ImageOcrRecord if kind == "image" else MediaReviewRecord
                record = cls.from_dict(payload)
                if payload.get("error"):
                    raise MediaJobError(str(payload["error"]), record=record)
                if not isinstance(record.text, str) or not record.text.strip():
                    raise MediaJobError("识别进程没有返回有效内容。", record=record)
                if record.media_hash != original_hash or _hash_file(source) != original_hash:
                    raise MediaJobError("附件已变化或识别结果与原附件不符，请重新识别。", record=record)
                record.confirmed = False
                record.applied_text = ""
                if kind == "formula":
                    from app.core.preprocess.formula_recognition_backend import normalize_options
                    record.recognition_options = normalize_options(options)
                return record
            except MediaJobError as exc:
                if exc.cancelled or exc.timed_out or (kind in self._servers and
                        self._servers[kind].process.poll() is not None):
                    self._remove(kind)
                raise
            except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
                self._remove(kind)
                raise MediaJobError("媒体识别失败：{}".format(exc)) from exc
