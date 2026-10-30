from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Callable, Optional, Union

from app.core.models import ImageOcrRecord, MediaReviewRecord

RecognitionRecord = Union[ImageOcrRecord, MediaReviewRecord]
ROOT = Path(__file__).resolve().parents[3]
TIMEOUTS = {"image": 180.0, "formula": 600.0, "audio": 600.0}
_ACTIVE_SESSION = ContextVar("media_batch_session", default=None)



@contextmanager
def media_session():
    """Reuse model processes for one batch and always close them afterwards."""
    current = _ACTIVE_SESSION.get()
    if current is not None:
        yield current
        return
    from app.core.preprocess.media_session import MediaSession
    session = MediaSession()
    token = _ACTIVE_SESSION.set(session)
    try:
        yield session
    finally:
        _ACTIVE_SESSION.reset(token)
        session.close()


class MediaJobError(ValueError):
    def __init__(self, message: str, *, record: Optional[RecognitionRecord] = None,
                 returncode: Optional[int] = None, timed_out: bool = False,
                 cancelled: bool = False) -> None:
        super().__init__(message)
        self.record = record
        self.returncode = returncode
        self.timed_out = timed_out
        self.cancelled = cancelled


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(kind: str, source: Path, output: Path) -> list[str]:
    if getattr(sys, "frozen", False):
        helper = Path(sys.executable).parent / "media_worker.exe"
        if helper.is_file():
            command = [str(helper), kind]
        else:
            command = [sys.executable, "--image-ocr-worker" if kind == "image" else
                       "--media-recognition-worker"]
            if kind != "image":
                command.append(kind)
    else:
        module = "app.core.preprocess.image_ocr_worker" if kind == "image" else "app.core.preprocess.media_recognition_worker"
        command = [sys.executable, "-m", module]
        if kind != "image":
            command.append(kind)
    return command + [str(source), str(output)]


def _terminate(process) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=15,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    if process.poll() is None:
        process.kill()
    try:
        process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()


def _communicate(process, timeout: float, should_cancel: Optional[Callable[[], bool]]):
    deadline = time.monotonic() + timeout
    while True:
        if should_cancel is not None and should_cancel():
            raise MediaJobError("媒体识别已取消，原题和已有识别结果保持不变。", cancelled=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MediaJobError("媒体识别超过 {:.0f} 秒，已结束本次任务。".format(timeout), timed_out=True)
        try:
            return process.communicate(timeout=min(0.25, remaining) if should_cancel else remaining)
        except subprocess.TimeoutExpired:
            continue


def recognize_media(path: Union[str, Path], kind: str, options=None, *,
                    timeout: Optional[float] = None,
                    should_cancel: Optional[Callable[[], bool]] = None) -> RecognitionRecord:
    session = _ACTIVE_SESSION.get()
    if session is not None:
        return session.recognize(path, kind, options, timeout=timeout, should_cancel=should_cancel)
    return _recognize_once(path, kind, options, timeout=timeout, should_cancel=should_cancel)


def _recognize_once(path: Union[str, Path], kind: str, options=None, *,
                    timeout: Optional[float] = None,
                    should_cancel: Optional[Callable[[], bool]] = None) -> RecognitionRecord:
    """起一个独立识别进程完成一次识别，进程随即退出。

    批量场景下图片识别**刻意**走这条路而不是复用常驻进程，原因见
    ``media_session.MediaSession`` 的类说明。
    """
    if kind not in TIMEOUTS:
        raise MediaJobError("不支持的媒体类型：{}".format(kind))
    if options is not None and kind != "formula":
        raise MediaJobError("裁剪、反色和缩放选项仅适用于公式图片。")
    if options is not None and not isinstance(options, dict):
        raise MediaJobError("公式识别选项必须是 JSON 对象。")
    timeout = TIMEOUTS[kind] if timeout is None else float(timeout)
    if timeout <= 0 or not math.isfinite(timeout):
        raise MediaJobError("识别超时时间必须是有限正数。")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise MediaJobError("媒体文件不存在：{}".format(source))
    if should_cancel is not None and should_cancel():
        raise MediaJobError("媒体识别已取消。", cancelled=True)
    try:
        original_hash = _hash_file(source)
        with tempfile.TemporaryDirectory(prefix="mc_media_job_") as temporary:
            output = Path(temporary) / "result.json"
            command = _command(kind, source, output)
            if options is not None:
                settings = Path(temporary) / "options.json"
                settings.write_text(json.dumps(options, ensure_ascii=False, allow_nan=False), encoding="utf-8")
                command.extend(["--options-file", str(settings)])
            environment = os.environ.copy()
            environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                               HF_DATASETS_OFFLINE="1", NO_ALBUMENTATIONS_UPDATE="1")
            environment.pop("PYTHONHOME", None)
            environment.pop("PYTHONPATH", None)
            process_options = ({"creationflags": subprocess.CREATE_NO_WINDOW |
                                subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else
                               {"start_new_session": True})
            process = subprocess.Popen(
                command, cwd=None if getattr(sys, "frozen", False) else str(ROOT),
                env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", **process_options)
            try:
                stdout, stderr = _communicate(process, timeout, should_cancel)
            except BaseException:
                _terminate(process)
                raise
            if not output.is_file():
                detail = (stderr or "").strip()[-2000:]
                raise MediaJobError(detail or "识别进程未返回结果文件。", returncode=process.returncode)
            try:
                payload = json.loads(output.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("结果不是 JSON 对象")
                cls = ImageOcrRecord if kind == "image" else MediaReviewRecord
                record = cls.from_dict(payload)
            except (OSError, ValueError, TypeError) as exc:
                raise MediaJobError("识别进程结果格式错误：{}".format(exc),
                                    returncode=process.returncode) from exc
            error = payload.get("error")
            if error or process.returncode:
                message = str(error or (stderr or "").strip()[-2000:] or
                              "识别进程异常退出，代码 {}".format(process.returncode))
                raise MediaJobError(message, record=record, returncode=process.returncode)
            if not isinstance(record.text, str) or not record.text.strip():
                raise MediaJobError("识别进程没有返回有效内容。", record=record)
            if record.media_hash != original_hash or _hash_file(source) != original_hash:
                raise MediaJobError("附件已变化或识别结果与原附件不符，请重新识别。", record=record)
            record.confirmed = False
            if isinstance(record, MediaReviewRecord):
                record.applied_text = ""
            return record
    except MediaJobError:
        raise
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        raise MediaJobError("媒体识别失败：{}".format(exc)) from exc
