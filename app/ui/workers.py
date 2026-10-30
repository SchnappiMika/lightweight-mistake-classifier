# -*- coding: utf-8 -*-
"""后台工作线程。

多模态预处理（尤其是 OCR / 音频转写）可能耗时数百毫秒到数秒，
若放在主线程会直接冻住界面。这里把耗时环节统一放进 QThread，
通过信号回传进度，对应测试项"全流程操作无卡顿、无崩溃"。
"""
from __future__ import annotations

import logging
import traceback
from typing import List, Optional, Sequence

from PyQt5.QtCore import QThread, pyqtSignal

from app.core.classify.engine import ClassificationEngine
from app.core.models import ClassifyResult, Question, Scheme
from app.core.preprocess import pipeline

log = logging.getLogger(__name__)

#: 关窗时等待单个后台线程收尾的上限（毫秒）
SHUTDOWN_WAIT_MS = 4000


def stop_worker(w: Optional[QThread], wait_ms: int = SHUTDOWN_WAIT_MS) -> None:
    """请求停止并等待一个后台线程结束。

    QThread 在仍然运行时被析构，Qt 会直接 abort 掉整个进程——表现为用户"关窗
    就闪退"。而预热线程要跑好几秒（jieba 初始化），这段窗口期里关窗是很常见的
    操作，所以退出前必须逐个把线程收干净。

    先尝试 ``stop()`` 打协作停止标志（分类线程支持），再 ``quit()`` + ``wait()``；
    实在等不到（比如卡在某个第三方识别引擎里）才 ``terminate()`` 兜底，
    保证退出时不会崩，也不会挂死。
    """
    if w is None:
        return
    try:
        if not w.isRunning():
            return
        stop = getattr(w, "stop", None)
        if callable(stop):
            stop()
        w.quit()
        if not w.wait(wait_ms):
            log.warning("后台线程 %s 未在 %dms 内退出，强制终止", type(w).__name__, wait_ms)
            w.terminate()
            w.wait(1000)
    except RuntimeError:
        pass          # 线程的 C++ 对象已被回收，忽略即可


class WarmupWorker(QThread):
    """启动预热：加载 jieba 词典与公式映射词典。

    放到后台做，界面能秒开；等用户点「一键分类」时词典早就绪了，
    这能减少首条错题的处理延迟。
    """

    done = pyqtSignal(bool, str)

    def run(self) -> None:                 # noqa: D102
        try:
            pipeline.warmup()
            self.done.emit(True, "分词词典与公式映射词典已就绪")
        except Exception as exc:           # noqa: BLE001
            log.exception("预热失败")
            self.done.emit(False, str(exc))


class ClassifyWorker(QThread):
    """批量分类工作线程。"""

    progress = pyqtSignal(int, int, str)          # done, total, 当前错题摘要
    one_done = pyqtSignal(object, object)         # Question, ClassifyResult
    finished_all = pyqtSignal(int, float, list)   # 成功数, 总耗时ms, 失败列表
    failed = pyqtSignal(str)

    def __init__(self, engine: ClassificationEngine, scheme: Scheme,
                 questions: Sequence[Question], parent=None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.scheme = scheme
        self.questions = list(questions)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:                 # noqa: D102
        from app.core.preprocess.media_jobs import media_session
        with media_session():
            self._run_batch()

    def _run_batch(self) -> None:
        import time
        from app.core.preprocess.media_jobs import MediaJobError
        from app.core.preprocess.media_preparation import prepare_question_media
        ok = 0
        errors: List[str] = []
        t0 = time.perf_counter()
        try:
            self.engine.sync_scheme()
        except Exception as exc:           # noqa: BLE001
            self.failed.emit("特征库构建失败：{}".format(exc))
            errors = ["{}：特征库构建失败：{}".format(q.display_text[:24], exc)
                      for q in self.questions]
            self.finished_all.emit(0, (time.perf_counter() - t0) * 1000.0, errors)
            return

        total = len(self.questions)
        initialization_ms = (time.perf_counter() - t0) * 1000.0
        for i, q in enumerate(self.questions, 1):
            if self._stop:
                break
            try:
                item_started = time.perf_counter()
                self.progress.emit(i - 1, total, "自动预处理：" + q.display_text)
                prepared = prepare_question_media(q, should_cancel=lambda: self._stop)
                if self._stop:
                    break
                fb = pipeline.preprocess(q, use_unreviewed_media=True)
                fb.media_elapsed_ms = prepared.elapsed_ms if prepared.recognized else 0.0
                res: ClassifyResult = self.engine.classify(q, fb=fb)
                fb.total_process_ms = ((time.perf_counter() - item_started) * 1000.0
                                       + (initialization_ms if i == 1 else 0.0))
                ClassificationEngine.apply_result(q, res, self.scheme, fb)
                if q.classified:
                    ok += 1
                    self.one_done.emit(q, res)
                else:
                    errors.append("{}：未获得有效分类，请补充题干或标注后重试".format(
                        q.display_text[:24] or "空白错题"))
            except MediaJobError as exc:
                if exc.cancelled:
                    break
                log.warning("错题 %s 媒体预处理失败：%s", q.id, exc)
                errors.append("{}：媒体识别失败：{}".format(q.display_text[:24], exc))
            except Exception as exc:       # noqa: BLE001
                # 单条失败不影响整批：记录后继续（"无崩溃、无数据丢失"）
                log.error("错题 %s 分类失败：%s\n%s", q.id, exc, traceback.format_exc())
                errors.append("{}：{}".format(q.display_text[:24], exc))
            self.progress.emit(i, total, q.display_text)
        self.finished_all.emit(ok, (time.perf_counter() - t0) * 1000.0, errors)


class ImportWorker(QThread):
    """批量导入 + 媒体预处理工作线程。"""

    progress = pyqtSignal(int, int, str)
    finished_all = pyqtSignal(list, str)   # questions, error

    def __init__(self, kind: str, paths: List[str], subject: str = "",
                 qtype: str = "", parent=None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.paths = paths
        self.subject = subject
        self.qtype = qtype
        self.failed_paths = []
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:                 # noqa: D102
        from app.core.dataio import importer
        out: List[Question] = []
        errors = []
        self.failed_paths = []
        for i, p in enumerate(self.paths, 1):
            if self._stop:
                self.failed_paths.extend(self.paths[i - 1:])
                errors.append("已停止，尚未处理 {} 个文件".format(len(self.paths) - i + 1))
                break
            try:
                if self.kind == "txt":
                    out.extend(importer.import_txt(p, self.subject, self.qtype))
                elif self.kind == "excel":
                    row_errors = []
                    out.extend(importer.import_excel(p, self.subject, self.qtype, errors=row_errors))
                    if row_errors:
                        self.failed_paths.append(p)
                        errors.extend("{}：{}".format(p, error) for error in row_errors)
                elif self.kind == "media":
                    out.extend(importer.import_media([p], self.subject, self.qtype))
                else:
                    raise ValueError("不支持的导入方式：{}".format(self.kind))
            except Exception as exc:
                log.exception("文件导入失败：%s", p)
                self.failed_paths.append(p)
                errors.append("{}：{}".format(p, exc))
            self.progress.emit(i, len(self.paths), p)
        self.finished_all.emit(out, "\n".join(errors))
