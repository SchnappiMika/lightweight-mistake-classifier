"""Batch recognition drafts with an explicit per-question review queue."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QAbstractItemView, QDialog, QHBoxLayout, QLabel,
                             QProgressBar, QTableWidget, QTableWidgetItem, QVBoxLayout)

from app.core.preprocess.media_preparation import media_kind
from app.core.preprocess.image_ocr import file_hash
from app.ui.widgets import button


def current_record(question, kind):
    return getattr(question, {"image": "image_ocr", "formula": "formula_ocr",
                              "audio": "audio_transcript"}[kind], None)


def record_matches(question, kind):
    record = current_record(question, kind)
    if record is None:
        return False
    try:
        return bool(record.media_hash and record.media_hash == file_hash(Path(question.media_path)))
    except OSError:
        return False


class MediaBatchWorker(QThread):
    item_done = pyqtSignal(str, str, object, str)
    progress = pyqtSignal(int, int)

    def __init__(self, questions, parent=None):
        super().__init__(parent)
        self.questions = deepcopy(questions)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        from app.core.preprocess.media_jobs import media_session
        with media_session():
            self._run_batch()

    def _run_batch(self):
        from app.core.preprocess.media_jobs import MediaJobError, recognize_media
        for index, question in enumerate(self.questions, 1):
            if self._stop:
                break
            kind = media_kind(question)
            if not kind:
                continue
            try:
                previous = current_record(question, kind)
                options = previous.recognition_options if kind == "formula" and previous else None
                record = recognize_media(question.media_path, kind, options=options,
                                         should_cancel=lambda: self._stop)
                self.item_done.emit(question.id, kind, record, "")
            except MediaJobError as exc:
                if exc.cancelled:
                    break
                self.item_done.emit(question.id, kind, None, str(exc))
            except Exception as exc:
                self.item_done.emit(question.id, kind, None, str(exc))
            self.progress.emit(index, len(self.questions))


class MediaBatchDialog(QDialog):
    """Recognize all pending media, then open the existing correction dialogs."""

    def __init__(self, ctx, questions, parent=None):
        super().__init__(parent.window() if parent is not None else None)
        self.ctx = ctx
        self.ids = [q.id for q in questions if media_kind(q)]
        self._worker = None
        self._stop_requested = False
        self._errors = {}
        self._match_cache = {}
        self.setWindowTitle("批量媒体识别与校对")
        self.resize(900, 560)
        layout = QVBoxLayout(self)
        description = QLabel("先批量生成识别草稿，再选择题目校对。识别成功会逐条保存；失败可单独重试。原图与上次确认内容保留。")
        description.setWordWrap(True)
        layout.addWidget(description)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["题目 / 附件", "识别类型", "状态"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 390)
        self.table.setColumnWidth(1, 100)
        layout.addWidget(self.table, 1)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)
        actions = QHBoxLayout()
        self.btn_run = button("识别缺失 / 失败项", "primary", "", self.start_pending)
        self.btn_retry = button("重新识别选中项", "", "", self.retry_selected)
        self.btn_review = button("校对选中项", "", "", self.review_selected)
        self.btn_stop = button("停止识别", "", "", self.stop)
        self.btn_close = button("完成并关闭", "", "", self.accept)
        for control in (self.btn_run, self.btn_retry, self.btn_review, self.btn_stop, self.btn_close):
            actions.addWidget(control)
        layout.addLayout(actions)
        self.refresh()
        self._set_busy(False)

    def questions(self):
        return [q for qid in self.ids if (q := self.ctx.repo.question_by_id(qid)) is not None]

    def refresh(self):
        selected = self.selected_ids() if self.table.rowCount() else set()
        self.table.setRowCount(0)
        pending = 0
        for question in self.questions():
            kind = media_kind(question)
            record = current_record(question, kind)
            if question.id in self._errors:
                state = "失败：" + self._errors[question.id]
            elif record is None:
                state = "尚未识别"
            elif not self._record_matches(question, kind):
                state = "附件已变化，请重新识别"
            else:
                applied = question.raw_text if kind == "image" else (record.applied_text or record.text)
                state = ("已确认" if record.confirmed and record.draft.strip() == applied.strip()
                         else "已有草稿，待校对")
            if state != "已确认":
                pending += 1
            row = self.table.rowCount()
            self.table.insertRow(row)
            for column, text in enumerate((question.display_text, {"image": "图片文字", "formula": "公式", "audio": "听力"}[kind], state)):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                if column == 0:
                    item.setData(Qt.UserRole, question.id)
                self.table.setItem(row, column, item)
            if question.id in selected:
                self.table.selectRow(row)
        self.message.setText("共 {} 个媒体附件，{} 个尚待确认；双击题目或点击校对选中项可逐题处理。".format(len(self.ids), pending))
        if not getattr(self, "_double_click_connected", False):
            self.table.cellDoubleClicked.connect(lambda *_: self.review_selected())
            self._double_click_connected = True

    def selected_ids(self):
        return {self.table.item(index.row(), 0).data(Qt.UserRole)
                for index in self.table.selectionModel().selectedRows()}

    def start_pending(self):
        self._start([q for q in self.questions() if not self._record_matches(q, media_kind(q))
                     or q.id in self._errors])

    def _record_matches(self, question, kind):
        record = current_record(question, kind)
        if record is None:
            return False
        try:
            path = Path(question.media_path)
            stat = path.stat()
            key = (question.id, str(path), stat.st_size, stat.st_mtime_ns, record.media_hash)
        except OSError:
            return False
        if key not in self._match_cache:
            self._match_cache[key] = record_matches(question, kind)
        return self._match_cache[key]

    def retry_selected(self):
        selected = self.selected_ids()
        self._start([q for q in self.questions() if q.id in selected])

    def _start(self, questions):
        if self._worker is not None and self._worker.isRunning():
            return
        if not questions:
            self.message.setText("没有待识别项。请选择已有草稿进行校对，或选中题目重新识别。")
            return
        if not self.ctx.begin_task("批量媒体识别"):
            return
        self._stop_requested = False
        self._set_busy(True)
        self.progress.setRange(0, len(questions))
        self.progress.setValue(0)
        self._worker = MediaBatchWorker(questions, self)
        self._worker.item_done.connect(self._save_result)
        self._worker.progress.connect(lambda done, _total: self.progress.setValue(done))
        self._worker.finished.connect(self._finished)
        self._worker.start()

    def _save_result(self, qid, kind, record, error):
        question = self.ctx.repo.question_by_id(qid)
        if question is None:
            return
        if error or record is None:
            self._errors[qid] = error or "识别没有返回结果"
        else:
            try:
                if kind == "image":
                    from app.core.image_review import save_review
                    # Retrying must not silently withdraw the user's last confirmation.
                    if question.image_ocr and question.image_ocr.confirmed:
                        record.confirmed = True
                    save_review(self.ctx.repo, question, record, apply=False)
                else:
                    from app.core.preprocess.media_recognition import save_media_review
                    save_media_review(self.ctx.repo, question, record, kind, apply=False)
                self._errors.pop(qid, None)
            except Exception as exc:
                self._errors[qid] = "保存失败：{}".format(exc)
                self.stop()
        self.refresh()

    def _finished(self):
        self.ctx.end_task("批量媒体识别")
        self._set_busy(False)
        self.ctx.questions_changed.emit()
        self.refresh()
        if self._stop_requested:
            self.message.setText("已停止识别；已完成的草稿已逐条保存，可校对或继续识别。")

    def _set_busy(self, busy):
        for control in (self.btn_run, self.btn_retry, self.btn_review, self.btn_close):
            control.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)
        self.progress.setVisible(busy)

    def stop(self):
        if self._worker is not None and self._worker.isRunning():
            self._stop_requested = True
            self._worker.stop()
            self.message.setText("正在取消当前识别并停止后续任务；已完成结果保留。")

    def review_selected(self):
        if self._worker is not None and self._worker.isRunning():
            return
        selected = self.selected_ids()
        for question in self.questions():
            if question.id not in selected:
                continue
            kind = media_kind(question)
            if kind == "image":
                from app.ui.image_ocr_dialog import ImageOcrDialog
                dialog = ImageOcrDialog(self.ctx, question, self)
            else:
                from app.ui.media_review_dialog import MediaReviewDialog
                dialog = MediaReviewDialog(self.ctx, question, kind, self)
            dialog.exec_()
        self.refresh()

    def done(self, result):
        if self._worker is not None and self._worker.isRunning():
            self.stop()
            return
        super().done(result)
