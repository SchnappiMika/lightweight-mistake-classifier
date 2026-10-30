from __future__ import annotations

from copy import deepcopy

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox,
                             QPlainTextEdit, QProgressBar, QScrollArea, QSplitter,
                             QTabWidget, QVBoxLayout, QWidget)

from app.core.image_review import save_review
from app.core.preprocess.image_ocr import get_engine
from app.ui.widgets import button


class ImageOcrWorker(QThread):
    completed = pyqtSignal(object, str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            self.completed.emit(get_engine().recognize_isolated(self.path), "")
        except Exception as exc:
            self.completed.emit(None, str(exc))


class ImageOcrDialog(QDialog):
    def __init__(self, ctx, question, parent=None):
        super().__init__(parent.window() if parent is not None else None)
        self.ctx = ctx
        self.question = question
        self.record = deepcopy(question.image_ocr)
        self._worker = None
        self._result = None
        self._dirty = False
        self._discard_result = False
        self._busy = False
        self.setWindowTitle("图片文字识别与校对")
        self.resize(1040, 720)
        self.setMinimumSize(760, 520)
        layout = QVBoxLayout(self)
        description = QLabel("一键分类会使用未校对的识别草稿并标记待复核；已有人工确认内容优先。请逐行核对印刷文字和选项，复杂公式、手写文字和图形含义请手动补充。")
        description.setWordWrap(True)
        layout.addWidget(description)
        splitter = QSplitter(Qt.Horizontal)
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("原图缩放"))
        self.zoom = QComboBox()
        self.zoom.addItem("适应宽度", 0.0)
        for text, factor in (("50%", 0.5), ("100%", 1.0), ("150%", 1.5), ("200%", 2.0)):
            self.zoom.addItem(text, factor)
        self.zoom.setCurrentIndex(0)
        zoom_row.addWidget(self.zoom)
        zoom_row.addStretch(1)
        image_layout.addLayout(zoom_row)
        self.scroll = QScrollArea()
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.pixmap = QPixmap(question.media_path)
        self.scroll.setWidget(self.image)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.zoom.currentIndexChanged.connect(self._zoom_image)
        self._zoom_image()
        image_layout.addWidget(self.scroll)
        splitter.addWidget(image_panel)
        tabs = QTabWidget()
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("点击“开始识别”，再对照原图修改文字。应用前不会替换已有题干。")
        self.editor.textChanged.connect(self._edited)
        self.original = QPlainTextEdit()
        self.original.setReadOnly(True)
        tabs.addTab(self.editor, "校对文本")
        tabs.addTab(self.original, "原始识别与逐行分数")
        splitter.addWidget(tabs)
        splitter.setSizes([520, 520])
        layout.addWidget(splitter, 1)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.PlainText)
        layout.addWidget(self.message)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        row = QHBoxLayout()
        self.btn_recognize = button("开始识别 / 重试", "primary", "", self._start)
        self.btn_save = button("保存校对草稿", "", "save", self._save_draft)
        self.btn_apply = button("应用为题干", "accent", "", self._apply)
        self.btn_cancel = button("取消本次识别", "", "", self._cancel)
        self.btn_close = button("关闭", "", "", self.reject)
        for widget in (self.btn_recognize, self.btn_save, self.btn_apply, self.btn_cancel, self.btn_close):
            row.addWidget(widget)
        layout.addLayout(row)
        if self.record:
            self._show_record()
        else:
            self.message.setText("尚无文字识别记录。组件未准备时，请先双击项目内“安装图片识别.bat”。")
        self._set_busy(False)

    def _zoom_image(self):
        if self.pixmap.isNull():
            self.image.setText("图片无法预览，请检查附件文件。")
            self.image.adjustSize()
            return
        factor = self.zoom.currentData()
        if not factor:
            factor = max(1, self.scroll.viewport().width() - 8) / self.pixmap.width()
        scaled = self.pixmap.scaled(max(1, int(self.pixmap.width() * factor)),
                                    max(1, int(self.pixmap.height() * factor)),
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image.setPixmap(scaled)
        self.image.resize(scaled.size())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "scroll") and self.zoom.currentData() == 0.0:
            self._zoom_image()

    def _edited(self):
        self._dirty = True

    def _review_status(self):
        if self.record.confirmed:
            if self.record.draft.strip() != self.question.raw_text.strip():
                return "已有题干仍用于分类；当前草稿尚未应用，点击“应用为题干”后替换题干。"
            return "已人工校对并应用为题干；一键分类使用已保存题干。"
        if self.question.classified and "image" in (self.question.feature or {}).get("unreviewed_media", []):
            return "已使用未校对草稿分类，待复核；修改并保存草稿会清除旧分类，请重新分类。"
        if not self.record.draft.strip():
            return "当前草稿为空；可填写校对内容或重新识别。下次一键分类将重新尝试识别附件。"
        return "草稿尚未人工确认；下次一键分类会使用草稿并标记待复核，点击“应用为题干”可确认并替换题干。"

    def _show_record(self):
        self.editor.blockSignals(True)
        self.editor.setPlainText(self.record.draft)
        self.editor.blockSignals(False)
        lines = ["原始识别文字：", self.record.text, "", "逐行模型分数（不是准确率）："]
        for line in self.record.lines:
            lines.append("{:.0%}  {}".format(line.get("score", 0), line.get("text", "")))
        self.original.setPlainText("\n".join(lines))
        uncertain = sum(line.get("score", 0) < 0.8 for line in self.record.lines)
        self.message.setText("{} 识别 {} 行，低分行 {} 行，耗时 {:.2f} 秒。请核对所有文字。".format(
            self._review_status(),
            len(self.record.lines), uncertain, self.record.elapsed_ms / 1000))

    def _set_busy(self, busy):
        self._busy = busy
        self.progress.setVisible(busy)
        self.editor.setEnabled(not busy and self.record is not None)
        self.btn_recognize.setEnabled(not busy)
        self.btn_save.setEnabled(not busy and self.record is not None)
        self.btn_apply.setEnabled(not busy and self.record is not None)
        self.btn_close.setEnabled(not busy)
        self.btn_cancel.setVisible(busy)
        self.btn_cancel.setEnabled(busy)

    def _save(self, apply=False):
        if self.record is None:
            return False
        updated = deepcopy(self.record)
        updated.draft = self.editor.toPlainText()
        try:
            save_review(self.ctx.repo, self.question, updated, apply)
        except (OSError, ValueError) as exc:
            self.message.setText("保存失败，已有题干和分类保持原样：{}".format(exc))
            self._dirty = True
            return False
        self.record = deepcopy(self.question.image_ocr)
        self._dirty = False
        self.ctx.questions_changed.emit()
        return True

    def _save_draft(self):
        was_classified = self.question.classified
        if self._save():
            message = "校对草稿已保存，已有题干保持原样。" + self._review_status()
            if was_classified and not self.question.classified:
                message = "校对草稿已保存；原分类使用的草稿已变更，旧分类已清除，请重新分类。" + self._review_status()
            self.message.setText(message)

    def _apply(self):
        text = self.editor.toPlainText().strip()
        if self.question.raw_text.strip() and self.question.raw_text.strip() != text:
            if QMessageBox.question(
                    self, "确认替换题干", "应用校对文本将替换现有题干，并清除旧分类，是否继续？",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
        if self._save(apply=True):
            self.message.setText("校对文本已应用为题干并保存。关闭此窗口后，可进入“一键分类与修正”进行分类。")
            self.ctx.toast.emit("图片文字已校对保存，旧分类已作废，请重新分类", "ok")

    def _start(self):
        if self._busy:
            return
        if self._dirty and not self._save():
            return
        if self.record and QMessageBox.question(
                self, "重新识别", "重试将生成新的识别文字和草稿，已有题干保持原样。是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        if not self.ctx.begin_task("图片识别"):
            return
        self._result = None
        self._discard_result = False
        self._set_busy(True)
        self.message.setText("正在本地识别，首次加载模型可能较慢。可取消本次结果，等待当前识别结束后返回。")
        self._worker = ImageOcrWorker(self.question.media_path, self)
        self._worker.completed.connect(self._receive)
        self._worker.finished.connect(self._finished)
        self._worker.start()

    def _receive(self, record, error):
        self._result = record, error

    def _finished(self):
        try:
            record, error = self._result or (None, "识别未正常结束，请检查组件后重试。")
            if self._discard_result:
                self.message.setText("已取消本次结果，原有题干和识别记录保持原样。")
            elif error:
                self.message.setText("识别失败，原有内容保持原样：{}".format(error))
            else:
                self.record = record
                self._show_record()
                self._dirty = True
                if self._save():
                    self._show_record()
        finally:
            self._set_busy(False)
            self.ctx.end_task("图片识别")

    def _cancel(self):
        self._discard_result = True
        self.btn_cancel.setEnabled(False)
        self.message.setText("已请求取消，将丢弃本次结果；请等待当前识别结束，不会覆盖已有内容。")

    def reject(self):
        if self._busy:
            return
        if self._dirty:
            answer = QMessageBox.question(
                self, "保存校对草稿", "校对内容尚未保存，是否保存草稿后关闭？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if answer == QMessageBox.Cancel or (answer == QMessageBox.Save and not self._save()):
                return
        super().reject()

    def closeEvent(self, event):
        if self._busy:
            event.ignore()
            return
        self.reject()
        event.setAccepted(self.result() == QDialog.Rejected and not self.isVisible())
