# -*- coding: utf-8 -*-
"""公式图片与听力音频的本地识别、校对和确认。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                             QProgressBar, QSplitter, QVBoxLayout)

from app.ui.widgets import button


class MediaReviewWorker(QThread):
    completed = pyqtSignal(object, str)

    def __init__(self, kind, path, parent=None, recognition_options=None):
        super().__init__(parent)
        self.kind = kind
        self.path = path
        self.recognition_options = deepcopy(recognition_options)

    def run(self):
        from app.core.preprocess.media_jobs import MediaJobError, recognize_media
        try:
            record = recognize_media(self.path, self.kind, self.recognition_options,
                                     should_cancel=self.isInterruptionRequested)
            self.completed.emit(record, "")
        except MediaJobError as exc:
            self.completed.emit(exc.record, str(exc))
        except Exception as exc:
            self.completed.emit(None, str(exc))


class MediaReviewDialog(QDialog):
    """展示自动识别的原文、草稿和人工确认状态。"""

    def __init__(self, ctx, question, kind, parent=None):
        super().__init__(parent.window() if parent is not None else None)
        self.ctx = ctx
        self.question = question
        self.kind = kind
        self._worker = None
        self._result = None
        self._discard_result = False
        self._dirty = False
        self._busy = False
        self.setWindowTitle("{}识别与校对".format("公式图片" if kind == "formula" else "听力音频"))
        self.resize(980, 680)
        self.setMinimumSize(700, 500)

        layout = QVBoxLayout(self)
        description = QLabel(self._description())
        description.setWordWrap(True)
        layout.addWidget(description)
        splitter = QSplitter()
        if kind == "formula":
            from app.ui.formula_crop_dialog import FormulaCropPanel
            self.formula_crop = FormulaCropPanel(question.media_path)
            splitter.addWidget(self.formula_crop)
        else:
            audio = QLabel("音频附件：{}\n{}".format(
                Path(question.media_path).name,
                "请先试听原音，再核对转写文本。"))
            audio.setWordWrap(True)
            splitter.addWidget(audio)
        tabs = QSplitter()
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("识别后请逐字核对，再保存草稿或确认应用。")
        self.editor.textChanged.connect(self._edited)
        self.original = QPlainTextEdit()
        self.original.setReadOnly(True)
        tabs.addWidget(self.editor)
        tabs.addWidget(self.original)
        splitter.addWidget(tabs)
        splitter.setSizes([440, 520])
        layout.addWidget(splitter, 1)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.PlainText)
        layout.addWidget(self.message)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        actions = QHBoxLayout()
        self.btn_recognize = button("开始识别 / 重试", "primary", "", self._start)
        self.btn_save = button("保存校对草稿", "", "save", self._save_draft)
        self.btn_apply = button("确认并应用", "accent", "", self._apply)
        self.btn_cancel = button("取消本次识别", "", "", self._cancel)
        self.btn_close = button("关闭", "", "", self.reject)
        for widget in (self.btn_recognize, self.btn_save, self.btn_apply,
                       self.btn_cancel, self.btn_close):
            actions.addWidget(widget)
        layout.addLayout(actions)
        self.record = deepcopy(getattr(question, "formula_ocr" if kind == "formula"
                                       else "audio_transcript", None))
        if self.record:
            if self.kind == "formula" and self.record.recognition_options:
                self.formula_crop.set_options(self.record.recognition_options)
            self._show_record()
        else:
            self.message.setText("尚无识别记录。请准备本地模型后再开始识别。")
        self._set_busy(False)

    def _description(self):
        if self.kind == "formula":
            return "一键分类会使用未校对的公式识别草稿并标记待复核；已有人工确认内容优先。请对照原图校对，点击“确认并应用”保存确认内容。"
        return "一键分类会使用未校对的听力转写草稿并标记待复核；已有人工确认内容优先。请试听原音校对，点击“确认并应用”保存确认内容。"

    def _review_status(self):
        if self.record.confirmed:
            applied = self.record.applied_text or self.record.draft or self.record.text
            if self.record.draft.strip() != applied.strip():
                return "已有人工确认内容仍用于分类；当前草稿尚未应用，点击“确认并应用”后替换确认内容。"
            return "已人工确认；一键分类优先使用确认内容。修改后点击“确认并应用”才会替换确认内容。"
        if self.question.classified and self.kind in (self.question.feature or {}).get("unreviewed_media", []):
            return "已使用未校对草稿分类，待复核；修改并保存草稿会清除旧分类，请重新分类。"
        if not self.record.draft.strip():
            return "当前草稿为空；可填写校对内容或重新识别。下次一键分类将重新尝试识别附件。"
        return "草稿尚未人工确认；下次一键分类会使用草稿并标记待复核，也可先点击“确认并应用”。"

    def _show_record(self):
        draft = self.record.draft
        self.editor.blockSignals(True)
        self.editor.setPlainText(draft)
        self.editor.blockSignals(False)
        original = getattr(self.record, "text", "")
        engine = getattr(self.record, "engine", "")
        elapsed = float(getattr(self.record, "elapsed_ms", 0.0) or 0.0)
        error = getattr(self.record, "error", "")
        self.original.setPlainText("原始识别结果：\n{}\n\n引擎：{}\n耗时：{:.2f} 秒{}".format(
            original, engine or "未知", elapsed / 1000.0,
            "\n错误：{}".format(error) if error else ""))
        self.message.setText(self._review_status())

    def _edited(self):
        self._dirty = True

    def _set_busy(self, busy):
        self._busy = busy
        self.progress.setVisible(busy)
        self.editor.setEnabled(not busy)
        if self.kind == "formula":
            self.formula_crop.setEnabled(not busy)
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
            from app.core.preprocess.media_recognition import save_media_review
            save_media_review(self.ctx.repo, self.question, updated, self.kind, apply=apply)
        except Exception as exc:
            self.message.setText("保存失败，已有内容保持原样：{}".format(exc))
            return False
        self.record = deepcopy(getattr(self.question, "formula_ocr" if self.kind == "formula"
                                      else "audio_transcript"))
        self._dirty = False
        self.ctx.questions_changed.emit()
        return True

    def _save_draft(self):
        was_classified = self.question.classified
        if self._save(False):
            message = "校对草稿已保存。" + self._review_status()
            if was_classified and not self.question.classified:
                message = "校对草稿已保存；原分类使用的草稿已变更，旧分类已清除，请重新分类。" + self._review_status()
            self.message.setText(message)

    def _apply(self):
        if not self.editor.toPlainText().strip():
            self.message.setText("确认内容不能为空，请先填写或重新识别。")
            return
        if self._save(True):
            self.message.setText("已确认并应用。关闭窗口后请重新分类，旧分类已作废。")
            self.ctx.toast.emit("{}识别结果已确认，请重新分类".format(
                "公式" if self.kind == "formula" else "听力"), "ok")

    def _start(self):
        if self._busy:
            return
        if self._dirty and not self._save(False):
            return
        if self.record and QMessageBox.question(
                self, "重新识别", "重试会替换当前草稿，已确认内容不会直接覆盖。是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        task = "公式识别" if self.kind == "formula" else "听力转写"
        if not self.ctx.begin_task(task):
            return
        self._result = None
        self._discard_result = False
        self._set_busy(True)
        self.message.setText("正在本地处理，不会联网下载模型。可取消并结束本次识别。")
        recognition_options = self.formula_crop.options() if self.kind == "formula" else None
        self._worker = MediaReviewWorker(self.kind, self.question.media_path, self,
                                         recognition_options=recognition_options)
        self._worker.completed.connect(self._receive)
        self._worker.finished.connect(self._finished)
        self._worker.start()

    def _receive(self, record, error):
        self._result = record, error

    def _finished(self):
        task = "公式识别" if self.kind == "formula" else "听力转写"
        try:
            record, error = self._result or (None, "识别未正常结束，请检查模型和附件。")
            if self._discard_result:
                self.message.setText("已取消本次结果，原有内容保持原样。")
            elif error or record is None or getattr(record, "error", ""):
                detail = error or getattr(record, "error", "") or "无结果"
                self.message.setText("识别失败，原有内容保持原样：{}".format(detail))
            else:
                previous_record = self.record
                previous_editor = self.editor.toPlainText()
                previous_original = self.original.toPlainText()
                self.record = record
                self._show_record()
                self._dirty = True
                if self._save(False):
                    self._show_record()
                else:
                    self.record = previous_record
                    self.editor.blockSignals(True)
                    self.editor.setPlainText(previous_editor)
                    self.editor.blockSignals(False)
                    self.original.setPlainText(previous_original)
                    self._dirty = False
        finally:
            self._set_busy(False)
            self.ctx.end_task(task)

    def _cancel(self):
        if not self._busy:
            return
        self._discard_result = True
        if self._worker is not None:
            self._worker.requestInterruption()
        self.btn_cancel.setEnabled(False)
        self.message.setText("正在结束本次识别进程；原题和已有结果保持不变。")

    def reject(self):
        if self._busy:
            return
        if self._dirty:
            answer = QMessageBox.question(
                self, "保存校对草稿", "校对内容尚未保存，是否保存草稿后关闭？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if answer == QMessageBox.Cancel or (answer == QMessageBox.Save and not self._save(False)):
                return
        super().reject()

    def closeEvent(self, event):
        if self._busy:
            event.ignore()
            return
        self.reject()
        event.setAccepted(not self.isVisible())
