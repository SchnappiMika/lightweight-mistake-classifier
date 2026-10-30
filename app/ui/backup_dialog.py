from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QFileDialog, QLabel, QMessageBox,
                             QProgressBar, QVBoxLayout)

from app import config
from app.core.dataio import backup
from app.ui.widgets import button


class BackupWorker(QThread):
    completed = pyqtSignal(bool, str)

    def __init__(self, repo, path, restore, parent=None):
        super().__init__(parent)
        self.repo = repo
        self.path = path
        self.restore = restore

    def run(self):
        try:
            action = backup.restore_backup if self.restore else backup.create_backup
            self.completed.emit(True, action(self.repo, self.path))
        except Exception as exc:
            self.completed.emit(False, str(exc))


class BackupDialog(QDialog):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._worker = None
        self._result = None
        self.setWindowTitle("数据备份与恢复")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        description = QLabel(
            "备份包含：错题、分类体系、人工修正、学习数据、设置、附件及导出文件。\n"
            "恢复会替换当前数据，并先自动保存一份恢复前备份。\n"
            "备份不包含 Python 运行环境、运行日志和历史备份文件。")
        description.setWordWrap(True)
        layout.addWidget(description)
        location = QLabel("当前数据目录：{}".format(config.WORKSPACE_DIR))
        location.setWordWrap(True)
        layout.addWidget(location)
        self.btn_backup = button("创建完整备份", "primary", "save", self._backup)
        self.btn_restore = button("从备份恢复", "", "folder", self._restore)
        self.btn_close = button("关闭", "", "", self.reject)
        for widget in (self.btn_backup, self.btn_restore):
            layout.addWidget(widget)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.message = QLabel("请选择操作。备份完成后，可以将 ZIP 文件复制到其他电脑。")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        layout.addWidget(self.btn_close)

    def _backup(self):
        default = config.WORKSPACE_DIR / "backups" / "错题备份_{}.zip".format(
            datetime.now().strftime("%Y%m%d_%H%M%S"))
        path, _ = QFileDialog.getSaveFileName(self, "保存完整备份", str(default), "错题备份 (*.zip)")
        if path:
            self._start(str(Path(path).with_suffix(".zip")), False)

    def _restore(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择错题软件备份", str(config.WORKSPACE_DIR / "backups"), "错题备份 (*.zip)")
        if not path:
            return
        answer = QMessageBox.question(
            self, "确认恢复数据",
            "将使用备份替换当前错题、体系、学习数据、设置、附件和导出文件。\n"
            "备份会先校验，当前数据会自动保存到工作区 backups 文件夹。\n\n"
            "确定从以下文件恢复吗？\n{}".format(path),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self._start(path, True)

    def _start(self, path, restore):
        task = "恢复" if restore else "备份"
        if not self.ctx.begin_task(task):
            return
        self._result = None
        for widget in (self.btn_backup, self.btn_restore, self.btn_close):
            widget.setEnabled(False)
        self.progress.show()
        self.message.setText("正在{}数据，请等待完成后再关闭窗口。".format(task))
        self._worker = BackupWorker(self.ctx.repo, path, restore, self)
        self._worker.completed.connect(self._receive_result)
        self._worker.finished.connect(self._finished)
        self._worker.start()

    def _receive_result(self, ok, message):
        self._result = (ok, message)

    def _finished(self):
        restore = self._worker.restore
        ok, message = self._result or (False, "任务未正常完成，请查看工作区数据及运行日志。")
        try:
            if ok and restore:
                self.ctx.workspace_restored()
            if ok:
                text = ("恢复完成。恢复前的数据备份保存在：\n" if restore
                        else "完整备份已保存：\n") + message
            else:
                text = "{}失败：{}".format("恢复" if restore else "备份", message)
            self.message.setText(text)
            self.ctx.toast.emit(text, "ok" if ok else "error")
        finally:
            self.ctx.end_task("恢复" if restore else "备份")
            self.progress.hide()
            for widget in (self.btn_backup, self.btn_restore, self.btn_close):
                widget.setEnabled(True)

    def reject(self):
        if self._worker is None or not self._worker.isRunning():
            super().reject()

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            event.ignore()
        else:
            super().closeEvent(event)
