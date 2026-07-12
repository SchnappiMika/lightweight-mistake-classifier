from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox, QFileDialog, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt
import os


class ExportPage(QWidget):
    def __init__(self):
        super().__init__()
        self.results = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("导出结果")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)

        opts = QGroupBox("导出选项")
        ol = QVBoxLayout(opts)
        self.by_subject = QCheckBox("按学科分表")
        self.by_subject.setChecked(True)
        ol.addWidget(self.by_subject)
        self.by_category = QCheckBox("按分类分表")
        ol.addWidget(self.by_category)
        layout.addWidget(opts)

        layout.addWidget(QLabel("预览:"))
        self.preview = QTableWidget()
        self.preview.setColumnCount(4)
        self.preview.setHorizontalHeaderLabels(["题目", "学科", "分类", "匹配度"])
        self.preview.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.preview)

        bl = QHBoxLayout()
        bl.addStretch()
        self.exp_btn = QPushButton("导出为 Excel")
        self.exp_btn.setStyleSheet("background-color: #2980b9; color: white; padding: 10px 25px; font-size: 14px;")
        self.exp_btn.clicked.connect(self._on_export)
        bl.addWidget(self.exp_btn)
        layout.addLayout(bl)

    def set_results(self, results):
        self.results = results
        self._update_preview()

    def _update_preview(self):
        self.preview.setRowCount(len(self.results))
        for i, r in enumerate(self.results):
            txt = r["text"][:30] + "..." if len(r["text"]) > 30 else r["text"]
            self.preview.setItem(i, 0, QTableWidgetItem(txt))
            self.preview.setItem(i, 1, QTableWidgetItem(r["subject"]))
            self.preview.setItem(i, 2, QTableWidgetItem(r["category"]))
            self.preview.setItem(i, 3, QTableWidgetItem("{:.1%}".format(r["score"])))

    def _on_export(self):
        if not self.results:
            QMessageBox.warning(self, "No Data", "No results to export")
            return
        fp, _ = QFileDialog.getSaveFileName(self, "保存为", "results.xlsx", "Excel Files (*.xlsx)")
        if fp:
            try:
                import pandas as pd
                df = pd.DataFrame(self.results)
                with pd.ExcelWriter(fp, engine="openpyxl") as w:
                    if self.by_subject.isChecked():
                        for subj in df["subject"].unique():
                            sdf = df[df["subject"] == subj]
                            sdf.to_excel(w, sheet_name=str(subj)[:31], index=False)
                    else:
                        df.to_excel(w, sheet_name="All Results", index=False)
                QMessageBox.information(self, "Success", "Exported to: " + fp)
            except Exception as e:
                QMessageBox.warning(self, "Error", "Export failed: " + str(e))