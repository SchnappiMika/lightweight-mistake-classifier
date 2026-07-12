from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QListWidget, QFileDialog,
    QComboBox, QSplitter, QFrame, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal
import json


class ImportPage(QWidget):
    """Multimodal question import page"""

    questions_imported = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.questions = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Multimodal Question Import & Preview")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title)

        # Import methods
        import_group = QGroupBox("Import Methods")
        import_layout = QHBoxLayout(import_group)

        # Manual input
        manual_btn = QPushButton("Manual Input")
        manual_btn.clicked.connect(self._on_manual_import)
        import_layout.addWidget(manual_btn)

        # TXT import
        txt_btn = QPushButton("Import TXT")
        txt_btn.clicked.connect(self._on_import_txt)
        import_layout.addWidget(txt_btn)

        # Excel import
        excel_btn = QPushButton("Import Excel")
        excel_btn.clicked.connect(self._on_import_excel)
        import_layout.addWidget(excel_btn)

        # Clear
        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        clear_btn.clicked.connect(self._on_clear)
        import_layout.addWidget(clear_btn)

        layout.addWidget(import_group)

        # Question list
        list_label = QLabel("Imported Questions:")
        list_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(list_label)

        self.question_list = QListWidget()
        layout.addWidget(self.question_list)

    def _on_manual_import(self):
        dialog = ManualInputDialog()
        if dialog.exec_():
            text, subject = dialog.get_data()
            if text:
                self.questions.append({
                    "text": text,
                    "subject": subject,
                    "type": "text"
                })
                self._refresh_list()

    def _on_import_txt(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Select TXT File", "", "Text Files (*.txt)")
        if fpath:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                items = [l.strip() for l in content.split("\n") if l.strip()]
                for item in items:
                    self.questions.append({
                        "text": item,
                        "subject": "auto",
                        "type": "text"
                    })
                self._refresh_list()
                QMessageBox.information(self, "Success", f"Imported {len(items)} questions")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    def _on_import_excel(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Select Excel File", "", "Excel Files (*.xlsx *.xls)")
        if fpath:
            try:
                import pandas as pd
                df = pd.read_excel(fpath)
                text_col = df.columns[0]
                for _, row in df.iterrows():
                    text = str(row[text_col]).strip()
                    if text and text != "nan":
                        self.questions.append({
                            "text": text,
                            "subject": "auto",
                            "type": "text"
                        })
                self._refresh_list()
                QMessageBox.information(self, "Success", f"Imported {len(df)} questions")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Excel import failed: {e}")

    def _on_clear(self):
        self.questions.clear()
        self._refresh_list()

    def _refresh_list(self):
        self.question_list.clear()
        for i, q in enumerate(self.questions):
            text = q["text"][:60] + "..." if len(q["text"]) > 60 else q["text"]
            self.question_list.addItem(f"[{i+1}] {text}")
        self.questions_imported.emit(self.questions)

    def get_questions(self):
        return self.questions


from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QComboBox, QDialogButtonBox


class ManualInputDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manual Input")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Question Text:"))
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Enter question text here...")
        layout.addWidget(self.text_edit)

        layout.addWidget(QLabel("Subject (optional):"))
        self.subject_combo = QComboBox()
        self.subject_combo.addItems(["auto", "\u9ad8\u7b49\u6570\u5b66", "\u5927\u5b66\u7269\u7406", "\u5927\u5b66\u82f1\u8bed"])
        layout.addWidget(self.subject_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self):
        return self.text_edit.toPlainText().strip(), self.subject_combo.currentText()