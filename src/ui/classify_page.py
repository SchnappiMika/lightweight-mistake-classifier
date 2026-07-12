from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QGroupBox, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal
import sys, os, json


class ClassifyPage(QWidget):
    results_ready = pyqtSignal(list)
    def __init__(self, classifier=None):
        super().__init__()
        self.classifier = classifier
        self.questions = []
        self.results = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("One-Click Classification & Correction")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by subject:"))
        self.subject_filter = QComboBox()
        self.subject_filter.addItems(["All", "\u9ad8\u7b49\u6570\u5b66", "\u5927\u5b66\u7269\u7406", "\u5927\u5b66\u82f1\u8bed"])
        filter_layout.addWidget(self.subject_filter)
        filter_layout.addStretch()
        self.classify_btn = QPushButton("Classify All")
        self.classify_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 8px 20px; font-size: 14px;")
        self.classify_btn.clicked.connect(self._on_classify)
        filter_layout.addWidget(self.classify_btn)
        layout.addLayout(filter_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["#", "Question", "Subject", "Classification", "Score"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status_label = QLabel("Ready - import questions first")
        self.status_label.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(self.status_label)

    def set_questions(self, questions):
        self.questions = questions
        self.results = []
        self._update_table()

    def set_classifier(self, classifier):
        self.classifier = classifier

    def _on_classify(self):
        if not self.questions:
            QMessageBox.warning(self, "No Data", "Please import questions first")
            return
        if not self.classifier:
            QMessageBox.warning(self, "Error", "Classifier not initialized")
            return
        self.classify_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setMaximum(len(self.questions))
        self.results = []
        for i, q in enumerate(self.questions):
            try:
                subj = None if q.get("subject") == "auto" else q["subject"]
                r = self.classifier.classify_text(q["text"], subj)
                cat, score = r["final_result"]
                self.results.append({"text": q["text"], "subject": r["detected_subject"],
                    "category": cat, "score": score, "corrected": False})
            except Exception as e:
                self.results.append({"text": q["text"], "subject": "error",
                    "category": str(e), "score": 0.0, "corrected": False})
            self.progress.setValue(i + 1)
        self._update_table()
        self.classify_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("Classification complete: " + str(len(self.results)) + " questions")
        self.results_ready.emit(self.results)

    def _update_table(self):
        self.table.setRowCount(len(self.results))
        for i, r in enumerate(self.results):
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            txt = r["text"][:40] + "..." if len(r["text"]) > 40 else r["text"]
            self.table.setItem(i, 1, QTableWidgetItem(txt))
            self.table.setItem(i, 2, QTableWidgetItem(r["subject"]))
            self.table.setItem(i, 3, QTableWidgetItem(r["category"]))
            score_str = "{:.1%}".format(r["score"])
            self.table.setItem(i, 4, QTableWidgetItem(score_str))

    def get_results(self):
        return self.results