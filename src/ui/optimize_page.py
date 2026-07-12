from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QGroupBox, QMessageBox, QTextEdit, QSplitter, QFrame)
from PyQt5.QtCore import Qt
import os, json


class OptimizePage(QWidget):
    def __init__(self, classifier=None):
        super().__init__()
        self.classifier = classifier
        self.results = []
        self.feedback_history = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        lbl = QLabel("Optimization")
        lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(lbl)

        splitter = QSplitter(Qt.Vertical)

        # Top: results table with correction
        top = QFrame()
        tl = QVBoxLayout(top)
        tl.addWidget(QLabel("Results:"))
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["#", "Question", "Category", "Priority", "Correct", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        tl.addWidget(self.table)

        # Correction tools
        ct = QHBoxLayout()
        ct.addWidget(QLabel("Correct to:"))
        self.correct_combo = QComboBox()
        self.correct_combo.setEditable(True)
        ct.addWidget(self.correct_combo)
        self.apply_btn = QPushButton("应用修正")
        self.apply_btn.setStyleSheet("background-color: #e67e22; color: white;")
        self.apply_btn.clicked.connect(self._on_correct)
        ct.addWidget(self.apply_btn)
        self.priority_btn = QPushButton("标记优先级")
        self.priority_btn.setStyleSheet("background-color: #9b59b6; color: white;")
        self.priority_btn.clicked.connect(self._on_mark_priority)
        ct.addWidget(self.priority_btn)
        tl.addLayout(ct)
        splitter.addWidget(top)

        # Bottom: feedback history
        bottom = QFrame()
        bl = QVBoxLayout(bottom)
        bl.addWidget(QLabel("修正记录:"))
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["Question", "Original", "Corrected", "Time"])
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        bl.addWidget(self.history_table)

        # Stats
        stats = QHBoxLayout()
        self.corrected_count = QLabel("已修正: 0")
        stats.addWidget(self.corrected_count)
        self.accuracy_label = QLabel("System accuracy: N/A")
        stats.addWidget(self.accuracy_label)
        stats.addStretch()
        bl.addLayout(stats)
        splitter.addWidget(bottom)
        layout.addWidget(splitter)

    def set_results(self, results):
        self.results = results
        self._update_table()
        # Populate correction combo
        self.correct_combo.clear()
        seen = set()
        for r in results:
            if r["category"] not in seen:
                self.correct_combo.addItem(r["category"])
                seen.add(r["category"])

    def _update_table(self):
        self.table.setRowCount(len(self.results))
        for i, r in enumerate(self.results):
            self.table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            txt = r["text"][:35] + "..." if len(r["text"]) > 35 else r["text"]
            self.table.setItem(i, 1, QTableWidgetItem(txt))
            self.table.setItem(i, 2, QTableWidgetItem(r["category"]))
            priority = r.get("priority", "")
            self.table.setItem(i, 3, QTableWidgetItem(priority))
            corrected = "Yes" if r.get("corrected") else "No"
            self.table.setItem(i, 4, QTableWidgetItem(corrected))
            status = r.get("status", "auto")
            self.table.setItem(i, 5, QTableWidgetItem(status))

    def _on_correct(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.results):
            QMessageBox.warning(self, "Select Row", "Please select a question first")
            return
        new_cat = self.correct_combo.currentText().strip()
        if not new_cat:
            return
        old_cat = self.results[row]["category"]
        self.results[row]["category"] = new_cat
        self.results[row]["corrected"] = True
        self.results[row]["status"] = "corrected"
        # Add to feedback history
        import time
        self.feedback_history.append({
            "text": self.results[row]["text"],
            "original": old_cat,
            "corrected": new_cat,
            "time": time.strftime("%H:%M")
        })
        self._update_table()
        self._update_history_table()
        self.corrected_count.setText(f"已修正: {len(self.feedback_history)}")
        # Incremental optimization
        if self.classifier:
            try:
                from ..preprocessing.text_cleaner import TextCleaner
                from ..preprocessing.tokenizer import Tokenizer
                from ..preprocessing.stopwords import StopwordsFilter
                cleaner = TextCleaner()
                tok = Tokenizer()
                sw = StopwordsFilter()
                cleaned = cleaner.clean(self.results[row]["text"])
                words = tok.segment(cleaned)
                cw = tok.extract_core_words(words)
                fw = sw.filter(cw, None)
                self.classifier.tfidf_matcher.add_sample(new_cat, " ".join(fw))
                self.accuracy_label.setText("System updated incrementally")
            except Exception as e:
                self.accuracy_label.setText(f"Update error: {e}")

    def _on_mark_priority(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.results):
            return
        current = self.results[row].get("priority", "")
        if current == "High":
            self.results[row]["priority"] = ""
        elif current == "":
            self.results[row]["priority"] = "High"
        else:
            self.results[row]["priority"] = "High"
        self._update_table()

    def _update_history_table(self):
        self.history_table.setRowCount(len(self.feedback_history))
        for i, h in enumerate(reversed(self.feedback_history)):
            txt = h["text"][:30] + "..." if len(h["text"]) > 30 else h["text"]
            self.history_table.setItem(i, 0, QTableWidgetItem(txt))
            self.history_table.setItem(i, 1, QTableWidgetItem(h["original"]))
            self.history_table.setItem(i, 2, QTableWidgetItem(h["corrected"]))
            self.history_table.setItem(i, 3, QTableWidgetItem(h["time"]))

    def get_results(self):
        return self.results