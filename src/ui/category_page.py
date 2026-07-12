from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QTextEdit, QLineEdit, QComboBox,
    QSplitter, QGroupBox, QMessageBox, QFrame, QFileDialog)
from PyQt5.QtCore import Qt
import os, json, shutil

class CategoryPage(QWidget):
    def __init__(self, dict_dir=None):
        super().__init__()
        if dict_dir is None:
            dict_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "knowledge_base", "data", "dictionaries")
        self.dict_dir = dict_dir
        self.current_data = None
        self.current_cat_idx = -1
        self._setup_ui()
        if os.path.exists(dict_dir):
            self._load_data()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        lbl = QLabel("Classification System Management")
        lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(lbl)

        top = QHBoxLayout()
        top.addWidget(QLabel("Subject:"))
        self.subject_combo = QComboBox()
        self.subject_combo.currentTextChanged.connect(self._on_subject_changed)
        top.addWidget(self.subject_combo)
        top.addStretch()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._load_data)
        top.addWidget(self.refresh_btn)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        left = QFrame()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Categories:"))
        self.cat_list = QListWidget()
        self.cat_list.currentRowChanged.connect(self._on_category_selected)
        ll.addWidget(self.cat_list)
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("+")
        self.add_btn.clicked.connect(self._on_add)
        btn_row.addWidget(self.add_btn)
        self.del_btn = QPushButton("-")
        self.del_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self.del_btn)
        ll.addLayout(btn_row)
        splitter.addWidget(left)

        right = QFrame()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("Category Name:"))
        self.name_edit = QLineEdit()
        rl.addWidget(self.name_edit)
        rl.addWidget(QLabel("Keywords (comma separated):"))
        self.kw_edit = QTextEdit()
        self.kw_edit.setMaximumHeight(100)
        rl.addWidget(self.kw_edit)
        rl.addWidget(QLabel("Sample text:"))
        self.sample_edit = QTextEdit()
        self.sample_edit.setMaximumHeight(80)
        rl.addWidget(self.sample_edit)
        self.save_btn = QPushButton("Save Changes")
        self.save_btn.setStyleSheet("background-color: #3498db; color: white;")
        self.save_btn.clicked.connect(self._on_save)
        rl.addWidget(self.save_btn)
        rl.addStretch()
        splitter.addWidget(right)
        layout.addWidget(splitter)

    def _load_data(self):
        self.subject_combo.blockSignals(True)
        self.subject_combo.clear()
        if os.path.exists(self.dict_dir):
            for fn in sorted(os.listdir(self.dict_dir)):
                if fn.endswith(".json"):
                    with open(os.path.join(self.dict_dir, fn), encoding="utf-8") as f:
                        d = json.load(f)
                        self.subject_combo.addItem(d["subject"] + "|" + fn)
        self.subject_combo.blockSignals(False)
        if self.subject_combo.count() > 0:
            self.subject_combo.setCurrentIndex(0)