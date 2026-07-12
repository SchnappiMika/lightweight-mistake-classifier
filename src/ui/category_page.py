from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QTextEdit, QLineEdit, QComboBox,
    QSplitter, QGroupBox, QMessageBox, QFrame)
from PyQt5.QtCore import Qt
import os, json


class CategoryPage(QWidget):
    def __init__(self, dict_dir=None):
        super().__init__()
        if dict_dir is None:
            dict_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "knowledge_base", "data", "dictionaries")
        self.dict_dir = dict_dir
        self._setup_ui()
        if os.path.exists(dict_dir):
            self._load_subjects()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(QLabel("Classification System Management"))
        .setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")

        top = QHBoxLayout()
        top.addWidget(QLabel("Subject:"))
        self.subject_combo = QComboBox()
        self.subject_combo.currentTextChanged.connect(self._on_subject_changed)
        top.addWidget(self.subject_combo)
        top.addStretch()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._load_subjects)
        top.addWidget(self.refresh_btn)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        
        # Left: category list
        left = QFrame()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Categories:"))
        self.cat_list = QListWidget()
        self.cat_list.currentRowChanged.connect(self._on_category_selected)
        ll.addWidget(self.cat_list)
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("+")
        self.add_btn.clicked.connect(self._on_add_category)
        btn_row.addWidget(self.add_btn)
        self.del_btn = QPushButton("-")
        self.del_btn.clicked.connect(self._on_delete_category)
        btn_row.addWidget(self.del_btn)
        ll.addLayout(btn_row)
        splitter.addWidget(left)

        # Right: category editor
        right = QFrame()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("Category Name:"))
        self.name_edit = QLineEdit()
        rl.addWidget(self.name_edit)
        rl.addWidget(QLabel("Keywords (comma separated):"))
        self.keywords_edit = QTextEdit()
        self.keywords_edit.setMaximumHeight(120)
        rl.addWidget(self.keywords_edit)
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

    def _load_subjects(self):
        self.subject_combo.clear()
        if os.path.exists(self.dict_dir):
            for fn in os.listdir(self.dict_dir):
                if fn.endswith(".json"):
                    with open(os.path.join(self.dict_dir, fn), encoding="utf-8") as f:
                        data = json.load(f)
                        self.subject_combo.addItem(data["subject"])