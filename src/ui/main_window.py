# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import sys, os

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_base))

from src.classification.classifier import Classifier
from src.ui.import_page import ImportPage
from src.ui.classify_page import ClassifyPage
from src.ui.export_page import ExportPage
from src.ui.category_page import CategoryPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("错题自动分类管理系统")
        self.resize(1200, 800)
        try:
            self.classifier = Classifier()
        except Exception as e:
            print("Classifier init error:", e)
            self.classifier = None

        self.import_page = ImportPage()
        self.classify_page = ClassifyPage(self.classifier)
        self.export_page = ExportPage()
        self.category_page = CategoryPage()

        self.import_page.questions_imported.connect(self.classify_page.set_questions)
        self._setup_ui()

    def _setup_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        ml = QHBoxLayout(cw)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)

        # Navigation
        nav = QFrame()
        nav.setObjectName("navPanel")
        nav.setFixedWidth(200)
        nl = QVBoxLayout(nav)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.setSpacing(0)
        title = QLabel("错题系统")
        title.setObjectName("navTitle")
        title.setAlignment(Qt.AlignCenter)
        title.setFixedHeight(80)
        f = QFont()
        f.setPointSize(14)
        f.setBold(True)
        title.setFont(f)
        nl.addWidget(title)

        self.nav = QListWidget()
        self.nav.addItems(["1. Import", "2. Manage", "3. Classify", "4. Export"])
        self.nav.setCurrentRow(0)
        self.nav.currentRowChanged.connect(self._switch_page)
        nl.addWidget(self.nav)
        nl.addStretch()
        v = QLabel("v1.0.0")
        v.setObjectName("versionLabel")
        v.setAlignment(Qt.AlignCenter)
        v.setFixedHeight(30)
        nl.addWidget(v)
        ml.addWidget(nav)

        # Content
        self.stack = QStackedWidget()
        self.stack.addWidget(self.import_page)
        self.stack.addWidget(self.category_page)
        self.stack.addWidget(self.classify_page)
        self.stack.addWidget(self.export_page)
        ml.addWidget(self.stack, 1)

        # Actions panel
        ap = QFrame()
        ap.setObjectName("actionPanel")
        ap.setFixedWidth(160)
        al = QVBoxLayout(ap)
        al.setContentsMargins(8, 16, 8, 16)
        at = QLabel("Quick Actions")
        at.setObjectName("actionTitle")
        at.setAlignment(Qt.AlignCenter)
        al.addWidget(at)
        al.addSpacing(10)

        def go(idx):
            return lambda: self.nav.setCurrentRow(idx)

        b1 = QPushButton("Import"); b1.clicked.connect(go(0)); al.addWidget(b1)
        b2 = QPushButton("Classify"); b2.setStyleSheet("background-color: #27ae60; color: white;"); b2.clicked.connect(go(2)); al.addWidget(b2)
        b3 = QPushButton("Export"); b3.clicked.connect(go(3)); al.addWidget(b3)
        al.addStretch()
        ml.addWidget(ap)

        # Status bar
        self.sb = QStatusBar()
        self.setStatusBar(self.sb)
        self.sb.showMessage("System ready")
        self._apply_style()

    def _switch_page(self, idx):
        self.stack.setCurrentIndex(idx)
        labels = ["Import questions", "Manage categories", "Classify questions", "Export results"]
        if idx < len(labels):
            self.sb.showMessage(labels[idx])

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            #navPanel { background-color: #2c3e50; border-right: 1px solid #34495e; }
            #navTitle { color: white; background-color: #1a252f; padding: 10px; }
            #navPanel QListWidget { background-color: transparent; color: #ecf0f1; border: none; font-size: 13px; padding: 5px; }
            #navPanel QListWidget::item { padding: 12px 10px; border-radius: 4px; }
            #navPanel QListWidget::item:selected { background-color: #3498db; }
            #navPanel QListWidget::item:hover { background-color: #34495e; }
            #versionLabel { color: #7f8c8d; font-size: 11px; }
            #actionPanel { background-color: #ecf0f1; border-left: 1px solid #bdc3c7; }
            #actionTitle { font-size: 14px; font-weight: bold; color: #2c3e50; }
            #actionPanel QPushButton { width: 100%; padding: 8px; margin: 3px 0; border: none; border-radius: 4px; font-size: 13px; background-color: #3498db; color: white; }
            #actionPanel QPushButton:hover { background-color: #2980b9; }
            QStatusBar { background-color: #2c3e50; color: white; font-size: 12px; }
        """)
