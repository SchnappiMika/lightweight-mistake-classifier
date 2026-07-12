"""
主窗口界面
"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QStackedWidget,
    QListWidget, QStatusBar
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class MainWindow(QMainWindow):
    """系统主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("错题自动分类管理系统")
        self.resize(1200, 800)
        self._setup_ui()

    def _setup_ui(self):
        """初始化界面布局"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ========== 左侧导航栏 ==========
        nav_panel = QFrame()
        nav_panel.setObjectName("navPanel")
        nav_panel.setFixedWidth(200)
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)

        # Logo / 标题区
        title_label = QLabel("错题分类系统")
        title_label.setObjectName("navTitle")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFixedHeight(80)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        title_label.setFont(font)
        nav_layout.addWidget(title_label)

        # 导航按钮列表
        self.nav_buttons = {}
        nav_items = [
            ("import", "📥 错题导入与预览"),
            ("manage", "📂 分类体系管理"),
            ("classify", "🏷️ 一键分类与修正"),
            ("optimize", "🔄 反馈增量优化"),
            ("export", "📤 结果导出"),
        ]

        self.nav_list = QListWidget()
        for key, label in nav_items:
            self.nav_list.addItem(label)
        self.nav_list.setCurrentRow(0)
        nav_layout.addWidget(self.nav_list)

        nav_layout.addStretch()

        # 版本信息
        version_label = QLabel("v1.0.0")
        version_label.setObjectName("versionLabel")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setFixedHeight(30)
        nav_layout.addWidget(version_label)

        main_layout.addWidget(nav_panel)

        # ========== 中间内容区 ==========
        self.content_stack = QStackedWidget()

        # 占位页面 - 后续替换为实际功能模块
        placeholder = QLabel("欢迎使用错题自动分类管理系统\n\n请从左侧导航栏选择功能")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setObjectName("placeholderLabel")
        self.content_stack.addWidget(placeholder)

        main_layout.addWidget(self.content_stack, 1)

        # ========== 右侧快捷操作区 ==========
        action_panel = QFrame()
        action_panel.setObjectName("actionPanel")
        action_panel.setFixedWidth(180)
        action_layout = QVBoxLayout(action_panel)
        action_layout.setContentsMargins(10, 20, 10, 20)

        action_title = QLabel("快捷操作")
        action_title.setObjectName("actionTitle")
        action_title.setAlignment(Qt.AlignCenter)
        action_layout.addWidget(action_title)

        action_layout.addSpacing(10)

        quick_actions = [
            ("一键导入", "import"),
            ("一键分类", "classify"),
            ("一键导出", "export"),
        ]
        for label, _ in quick_actions:
            btn = QPushButton(label)
            btn.setObjectName("quickActionBtn")
            action_layout.addWidget(btn)

        action_layout.addStretch()
        main_layout.addWidget(action_panel)

        # ========== 状态栏 ==========
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        # 样式
        self._apply_style()

    def _apply_style(self):
        """应用界面样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            #navPanel {
                background-color: #2c3e50;
                border-right: 1px solid #34495e;
            }
            #navTitle {
                color: white;
                background-color: #1a252f;
                padding: 10px;
            }
            #navPanel QListWidget {
                background-color: transparent;
                color: #ecf0f1;
                border: none;
                font-size: 13px;
                padding: 5px;
            }
            #navPanel QListWidget::item {
                padding: 12px 10px;
                border-radius: 4px;
            }
            #navPanel QListWidget::item:selected {
                background-color: #3498db;
            }
            #navPanel QListWidget::item:hover {
                background-color: #34495e;
            }
            #versionLabel {
                color: #7f8c8d;
                font-size: 11px;
            }
            #actionPanel {
                background-color: #ecf0f1;
                border-left: 1px solid #bdc3c7;
            }
            #actionTitle {
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
            }
            #quickActionBtn {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-size: 13px;
            }
            #quickActionBtn:hover {
                background-color: #2980b9;
            }
            #placeholderLabel {
                font-size: 18px;
                color: #7f8c8d;
            }
            QStatusBar {
                background-color: #2c3e50;
                color: white;
                font-size: 12px;
            }
        """)
