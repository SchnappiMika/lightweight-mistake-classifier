"""
基于传统文本分类方法的大学生轻量化错题自动分类管理系统
入口文件
"""
import sys
import os

# 将项目根目录加入 sys.path，确保模块导入正常
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication
from src.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("错题自动分类管理系统")
    app.setApplicationDisplayName("错题自动分类管理系统")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
