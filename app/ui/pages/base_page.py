# -*- coding: utf-8 -*-
"""页面基类。

中间核心内容展示区的每个页面都继承本类，向主窗口暴露统一契约：
    quick_actions()  -> 右侧快捷操作区要展示什么
    on_show()        -> 被切换到时刷新数据
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from app.ui.context import AppContext
from app.ui.workers import stop_worker


class BasePage(QWidget):
    """所有功能页的基类。"""

    #: 页面标题与副标题，显示在顶部栏
    title: str = ""
    subtitle: str = ""

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0, 0, 0, 0)
        self.root.setSpacing(10)

    # ------------------------------------------------------------------ #
    def quick_actions(self) -> Tuple[str, List[QWidget]]:
        """返回 (提示文案, 快捷操作按钮列表)，由主窗口装进右侧面板。"""
        return "", []

    def on_show(self) -> None:
        """页面被切换到时调用，用于刷新数据。"""

    def shutdown(self) -> None:
        """窗口关闭前调用：停掉本页启动的后台线程。

        QThread 若在对象析构时仍在运行，Qt 会直接 abort 掉整个进程。
        预热线程要跑好几秒（jieba 初始化），这段时间里用户完全可能关窗，
        所以每个启动了线程的页面都必须在这里把它们收干净。
        """
        for w in self.workers():
            stop_worker(w)

    def workers(self) -> List[QThread]:
        """返回本页当前持有的后台线程（供 shutdown 统一处理）。"""
        return []

    def toast(self, msg: str, kind: str = "info") -> None:
        self.ctx.toast.emit(msg, kind)

    def status(self, msg: str) -> None:
        self.ctx.status.emit(msg)
