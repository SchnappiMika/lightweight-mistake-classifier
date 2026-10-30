# -*- coding: utf-8 -*-
"""新手引导（需求 4.3.3「新手用户首次使用时可触发自动新手引导」）。

实现方式：在主窗口上盖一层半透明遮罩，对当前步骤的目标控件"挖孔"高亮，
旁边浮出说明气泡，配「上一步 / 下一步 / 跳过」。首次启动自动触发，
之后可从左下角问号按钮随时重放。
"""
from __future__ import annotations

from typing import Callable, List, Optional

from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.ui import theme


class GuideStep:
    """一步引导。

    Args:
        target: 返回目标控件的可调用对象（延迟求值，避免页面尚未创建时取不到）
        page:   该步骤需要先切到的页面索引，None 表示不切页
    """

    def __init__(self, title: str, text: str,
                 target: Optional[Callable[[], Optional[QWidget]]] = None,
                 page: Optional[int] = None) -> None:
        self.title = title
        self.text = text
        self.target = target
        self.page = page


class GuideOverlay(QWidget):
    """引导遮罩层。"""

    finished = pyqtSignal()
    page_requested = pyqtSignal(int)

    def __init__(self, parent: QWidget, steps: List[GuideStep]) -> None:
        super().__init__(parent)
        self.steps = steps
        self.index = 0
        self._hole = QRect()
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setFocusPolicy(Qt.StrongFocus)

        p = theme.palette()
        self.bubble = QWidget(self)
        self.bubble.setStyleSheet(
            "background: {}; border: 1px solid {}; border-radius: 11px;".format(
                p["surface"], p["border_strong"]))
        bv = QVBoxLayout(self.bubble)
        bv.setContentsMargins(16, 14, 16, 12)
        bv.setSpacing(7)

        self.lb_step = QLabel()
        self.lb_step.setStyleSheet(
            "color: {}; font-size: 12px; font-weight: bold; background: transparent;".format(
                p["primary"]))
        self.lb_title = QLabel()
        self.lb_title.setStyleSheet(
            "font-size: 16px; font-weight: bold; background: transparent;")
        self.lb_text = QLabel()
        self.lb_text.setWordWrap(True)
        self.lb_text.setStyleSheet(
            "color: {}; font-size: 13px; background: transparent;".format(p["text_dim"]))
        bv.addWidget(self.lb_step)
        bv.addWidget(self.lb_title)
        bv.addWidget(self.lb_text)

        row = QHBoxLayout()
        row.setSpacing(7)
        self.btn_skip = QPushButton("跳过引导")
        self.btn_skip.setProperty("kind", "ghost")
        self.btn_skip.setCursor(Qt.PointingHandCursor)
        self.btn_skip.clicked.connect(self._finish)
        self.btn_prev = QPushButton("上一步")
        self.btn_prev.setCursor(Qt.PointingHandCursor)
        self.btn_prev.clicked.connect(self.prev)
        self.btn_next = QPushButton("下一步")
        self.btn_next.setProperty("kind", "primary")
        self.btn_next.setCursor(Qt.PointingHandCursor)
        self.btn_next.clicked.connect(self.next)
        row.addWidget(self.btn_skip)
        row.addStretch(1)
        row.addWidget(self.btn_prev)
        row.addWidget(self.btn_next)
        bv.addLayout(row)

        self.bubble.setFixedWidth(330)
        self.hide()

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if not self.steps:
            return
        self.index = 0
        self.setGeometry(self.parent().rect())
        self.show()
        self.raise_()
        self.setFocus()
        self._refresh()

    def next(self) -> None:
        if self.index >= len(self.steps) - 1:
            self._finish()
            return
        self.index += 1
        self._refresh()

    def prev(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._refresh()

    def _finish(self) -> None:
        self.hide()
        self.finished.emit()

    # ------------------------------------------------------------------ #
    def _refresh(self) -> None:
        st = self.steps[self.index]
        if st.page is not None:
            self.page_requested.emit(st.page)
        self.lb_step.setText("第 {} / {} 步".format(self.index + 1, len(self.steps)))
        self.lb_title.setText(st.title)
        self.lb_text.setText(st.text)
        self.btn_prev.setEnabled(self.index > 0)
        self.btn_next.setText("完成，开始使用" if self.index == len(self.steps) - 1 else "下一步")

        self._hole = QRect()
        if st.target:
            try:
                w = st.target()
            except Exception:              # noqa: BLE001 目标控件可能尚未构建
                w = None
            if w is not None and w.isVisible():
                # 必须走全局坐标绕一圈，不能用 w.mapTo(self, ...)。
                #
                # Qt 的 mapTo(parent, pos) 要求 parent 是调用控件的**祖先**，
                # 传别的东西是未定义行为——实测直接 SIGSEGV，整个程序没了。
                # 而遮罩层是主窗口的子控件，与导航按钮是叔侄关系，并不是它的祖先。
                # mapToGlobal / mapFromGlobal 不依赖控件树的层级关系，任何两个
                # 控件之间都能安全换算。
                #
                # 这个坑很要命：新手引导是首次启动时自动弹的，等于每个新用户
                # 打开程序就崩。
                tl = self.mapFromGlobal(w.mapToGlobal(QPoint(0, 0)))
                self._hole = QRect(tl, w.size()).adjusted(-6, -6, 6, 6)
        self._place_bubble()
        self.update()

    def _place_bubble(self) -> None:
        self.bubble.adjustSize()
        bw, bh = self.bubble.width(), self.bubble.height()
        r = self.rect()
        if self._hole.isNull():                       # 无目标：居中
            self.bubble.move((r.width() - bw) // 2, (r.height() - bh) // 2)
            return
        x = self._hole.right() + 16
        if x + bw > r.width() - 12:                   # 右侧放不下就放左侧
            x = self._hole.left() - bw - 16
        x = max(12, min(x, r.width() - bw - 12))
        y = self._hole.center().y() - bh // 2
        y = max(12, min(y, r.height() - bh - 12))
        self.bubble.move(x, y)

    # ------------------------------------------------------------------ #
    def paintEvent(self, _e) -> None:      # noqa: N802 Qt 命名约定
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        path = QPainterPath()
        path.addRect(0, 0, self.width(), self.height())
        if not self._hole.isNull():
            hole = QPainterPath()
            hole.addRoundedRect(float(self._hole.x()), float(self._hole.y()),
                                float(self._hole.width()), float(self._hole.height()), 9, 9)
            path = path.subtracted(hole)               # 挖孔：目标控件保持原色
        p.fillPath(path, QColor(10, 14, 22, 168))
        if not self._hole.isNull():
            pen = QPen(QColor(theme.palette()["primary"]))
            pen.setWidth(2)
            p.setPen(pen)
            p.drawRoundedRect(self._hole, 9, 9)
        p.end()

    def resizeEvent(self, _e) -> None:     # noqa: N802
        self._place_bubble()

    def mousePressEvent(self, _e) -> None:  # noqa: N802
        pass                                # 吞掉点击，避免误触底层控件

    def keyPressEvent(self, e) -> None:     # noqa: N802
        if e.key() == Qt.Key_Escape:
            self._finish()
        elif e.key() in (Qt.Key_Right, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.next()
        elif e.key() == Qt.Key_Left:
            self.prev()


def default_steps(win) -> List[GuideStep]:
    """默认引导脚本：串起五大模块的标准使用路径。"""
    return [
        GuideStep(
            "欢迎使用错题自动分类系统",
            "整个流程只有四步：导入错题 → 选分类体系 → 一键分类 → 导出归档。\n"
            "全程本地离线运行，不联网、不收费，也不需要任何编程基础。",
            None, 0),
        GuideStep(
            "第一步：导入错题",
            "支持单条手动录入，也支持批量导入 TXT / Excel，还可以直接导入公式图片、"
            "工科图形题截图和英语听力音频。导入后可以在右侧预览并补充标注。",
            lambda: win.nav.buttons[0], 0),
        GuideStep(
            "第二步：搭建你自己的分类体系",
            "系统内置了高数、大物、大英等课程的默认体系，你也可以零代码新建：\n"
            "加标签、填关键词、贴标准样本即可，改完立即生效，不用重启也不用改代码。",
            lambda: win.nav.buttons[1], 1),
        GuideStep(
            "第三步：一键分类",
            "选好分类体系后点「一键分类」，系统会自动完成多模态预处理、关键词粗筛与"
            "多特征融合精匹配。每条错题的分类依据都可以展开查看。",
            lambda: win.nav.buttons[2], 2),
        GuideStep(
            "分类不准？改一下它就会记住",
            "在分类页直接把标签改成正确的，系统会立刻增量更新该分类的特征库与权重，"
            "同类错题下次就更准 —— 不需要重新训练，也不需要联网。",
            lambda: win.nav.buttons[3], 3),
        GuideStep(
            "第四步：导出归档",
            "分类结果可以按任意维度分 sheet 导出成 Excel，图片和音频也能按标签"
            "分文件夹归档，打印出来就是一份现成的错题本。",
            lambda: win.nav.buttons[4], 4),
        GuideStep(
            "准备好了",
            "左下角的问号按钮可以随时重看这份引导，旁边的月亮/太阳按钮可以切换深浅主题。\n"
            "想快速体验的话，去导入页点「载入示例错题」就能一键灌入 155 条演示数据。",
            None, 0),
    ]
