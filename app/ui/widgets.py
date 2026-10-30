# -*- coding: utf-8 -*-
"""共享界面组件（需求 4.3.3 交互界面设计）。

* ``NavBar``          左侧功能导航栏：五大入口 + 图标 + 悬浮操作提示
* ``QuickActionPanel`` 右侧快捷操作按钮区：随当前页面动态切换
* ``Card`` / ``SectionTitle`` / ``Toast`` 等基础件
"""
from __future__ import annotations

from functools import wraps
from typing import Callable, List, Optional, Tuple

from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QButtonGroup, QFrame, QGraphicsOpacityEffect, QHBoxLayout,
                             QHeaderView, QLabel, QPushButton, QSizePolicy, QToolButton, QVBoxLayout,
                             QWidget)

from app import config
from app.ui import icons, theme


def batch_view_updates(*attributes):
    """重建视图期间暂停逐项重绘，结束或异常时恢复原状态。"""
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            views = [getattr(self, name) for name in attributes]
            states = [(view, view.updatesEnabled(), view.isSortingEnabled()) for view in views]
            headers = []
            for view in views:
                horizontal = getattr(view, "horizontalHeader", None)
                if horizontal is None:
                    horizontal = view.header
                for header in (horizontal(), getattr(view, "verticalHeader", lambda: None)()):
                    if header is not None and header.count():
                        headers.append((header, [header.sectionResizeMode(i)
                                                  for i in range(header.count())],
                                        header.stretchLastSection()))
            try:
                for view, _updates, _sorting in states:
                    view.setUpdatesEnabled(False)
                    view.setSortingEnabled(False)
                # ResizeToContents can rescan every populated row after each setItem,
                # even with painting disabled. Freeze section geometry for the batch.
                for header, _modes, _stretch in headers:
                    header.setStretchLastSection(False)
                    header.setSectionResizeMode(QHeaderView.Fixed)
                return method(self, *args, **kwargs)
            finally:
                for header, modes, stretch in headers:
                    if len(set(modes)) == 1:
                        header.setSectionResizeMode(modes[0])
                    else:
                        for index, mode in enumerate(modes):
                            if index < header.count():
                                header.setSectionResizeMode(index, mode)
                    header.setStretchLastSection(stretch)
                for view, updates, sorting in states:
                    view.setSortingEnabled(sorting)
                    view.setUpdatesEnabled(updates)
        return wrapped
    return decorate


# --------------------------------------------------------------------------- #
# 基础件
# --------------------------------------------------------------------------- #
class Card(QFrame):
    """带边框圆角的内容卡片。"""

    def __init__(self, parent: Optional[QWidget] = None, margin: int = 12,
                 spacing: int = 8) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(margin, margin, margin, margin)
        self.box.setSpacing(spacing)


def label(text: str, role: str = "", parent: Optional[QWidget] = None) -> QLabel:
    lb = QLabel(text, parent)
    if role:
        lb.setProperty("role", role)
    return lb


def button(text: str, kind: str = "", icon_name: str = "",
           on_click: Optional[Callable] = None, tip: str = "") -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setProperty("kind", kind)
    if icon_name:
        col = "#FFFFFF" if kind in ("primary", "accent") else ""
        b.setIcon(icons.icon(icon_name, 16, col))
        b.setIconSize(QSize(16, 16))
    if on_click:
        b.clicked.connect(on_click)
    if tip:
        b.setToolTip(tip)
    b.setCursor(Qt.PointingHandCursor)
    return b


def hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color: {}; background: {};".format(
        theme.palette()["border"], theme.palette()["border"]))
    f.setFixedHeight(1)
    return f


class SectionTitle(QWidget):
    """小节标题 + 右侧说明文字。"""

    def __init__(self, title: str, hint: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        self.title = label(title, "h2")
        h.addWidget(self.title)
        self.hint = label(hint, "faint")
        h.addWidget(self.hint)
        h.addStretch(1)

    def set_hint(self, text: str) -> None:
        self.hint.setText(text)


class Toast(QLabel):
    """轻量浮层提示：操作反馈不打断流程，比模态弹窗更适合高频操作。"""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.hide()
        self._effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._effect)
        self._anim = QPropertyAnimation(self._effect, b"opacity", self)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fade_out)

    def show_msg(self, text: str, kind: str = "info", msec: int = 2600) -> None:
        p = theme.palette()
        color = {"info": p["primary"], "ok": p["success"],
                 "warn": p["warn"], "error": p["danger"]}.get(kind, p["primary"])
        self.setText(text)
        self.setStyleSheet(
            "background: {}; color: #FFFFFF; border-radius: 8px; padding: 9px 16px;"
            "font-size: 13px; font-weight: bold;".format(color))
        self.adjustSize()
        self.setFixedWidth(min(max(self.sizeHint().width() + 8, 180),
                               max(self.parent().width() - 80, 200)))
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()
        self._effect.setOpacity(1.0)
        self._anim.stop()
        self._timer.start(msec)

    def _reposition(self) -> None:
        par = self.parent()
        if not par:
            return
        self.move(max((par.width() - self.width()) // 2, 10), par.height() - self.height() - 26)

    def _fade_out(self) -> None:
        self._anim.stop()
        self._anim.setDuration(380)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        try:
            self._anim.finished.disconnect()
        except TypeError:
            pass
        self._anim.finished.connect(self.hide)
        self._anim.start()


# --------------------------------------------------------------------------- #
# 左侧功能导航栏
# --------------------------------------------------------------------------- #
class NavItem(QFrame):
    """单个功能入口：序号 + 图标 + 文字，整行可点。

    做成自绘 QFrame（而非 QToolButton）是为了让序号用等宽弱化色、文字用主色，
    并支持激活态的「左色条 + 浅色高亮」——这些用 QToolButton 的单一文本样式做不到。
    激活/常态通过动态属性 active + QSS 子选择器换色；图标是烘焙进 pixmap 的，
    换色需重渲，故 set_active / refresh 里手动重画。
    """

    clicked = pyqtSignal(int)

    def __init__(self, index: int, number: str, icon_name: str, text: str,
                 tip: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("NavItem")
        # 与 NavBar 同理：QFrame 子类必须显式打开，QSS 的底色/左色条才会被绘制。
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("active", "false")
        self.index = index
        self.icon_name = icon_name
        self.setFixedHeight(42)
        self.setCursor(Qt.PointingHandCursor)
        if tip:
            self.setToolTip(tip)

        h = QHBoxLayout(self)
        # 左内边距 11 + 3px 左色条 = 内容左起 14；色条恒占位（透明→有色），避免激活时位移。
        h.setContentsMargins(11, 0, 12, 0)
        h.setSpacing(11)

        self.num = QLabel(number)
        self.num.setObjectName("NavNum")
        self.num.setFixedWidth(16)
        self.ico = QLabel()
        self.ico.setObjectName("NavIco")
        self.ico.setFixedSize(20, 20)
        self.ico.setAlignment(Qt.AlignCenter)
        self.txt = QLabel(text)
        self.txt.setObjectName("NavText")

        h.addWidget(self.num)
        h.addWidget(self.ico)
        h.addWidget(self.txt)
        h.addStretch(1)
        self._render_icon()

    def _render_icon(self) -> None:
        p = theme.palette()
        active = self.property("active") == "true"
        col = p["nav_text_active"] if active else p["nav_text"]
        self.ico.setPixmap(icons.pixmap(self.icon_name, 19, col))

    def set_active(self, on: bool) -> None:
        if (self.property("active") == "true") == on:
            return
        self.setProperty("active", "true" if on else "false")
        # 动态属性变化后必须 unpolish/polish，QSS 的 [active] 分支才会重新命中。
        self.style().unpolish(self)
        self.style().polish(self)
        self._render_icon()

    def refresh(self) -> None:
        """主题切换后重渲图标（颜色随主题变）。"""
        self._render_icon()

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.index)
        super().mousePressEvent(e)


class NavBar(QWidget):
    """左侧功能导航栏：五大核心功能入口。

    每个入口配清晰图标与悬浮操作提示，对应需求 4.3.3「左侧为功能导航栏」。
    结构：品牌区（标记+名称+副标题）→ 离线状态 → 「工作流」小节 → 序号步骤入口
    → 底部（主题切换 / 新手引导 / 版本）。导航栏跟随主题换浅色 / 深色。
    """

    switched = pyqtSignal(int)
    theme_toggled = pyqtSignal()
    guide_requested = pyqtSignal()

    def __init__(self, entries: List[Tuple[str, str, str]],
                 parent: Optional[QWidget] = None) -> None:
        """entries: [(标题, 图标名, 悬浮提示)]"""
        super().__init__(parent)
        self.setObjectName("NavBar")
        # ↓↓↓ 关键：没有这一行，QSS 里 QWidget#NavBar 的底色/右边框根本不会被绘制。
        # Qt 只对「纯 QWidget 实例」自动打开 WA_StyledBackground（源码中判断
        # metaObject() == QWidget::staticMetaObject），NavBar 是子类，不满足该条件。
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedWidth(244)
        # 供新手引导 guide.py 按索引定位高亮，名字沿用 buttons 保持兼容。
        self.buttons: List[NavItem] = []
        self._current = 0

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # ---- 品牌区：标记 + 名称 + 副标题 ---- #
        brand = QWidget()
        brand.setStyleSheet("background: transparent;")
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(16, 18, 16, 10)
        bl.setSpacing(11)
        self.mark = QLabel()
        self.mark.setObjectName("NavMark")
        self.mark.setFixedSize(38, 38)
        self.mark.setAlignment(Qt.AlignCenter)
        self.mark.setPixmap(icons.brand_mark(38))
        bt = QVBoxLayout()
        bt.setSpacing(2)
        self.title = QLabel("错题自动分类")
        self.title.setObjectName("NavTitle")
        self.sub = QLabel("轻量化 · 本地运行 · 零代码")
        self.sub.setObjectName("NavSubtitle")
        bt.addWidget(self.title)
        bt.addWidget(self.sub)
        bl.addWidget(self.mark)
        bl.addLayout(bt)
        bl.addStretch(1)
        v.addWidget(brand)

        # ---- 离线状态药丸 ---- #
        sw = QWidget()
        sw.setStyleSheet("background: transparent;")
        sl = QHBoxLayout(sw)
        sl.setContentsMargins(16, 0, 16, 6)
        sl.setSpacing(0)
        self.status = QLabel("●  本地文本分类")
        self.status.setObjectName("NavStatus")
        sl.addWidget(self.status)
        sl.addStretch(1)
        v.addWidget(sw)

        # ---- 「工作流」小节标签 + 进度分数 ---- #
        eb = QWidget()
        eb.setStyleSheet("background: transparent;")
        el = QHBoxLayout(eb)
        el.setContentsMargins(19, 14, 18, 8)
        el.setSpacing(0)
        self.eyebrow = QLabel("工 作 流")
        self.eyebrow.setObjectName("NavEyebrow")
        self.frac = QLabel("1 / {}".format(len(entries)))
        self.frac.setObjectName("NavFrac")
        el.addWidget(self.eyebrow)
        el.addStretch(1)
        el.addWidget(self.frac)
        v.addWidget(eb)

        # ---- 五大功能入口 ---- #
        wrap = QWidget()
        wrap.setStyleSheet("background: transparent;")
        wv = QVBoxLayout(wrap)
        wv.setContentsMargins(10, 0, 10, 4)
        wv.setSpacing(3)
        for i, (text, ic, tip) in enumerate(entries):
            it = NavItem(i, "{:02d}".format(i + 1), ic, text, tip)
            it.clicked.connect(self._on_click)
            it.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.buttons.append(it)
            wv.addWidget(it)
        v.addWidget(wrap)
        v.addStretch(1)

        # ---- 底部分隔线 ---- #
        dw = QWidget()
        dw.setStyleSheet("background: transparent;")
        dl = QVBoxLayout(dw)
        dl.setContentsMargins(16, 6, 16, 0)
        dl.setSpacing(0)
        self._divider = QFrame()
        self._divider.setObjectName("NavDivider")
        self._divider.setFixedHeight(1)
        dl.addWidget(self._divider)
        v.addWidget(dw)

        # ---- 底部工具条：主题切换 + 新手引导 ---- #
        bar = QWidget()
        bar.setStyleSheet("background: transparent;")
        bh = QHBoxLayout(bar)
        bh.setContentsMargins(12, 8, 12, 2)
        bh.setSpacing(4)
        self.btn_theme = self._mini("moon", "切换浅色 / 深色主题")
        self.btn_theme.clicked.connect(self.theme_toggled.emit)
        self.btn_guide = self._mini("help", "重新播放新手引导")
        self.btn_guide.clicked.connect(self.guide_requested.emit)
        bh.addWidget(self.btn_theme)
        bh.addWidget(self.btn_guide)
        bh.addStretch(1)
        v.addWidget(bar)

        foot = QLabel("v{} · 本地离线运行".format(config.APP_VERSION))
        foot.setObjectName("NavFooter")
        foot.setContentsMargins(14, 0, 14, 10)
        v.addWidget(foot)

        if self.buttons:
            self.buttons[0].set_active(True)

    def _mini(self, name: str, tip: str) -> QToolButton:
        b = QToolButton()
        b.setObjectName("NavMini")
        b.setIcon(icons.icon(name, 17, theme.palette()["nav_text"]))
        b.setIconSize(QSize(17, 17))
        b.setToolTip(tip)
        b.setCursor(Qt.PointingHandCursor)
        b.setFixedSize(34, 30)
        return b

    # ------------------------------------------------------------------ #
    def _on_click(self, index: int) -> None:
        self.select(index)
        self.switched.emit(index)

    def select(self, index: int) -> None:
        if not (0 <= index < len(self.buttons)):
            return
        self._current = index
        for i, it in enumerate(self.buttons):
            it.set_active(i == index)
        self.frac.setText("{} / {}".format(index + 1, len(self.buttons)))

    def refresh_icons(self, entries: List[Tuple[str, str, str]]) -> None:
        """主题切换后重建图标（导航项图标、品牌标记、底部迷你按钮、分隔线）。"""
        self.mark.setPixmap(icons.brand_mark(38))
        for it in self.buttons:
            it.refresh()
        p = theme.palette()
        self.btn_theme.setIcon(icons.icon(
            "sun" if theme.current() == config.THEME_DARK else "moon", 17, p["nav_text"]))
        self.btn_guide.setIcon(icons.icon("help", 17, p["nav_text"]))
        self._divider.setStyleSheet("background: {};".format(p["nav_border"]))


# --------------------------------------------------------------------------- #
# 右侧快捷操作区
# --------------------------------------------------------------------------- #
class QuickActionPanel(QWidget):
    """右侧快捷操作按钮区：一键导入 / 一键分类 / 一键导出 / 一键标记等。

    对应需求 4.3.3「右侧为快捷操作按钮区」，内容随中间区当前功能动态切换。
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(212)
        self._v = QVBoxLayout(self)
        self._v.setContentsMargins(0, 0, 0, 0)
        self._v.setSpacing(10)
        self._card = Card(margin=12, spacing=7)
        self._v.addWidget(self._card)
        self._v.addStretch(1)
        self._title = label("快捷操作", "h2")
        self._card.box.addWidget(self._title)
        self._hint = label("", "faint")
        self._hint.setWordWrap(True)
        self._card.box.addWidget(self._hint)
        self._card.box.addWidget(hline())
        self._slot = QVBoxLayout()
        self._slot.setSpacing(7)
        self._card.box.addLayout(self._slot)

    def set_actions(self, hint: str, widgets: List[QWidget]) -> None:
        """替换当前展示的快捷操作按钮。"""
        current = [self._slot.itemAt(i).widget() for i in range(self._slot.count())]
        if current == widgets:
            return
        self.clear_actions()
        self._hint.setText(hint)
        self._hint.setVisible(bool(hint))
        for w in widgets:
            self._slot.addWidget(w)
            w.show()

    def clear_actions(self) -> None:
        """移除布局并隐藏缓存控件，保留 Qt 父对象管理其生命周期。"""
        while self._slot.count():
            it = self._slot.takeAt(0)
            w = it.widget()
            if w:
                w.hide()


class StatChip(QWidget):
    """统计小卡：一个数字 + 一行说明。"""

    def __init__(self, title: str, value: str = "-", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # 与 NavBar 同一个坑：QWidget 子类必须显式打开，下面 setStyleSheet 里的
        # 底色和圆角边框才会被真正绘制出来。
        self.setAttribute(Qt.WA_StyledBackground, True)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(1)
        self.val = QLabel(value)
        self.val.setStyleSheet("font-size: 20px; font-weight: bold; background: transparent;")
        self.cap = label(title, "faint")
        self.cap.setStyleSheet("background: transparent;")
        v.addWidget(self.val)
        v.addWidget(self.cap)
        self.setStyleSheet("background: {}; border: 1px solid {}; border-radius: 9px;".format(
            theme.palette()["surface_alt"], theme.palette()["border"]))

    def set_value(self, v: str, color: str = "") -> None:
        self.val.setText(v)
        self.val.setStyleSheet(
            "font-size: 20px; font-weight: bold; background: transparent;"
            + ("color: {};".format(color) if color else ""))
