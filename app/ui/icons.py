# -*- coding: utf-8 -*-
"""内联 SVG 图标集。

图标以路径数据内嵌在代码里，随主题色动态着色：
* 打包时不需要额外的图片资源，PyInstaller 配置更简单，体积更小；
* 换主题时图标颜色跟着变，不用维护两套 png。
"""
from __future__ import annotations

from typing import Dict

from PyQt5.QtCore import QByteArray, QSize, Qt
from PyQt5.QtGui import QIcon, QPainter, QPixmap
from PyQt5.QtSvg import QSvgRenderer

from app.ui import theme

# 24x24 视口、stroke 风格的路径数据
PATHS: Dict[str, str] = {
    "import": ("M12 3v12M12 15l-4-4M12 15l4-4"
               "M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"),
    "scheme": ("M4 5h6v6H4zM14 5h6v6h-6zM4 13h6v6H4zM14 13h6v6h-6z"),
    "classify": ("M4 6h16M4 12h10M4 18h6M17 14l2.5 2.5L23 12"),
    "feedback": ("M21 11.5a8.4 8.4 0 0 1-9 8.4 8.5 8.5 0 0 1-3.8-.9L3 21l1.9-5.2"
                 "A8.4 8.4 0 0 1 4 11.5a8.5 8.5 0 0 1 8.5-8.5 8.4 8.4 0 0 1 8.5 8.5z"
                 "M8.5 11.5h.01M12.5 11.5h.01M16.5 11.5h.01"),
    "export": ("M12 21V9M12 9l-4 4M12 9l4 4M4 7V5a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v2"),
    "add": "M12 5v14M5 12h14",
    "remove": "M5 12h14",
    "trash": ("M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2"
              "M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6M10 11v6M14 11v6"),
    "save": ("M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"
             "M17 21v-8H7v8M7 3v5h8"),
    "play": "M6 4l14 8-14 8z",
    "stop": "M6 6h12v12H6z",
    "sun": ("M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10zM12 1v3M12 20v3M4.2 4.2l2.1 2.1"
            "M17.7 17.7l2.1 2.1M1 12h3M20 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"),
    "moon": "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z",
    "help": ("M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"
             "M12 17h.01"),
    "check": "M20 6L9 17l-5-5",
    "search": "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM21 21l-4.3-4.3",
    "folder": ("M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5"
               "a2 2 0 0 1-2-2z"),
    "refresh": ("M21 12a9 9 0 1 1-2.6-6.4M21 3v6h-6"),
    "image": ("M3 5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"
              "M8.5 10a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM21 15l-5-5L5 21"),
    "audio": ("M11 5L6 9H2v6h4l5 4zM15.5 8.5a5 5 0 0 1 0 7M19 5a9 9 0 0 1 0 14"),
    "text": "M4 6h16M4 10h16M4 14h12M4 18h8",
    "chart": "M4 20V10M10 20V4M16 20v-7M22 20H2",
    "sliders": ("M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3"
                "M1 14h6M9 8h6M17 16h6"),
    "warning": ("M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"
                "M12 9v4M12 17h.01"),
    "info": "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 16v-4M12 8h.01",
    "wand": ("M15 4V2M15 16v-2M8 9h2M20 9h2M17.8 11.8l1.4 1.4M17.8 6.2l1.4-1.4"
             "M3 21l9-9M12.2 6.2l-1.4-1.4"),
}

_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="{color}" stroke-width="{w}" stroke-linecap="round" '
        'stroke-linejoin="round"><path d="{d}"/></svg>')

_cache: Dict[tuple, QIcon] = {}


def pixmap(name: str, size: int = 18, color: str = "", width: float = 1.8) -> QPixmap:
    """渲染单个图标为 QPixmap（自动适配高分屏 devicePixelRatio）。"""
    from PyQt5.QtWidgets import QApplication

    d = PATHS.get(name, PATHS["info"])
    col = color or theme.palette()["text"]
    svg = _SVG.format(color=col, d=d, w=width)
    app = QApplication.instance()
    dpr = app.devicePixelRatio() if app else 1.0
    r = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    r.render(p)
    p.end()
    return pm


def icon(name: str, size: int = 18, color: str = "", width: float = 1.8) -> QIcon:
    key = (name, size, color or theme.current(), width)
    if key not in _cache:
        _cache[key] = QIcon(pixmap(name, size, color, width))
    return _cache[key]


def nav_icon(name: str, size: int = 19) -> QIcon:
    """导航栏图标：常态与选中态用不同颜色，Qt 会按状态自动取用。"""
    p = theme.palette()
    ic = QIcon()
    ic.addPixmap(pixmap(name, size, p["nav_text"]), QIcon.Normal, QIcon.Off)
    ic.addPixmap(pixmap(name, size, p["nav_text_active"]), QIcon.Normal, QIcon.On)
    ic.addPixmap(pixmap(name, size, p["nav_text_active"]), QIcon.Active, QIcon.Off)
    ic.addPixmap(pixmap(name, size, p["nav_text_active"]), QIcon.Selected, QIcon.On)
    return ic


def app_icon() -> QIcon:
    """程序图标：一个带勾的书页，程序化生成，免外部 ico 文件。"""
    p = theme.palette()
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<rect x="4" y="4" width="56" height="56" rx="13" fill="{pri}"/>'
        '<path d="M20 17h18l8 8v22a3 3 0 0 1-3 3H20a3 3 0 0 1-3-3V20a3 3 0 0 1 3-3z" '
        'fill="#FFFFFF" opacity="0.93"/>'
        '<path d="M24 30h12M24 36h9" stroke="{pri}" stroke-width="2.6" '
        'stroke-linecap="round"/>'
        '<path d="M32 43l5 5 10-11" stroke="{acc}" stroke-width="4.5" fill="none" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        "</svg>"
    ).format(pri=p["primary"], acc=p["accent"])
    r = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    ic = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        pm = QPixmap(QSize(s, s))
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, True)
        r.render(painter)
        painter.end()
        ic.addPixmap(pm)
    return ic


def brand_mark(size: int = 38) -> QPixmap:
    """左侧导航的品牌标记：圆角渐变方块 + 书页，随主题主色/强调色换色。

    单独一个 pixmap（不进 QIcon 缓存），切换主题时由 NavBar 主动重渲。
    """
    from PyQt5.QtWidgets import QApplication

    p = theme.palette()
    c1 = p["primary"]
    c2 = p.get("accent", p["primary"])
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/>'
        '</linearGradient></defs>'
        '<rect x="1" y="1" width="38" height="38" rx="11" fill="url(#g)"/>'
        '<path d="M11.5 12.5h10l3.5 3.5v11a1.6 1.6 0 0 1-1.6 1.6H11.5'
        'A1.6 1.6 0 0 1 9.9 27V14.1A1.6 1.6 0 0 1 11.5 12.5z" '
        'fill="#FFFFFF" opacity="0.95"/>'
        '<path d="M13.6 19.6h7.8M13.6 23.2h5" stroke="{c1}" stroke-width="1.7" '
        'stroke-linecap="round"/>'
        "</svg>"
    ).format(c1=c1, c2=c2)
    app = QApplication.instance()
    dpr = app.devicePixelRatio() if app else 1.0
    r = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    r.render(painter)
    painter.end()
    return pm


def clear_cache() -> None:
    """切换主题后清空图标缓存，让图标跟着换色。"""
    _cache.clear()
