# -*- coding: utf-8 -*-
"""界面主题（需求 4.3.3「可选功能：支持界面主题切换（浅色/深色）」）。

集中管理配色与 QSS，页面代码里不出现硬编码颜色，切换主题时只需重新 apply 一次。
字体栈针对三大平台分别兜底，避免 Windows 上中文界面掉成宋体锯齿。
"""
from __future__ import annotations

import sys
from typing import Dict

from app import config

LIGHT: Dict[str, str] = {
    "bg": "#F4F6F8",
    "surface": "#FFFFFF",
    "surface_alt": "#FAFBFC",
    "surface_hover": "#EEF2F7",
    "border": "#DFE3E9",
    "border_strong": "#C6CCD5",
    "text": "#1F2328",
    "text_dim": "#6B7280",
    "text_faint": "#9AA3AF",
    "primary": "#2F6FED",
    "primary_hover": "#2560D8",
    "primary_soft": "#E8F0FE",
    "accent": "#0E9F9A",
    "accent_soft": "#E3F6F5",
    "success": "#16A34A",
    "warn": "#D97706",
    "warn_soft": "#FEF3C7",
    "danger": "#DC2626",
    "danger_soft": "#FEE2E2",
    "nav_bg": "#FFFFFF",
    "nav_border": "#E8ECF2",
    "nav_title": "#1B2330",
    "nav_text": "#5A6472",
    "nav_text_active": "#2F6FED",
    "nav_muted": "#9AA3AF",
    "nav_hover": "#F2F5FA",
    "nav_active_bg": "#EAF1FD",
    "nav_bar": "#2F6FED",
    "nav_active": "#2F6FED",
    "shadow": "rgba(15, 23, 42, 0.06)",
}

DARK: Dict[str, str] = {
    "bg": "#14171C",
    "surface": "#1B1F26",
    "surface_alt": "#21262E",
    "surface_hover": "#282E38",
    "border": "#2E343D",
    "border_strong": "#3D444F",
    "text": "#E6E9EF",
    "text_dim": "#98A2B3",
    "text_faint": "#6B7480",
    "primary": "#5B8DEF",
    "primary_hover": "#7AA3F5",
    "primary_soft": "#1E2A44",
    "accent": "#2DD4BF",
    "accent_soft": "#12312F",
    "success": "#4ADE80",
    "warn": "#FBBF24",
    "warn_soft": "#3A2E12",
    "danger": "#F87171",
    "danger_soft": "#3B1D1D",
    "nav_bg": "#12151B",
    "nav_border": "#242A33",
    "nav_title": "#F1F4F9",
    "nav_text": "#98A2B3",
    "nav_text_active": "#FFFFFF",
    "nav_muted": "#6B7480",
    "nav_hover": "rgba(255,255,255,0.06)",
    "nav_active_bg": "rgba(91,141,239,0.18)",
    "nav_bar": "#5B8DEF",
    "nav_active": "#5B8DEF",
    "shadow": "rgba(0, 0, 0, 0.35)",
}

PALETTES = {config.THEME_LIGHT: LIGHT, config.THEME_DARK: DARK}
_current = config.DEFAULT_THEME


def font_stack() -> str:
    """跨平台中文字体栈（对应"跨 Windows / MacOS 兼容"目标）。"""
    if sys.platform.startswith("win"):
        return '"Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif'
    if sys.platform == "darwin":
        return '"PingFang SC", "Helvetica Neue", "Hiragino Sans GB", sans-serif'
    return '"Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei", sans-serif'


def palette(name: str = "") -> Dict[str, str]:
    return PALETTES.get(name or _current, LIGHT)


def current() -> str:
    return _current


def set_current(name: str) -> None:
    global _current
    _current = name if name in PALETTES else config.DEFAULT_THEME


QSS_TEMPLATE = """
* {{ font-family: {font}; outline: none; }}

QWidget {{ background: {bg}; color: {text}; font-size: 14px; }}
QMainWindow, QDialog {{ background: {bg}; }}
QToolTip {{
    background: {surface}; color: {text}; border: 1px solid {border_strong};
    padding: 6px 8px; border-radius: 6px; font-size: 13px;
}}

/* ---------- 卡片容器 ---------- */
QFrame#Card, QWidget#Card {{
    background: {surface}; border: 1px solid {border}; border-radius: 10px;
}}
QFrame#TopBar {{
    background: {surface}; border: none; border-bottom: 1px solid {border};
}}

/* ---------- 左侧功能导航栏 ---------- */
QWidget#NavBar {{ background: {nav_bg}; border: none; border-right: 1px solid {nav_border}; }}
QLabel#NavTitle {{ color: {nav_title}; font-size: 16px; font-weight: bold; background: transparent; }}
QLabel#NavSubtitle {{ color: {nav_muted}; font-size: 12px; background: transparent; }}
QLabel#NavStatus {{
    color: {success}; font-size: 12px; font-weight: bold;
    background: rgba(22,163,74,0.10); border: 1px solid rgba(22,163,74,0.30);
    border-radius: 10px; padding: 3px 11px;
}}
QLabel#NavEyebrow {{ color: {nav_muted}; font-size: 11px; font-weight: bold; background: transparent; }}
QLabel#NavFrac {{ color: {nav_text}; font-size: 12px; font-weight: bold; background: transparent; }}

/* 每个功能入口是一个 QFrame（内含序号 + 图标 + 文字），激活态用左色条 + 浅色高亮 */
QFrame#NavItem {{ background: transparent; border: none; border-left: 3px solid transparent; border-radius: 9px; }}
QFrame#NavItem:hover {{ background: {nav_hover}; }}
QFrame#NavItem[active="true"] {{ background: {nav_active_bg}; border-left: 3px solid {nav_bar}; }}
QLabel#NavNum {{ color: {nav_muted}; font-size: 12px; font-weight: bold; background: transparent; }}
QLabel#NavText {{ color: {nav_text}; font-size: 14px; background: transparent; }}
QLabel#NavIco {{ background: transparent; }}
QFrame#NavItem[active="true"] QLabel#NavNum {{ color: {nav_text_active}; }}
QFrame#NavItem[active="true"] QLabel#NavText {{ color: {nav_text_active}; font-weight: bold; }}

QToolButton#NavMini {{ background: transparent; border: none; border-radius: 8px; }}
QToolButton#NavMini:hover {{ background: {nav_hover}; }}
QFrame#NavDivider {{ background: {nav_border}; border: none; }}
QLabel#NavFooter {{ color: {nav_muted}; font-size: 12px; background: transparent; }}

/* ---------- 按钮 ---------- */
QPushButton {{
    background: {surface}; color: {text}; border: 1px solid {border_strong};
    border-radius: 7px; padding: 7px 14px; font-size: 14px;
}}
QPushButton:hover {{ background: {surface_hover}; border-color: {primary}; }}
QPushButton:pressed {{ background: {primary_soft}; }}
QPushButton:disabled {{ color: {text_faint}; border-color: {border}; background: {surface_alt}; }}
QPushButton[kind="primary"] {{
    background: {primary}; color: #FFFFFF; border: 1px solid {primary}; font-weight: bold;
}}
QPushButton[kind="primary"]:hover {{ background: {primary_hover}; border-color: {primary_hover}; }}
QPushButton[kind="primary"]:disabled {{ background: {border_strong}; border-color: {border_strong}; color: {surface}; }}
QPushButton[kind="accent"] {{ background: {accent}; color: #FFFFFF; border: 1px solid {accent}; font-weight: bold; }}
QPushButton[kind="danger"] {{ color: {danger}; border-color: {danger}; }}
QPushButton[kind="danger"]:hover {{ background: {danger_soft}; }}
QPushButton[kind="ghost"] {{ background: transparent; border: none; color: {text_dim}; }}
QPushButton[kind="ghost"]:hover {{ background: {surface_hover}; color: {text}; }}

/* ---------- 输入 ---------- */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background: {surface}; color: {text}; border: 1px solid {border_strong};
    border-radius: 7px; padding: 6px 9px; selection-background-color: {primary};
    selection-color: #FFFFFF;
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {primary}; }}
QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {{
    background: {surface_alt}; color: {text_faint};
}}
QComboBox {{
    background: {surface}; color: {text}; border: 1px solid {border_strong};
    border-radius: 7px; padding: 6px 9px; min-height: 18px;
}}
QComboBox:hover {{ border-color: {primary}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: none; border-left: 4px solid transparent;
    border-right: 4px solid transparent; border-top: 5px solid {text_dim}; margin-right: 8px; }}
QComboBox QAbstractItemView {{
    background: {surface}; color: {text}; border: 1px solid {border_strong};
    selection-background-color: {primary_soft}; selection-color: {text}; outline: none;
    padding: 4px;
}}

/* ---------- 列表 / 树 / 表格 ---------- */
QListWidget, QTreeWidget, QTableWidget, QTableView, QTreeView, QListView {{
    background: {surface}; color: {text}; border: 1px solid {border};
    border-radius: 8px; alternate-background-color: {surface_alt};
    selection-background-color: {primary_soft}; selection-color: {text}; outline: none;
}}
QListWidget::item, QTreeWidget::item {{ padding: 6px 4px; border-radius: 5px; }}
QListWidget::item:hover, QTreeWidget::item:hover {{ background: {surface_hover}; }}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: {primary_soft}; color: {text};
}}
QTableWidget::item {{ padding: 4px 6px; }}
QTableWidget::item:selected {{ background: {primary_soft}; color: {text}; }}
QHeaderView::section {{
    background: {surface_alt}; color: {text_dim}; border: none;
    border-bottom: 1px solid {border}; border-right: 1px solid {border};
    padding: 7px 6px; font-weight: bold; font-size: 13px;
}}
QTableCornerButton::section {{ background: {surface_alt}; border: none; }}

/* ---------- 分组框 ---------- */
QGroupBox {{
    background: {surface}; border: 1px solid {border}; border-radius: 9px;
    margin-top: 14px; padding: 12px 10px 10px 10px; font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left; left: 12px; top: 2px;
    padding: 0 6px; color: {text_dim}; font-size: 13px;
}}

/* ---------- 滚动条 ---------- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {border_strong}; border-radius: 5px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {text_faint}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {border_strong}; border-radius: 5px; min-width: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---------- 进度条 ---------- */
QProgressBar {{
    background: {surface_alt}; border: 1px solid {border}; border-radius: 7px;
    height: 18px; text-align: center; color: {text_dim}; font-size: 12px;
}}
QProgressBar::chunk {{ background: {primary}; border-radius: 6px; }}

/* ---------- 其他 ---------- */
QSplitter::handle {{ background: {border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QCheckBox, QRadioButton {{ spacing: 7px; color: {text}; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px; border: 1px solid {border_strong};
    background: {surface}; border-radius: 4px;
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {primary}; border-color: {primary};
}}
QStatusBar {{ background: {surface}; color: {text_dim}; border-top: 1px solid {border}; }}
QStatusBar::item {{ border: none; }}
QTabWidget::pane {{ border: 1px solid {border}; border-radius: 8px; background: {surface}; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {text_dim}; padding: 7px 16px;
    border: 1px solid transparent; border-bottom: none;
    border-top-left-radius: 7px; border-top-right-radius: 7px;
}}
QTabBar::tab:selected {{
    background: {surface}; color: {primary}; font-weight: bold;
    border-color: {border}; border-bottom-color: {surface};
}}
QMenu {{ background: {surface}; border: 1px solid {border_strong}; border-radius: 8px; padding: 5px; }}
QMenu::item {{ padding: 6px 22px 6px 14px; border-radius: 5px; }}
QMenu::item:selected {{ background: {primary_soft}; }}

/* ---------- 语义化文本 ---------- */
QLabel[role="h1"] {{ font-size: 18px; font-weight: bold; color: {text}; }}
QLabel[role="h2"] {{ font-size: 15px; font-weight: bold; color: {text}; }}
QLabel[role="dim"] {{ color: {text_dim}; font-size: 13px; }}
QLabel[role="faint"] {{ color: {text_faint}; font-size: 12px; }}
QLabel[role="ok"] {{ color: {success}; font-weight: bold; }}
QLabel[role="warn"] {{ color: {warn}; font-weight: bold; }}
QLabel[role="danger"] {{ color: {danger}; font-weight: bold; }}
QLabel#Badge {{
    background: {primary_soft}; color: {primary}; border-radius: 9px;
    padding: 3px 9px; font-size: 12px; font-weight: bold;
}}
QLabel#BadgeWarn {{
    background: {warn_soft}; color: {warn}; border-radius: 9px;
    padding: 3px 9px; font-size: 12px; font-weight: bold;
}}
"""


def stylesheet(name: str = "") -> str:
    p = dict(palette(name))
    p["font"] = font_stack()
    return QSS_TEMPLATE.format(**p)


def apply(app, name: str = "") -> None:
    """把主题应用到整个 QApplication。"""
    if name:
        set_current(name)
    app.setStyleSheet(stylesheet(current()))
