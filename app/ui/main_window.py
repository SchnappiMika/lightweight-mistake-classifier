# -*- coding: utf-8 -*-
"""主窗口（需求 4.3.3 交互界面设计）。

三栏布局：
    左：功能导航栏（五大核心功能入口 + 图标 + 悬浮提示 + 新手引导入口 + 主题切换）
    中：核心内容展示区（QStackedWidget，随导航动态切换）
    右：快捷操作按钮区（内容随当前页动态替换）
顶部为标题栏（页面标题 + 环境自检），底部为状态栏。
"""
from __future__ import annotations

import logging
from html import escape
from typing import List, Optional

from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QFrame, QHBoxLayout, QLabel,
                             QMainWindow, QStackedWidget, QTextBrowser, QVBoxLayout,
                             QWidget)

from app import config
from app.ui import guide as guide_mod
from app.ui import icons, theme
from app.ui.context import AppContext
from app.ui.pages.page_classify import ClassifyPage
from app.ui.pages.page_export import ExportPage
from app.ui.pages.page_feedback import FeedbackPage
from app.ui.pages.page_import import ImportPage
from app.ui.pages.page_scheme import SchemePage
from app.ui.widgets import NavBar, QuickActionPanel, Toast, button, label
from app.ui.workers import WarmupWorker, stop_worker

log = logging.getLogger(__name__)

# (标题, 图标, 悬浮提示)
NAV_ENTRIES = [
    ("导入与预览", "import", "单条录入 / TXT·Excel 批量导入 / 公式图片·图形题·听力音频导入"),
    ("分类体系管理", "scheme", "零代码搭建你自己的分类体系：标签、关键词、标准样本、权重"),
    ("一键分类与修正", "classify", "多模态预处理 + 关键词粗筛 + 多特征融合精匹配"),
    ("反馈与增量优化", "feedback", "查看系统从你的修正里学到了什么，权重怎么变的"),
    ("导出与归档", "export", "按任意维度分 sheet 导出 Excel，图片音频分文件夹归档"),
]


class EnvDialog(QDialog):
    """环境自检：把可选识别引擎的可用性摊开告诉用户。

    这些引擎（pix2tex / PaddleOCR / Whisper）体积大，不是每个人都会装。
    系统的设计是"没装也能用，只是相应模态降级"，所以必须让用户清楚
    现在处于哪种状态，而不是默默地识别不出来。
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("环境自检")
        self.setMinimumSize(600, 400)
        v = QVBoxLayout(self)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(False)
        v.addWidget(self.view)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.button(QDialogButtonBox.Close).setText("关闭")
        bb.addButton("刷新状态", QDialogButtonBox.ActionRole).clicked.connect(self._load)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        v.addWidget(bb)
        QTimer.singleShot(0, self._load)

    def _load(self) -> None:
        from app.core.preprocess import pipeline
        p = theme.palette()
        rows: List[str] = []
        for st in pipeline.engine_report():
            color = p["success"] if st.available else p["warn"]
            mark = st.state_label
            detail = escape(st.detail).replace("\n", "<br>")
            action = escape(st.action).replace("\n", "<br>")
            rows.append(
                '<tr><td style="padding:7px 12px 7px 0;"><b>{}</b></td>'
                '<td style="padding:7px 12px 7px 0;color:{};white-space:nowrap;">{}</td>'
                '<td style="padding:7px 0;color:{};">{}</td></tr>'.format(
                    escape(st.name), color, escape(mark), p["text_dim"],
                    detail + ("<br><b>下一步：</b>" + action if action else "")))
        html = """
        <div style="font-family:{font};color:{text};">
        <h3 style="margin:0 0 4px 0;">多模态识别引擎</h3>
        <p style="color:{dim};margin:0 0 10px 0;font-size:13px;">
        以下识别与英文增强组件为<b>可选</b>组件。自检仅读取安装信息、本地模型文件状态和已有识别状态，
        不加载模型、不下载模型。检测到组件不等于模型可用；加载成功也不代表实际识别效果已经验收。
        缺少组件时，仍可手动录入题干、内联公式和标注进行分类。
        </p>
        <table style="font-size:13px;">{rows}</table>
        <h3 style="margin:16px 0 4px 0;">核心组件</h3>
        <p style="color:{dim};font-size:13px;margin:0;">
        jieba 分词、scikit-learn（TF-IDF + 余弦相似度）、pandas（Excel 读写）、
        PyQt5（界面）为必需依赖，随程序一起安装，全部本地运行。
        </p>
        <h3 style="margin:16px 0 4px 0;">网络</h3>
        <p style="color:{ok};font-size:13px;margin:0;">
        核心文本分类在本地运行。印刷文字识别通过“安装图片识别.bat”准备本地组件和模型，
        识别使用独立进程，不联网下载。旧公式和音频引擎仍需另外验证，首次初始化可能下载模型。
        本窗口不执行安装、下载或识别。
        </p>
        <h3 style="margin:16px 0 4px 0;">数据位置</h3>
        <p style="color:{dim};font-size:13px;margin:0;">
        {ws}<br>错题、分类体系、反馈记录与导出文件保存在这里。
        请使用主窗口顶部“备份与恢复”创建备份或迁移数据。
        </p>
        </div>
        """.format(font=theme.font_stack(), text=p["text"], dim=p["text_dim"],
                   ok=p["success"], rows="".join(rows), ws=escape(str(config.WORKSPACE_DIR)))
        self.view.setHtml(html)


class MainWindow(QMainWindow):
    """主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.ctx = AppContext(self)
        theme.set_current(self.ctx.theme_name())
        self._quick_actions_cache = {}
        self._dirty_pages = set(range(5))

        self.setWindowTitle("{} v{}".format(config.APP_NAME, config.APP_VERSION))
        self.setWindowIcon(icons.app_icon())
        self.resize(1380, 850)
        self.setMinimumSize(1120, 700)

        self._build()
        self._connect()
        self._warmup()
        QTimer.singleShot(320, self._maybe_guide)

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        # ---- 左：功能导航栏 ---- #
        self.nav = NavBar(NAV_ENTRIES)
        h.addWidget(self.nav)

        # ---- 中 + 右 ---- #
        rightwrap = QWidget()
        rv = QVBoxLayout(rightwrap)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)

        # 顶部标题栏
        top = QFrame()
        top.setObjectName("TopBar")
        top.setFixedHeight(68)
        th = QHBoxLayout(top)
        th.setContentsMargins(18, 8, 14, 8)
        th.setSpacing(10)
        tv = QVBoxLayout()
        tv.setSpacing(1)
        self.lb_title = label("", "h1")
        self.lb_sub = label("", "dim")
        tv.addWidget(self.lb_title)
        tv.addWidget(self.lb_sub)
        th.addLayout(tv)
        th.addStretch(1)
        self.lb_env = QLabel("")
        self.lb_env.setObjectName("Badge")
        th.addWidget(self.lb_env)
        self.btn_backup = button("备份与恢复", "ghost", "save", self._show_backup)
        th.addWidget(self.btn_backup)
        self.btn_stop_task = button("停止任务", "", "stop", self._stop_task)
        self.btn_stop_task.hide()
        th.addWidget(self.btn_stop_task)
        th.addWidget(button("环境自检", "ghost", "info", self._show_env,
                            "查看公式识别 / 图形 OCR / 听力转写引擎是否就绪"))
        rv.addWidget(top)

        # 内容 + 快捷操作
        body = QWidget()
        bh = QHBoxLayout(body)
        bh.setContentsMargins(14, 12, 14, 12)
        bh.setSpacing(12)

        self.stack = QStackedWidget()
        self.pages = [
            ImportPage(self.ctx), SchemePage(self.ctx), ClassifyPage(self.ctx),
            FeedbackPage(self.ctx), ExportPage(self.ctx),
        ]
        for p in self.pages:
            self.stack.addWidget(p)
        bh.addWidget(self.stack, 1)

        self.quick = QuickActionPanel()
        bh.addWidget(self.quick)
        rv.addWidget(body, 1)
        h.addWidget(rightwrap, 1)
        self.setCentralWidget(central)

        self.statusBar().showMessage("就绪")
        self.toast = Toast(self)
        self.overlay = guide_mod.GuideOverlay(self, guide_mod.default_steps(self))
        self._switch(0)

    def _connect(self) -> None:
        self.nav.switched.connect(self._switch)
        self.nav.theme_toggled.connect(self._toggle_theme)
        self.nav.guide_requested.connect(self._start_guide)
        self.ctx.status.connect(lambda m: self.statusBar().showMessage(m, 8000))
        self.ctx.toast.connect(lambda m, k: self.toast.show_msg(m, k))
        self.overlay.page_requested.connect(self._switch_from_guide)
        self.overlay.finished.connect(
            lambda: self.ctx.set_setting("guide_done", True))
        self.ctx.task_changed.connect(self._on_task_changed)
        self.ctx.workspace_reloaded.connect(self._workspace_reloaded)
        self.ctx.questions_changed.connect(lambda: self._invalidate_pages((0, 2, 3, 4)))
        self.ctx.question_edited.connect(lambda: self._invalidate_pages((0, 2, 3, 4)))
        self.ctx.scheme_changed.connect(lambda: self._invalidate_pages((1, 2, 3, 4)))
        self.ctx.feedback_changed.connect(lambda: self._invalidate_pages((2, 3)))

    def _invalidate_pages(self, indices):
        self._dirty_pages.update(indices)

    def _clear_quick_actions(self):
        self.quick.clear_actions()
        for _hint, widgets in self._quick_actions_cache.values():
            for widget in widgets:
                widget.hide()
                widget.deleteLater()
        self._quick_actions_cache.clear()

    def _workspace_reloaded(self):
        from PyQt5.QtWidgets import QApplication
        theme.apply(QApplication.instance(), self.ctx.theme_name())
        icons.clear_cache()
        self.nav.refresh_icons(NAV_ENTRIES)
        self.setWindowIcon(icons.app_icon())
        self._invalidate_pages(range(len(self.pages)))
        self._clear_quick_actions()

    def _on_task_changed(self, task):
        for widget in (self.stack, self.quick, self.nav, self.btn_backup):
            widget.setEnabled(not task)
        self.btn_stop_task.setVisible(task in ("导入", "分类"))
        self.statusBar().showMessage("正在{}，编辑暂时锁定".format(task) if task else "就绪")

    def _stop_task(self):
        if self.ctx.active_task == "分类":
            self.pages[2]._stop()
        elif self.ctx.active_task == "导入" and self.pages[0]._worker is not None:
            self.pages[0]._worker.stop()
            self.statusBar().showMessage("已请求停止导入，当前文件处理完后结束")

    def _show_backup(self):
        if self.ctx.active_task:
            self.toast.show_msg("请等待当前任务结束", "warn")
            return
        from app.ui.backup_dialog import BackupDialog
        BackupDialog(self.ctx, self).exec_()
        self._switch(self.stack.currentIndex())

    # ------------------------------------------------------------------ #
    def _switch(self, index: int) -> None:
        if not (0 <= index < len(self.pages)):
            return
        self.stack.setCurrentIndex(index)
        self.nav.select(index)
        page = self.pages[index]
        self.lb_title.setText(page.title)
        self.lb_sub.setText(page.subtitle)
        if index in self._dirty_pages:
            page.on_show()
            self._dirty_pages.discard(index)
        cached = self._quick_actions_cache.get(index)
        if cached is None:
            cached = page.quick_actions()
            self._quick_actions_cache[index] = cached
        hint, widgets = cached
        self.quick.set_actions(hint, widgets)

    def _switch_from_guide(self, index: int) -> None:
        self._switch(index)
        self.overlay.raise_()

    # ------------------------------------------------------------------ #
    def _toggle_theme(self) -> None:
        new = (config.THEME_DARK if theme.current() == config.THEME_LIGHT
               else config.THEME_LIGHT)
        from PyQt5.QtWidgets import QApplication
        theme.set_current(new)
        icons.clear_cache()
        theme.apply(QApplication.instance(), new)
        self.nav.refresh_icons(NAV_ENTRIES)
        self.setWindowIcon(icons.app_icon())
        self.ctx.set_setting("theme", new)
        # 快捷操作按钮包含主题色图标，主题切换时才重建；普通页面切换复用缓存。
        self._clear_quick_actions()
        self._invalidate_pages((2, 3))
        # 快捷操作按钮内嵌了图标，需重建才能跟着换色
        self._switch(self.stack.currentIndex())
        self.toast.show_msg("已切换到{}主题".format("深色" if new == config.THEME_DARK else "浅色"), "ok")

    # ------------------------------------------------------------------ #
    def _warmup(self) -> None:
        self.lb_env.setText("正在加载词典…")
        self._w = WarmupWorker(self)
        self._w.done.connect(self._on_warm)
        self._w.start()

    def _on_warm(self, ok: bool, msg: str) -> None:
        from app.core.preprocess import pipeline
        if not ok:
            self.lb_env.setObjectName("BadgeWarn")
            self.lb_env.setText("词典加载失败")
            self.lb_env.setToolTip(msg)
            return
        report = pipeline.engine_report()
        n_ok = sum(1 for r in report if r.available)
        self.lb_env.setObjectName("Badge" if n_ok == len(report) else "BadgeWarn")
        self.lb_env.setStyleSheet("")     # 触发 QSS 按新 objectName 重刷
        self.lb_env.setText("文本就绪 · 增强组件 {}/{}".format(n_ok, len(report)))
        self.lb_env.setToolTip(
            "{}\n\n" .format(msg)
            + "已加载组件数不代表真实媒体识别已验收。\n"
            + "\n".join("{}：{}".format(r.name, r.state_label)
                        for r in report)
            + "\n\n点击右侧「环境自检」查看详情")
        self.statusBar().showMessage("文本分类已就绪；图片与音频识别状态请查看环境自检", 6000)

    def _show_env(self) -> None:
        EnvDialog(self).exec_()

    # ------------------------------------------------------------------ #
    def _maybe_guide(self) -> None:
        if not self.ctx.setting("guide_done", False):
            self._start_guide()

    def _start_guide(self) -> None:
        self.overlay.setGeometry(self.rect())
        self.overlay.start()

    # ------------------------------------------------------------------ #
    def resizeEvent(self, e) -> None:      # noqa: N802
        super().resizeEvent(e)
        if self.overlay.isVisible():
            self.overlay.setGeometry(self.rect())

    def closeEvent(self, e) -> None:       # noqa: N802
        if self.ctx.active_task in ("备份", "恢复", "图片识别", "公式识别", "听力转写"):
            e.ignore()
            return
        if self.ctx.active_task in ("导入", "分类"):
            self._stop_task()
            self.toast.show_msg("正在停止任务并保存已完成内容，任务结束后请再次关闭窗口", "warn")
            e.ignore()
            return
        # 先收线程再存盘：QThread 在仍运行时被析构会让 Qt 直接 abort 掉进程
        # （用户表现为"关窗闪退"）。预热线程要跑好几秒，这段时间里关窗很常见。
        for p in self.pages:
            try:
                p.shutdown()
            except Exception:              # noqa: BLE001
                log.exception("关闭页面 %s 的后台线程失败", type(p).__name__)
        stop_worker(getattr(self, "_w", None))
        try:
            self.ctx.repo.save_all()
        except Exception:                  # noqa: BLE001
            log.exception("退出时保存失败")
        super().closeEvent(e)
