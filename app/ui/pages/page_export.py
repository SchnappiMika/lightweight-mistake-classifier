# -*- coding: utf-8 -*-
"""功能模块 5：个性化分类结果导出（需求 4.3.2 第 5 点）。

* 按任意分类维度分 sheet 导出 Excel（知识点 / 错误类型 / 难度 / 复习优先级 / 自定义）
* 按分类标签把图片、音频分文件夹归档，并生成 index.csv 索引
* 导出前提供结构预览，避免导完才发现分错了地方
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLineEdit,
                             QMessageBox, QSplitter, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from app import config
from app.core.dataio import exporter
from app.ui.pages.base_page import BasePage
from app.ui.widgets import Card, SectionTitle, StatChip, batch_view_updates, button, hline, label


class ExportPage(BasePage):
    """个性化分类结果导出页。"""

    title = "个性化分类结果导出"
    subtitle = "按你自己的维度分 sheet 归档；图片音频按标签分文件夹，便于后续打印整理"

    def __init__(self, ctx, parent: Optional[QWidget] = None) -> None:
        super().__init__(ctx, parent)
        self._last_out: str = ""
        self._build()
        ctx.questions_changed.connect(self.refresh)
        ctx.question_edited.connect(self.refresh)
        ctx.scheme_changed.connect(self._refresh_scheme_box)

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        chips = QHBoxLayout()
        chips.setSpacing(8)
        self.chip_total = StatChip("错题总数")
        self.chip_done = StatChip("已分类")
        self.chip_media = StatChip("含图片/音频")
        self.chip_fixed = StatChip("已人工确认")
        for c in (self.chip_total, self.chip_done, self.chip_media, self.chip_fixed):
            chips.addWidget(c)
        chips.addStretch(1)
        self.root.addLayout(chips)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        # ---------------- 左：导出设置 ---------------- #
        left = Card(margin=12, spacing=9)
        left.box.addWidget(SectionTitle("导出设置", ""))

        left.box.addWidget(label("分类体系", "dim"))
        self.cb_scheme = QComboBox()
        self.cb_scheme.currentIndexChanged.connect(self._on_scheme_changed)
        left.box.addWidget(self.cb_scheme)

        left.box.addWidget(label("按哪个维度分 sheet 归档", "dim"))
        self.cb_dim = QComboBox()
        self.cb_dim.currentIndexChanged.connect(self.refresh)
        left.box.addWidget(self.cb_dim)

        self.ck_only_done = QCheckBox("只导出已完成分类的错题")
        self.ck_only_done.setChecked(True)
        self.ck_only_done.stateChanged.connect(self.refresh)
        left.box.addWidget(self.ck_only_done)

        self.ck_filter_subject = QCheckBox("只导出当前体系对应学科的错题")
        self.ck_filter_subject.setChecked(True)
        self.ck_filter_subject.stateChanged.connect(self.refresh)
        left.box.addWidget(self.ck_filter_subject)

        left.box.addWidget(hline())
        left.box.addWidget(label("输出位置", "dim"))
        row = QHBoxLayout()
        row.setSpacing(5)
        self.ed_out = QLineEdit(str(config.EXPORT_DIR))
        row.addWidget(self.ed_out, 1)
        row.addWidget(button("浏览", "", "folder", self._pick_dir))
        left.box.addLayout(row)
        left.box.addWidget(label(
            "默认导出到工作区的 exports 目录；这个目录始终可写，"
            "打包成 exe 后也不会有权限问题。", "faint"))
        left.box.addStretch(1)
        left.setFixedWidth(298)
        split.addWidget(left)

        # ---------------- 右：结构预览 ---------------- #
        right = Card(margin=10, spacing=8)
        self.sec_prev = SectionTitle("导出结构预览", "")
        right.box.addWidget(self.sec_prev)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Sheet / 文件夹", "错题数", "说明"])
        self.tree.setColumnWidth(0, 300)
        self.tree.setColumnWidth(1, 72)
        self.tree.setAlternatingRowColors(True)
        right.box.addWidget(self.tree, 1)
        self.lb_tip = label("", "faint")
        self.lb_tip.setWordWrap(True)
        right.box.addWidget(self.lb_tip)
        split.addWidget(right)

        split.setStretchFactor(1, 1)
        self.root.addWidget(split)
        self._refresh_scheme_box()
        self.refresh()

    # ------------------------------------------------------------------ #
    def quick_actions(self) -> Tuple[str, List[QWidget]]:
            return ("导出的 Excel 自带「全部错题」总表与「分类统计」汇总表，"
                "列宽已按内容估算，便于检查和后续打印整理。", [
            button("一键导出 Excel", "primary", "export", self._export_excel),
            button("导出媒体归档", "", "folder", self._export_media,
                   "按分类标签把图片 / 音频分文件夹复制出来"),
            button("同时导出两者", "accent", "save", self._export_both),
            hline(),
            button("打开输出目录", "", "folder", self._open_dir),
            button("刷新预览", "ghost", "refresh", self.refresh),
        ])

    # ------------------------------------------------------------------ #
    def _refresh_scheme_box(self) -> None:
        cur = self.ctx.current_scheme()
        self.cb_scheme.blockSignals(True)
        self.cb_scheme.clear()
        for s in self.ctx.schemes:
            self.cb_scheme.addItem("{}（{}）".format(s.name, s.subject), s.id)
        if cur:
            i = self.cb_scheme.findData(cur.id)
            if i >= 0:
                self.cb_scheme.setCurrentIndex(i)
        self.cb_scheme.blockSignals(False)
        self._refresh_dims()

    def _refresh_dims(self) -> None:
        sc = self._scheme()
        self.cb_dim.blockSignals(True)
        prev = self.cb_dim.currentData()
        self.cb_dim.clear()
        self.cb_dim.addItem("不分 sheet（只出一张总表）", "")
        if sc:
            for d in sc.dims:
                if d.enabled and d.labels:
                    self.cb_dim.addItem(d.name, d.key)
        i = self.cb_dim.findData(prev or config.DIM_KNOWLEDGE)
        self.cb_dim.setCurrentIndex(i if i >= 0 else min(1, self.cb_dim.count() - 1))
        self.cb_dim.blockSignals(False)
        self.refresh()

    def _on_scheme_changed(self) -> None:
        sid = self.cb_scheme.currentData()
        if sid:
            self.ctx.set_current_scheme(sid)
        self._refresh_dims()

    def _scheme(self):
        sid = self.cb_scheme.currentData()
        return self.ctx.repo.scheme_by_id(sid) if sid else self.ctx.current_scheme()

    def _items(self):
        sc = self._scheme()
        out = []
        # 分类过的题目带有 scheme_id；按体系筛选后不会把同学科的另一套体系混入。
        # 对旧工作区中尚未写入 scheme_id 的全量记录，exporter 保留兼容回退。
        scoped = exporter.scope_questions(self.ctx.questions, sc)
        for q in scoped:
            if self.ck_only_done.isChecked() and not q.classified:
                continue
            if (self.ck_filter_subject.isChecked() and sc
                    and sc.subject != config.SUBJECT_GENERAL and q.subject != sc.subject):
                continue
            out.append(q)
        return out

    # ------------------------------------------------------------------ #
    @batch_view_updates("tree")
    def refresh(self, *_signal_args) -> None:
        sc = self._scheme()
        items = self._items()
        dim_key = self.cb_dim.currentData()

        self.chip_total.set_value(str(len(self.ctx.questions)))
        self.chip_done.set_value(str(sum(1 for q in self.ctx.questions if q.classified)))
        self.chip_media.set_value(str(sum(1 for q in self.ctx.questions if q.media_path)))
        self.chip_fixed.set_value(str(sum(1 for q in self.ctx.questions
                                          if any(q.corrected.values()))))

        self.tree.clear()
        if not items:
            self.sec_prev.set_hint("当前筛选条件下没有可导出的错题")
            self.lb_tip.setText(
                "先去「导入」页导入错题，再去「一键分类」页分类，然后回来导出。")
            return
        self.sec_prev.set_hint("待导出 {} 条".format(len(items)))

        root = QTreeWidgetItem(self.tree, ["错题分类结果.xlsx", str(len(items)), "Excel 文件"])
        f = root.font(0)
        f.setBold(True)
        root.setFont(0, f)
        QTreeWidgetItem(root, ["全部错题", str(len(items)), "总表，含全部维度的分类结果与置信度"])
        if sc:
            n_dims = sum(1 for d in sc.dims if d.enabled)
            QTreeWidgetItem(root, ["分类统计", str(n_dims), "各维度下每个标签的错题数与占比"])
        if dim_key and sc:
            dim = sc.dim(dim_key)
            for name, cnt in exporter.preview_structure(items, sc, dim_key):
                QTreeWidgetItem(root, ["{}-{}".format(dim.name if dim else dim_key, name),
                                       str(cnt), "独立 sheet"])
        root.setExpanded(True)

        media = [q for q in items if q.media_path and Path(q.media_path).exists()]
        if media:
            mroot = QTreeWidgetItem(self.tree, ["错题媒体归档/", str(len(media)), "文件夹"])
            mroot.setFont(0, f)
            if dim_key and sc:
                dim = sc.dim(dim_key)
                sub = {}
                for q in media:
                    sub.setdefault(q.label_names.get(dim_key, "") or "未分类", 0)
                    sub[q.label_names.get(dim_key, "") or "未分类"] += 1
                for name, cnt in sub.items():
                    QTreeWidgetItem(mroot, ["{}/{}/".format(
                        dim.name if dim else dim_key, name), str(cnt), "图片 / 音频"])
            QTreeWidgetItem(mroot, ["index.csv", str(len(media)), "归档索引，Excel 可直接打开"])
            mroot.setExpanded(True)

        undone = sum(1 for q in items if not q.classified)
        tips = []
        if undone:
            tips.append("有 {} 条还没分类，导出后对应标签会是空的。".format(undone))
        unclassified = sum(1 for q in items
                           if dim_key and not q.label_names.get(dim_key))
        if dim_key and unclassified:
            tips.append("有 {} 条在该维度下没有标签，会被归到「未分类」sheet。".format(unclassified))
        if not media:
            tips.append("当前没有带图片 / 音频的错题，媒体归档会跳过。")
        self.lb_tip.setText(" ".join(tips) if tips else
                            "一切就绪，点右侧「一键导出 Excel」即可。")

    # ------------------------------------------------------------------ #
    def _pick_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", self.ed_out.text())
        if d:
            self.ed_out.setText(d)

    def _out_dir(self) -> Path:
        p = Path(self.ed_out.text().strip() or str(config.EXPORT_DIR))
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            p = config.EXPORT_DIR
            p.mkdir(parents=True, exist_ok=True)
            self.ed_out.setText(str(p))
        return p

    def _export_excel(self) -> Optional[str]:
        items = self._items()
        if not items:
            self.toast("没有可导出的错题", "warn")
            return None
        sc = self._scheme()
        out = self._out_dir() / exporter.default_export_path().name
        try:
            path = exporter.export_excel(items, sc, str(out),
                                         split_dim=self.cb_dim.currentData() or None,
                                         only_classified=self.ck_only_done.isChecked(),
                                         strict_scheme=False)
        except Exception as exc:            # noqa: BLE001
            QMessageBox.warning(self, "导出失败", str(exc))
            self.toast("导出失败：{}".format(exc), "error")
            return None
        self._last_out = str(Path(path).parent)
        self.toast("已导出 {} 条错题 → {}".format(len(items), Path(path).name), "ok")
        self.status("导出完成：{}".format(path))
        return path

    def _export_media(self) -> Optional[str]:
        items = self._items()
        sc = self._scheme()
        out = self._out_dir() / exporter.default_export_path("错题媒体归档", "").name
        try:
            path, n = exporter.export_media_archive(
                items, sc, str(out), dim_key=self.cb_dim.currentData() or config.DIM_KNOWLEDGE,
                strict_scheme=False)
        except ValueError as exc:
            self.toast(str(exc), "warn")
            return None
        except Exception as exc:            # noqa: BLE001
            QMessageBox.warning(self, "归档失败", str(exc))
            return None
        self._last_out = path
        self.toast("已归档 {} 个媒体文件 → {}".format(n, Path(path).name), "ok")
        return path

    def _export_both(self) -> None:
        a = self._export_excel()
        b = self._export_media()
        if a or b:
            self.toast("导出完成，点「打开输出目录」查看", "ok")

    def _open_dir(self) -> None:
        d = self._last_out or self.ed_out.text().strip() or str(config.EXPORT_DIR)
        p = Path(d)
        if not p.exists():
            p = config.EXPORT_DIR
            p.mkdir(parents=True, exist_ok=True)
        # 跨平台打开文件管理器（对应 Windows / MacOS 双平台目标）
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))        # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
        except Exception:                   # noqa: BLE001
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def on_show(self) -> None:
        self._refresh_scheme_box()
        self.refresh()
