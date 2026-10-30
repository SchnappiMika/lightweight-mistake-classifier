# -*- coding: utf-8 -*-
"""功能模块 3：多特征融合一键分类与结果修正（需求 4.3.2 第 3 点）。

布局对应需求 4.3.3：中间区在本模式下分为
    「错题列表区」+「分类结果展示区」+「修正操作区」。

分类过程放在后台线程执行，并实时回填结果，保证批量分类时界面不卡顿。
分类依据（粗筛命中的关键词、各特征通道得分与权重）全部可展开查看——
这是传统黑盒工具做不到的，也是本系统"可解释"的体现。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QInputDialog,
                             QMessageBox, QProgressBar, QSplitter, QTableWidget,
                             QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                             QWidget)

from app import config
from app.core.models import ClassifyResult, Question
from app.ui import theme
from app.ui.pages.base_page import BasePage
from app.ui.widgets import Card, SectionTitle, batch_view_updates, button, hline, label
from app.ui.workers import ClassifyWorker, stop_worker


class ClassifyPage(BasePage):
    """一键分类与结果修正页。"""

    title = "多特征融合一键分类与结果修正"
    subtitle = "关键词粗筛 → 多特征融合精匹配；改错一次，系统就记住一次"

    def __init__(self, ctx, parent: Optional[QWidget] = None) -> None:
        super().__init__(ctx, parent)
        self._worker: Optional[ClassifyWorker] = None
        self._current: Optional[Question] = None
        self._results: Dict[str, ClassifyResult] = {}
        self._fix_boxes: Dict[str, QComboBox] = {}
        self._build()
        ctx.questions_changed.connect(self.refresh_list)
        ctx.question_edited.connect(self.refresh_list)
        ctx.scheme_changed.connect(self._refresh_scheme_box)
        ctx.workspace_reloaded.connect(self._workspace_reloaded)

    def _workspace_reloaded(self):
        self._current = None
        self.table.clearSelection()
        self._results.clear()
        self.lb_perf.clear()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        # 顶部：体系选择 + 一键分类
        bar = QHBoxLayout()
        bar.setSpacing(7)
        bar.addWidget(label("分类体系", "dim"))
        self.cb_scheme = QComboBox()
        self.cb_scheme.setMinimumWidth(230)
        self.cb_scheme.currentIndexChanged.connect(self._on_scheme_changed)
        bar.addWidget(self.cb_scheme)
        self.lb_scheme_stat = label("", "faint")
        bar.addWidget(self.lb_scheme_stat)
        bar.addStretch(1)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(190)
        self.progress.setVisible(False)
        bar.addWidget(self.progress)
        self.btn_run = button("一键分类", "primary", "wand", self._run_all,
                              "对全部未分类错题执行多模态预处理与双层分级分类")
        bar.addWidget(self.btn_run)
        self.root.addLayout(bar)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        # ---------------- 左：错题列表区 ---------------- #
        left = Card(margin=10, spacing=8)
        self.sec_list = SectionTitle("错题列表", "")
        left.box.addWidget(self.sec_list)
        flt = QHBoxLayout()
        flt.setSpacing(6)
        self.cb_filter = QComboBox()
        self.cb_filter.addItem("全部错题", "all")
        self.cb_filter.addItem("未分类", "todo")
        self.cb_filter.addItem("已分类", "done")
        self.cb_filter.addItem("媒体未校对", "unreviewed")
        self.cb_filter.addItem("置信度偏低（建议复核）", "low")
        self.cb_filter.addItem("已人工修正", "fixed")
        self.cb_filter.currentIndexChanged.connect(self.refresh_list)
        flt.addWidget(self.cb_filter)
        flt.addStretch(1)
        left.box.addLayout(flt)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["题干摘要", "学科", "主要分类", "置信度"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in (1, 2, 3):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_select)
        left.box.addWidget(self.table, 1)
        split.addWidget(left)

        # ---------------- 中：分类结果展示区 ---------------- #
        mid = Card(margin=10, spacing=8)
        self.sec_result = SectionTitle("分类结果与依据", "从左侧选择一条错题")
        mid.box.addWidget(self.sec_result)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["维度 / 候选标签", "得分", "说明"])
        self.tree.setColumnWidth(0, 230)
        self.tree.setColumnWidth(1, 62)
        self.tree.setAlternatingRowColors(True)
        mid.box.addWidget(self.tree, 1)
        self.lb_perf = label("", "faint")
        self.lb_perf.setWordWrap(True)
        mid.box.addWidget(self.lb_perf)
        split.addWidget(mid)

        # ---------------- 右：修正操作区 ---------------- #
        right = Card(margin=10, spacing=8)
        self.sec_fix = SectionTitle("修正操作", "选错了？直接改，系统会立刻学")
        right.box.addWidget(self.sec_fix)
        self.fix_area = QVBoxLayout()
        self.fix_area.setSpacing(7)
        right.box.addLayout(self.fix_area)
        right.box.addWidget(hline())
        right.box.addWidget(label("一键标记复习优先级", "dim"))
        prow = QHBoxLayout()
        prow.setSpacing(5)
        for name, kind in (("高", "danger"), ("中", ""), ("低", "ghost")):
            prow.addWidget(button(name, kind, "", lambda _c=False, n=name: self._mark_priority(n)))
        right.box.addLayout(prow)
        right.box.addStretch(1)
        self.lb_fix_tip = label("", "faint")
        self.lb_fix_tip.setWordWrap(True)
        right.box.addWidget(self.lb_fix_tip)
        right.setFixedWidth(258)
        split.addWidget(right)

        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 5)
        self.root.addWidget(split)
        self._refresh_scheme_box()
        self.refresh_list()

    # ------------------------------------------------------------------ #
    def quick_actions(self) -> Tuple[str, List[QWidget]]:
        return ("分类依据是透明的：中间区能看到粗筛命中了哪些关键词、"
                "四个特征通道各打了多少分。", [
            button("一键分类（未分类）", "primary", "wand", self._run_all),
            button("重新分类全部", "", "refresh", lambda: self._run_all(force=True),
                   "忽略已有结果，对全部错题重跑一遍（已人工修正的维度会被保留）"),
            button("只分类选中项", "", "check", self._run_selected),
            button("批量媒体识别与校对", "", "", self._batch_media),
            hline(),
            button("采纳为标准样本", "accent", "save", self._promote,
                   "把当前错题固化为该分类的标准样本，长期提升同类题的准确率"),
            button("停止", "danger", "stop", self._stop),
        ])

    # ------------------------------------------------------------------ #
    # 体系
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
        self._update_scheme_stat()

    def _update_scheme_stat(self) -> None:
        eng = self.ctx.engine()
        if eng is None:
            self.lb_scheme_stat.setText("")
            return
        s = eng.stats()
        self.lb_scheme_stat.setText(
            "{} 个标签 · {} 个关键词 · {} 条样本 · 已从 {} 次修正中学习".format(
                s["labels"], s["keywords"], s["samples"], s["feedback"]))

    def _on_scheme_changed(self) -> None:
        sid = self.cb_scheme.currentData()
        if sid:
            self.ctx.set_current_scheme(sid)
        self._update_scheme_stat()

    # ------------------------------------------------------------------ #
    # 列表
    # ------------------------------------------------------------------ #
    def _primary_dimension(self, q: Question) -> str:
        if config.DIM_KNOWLEDGE in q.labels:
            return config.DIM_KNOWLEDGE
        scheme = self.ctx.repo.scheme_by_id(q.scheme_id)
        return next((d.key for d in (scheme.enabled_dims() if scheme else []) if d.key in q.labels),
                    next(iter(q.labels), config.DIM_KNOWLEDGE))

    def _filtered(self) -> List[Question]:
        mode = self.cb_filter.currentData()
        out = []
        for q in self.ctx.questions:
            if mode == "todo" and q.classified:
                continue
            if mode == "done" and not q.classified:
                continue
            if mode == "unreviewed" and not (q.feature or {}).get("unreviewed_media"):
                continue
            if mode == "fixed" and not any(q.corrected.values()):
                continue
            if mode == "low":
                sc = q.scores.get(self._primary_dimension(q), 1.0)
                if not q.classified or sc >= config.LOW_CONFIDENCE_THRESHOLD:
                    continue
            out.append(q)
        return out

    @batch_view_updates("table")
    def refresh_list(self, *_signal_args) -> None:
        keep = self._current.id if self._current else ""
        items = self._filtered()
        p = theme.palette()
        self.table.blockSignals(True)
        self.table.setRowCount(len(items))
        for r, q in enumerate(items):
            needs_review = bool((q.feature or {}).get("unreviewed_media"))
            it = QTableWidgetItem(("【未校对】" if needs_review else "")
                                 + (q.display_text or "（空题干，仅附件）"))
            it.setData(Qt.UserRole, q.id)
            if needs_review:
                it.setForeground(Qt.darkYellow)
                it.setToolTip("分类使用了自动识别内容，请通过「批量媒体识别与校对」核对原始附件。")
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, QTableWidgetItem(q.subject))
            primary = self._primary_dimension(q)
            name = q.label_names.get(primary, "")
            ki = QTableWidgetItem(name + ("  ✓" if q.corrected.get(primary) else ""))
            self.table.setItem(r, 2, ki)
            sc = q.scores.get(primary, 0.0)
            si = QTableWidgetItem("{:.2f}".format(sc) if q.classified else "-")
            if q.classified and sc < config.LOW_CONFIDENCE_THRESHOLD:
                si.setForeground(Qt.red if theme.current() == config.THEME_LIGHT
                                 else Qt.magenta)
                si.setToolTip("置信度偏低，建议人工复核")
            self.table.setItem(r, 3, si)
        self.table.blockSignals(False)

        total = len(self.ctx.questions)
        done = sum(1 for q in self.ctx.questions if q.classified)
        fixed = sum(1 for q in self.ctx.questions if any(q.corrected.values()))
        unreviewed = sum(1 for q in self.ctx.questions if q.classified
                         and (q.feature or {}).get("unreviewed_media"))
        self.sec_list.set_hint("共 {} 条 ｜ 已分类 {} ｜ 人工修正 {} ｜ 媒体未校对 {}".format(
            total, done, fixed, unreviewed))
        self.btn_run.setEnabled(total > 0)
        if keep:
            for r in range(self.table.rowCount()):
                if self.table.item(r, 0).data(Qt.UserRole) == keep:
                    self.table.blockSignals(True)
                    self.table.selectRow(r)
                    self.table.blockSignals(False)
                    self._show_result(self._current)
                    self._build_fix_area(self._current)
                    return
        self._current = None
        self.tree.clear()
        self._build_fix_area(None)
        _ = p

    def _on_select(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        qid = self.table.item(rows[0].row(), 0).data(Qt.UserRole)
        q = self.ctx.repo.question_by_id(qid)
        if q is None:
            return
        self._current = q
        self._show_result(q)
        self._build_fix_area(q)

    # ------------------------------------------------------------------ #
    # 分类结果展示
    # ------------------------------------------------------------------ #
    def _show_result(self, q: Question) -> None:
        self.tree.clear()
        sc = self.ctx.repo.scheme_by_id(q.scheme_id) or self.ctx.current_scheme()
        if not q.classified and not q.feature:
            self.sec_result.set_hint("这条错题还没有分类，点右上角「一键分类」")
            self.lb_perf.setText("")
            return
        self.sec_result.set_hint("{} ｜ {}".format(
            config.QTYPE_LABELS.get(q.qtype, q.qtype) if q.classified else
            "未匹配有效分类，请补充内容或检查体系标签", sc.name if sc else ""))
        p = theme.palette()
        unreviewed = (q.feature or {}).get("unreviewed_media", [])
        if unreviewed:
            kinds = {"image": "图片文字", "formula": "公式", "audio": "听力音频"}
            node = QTreeWidgetItem(self.tree, ["媒体识别：未校对", "",
                "、".join(kinds.get(kind, kind) for kind in unreviewed) + "；请对照原始附件复核"])
            node.setForeground(0, Qt.darkYellow)

        for dim_key, name in [(d.key, d.name) for d in (sc.dims if sc else [])]:
            if dim_key not in q.label_names:
                dimension = sc.dim(dim_key)
                if dimension is None or not dimension.enabled or not dimension.labels:
                    continue
                detail = ("请手动标记复习优先级" if dim_key == config.DIM_PRIORITY else
                          "请补充实际错因标注或手动选择" if dim_key == config.DIM_ERROR else
                          "请补充难度依据或手动选择" if dim_key == config.DIM_DIFFICULTY else
                          "缺少有效依据，请补充内容或手动选择")
                QTreeWidgetItem(self.tree, ["{}：待确定".format(name), "", detail])
                continue
            top = QTreeWidgetItem(self.tree, [
                "{}：{}".format(name, q.label_names.get(dim_key, "")),
                "{:.3f}".format(q.scores.get(dim_key, 0.0)),
                "人工已确认" if q.corrected.get(dim_key) else ""])
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            if q.corrected.get(dim_key):
                top.setForeground(2, Qt.darkGreen)
            elif q.scores.get(dim_key, 0) < config.LOW_CONFIDENCE_THRESHOLD:
                top.setText(2, "置信度偏低，建议复核")
                top.setForeground(2, Qt.darkYellow)

            for c in q.candidates.get(dim_key, [])[:config.COARSE_TOP_K]:
                tag = "召回兜底" if c.get("fallback") else "粗筛命中：{}".format(
                    "、".join(c.get("hit_keywords", [])[:5]) or "无")
                node = QTreeWidgetItem(top, [
                    "    " + c.get("label_name", ""),
                    "{:.3f}".format(c.get("final_score", 0.0)), tag])
                node.setForeground(2, Qt.gray)
                # 各特征通道明细：让"多特征融合"看得见
                chs = c.get("channel_scores", {})
                ws = c.get("channel_weights", {})
                detail = "  ".join(
                    "{} {:.2f}×{:.0f}%".format(config.CHANNEL_LABELS[ch],
                                               chs.get(ch, 0.0), ws.get(ch, 0.0) * 100)
                    for ch in config.CHANNELS if ws.get(ch, 0) > 0.001)
                if detail:
                    sub = QTreeWidgetItem(node, ["        特征通道", "", detail])
                    sub.setForeground(2, Qt.gray)
            top.setExpanded(True)

        fb = q.feature or {}
        parts = []
        if fb.get("text_tokens"):
            parts.append("文本特征词 {} 个：{}".format(
                len(fb["text_tokens"]), "、".join(fb["text_tokens"][:12])))
        if fb.get("formula_tokens"):
            uniq = list(dict.fromkeys(fb["formula_tokens"]))
            parts.append("公式映射特征词：{}".format("、".join(uniq[:10])))
        if fb.get("latex"):
            parts.append("识别到的 LaTeX：{}".format(fb["latex"][:70]))
        if fb.get("transcript"):
            parts.append("听力转写：{}".format(fb["transcript"][:70]))
        if fb.get("ocr_text"):
            parts.append("图内 OCR：{}".format(fb["ocr_text"][:70]))
        if fb.get("figure_parameters"):
            parameters = QTreeWidgetItem(self.tree, ["图形参数", "", "保留原值，题干和标注分别列出"])
            for item in fb["figure_parameters"]:
                origin = {"reviewed_text": "校对题干", "question_text": "题干", "automatic_ocr": "自动识别（未校对）",
                          "annotation": "人工标注"}.get(item.get("source"), item.get("source", ""))
                node = QTreeWidgetItem(parameters, [item.get("symbol") or "未命名量",
                    "{} {}".format(item.get("value", ""), item.get("unit", "")).strip(), origin])
                node.setToolTip(0, item.get("raw", ""))
            parameters.setExpanded(True)
        if fb.get("notes"):
            parts.append("提示：{}".format("；".join(fb["notes"][:3])))
        parts.append("本次预处理 {:.1f}ms ｜ 特征分类 {:.1f}ms（不含媒体识别）".format(
            fb.get("elapsed_ms", 0.0), q.elapsed_ms))
        if fb.get("total_process_ms"):
            parts.append("本次自动处理 {:.1f}ms（识别、预处理与分类）；本次媒体识别 {:.1f}ms".format(
                fb["total_process_ms"], fb.get("media_elapsed_ms", 0.0)))
        self.lb_perf.setText("\n".join(parts))
        _ = p

    # ------------------------------------------------------------------ #
    # 修正区
    # ------------------------------------------------------------------ #
    def _build_fix_area(self, q: Optional[Question]) -> None:
        while self.fix_area.count():
            it = self.fix_area.takeAt(0)
            w = it.widget()
            if w:
                w.hide()
                w.deleteLater()
        self._fix_boxes = {}
        if q is None or not q.classified:
            self.lb_fix_tip.setText("选中一条已分类的错题后，这里会出现每个维度的下拉框。")
            return
        sc = self.ctx.repo.scheme_by_id(q.scheme_id) or self.ctx.current_scheme()
        if sc is None:
            return
        for d in sc.dims:
            if not d.enabled or not d.labels:
                continue
            self.fix_area.addWidget(label(d.name, "dim"))
            cb = QComboBox()
            cb.addItem("待确定 / 请选择", "")
            for lb in d.labels:
                cb.addItem(lb.name, lb.id)
            cur = q.labels.get(d.key, "")
            i = cb.findData(cur)
            cb.setCurrentIndex(i if i >= 0 else 0)
            cb.currentIndexChanged.connect(
                lambda _i, key=d.key, box=cb: self._apply_fix(key, box))
            self._fix_boxes[d.key] = cb
            self.fix_area.addWidget(cb)
        self.lb_fix_tip.setText(
            "修正会立即保存。有有效依据时，系统会增量更新分类特征和权重。"
            "错因与难度使用人工标注中的依据；复习优先级由你指定。整个过程不联网。")

    def _apply_fix(self, dim_key: str, box: QComboBox) -> None:
        if self._current is None:
            return
        lid = box.currentData()
        if not lid or self._current.labels.get(dim_key) == lid:
            return
        old = self._current.label_names.get(dim_key, "(空)")
        if self.ctx.apply_correction(self._current, dim_key, lid):
            self.toast("已修正：{} → {}，特征库与权重已增量更新".format(
                old, box.currentText()), "ok")
        else:
            self.toast("已修正：{} → {}".format(old, box.currentText()), "ok")
        self._show_result(self._current)
        self._update_scheme_stat()

    def _mark_priority(self, name: str) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.toast("请先选择错题", "warn")
            return
        n = 0
        for r in rows:
            qid = self.table.item(r.row(), 0).data(Qt.UserRole)
            q = self.ctx.repo.question_by_id(qid)
            if q is None:
                continue
            q.priority = name
            sc = self.ctx.repo.scheme_by_id(q.scheme_id) or self.ctx.current_scheme()
            d = sc.dim(config.DIM_PRIORITY) if sc else None
            lb = d.label_by_name(name) if d else None
            if lb is not None and q.classified:
                self.ctx.apply_correction(q, config.DIM_PRIORITY, lb.id)
            n += 1
        self.ctx.save_questions()
        self.ctx.questions_changed.emit()
        self.toast("已把 {} 条错题标记为「{}」优先级".format(n, name), "ok")

    def _promote(self) -> None:
        if self._current is None or not self._current.classified:
            self.toast("请先选择一条已分类的错题", "warn")
            return
        sc = self.ctx.repo.scheme_by_id(self._current.scheme_id) or self.ctx.current_scheme()
        dimensions = [d for d in (sc.enabled_dims() if sc else [])
                      if d.key != config.DIM_PRIORITY and self._current.labels.get(d.key)]
        if not dimensions:
            self.toast("这条题暂无可采纳的分类标签", "warn")
            return
        dimension = dimensions[0]
        if len(dimensions) > 1:
            choices = ["{}：{}".format(d.name, self._current.label_names.get(d.key, ""))
                       for d in dimensions]
            selected, accepted = QInputDialog.getItem(self, "采纳为标准样本", "选择目标分类维度", choices,
                next((i for i, d in enumerate(dimensions) if d.key == config.DIM_KNOWLEDGE), 0), False)
            if not accepted:
                return
            dimension = dimensions[choices.index(selected)]
        opt = self.ctx.optimizer(sc)
        if opt and opt.promote_sample(self._current, dimension.key):
            self.ctx.notify_scheme_changed(sc)
            self.toast("已把这条错题采纳为「{}」的标准样本".format(
                self._current.label_names.get(dimension.key, "")), "ok")
        else:
            self.toast("采纳失败：缺少已确认内容、相关标注，或该样本已存在", "warn")

    # ------------------------------------------------------------------ #
    # 分类执行
    # ------------------------------------------------------------------ #
    def _run_all(self, force: bool = False) -> None:
        items = self.ctx.questions if force else [q for q in self.ctx.questions
                                                  if not q.classified]
        if not items:
            self.toast("没有需要分类的错题" if not force else "还没有导入错题", "warn")
            return
        self._start(items)

    def _run_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.toast("请先在左侧选择错题", "warn")
            return
        ids = [self.table.item(r.row(), 0).data(Qt.UserRole) for r in rows]
        items = [q for q in self.ctx.questions if q.id in set(ids)]
        self._start(items)

    def _batch_media(self):
        from app.ui.media_batch_dialog import MediaBatchDialog, media_kind
        selected = {self.table.item(index.row(), 0).data(Qt.UserRole)
                    for index in self.table.selectionModel().selectedRows()}
        questions = [q for q in self.ctx.questions if media_kind(q)
                     and (not selected or q.id in selected)]
        if not questions:
            self.toast("当前范围没有图片或音频附件", "warn")
            return
        MediaBatchDialog(self.ctx, questions, self).exec_()

    def _start(self, items: List[Question]) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.toast("上一批还在分类中，请稍候", "warn")
            return
        # 按错题自身的学科挑选对应体系；找不到就用当前选中的体系
        groups: Dict[str, List[Question]] = {}
        for q in items:
            sc = self.ctx.scheme_for_subject(q.subject)
            if sc is None:
                continue
            groups.setdefault(sc.id, []).append(q)
        if not groups:
            self.toast("没有可用的分类体系", "error")
            return
        if not self.ctx.begin_task("分类"):
            return
        self._queue = list(groups.items())
        self._total_all = len(items)
        self._done_all = 0
        self._success_all = 0
        self._errors = []
        self._save_failed = ""
        self._ms = 0.0
        self.progress.setVisible(True)
        self.progress.setRange(0, self._total_all)
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)
        self._next_group()

    def _next_group(self) -> None:
        if not self._queue:
            self._on_all_done()
            return
        sid, items = self._queue.pop(0)
        sc = self.ctx.repo.scheme_by_id(sid)
        eng = self.ctx.engine(sc)
        if eng is None or sc is None:
            self._next_group()
            return
        self._worker = ClassifyWorker(eng, sc, items, self)
        self._worker.progress.connect(self._on_progress)
        self._group_result = None
        self._worker.finished_all.connect(self._store_group_result)
        self._worker.finished.connect(self._finish_group)
        self._worker.failed.connect(lambda m: self.toast(m, "error"))
        self._worker.start()
        self.status("正在分类「{}」下的 {} 条错题…".format(sc.name, len(items)))

    def _on_progress(self, done: int, _total: int, text: str) -> None:
        self.progress.setValue(self._done_all + done)
        self.status("分类中（{}/{}）：{}".format(
            self._done_all + done, self._total_all, text[:30]))

    def _store_group_result(self, ok, ms, errors) -> None:
        self._group_result = (ok, ms, errors)

    def _finish_group(self) -> None:
        if self._group_result is not None:
            self._on_group_done(*self._group_result)

    def _on_group_done(self, ok: int, ms: float, errors: List[str]) -> None:
        self._done_all += ok + len(errors)
        self._success_all += ok
        self._errors = getattr(self, "_errors", []) + errors
        self._ms = getattr(self, "_ms", 0.0) + ms
        try:
            self.ctx.save_questions()
        except OSError as exc:
            self._save_failed = str(exc)
            self._queue = []
        self._next_group()

    def _on_all_done(self) -> None:
        self.ctx.end_task("分类")
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        errors = getattr(self, "_errors", [])
        ms = getattr(self, "_ms", 0.0)
        self._errors, self._ms = [], 0.0
        n = self._done_all
        avg = ms / n if n else 0
        self.ctx.questions_changed.emit()
        self._update_scheme_stat()
        msg = "分类结束：成功 {} 条，失败 {} 条，未处理 {} 条，自动处理平均 {:.0f}ms/条".format(
            self._success_all, len(errors), self._total_all - n, avg)
        save_failed = getattr(self, "_save_failed", "")
        if save_failed:
            msg += "；结果尚未保存，请保持窗口打开，检查磁盘后创建完整备份"
            QMessageBox.warning(self, "分类结果尚未保存", msg + "\n\n" + save_failed)
        if errors:
            QMessageBox.warning(self, "部分错题分类失败",
                                "以下错题处理失败，其余结果已正常保存：\n\n"
                                + "\n".join(errors[:10]))
        self.toast(msg, "ok" if not errors and not save_failed else "warn")
        self.status(msg)

    def _stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._queue = []
            self.toast("已请求停止；正在识别的媒体任务会取消，已完成结果保留", "warn")

    def workers(self):
        return [self._worker]

    def shutdown(self) -> None:
        self._queue = []          # 别让收尾后又去启动下一批
        super().shutdown()
        self.ctx.end_task("分类")

    def on_show(self) -> None:
        self._refresh_scheme_box()
        self.refresh_list()
