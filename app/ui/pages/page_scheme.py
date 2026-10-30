# -*- coding: utf-8 -*-
"""功能模块 2：零代码分类体系管理（需求 4.3.2 第 2 点 / 4.2.3 第 1 点）。

可视化分栏编辑：
    左：体系树（分类体系 → 分类维度 → 分类标签）
    中：标签编辑（名称 / 关键词库 / 特征权重覆盖）
    右：标准错题样本（支持批量导入）

用户全程不写一行代码，保存后系统自动同步特征库与匹配规则，无需重启。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                             QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                             QInputDialog, QLabel, QLineEdit, QListWidget, QMessageBox,
                             QPlainTextEdit, QSlider, QSplitter, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from app import config
from app.core.models import CategoryLabel, Dimension, Scheme, new_id
from app.ui.pages.base_page import BasePage
from app.ui.widgets import Card, SectionTitle, batch_view_updates, button, hline, label

ROLE_KIND = Qt.UserRole          # "scheme" / "dim" / "label"
ROLE_ID = Qt.UserRole + 1


class NewSchemeDialog(QDialog):
    """新建分类体系向导。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("新建分类体系")
        self.setMinimumWidth(400)
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.ed_name = QLineEdit()
        self.ed_name.setPlaceholderText("例：高等数学（张老师班 2026 春）")
        self.cb_subject = QComboBox()
        self.cb_subject.addItems(config.SUBJECTS)
        self.ed_desc = QLineEdit()
        self.ed_desc.setPlaceholderText("选填，一句话说明这套体系的用途")
        form.addRow("体系名称", self.ed_name)
        form.addRow("所属学科", self.cb_subject)
        form.addRow("说明", self.ed_desc)
        v.addLayout(form)

        v.addWidget(label("要包含哪些分类维度", "dim"))
        self.checks = {}
        for k, name in config.DIM_LABELS.items():
            cb = QCheckBox(name)
            cb.setChecked(True)
            self.checks[k] = cb
            v.addWidget(cb)
        v.addWidget(label("提示：知识点维度需要你自己添加标签；错误类型 / 难度 / 优先级"
                          "会预置一套通用标签，可随时改。", "faint"))

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def result_scheme(self, template: Optional[Scheme]) -> Optional[Scheme]:
        name = self.ed_name.text().strip()
        if not name:
            return None
        sc = Scheme(name=name, subject=self.cb_subject.currentText(),
                    description=self.ed_desc.text().strip(), builtin=False)
        for k, cb in self.checks.items():
            if not cb.isChecked():
                continue
            d = Dimension(key=k, name=config.DIM_LABELS[k], enabled=True, labels=[])
            # 通用维度从模板体系里拷一份预置标签，省去用户从零敲
            if k != config.DIM_KNOWLEDGE and template is not None:
                src = template.dim(k)
                if src is not None:
                    for lb in src.labels:
                        d.labels.append(CategoryLabel(
                            id=new_id("lb"), name=lb.name, dim=k,
                            keywords=list(lb.keywords), samples=list(lb.samples)))
            sc.dims.append(d)
        return sc


class SchemePage(BasePage):
    """零代码分类体系管理页。"""

    title = "零代码分类体系管理"
    subtitle = "自主搭建分学科分类体系：加标签、填关键词、贴样本，改完立即生效"

    def __init__(self, ctx, parent: Optional[QWidget] = None) -> None:
        super().__init__(ctx, parent)
        self._label: Optional[CategoryLabel] = None
        self._dim: Optional[Dimension] = None
        self._scheme: Optional[Scheme] = None
        self._selection_id = ""
        self._loading = False
        self._build()
        ctx.scheme_changed.connect(self.refresh_tree)
        ctx.workspace_reloaded.connect(self._workspace_reloaded)

    def _workspace_reloaded(self):
        self._scheme = self._dim = self._label = None
        self._selection_id = ""
        self._clear_editor()
        self._set_enabled(False)

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        # ---------------- 左：体系树 ---------------- #
        left = Card(margin=10, spacing=8)
        self.sec_tree = SectionTitle("分类体系", "")
        left.box.addWidget(self.sec_tree)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.currentItemChanged.connect(self._on_tree_select)
        left.box.addWidget(self.tree, 1)
        row = QHBoxLayout()
        row.setSpacing(5)
        row.addWidget(button("新建体系", "", "add", self._new_scheme))
        row.addWidget(button("删除", "danger", "trash", self._delete_node))
        left.box.addLayout(row)
        split.addWidget(left)

        # ---------------- 中：标签编辑 ---------------- #
        mid = Card(margin=10, spacing=8)
        self.sec_edit = SectionTitle("标签编辑", "从左侧选择一个分类标签")
        mid.box.addWidget(self.sec_edit)

        form = QFormLayout()
        form.setSpacing(7)
        self.ed_name = QLineEdit()
        self.ed_name.textChanged.connect(self._on_name_changed)
        form.addRow("标签名称", self.ed_name)
        mid.box.addLayout(form)

        mid.box.addWidget(label("知识点关键词库 —— 第一层粗筛依据，一行一个", "dim"))
        self.ed_keywords = QPlainTextEdit()
        self.ed_keywords.setPlaceholderText("定积分\n牛顿莱布尼茨公式\n变限积分\n反常积分")
        self.ed_keywords.textChanged.connect(self._on_kw_changed)
        mid.box.addWidget(self.ed_keywords, 3)
        self.lb_kw_stat = label("", "faint")
        mid.box.addWidget(self.lb_kw_stat)

        mid.box.addWidget(hline())
        mid.box.addWidget(label("特征权重覆盖 —— 仅对本标签生效，不勾选则用学科/题型默认规则",
                                "dim"))
        self.cb_override = QCheckBox("为该标签单独指定各特征通道的权重")
        self.cb_override.stateChanged.connect(self._on_override_toggle)
        mid.box.addWidget(self.cb_override)
        mid.box.addWidget(label("滑块设置相对大小，右侧显示归一化比例；无内容的通道会分配给其他通道。", "faint"))
        self.sliders = {}
        for ch in config.CHANNELS:
            r = QHBoxLayout()
            r.setSpacing(6)
            cap = QLabel(config.CHANNEL_LABELS[ch])
            cap.setFixedWidth(58)
            cap.setProperty("role", "dim")
            s = QSlider(Qt.Horizontal)
            s.setRange(0, 100)
            s.setValue(25)
            val = QLabel("25%")
            val.setFixedWidth(58)
            val.setProperty("role", "faint")
            s.valueChanged.connect(self._on_weight_changed)
            r.addWidget(cap)
            r.addWidget(s, 1)
            r.addWidget(val)
            self.sliders[ch] = (s, val)
            mid.box.addLayout(r)
        split.addWidget(mid)

        # ---------------- 右：标准错题样本 ---------------- #
        right = Card(margin=10, spacing=8)
        self.sec_sample = SectionTitle("标准错题样本", "参与第二层精匹配的标签画像")
        right.box.addWidget(self.sec_sample)
        self.list_samples = QListWidget()
        self.list_samples.setWordWrap(True)
        self.list_samples.currentRowChanged.connect(self._on_sample_select)
        right.box.addWidget(self.list_samples, 2)
        self.ed_sample = QPlainTextEdit()
        self.ed_sample.setPlaceholderText(
            "粘贴一条典型错题的题干作为该分类的标准样本。\n"
            "样本越贴近你自己的题目，分类越准。")
        self.ed_sample.setMaximumHeight(96)
        right.box.addWidget(self.ed_sample, 1)
        srow = QHBoxLayout()
        srow.setSpacing(5)
        srow.addWidget(button("添加", "primary", "add", self._add_sample))
        srow.addWidget(button("更新", "", "save", self._update_sample))
        srow.addWidget(button("删除", "danger", "remove", self._del_sample))
        right.box.addLayout(srow)
        right.box.addWidget(button("批量导入样本（TXT / Excel）", "", "import",
                                   self._import_samples,
                                   "把一批已归好类的错题一次性导入为该标签的标准样本"))
        split.addWidget(right)

        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 4)
        split.setStretchFactor(2, 3)
        self.root.addWidget(split)
        self._set_enabled(False)
        self.refresh_tree()

    # ------------------------------------------------------------------ #
    def quick_actions(self) -> Tuple[str, List[QWidget]]:
        return ("这里改的东西立刻生效：不用重启，不用改代码，也不会覆盖系统已经"
                "从你的修正里学到的东西。", [
            button("新建分类体系", "primary", "add", self._new_scheme),
            button("新建分类维度", "", "add", self._new_dim,
                   "除了知识点/错因/难度/优先级，你也能自定义任意维度"),
            button("新建分类标签", "", "add", self._new_label),
            hline(),
            button("导出体系（分享给同学）", "", "export", self._export_scheme),
            button("导入他人的体系", "", "import", self._import_scheme),
            hline(),
            button("恢复内置体系", "", "refresh", self._reset_builtin,
                   "恢复出厂的默认体系，你自建的体系不受影响"),
        ])

    # ------------------------------------------------------------------ #
    # 体系树
    # ------------------------------------------------------------------ #
    @batch_view_updates("tree")
    def refresh_tree(self) -> None:
        keep = self._selection_id
        self.tree.blockSignals(True)
        self.tree.clear()
        for sc in self.ctx.schemes:
            si = QTreeWidgetItem(self.tree, ["{}{}".format(
                sc.name, "  ·内置" if sc.builtin else "")])
            si.setData(0, ROLE_KIND, "scheme")
            si.setData(0, ROLE_ID, sc.id)
            si.setToolTip(0, sc.description or sc.name)
            for d in sc.dims:
                di = QTreeWidgetItem(si, ["{}（{}）".format(d.name, len(d.labels))])
                di.setData(0, ROLE_KIND, "dim")
                di.setData(0, ROLE_ID, "{}::{}".format(sc.id, d.key))
                if not d.enabled:
                    di.setForeground(0, Qt.gray)
                for lb in d.labels:
                    li = QTreeWidgetItem(di, [lb.name])
                    li.setData(0, ROLE_KIND, "label")
                    li.setData(0, ROLE_ID, "{}::{}::{}".format(sc.id, d.key, lb.id))
                    li.setToolTip(0, "关键词 {} 个 / 标准样本 {} 条".format(
                        len(lb.keywords), len(lb.samples)))
            si.setExpanded(sc.id == (self._scheme.id if self._scheme else ""))
        self.tree.blockSignals(False)
        n_sc = len(self.ctx.schemes)
        n_lb = sum(len(d.labels) for s in self.ctx.schemes for d in s.dims)
        self.sec_tree.set_hint("{} 套体系 / {} 个标签".format(n_sc, n_lb))
        # A deleted node falls back to its surviving parent. Resolve every
        # selection against the repository again after restore/reset/import.
        while keep and not self._select_node(keep, reload_editor=not self._loading):
            keep = keep.rpartition("::")[0]
        if not keep:
            current = self.ctx.current_scheme()
            if current is None or not self._select_node(current.id, reload_editor=not self._loading):
                self._bind_selection(None, reload_editor=not self._loading)

    def _select_node(self, rid: str, reload_editor: bool = True) -> bool:
        def walk(node):
            for i in range(node.childCount()):
                child = node.child(i)
                if child.data(0, ROLE_ID) == rid:
                    return child
                found = walk(child)
                if found is not None:
                    return found
            return None

        found = walk(self.tree.invisibleRootItem())
        if found is None:
            return False
        previous = self.tree.blockSignals(True)
        self.tree.setCurrentItem(found)
        self.tree.blockSignals(previous)
        parent = found.parent()
        while parent is not None:
            parent.setExpanded(True)
            parent = parent.parent()
        self._bind_selection(found, reload_editor=reload_editor)
        return True

    def _select_label_by_id(self, lid: str) -> None:
        it = self.tree.invisibleRootItem()

        def walk(node) -> Optional[QTreeWidgetItem]:
            for i in range(node.childCount()):
                c = node.child(i)
                if (c.data(0, ROLE_KIND) == "label" and
                        str(c.data(0, ROLE_ID)).split("::")[-1] == lid):
                    return c
                r = walk(c)
                if r:
                    return r
            return None

        found = walk(it)
        if found:
            self._select_node(str(found.data(0, ROLE_ID)))

    def _on_tree_select(self, cur: Optional[QTreeWidgetItem], _prev) -> None:
        if self._loading:
            return
        self._bind_selection(cur)
        if self._scheme:
            self.ctx.set_current_scheme(self._scheme.id)

    def _bind_selection(self, cur, reload_editor=True):
        rid = str(cur.data(0, ROLE_ID) or "") if cur is not None else ""
        self._selection_id = rid
        parts = rid.split("::")
        self._scheme = self.ctx.repo.scheme_by_id(parts[0])
        self._dim = self._scheme.dim(parts[1]) if (self._scheme and len(parts) > 1) else None
        self._label = (self._dim.label_by_id(parts[2])
                       if (self._dim and len(parts) > 2) else None)
        if not reload_editor:
            return
        if self._label is None:
            self._set_enabled(False)
            self.sec_edit.set_hint("从左侧选择一个分类标签")
            self._clear_editor()
            return
        self._set_enabled(True)
        self._load_label(self._label)

    # ------------------------------------------------------------------ #
    # 标签编辑
    # ------------------------------------------------------------------ #
    def _set_enabled(self, on: bool) -> None:
        for w in (self.ed_name, self.ed_keywords, self.cb_override,
                  self.list_samples, self.ed_sample):
            w.setEnabled(on)
        for s, _v in self.sliders.values():
            s.setEnabled(on and self.cb_override.isChecked())

    def _clear_editor(self) -> None:
        self._loading = True
        self.ed_name.clear()
        self.ed_keywords.clear()
        self.list_samples.clear()
        self.ed_sample.clear()
        self.lb_kw_stat.setText("")
        self.cb_override.setChecked(False)
        self.sec_sample.set_hint("选择分类标签后可管理标准样本")
        self._loading = False

    def _load_label(self, lb: CategoryLabel) -> None:
        self._loading = True
        self.ed_name.setText(lb.name)
        self.ed_keywords.setPlainText("\n".join(lb.keywords))
        self.cb_override.setChecked(bool(lb.weight_override))
        eng = self.ctx.engine(self._scheme)
        w = (eng.rules.get(self._scheme.subject, config.QTYPE_TEXT, lb.weight_override)
             if eng else {})
        for ch, (s, v) in self.sliders.items():
            val = int(round(float(w.get(ch, 0.25)) * 100))
            s.setValue(val)
            s.setEnabled(self.cb_override.isChecked())
        self._refresh_weight_display()
        self.list_samples.clear()
        for s_ in lb.samples:
            self.list_samples.addItem(s_)
        self.ed_sample.clear()
        self.sec_edit.set_hint("{} ｜ {}".format(
            self._scheme.name if self._scheme else "",
            self._dim.name if self._dim else ""))
        self._update_kw_stat()
        prof = eng.profiles.get(lb.id) if eng else None
        extra = "（含 {} 次反馈学到的增量特征）".format(prof.n_feedback) if (
            prof and prof.n_feedback) else ""
        self.sec_sample.set_hint("{} 条样本{}".format(len(lb.samples), extra))
        self._loading = False

    def _update_kw_stat(self) -> None:
        if self._label is None:
            return
        n = len(self._label.keywords)
        tip = "{} 个关键词".format(n)
        if n == 0:
            tip += " —— 没有关键词，这个标签在粗筛阶段永远不会被命中"
        elif n < 3:
            tip += " —— 建议至少填 3~5 个，覆盖同一知识点的不同表述"
        self.lb_kw_stat.setText(tip)

    def _on_name_changed(self, text: str) -> None:
        if self._loading or self._label is None:
            return
        self._label.name = text.strip()
        it = self.tree.currentItem()
        if it is not None:
            it.setText(0, self._label.name)
        self._commit()

    def _on_kw_changed(self) -> None:
        if self._loading or self._label is None:
            return
        kws = [k.strip() for k in self.ed_keywords.toPlainText().splitlines() if k.strip()]
        seen, out = set(), []
        for k in kws:                       # 去重但保序
            if k not in seen:
                seen.add(k)
                out.append(k)
        self._label.keywords = out
        self._update_kw_stat()
        self._commit()

    def _on_override_toggle(self) -> None:
        on = self.cb_override.isChecked()
        for s, _v in self.sliders.values():
            s.setEnabled(on and self._label is not None)
        if self._loading or self._label is None:
            return
        if not on:
            self._label.weight_override = {}
            self._refresh_weight_display()
            self._commit()
        else:
            self._on_weight_changed()

    def _on_weight_changed(self) -> None:
        if self._loading or self._label is None or not self.cb_override.isChecked():
            return
        self._label.weight_override = {
            ch: s.value() / 100.0 for ch, (s, _v) in self.sliders.items()}
        self._refresh_weight_display()
        self._commit()

    def _refresh_weight_display(self) -> None:
        if self._label is None or self._scheme is None:
            return
        engine = self.ctx.engine(self._scheme)
        if engine is None:
            return
        effective = engine.rules.get(self._scheme.subject, config.QTYPE_TEXT,
                                     self._label.weight_override)
        for channel, (_slider, value) in self.sliders.items():
            value.setText("{:.1f}%".format(effective[channel] * 100))

    def _commit(self) -> None:
        """把改动落盘并让特征库失效（下次分类自动热重建）。"""
        self._loading = True
        try:
            self.ctx.notify_scheme_changed(self._scheme)
        finally:
            self._loading = False
        self.status("分类体系已保存，特征库将在下次分类时自动同步")

    # ------------------------------------------------------------------ #
    # 增删
    # ------------------------------------------------------------------ #
    def _template(self) -> Optional[Scheme]:
        for s in self.ctx.schemes:
            if s.builtin and s.dim(config.DIM_ERROR):
                return s
        return None

    def _new_scheme(self) -> None:
        dlg = NewSchemeDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        sc = dlg.result_scheme(self._template())
        if sc is None:
            self.toast("体系名称不能为空", "warn")
            return
        self.ctx.repo.add_scheme(sc)
        self._scheme = sc
        self._dim = self._label = None
        self._selection_id = sc.id
        self.ctx.set_current_scheme(sc.id)
        self.refresh_tree()
        self.toast("已创建「{}」，接下来给「知识点」维度添加标签".format(sc.name), "ok")

    def _new_dim(self) -> None:
        if self._scheme is None:
            self.toast("请先在左侧选中一套分类体系", "warn")
            return
        name, ok = QInputDialog.getText(self, "新建分类维度",
                                        "维度名称（例：题目来源、掌握程度）：")
        if not ok or not name.strip():
            return
        key = "custom_" + new_id("d")[2:8]
        self._dim = Dimension(key=key, name=name.strip(), enabled=True, labels=[])
        self._scheme.dims.append(self._dim)
        self._label = None
        self._selection_id = "{}::{}".format(self._scheme.id, key)
        self._commit()
        self.refresh_tree()
        self.toast("已新建维度「{}」，接着给它添加标签".format(name.strip()), "ok")

    def _new_label(self) -> None:
        if self._dim is None or self._scheme is None:
            self.toast("请先在左侧选中一个分类维度", "warn")
            return
        name, ok = QInputDialog.getText(self, "新建分类标签",
                                        "标签名称（例：定积分及其应用）：")
        if not ok or not name.strip():
            return
        lb = CategoryLabel(name=name.strip(), dim=self._dim.key, keywords=[name.strip()])
        self._dim.labels.append(lb)
        self._label = lb
        self._selection_id = "{}::{}::{}".format(self._scheme.id, self._dim.key, lb.id)
        self._commit()
        self.refresh_tree()
        self._select_label_by_id(lb.id)
        self.ed_keywords.setFocus()
        self.toast("已新建标签，请在中间补充关键词", "ok")

    def _delete_node(self) -> None:
        it = self.tree.currentItem()
        if it is None:
            return
        kind = it.data(0, ROLE_KIND)
        name = it.text(0)
        if QMessageBox.question(self, "确认删除", "确定删除「{}」吗？此操作不可撤销。".format(name),
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return
        if kind == "scheme" and self._scheme:
            self.ctx.repo.remove_scheme(self._scheme.id)
            self._selection_id = ""
            self._scheme = None
            self._dim = None
            self._label = None
            self.ctx.questions_changed.emit()
            self.ctx.scheme_changed.emit()
        elif kind == "dim" and self._scheme and self._dim:
            self._scheme.dims = [d for d in self._scheme.dims if d.key != self._dim.key]
            self._selection_id = self._scheme.id
            self._dim = None
            self._label = None
            self._commit()
        elif kind == "label" and self._dim and self._label:
            self._dim.labels = [x for x in self._dim.labels if x.id != self._label.id]
            self._selection_id = "{}::{}".format(self._scheme.id, self._dim.key)
            self._label = None
            self._commit()
        self._set_enabled(False)
        self._clear_editor()
        self.refresh_tree()
        self.toast("已删除「{}」".format(name), "ok")

    # ------------------------------------------------------------------ #
    # 标准样本
    # ------------------------------------------------------------------ #
    def _on_sample_select(self, row: int) -> None:
        if self._label is None or row < 0 or row >= len(self._label.samples):
            return
        self._loading = True
        self.ed_sample.setPlainText(self._label.samples[row])
        self._loading = False

    def _add_sample(self) -> None:
        if self._label is None:
            return
        t = self.ed_sample.toPlainText().strip()
        if not t:
            self.toast("请先在下方输入样本内容", "warn")
            return
        if t in self._label.samples:
            self.toast("该样本已存在", "warn")
            return
        self._label.samples.append(t)
        self.list_samples.addItem(t)
        self.ed_sample.clear()
        self._commit()
        self.sec_sample.set_hint("{} 条样本".format(len(self._label.samples)))
        self.toast("样本已添加，特征库将自动同步", "ok")

    def _update_sample(self) -> None:
        row = self.list_samples.currentRow()
        if self._label is None or row < 0:
            return
        t = self.ed_sample.toPlainText().strip()
        if not t:
            return
        if t in self._label.samples and t != self._label.samples[row]:
            self.toast("该样本已存在，不能重复更新", "warn")
            return
        self._label.samples[row] = t
        self.list_samples.item(row).setText(t)
        self._commit()
        self.toast("样本已更新", "ok")

    def _del_sample(self) -> None:
        row = self.list_samples.currentRow()
        if self._label is None or row < 0:
            return
        self._label.samples.pop(row)
        self.list_samples.takeItem(row)
        self.ed_sample.clear()
        self._commit()
        self.sec_sample.set_hint("{} 条样本".format(len(self._label.samples)))

    def _import_samples(self) -> None:
        if self._label is None:
            self.toast("请先选中一个分类标签", "warn")
            return
        p, _ = QFileDialog.getOpenFileName(
            self, "选择样本文件", str(Path.home()),
            "文本/表格 (*.txt *.xlsx *.csv *.tsv);;所有文件 (*)")
        if not p:
            return
        from app.core.dataio import sample_import
        try:
            items = sample_import.read_samples(p)
        except sample_import.ImportError_ as exc:
            self.toast("导入失败：{}".format(exc), "error")
            return
        n = 0
        existing = set(self._label.samples)
        for t in items:
            if t not in existing:
                self._label.samples.append(t)
                self.list_samples.addItem(t)
                existing.add(t)
                n += 1
        if n:
            self._commit()
        self.sec_sample.set_hint("{} 条样本".format(len(self._label.samples)))
        self.toast("已批量导入 {} 条标准样本".format(n), "ok")

    # ------------------------------------------------------------------ #
    # 体系导入导出
    # ------------------------------------------------------------------ #
    def _export_scheme(self) -> None:
        if self._scheme is None:
            self.toast("请先在左侧选中一套体系", "warn")
            return
        from app.core.dataio import exporter
        p, _ = QFileDialog.getSaveFileName(
            self, "导出分类体系",
            str(config.EXPORT_DIR / "分类体系_{}.json".format(self._scheme.name)),
            "JSON (*.json)")
        if not p:
            return
        exporter.export_scheme(self._scheme, p)
        self.toast("体系已导出，可以直接发给同学：{}".format(Path(p).name), "ok")

    def _import_scheme(self) -> None:
        from app.core.dataio import exporter
        p, _ = QFileDialog.getOpenFileName(self, "导入分类体系", str(Path.home()),
                                           "JSON (*.json)")
        if not p:
            return
        try:
            sc = exporter.import_scheme(p)
        except Exception as exc:            # noqa: BLE001
            self.toast("导入失败：{}".format(exc), "error")
            return
        self.ctx.repo.add_scheme(sc)
        self._selection_id = sc.id
        self._scheme, self._dim, self._label = sc, None, None
        self.ctx.set_current_scheme(sc.id)
        self.ctx.scheme_changed.emit()
        self.refresh_tree()
        self.toast("已导入体系「{}」".format(sc.name), "ok")

    def _reset_builtin(self) -> None:
        if QMessageBox.question(
                self, "恢复内置体系",
                "将把内置的默认体系恢复到出厂状态（你自己新建的体系不受影响），确定吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.ctx.repo.reset_builtin_schemes()
        self.ctx.scheme_changed.emit()
        self.refresh_tree()
        self.toast("内置体系已恢复出厂状态", "ok")

    def on_show(self) -> None:
        self.refresh_tree()
