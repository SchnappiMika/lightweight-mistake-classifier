# -*- coding: utf-8 -*-
"""功能模块 4：分类结果反馈与增量优化（需求 4.3.2 第 4 点 / 4.2.3 第 2 点）。

本页把"系统到底从我的修正里学到了什么"摊开给用户看：
    * 反馈日志：每一次修正，以及它把哪个 (学科, 题型) 桶的权重推向了哪边
    * 权重规则可视化：默认规则 vs 被反馈改写过的规则，一眼看出差异
    * 特征库增量统计：每个标签累积了多少条增量特征
    * 准确率趋势：以"新分类结果被人工修正的比例"为代理指标的滚动曲线

这解决的是传统工具的黑盒问题——用户不知道系统凭什么这么分，也不知道自己
的修正有没有起作用。
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView,
                             QMessageBox, QSplitter, QTableWidget, QTableWidgetItem,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from app import config
from app.ui import theme
from app.ui.pages.base_page import BasePage
from app.ui.widgets import Card, SectionTitle, StatChip, batch_view_updates, button, hline, label


class WeightBar(QWidget):
    """四通道权重条：把抽象的权重向量画成一根堆叠条。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.weights: Dict[str, float] = {}
        self.setFixedHeight(20)
        self.setMinimumWidth(150)

    def set_weights(self, w: Dict[str, float]) -> None:
        self.weights = dict(w)
        self.setToolTip("  ".join("{} {:.0f}%".format(config.CHANNEL_LABELS[c],
                                                      w.get(c, 0) * 100)
                                  for c in config.CHANNELS))
        self.update()

    def paintEvent(self, _e) -> None:      # noqa: N802
        if not self.weights:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        pal = theme.palette()
        colors = {config.CHANNEL_TEXT: pal["primary"], config.CHANNEL_FORMULA: pal["accent"],
                  config.CHANNEL_QTYPE: pal["warn"], config.CHANNEL_SUBJECT: pal["text_faint"]}
        total = sum(max(self.weights.get(c, 0.0), 0.0) for c in config.CHANNELS) or 1.0
        x, w_all = 0.0, float(self.width())
        p.setPen(Qt.NoPen)
        for c in config.CHANNELS:
            frac = max(self.weights.get(c, 0.0), 0.0) / total
            w = frac * w_all
            if w <= 0.4:
                continue
            p.setBrush(QColor(colors[c]))
            p.drawRoundedRect(QRectF(x, 3, max(w - 1.2, 1.0), self.height() - 6), 3, 3)
            if w > 34:
                p.setPen(QColor("#FFFFFF"))
                p.drawText(QRectF(x, 3, w, self.height() - 6), Qt.AlignCenter,
                           "{:.0f}%".format(frac * 100))
                p.setPen(Qt.NoPen)
            x += w
        p.end()


class TrendChart(QWidget):
    """人工修正率滚动曲线（自绘，不引入额外图表依赖，保持轻量化）。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.points: List[float] = []
        self.setMinimumHeight(112)

    def set_points(self, pts: List[float]) -> None:
        self.points = list(pts)
        self.update()

    def paintEvent(self, _e) -> None:      # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        pal = theme.palette()
        r = self.rect().adjusted(28, 8, -8, -18)
        p.setPen(QPen(QColor(pal["border"]), 1))
        p.drawRect(r)
        p.setPen(QColor(pal["text_faint"]))
        for frac, txt in ((0.0, "100%"), (0.5, "50%"), (1.0, "0%")):
            y = r.top() + r.height() * frac
            p.drawLine(int(r.left()), int(y), int(r.right()), int(y))
            p.drawText(0, int(y) + 4, "{:>5}".format(txt))
        if len(self.points) < 2:
            p.setPen(QColor(pal["text_faint"]))
            p.drawText(r, Qt.AlignCenter, "累计 2 次以上人工修正后，这里会显示趋势")
            p.end()
            return
        n = len(self.points)
        pen = QPen(QColor(pal["primary"]), 2)
        p.setPen(pen)
        prev = None
        for i, v in enumerate(self.points):
            x = r.left() + r.width() * (i / max(n - 1, 1))
            y = r.bottom() - r.height() * max(0.0, min(1.0, v))
            if prev is not None:
                p.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
            prev = (x, y)
        p.setPen(QColor(pal["text_dim"]))
        p.drawText(int(r.left()), int(r.bottom()) + 14, "早")
        p.drawText(int(r.right()) - 14, int(r.bottom()) + 14, "近")
        p.end()


class FeedbackPage(BasePage):
    """分类结果反馈与增量优化页。"""

    title = "分类结果反馈与增量优化"
    subtitle = "每一次修正都会即时更新特征库与权重规则，无需全量重训、无需联网"

    def __init__(self, ctx, parent: Optional[QWidget] = None) -> None:
        super().__init__(ctx, parent)
        self._build()
        ctx.feedback_changed.connect(self.refresh)
        ctx.questions_changed.connect(self.refresh)
        ctx.question_edited.connect(self.refresh)
        ctx.scheme_changed.connect(self.refresh)

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        # 顶部统计
        chips = QHBoxLayout()
        chips.setSpacing(8)
        self.chip_fb = StatChip("累计人工修正")
        self.chip_rate = StatChip("最近修正率")
        self.chip_terms = StatChip("增量特征词")
        self.chip_rules = StatChip("被改写的权重桶")
        for c in (self.chip_fb, self.chip_rate, self.chip_terms, self.chip_rules):
            chips.addWidget(c)
        chips.addStretch(1)
        self.root.addLayout(chips)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        # ---------------- 左：反馈日志 ---------------- #
        left = Card(margin=10, spacing=8)
        self.sec_log = SectionTitle("人工反馈日志", "")
        left.box.addWidget(self.sec_log)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["时间", "学科", "题型", "原判", "修正为", "权重变化"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        hh.setSectionResizeMode(4, QHeaderView.Stretch)
        left.box.addWidget(self.table, 1)
        split.addWidget(left)

        # ---------------- 右：权重规则 + 趋势 ---------------- #
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(10)

        card_w = Card(margin=10, spacing=6)
        self.sec_w = SectionTitle("差异化特征权重规则", "按 学科 × 题型 分桶")
        card_w.box.addWidget(self.sec_w)
        lg = QHBoxLayout()
        lg.setSpacing(10)
        pal = theme.palette()
        for ch, col in ((config.CHANNEL_TEXT, pal["primary"]),
                        (config.CHANNEL_FORMULA, pal["accent"]),
                        (config.CHANNEL_QTYPE, pal["warn"]),
                        (config.CHANNEL_SUBJECT, pal["text_faint"])):
            dot = QWidget()
            dot.setFixedSize(9, 9)
            dot.setStyleSheet("background:{};border-radius:4px;".format(col))
            lg.addWidget(dot)
            lg.addWidget(label(config.CHANNEL_LABELS[ch], "faint"))
        lg.addStretch(1)
        card_w.box.addLayout(lg)
        self.tree_w = QTreeWidget()
        self.tree_w.setHeaderLabels(["学科 × 题型", "权重分配", "反馈次数"])
        self.tree_w.setColumnWidth(0, 190)
        self.tree_w.setColumnWidth(1, 168)
        self.tree_w.setRootIsDecorated(False)
        self.tree_w.setAlternatingRowColors(True)
        card_w.box.addWidget(self.tree_w, 1)
        rv.addWidget(card_w, 3)

        card_t = Card(margin=10, spacing=6)
        self.sec_t = SectionTitle("人工修正率趋势", "越低说明系统判得越准")
        card_t.box.addWidget(self.sec_t)
        self.chart = TrendChart()
        card_t.box.addWidget(self.chart)
        self.lb_trend = label("", "faint")
        self.lb_trend.setWordWrap(True)
        card_t.box.addWidget(self.lb_trend)
        rv.addWidget(card_t, 2)
        split.addWidget(right)

        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 4)
        self.root.addWidget(split)
        self.refresh()

    # ------------------------------------------------------------------ #
    def quick_actions(self) -> Tuple[str, List[QWidget]]:
        return ("系统学到的东西全在这儿，看得见也删得掉——不是黑盒。", [
            button("刷新", "", "refresh", self.refresh),
            button("导出反馈日志", "", "export", self._export_log,
                   "导出成 Excel，可直接放进结题报告"),
            hline(),
            button("重置权重规则", "danger", "sliders", self._reset_weights,
                   "只把权重恢复默认，已学到的特征增量保留"),
            button("清空全部学习成果", "danger", "trash", self._reset_all,
                   "清空反馈日志、特征增量与权重改写，回到出厂状态"),
        ])

    # ------------------------------------------------------------------ #
    @batch_view_updates("table", "tree_w")
    def refresh(self, *_signal_args) -> None:
        fb = self.ctx.repo.feedback
        self.sec_log.set_hint("共 {} 条".format(len(fb)) if fb else
                              "还没有任何修正记录——去分类页改一条试试")

        self.table.setRowCount(len(fb))
        for r, rec in enumerate(reversed(fb)):
            self.table.setItem(r, 0, QTableWidgetItem(
                datetime.fromtimestamp(rec.created_at).strftime("%m-%d %H:%M")))
            self.table.setItem(r, 1, QTableWidgetItem(rec.subject))
            self.table.setItem(r, 2, QTableWidgetItem(
                config.QTYPE_LABELS.get(rec.qtype, rec.qtype)))
            self.table.setItem(r, 3, QTableWidgetItem(rec.from_label_name or "（空）"))
            self.table.setItem(r, 4, QTableWidgetItem(rec.to_label_name))
            self.table.setItem(r, 5, QTableWidgetItem(self._weight_delta_text(rec)))
            it = self.table.item(r, 3)
            it.setToolTip("参与增量更新的特征词：\n" + "、".join(rec.tokens[:40]))

        self._refresh_weights()
        self._refresh_stats()
        self._refresh_trend()

    @staticmethod
    def _weight_delta_text(rec) -> str:
        if not rec.weight_before or not rec.weight_after:
            return "—"
        parts = []
        for ch in config.CHANNELS:
            d = rec.weight_after.get(ch, 0) - rec.weight_before.get(ch, 0)
            if abs(d) >= 0.002:
                parts.append("{}{:+.1f}%".format(config.CHANNEL_LABELS[ch][:2], d * 100))
        return "  ".join(parts) or "—"

    def _refresh_weights(self) -> None:
        self.tree_w.clear()
        table = self.ctx.rules.effective_table()
        # 被反馈改写过的桶排前面
        for (subj, qt), (w, hits, changed) in sorted(
                table.items(), key=lambda x: (not x[1][2], x[0])):
            name = "{} × {}".format(subj if subj != "*" else "全部学科",
                                    config.QTYPE_LABELS.get(qt, "全部题型")
                                    if qt != "*" else "全部题型")
            it = QTreeWidgetItem(self.tree_w, [name, "", str(hits) if hits else ""])
            bar = WeightBar()
            bar.set_weights(w)
            self.tree_w.setItemWidget(it, 1, bar)
            if changed:
                f = it.font(0)
                f.setBold(True)
                it.setFont(0, f)
                it.setToolTip(0, "该桶的权重已被 {} 次人工修正在线调整过".format(hits))
        n_changed = sum(1 for _k, v in table.items() if v[2])
        self.sec_w.set_hint("{} 个桶，其中 {} 个已被反馈改写".format(len(table), n_changed))

    def _refresh_stats(self) -> None:
        fb = self.ctx.repo.feedback
        self.chip_fb.set_value(str(len(fb)))

        n_terms = 0
        for sc in self.ctx.schemes:
            eng = self.ctx._engines.get(sc.id)     # noqa: SLF001 只读统计，不触发建库
            if eng is None:
                continue
            for p in eng.profiles.values():
                n_terms += sum(1 for v in p.delta_tf.values() if abs(v) > 1e-9)
        self.chip_terms.set_value(str(n_terms))

        table = self.ctx.rules.effective_table()
        self.chip_rules.set_value(str(sum(1 for _k, v in table.items() if v[2])))

        done = [q for q in self.ctx.questions if q.classified]
        if done:
            fixed = sum(1 for q in done if any(q.corrected.values()))
            rate = fixed / len(done)
            pal = theme.palette()
            self.chip_rate.set_value(
                "{:.0%}".format(rate),
                pal["success"] if rate < 0.15 else (pal["warn"] if rate < 0.35
                                                    else pal["danger"]))
        else:
            self.chip_rate.set_value("-")

    def _refresh_trend(self) -> None:
        """以"最近 N 条分类中被修正的比例"作为准确率的代理指标画滚动曲线。

        说明：这是**代理指标**而非真实准确率——用户不一定会修正每一条错判。
        它反映的是"系统判得让用户不满意的频率"，趋势向下就说明增量优化在起效。
        """
        done = sorted([q for q in self.ctx.questions if q.classified],
                      key=lambda q: q.created_at)
        if len(done) < 4:
            self.chart.set_points([])
            self.lb_trend.setText("分类的错题还太少，累计几条之后这里会显示趋势。")
            return
        win = max(5, len(done) // 8)
        pts: List[float] = []
        for i in range(win, len(done) + 1):
            chunk = done[i - win:i]
            pts.append(sum(1 for q in chunk if any(q.corrected.values())) / len(chunk))
        self.chart.set_points(pts)
        if len(pts) >= 2:
            delta = pts[-1] - pts[0]
            trend = ("修正率在下降，系统越判越准了" if delta < -0.02 else
                     "修正率基本持平" if abs(delta) <= 0.02 else
                     "修正率在上升，建议去分类体系页补充关键词或标准样本")
            self.lb_trend.setText(
                "滑动窗口 {} 条 ｜ 从 {:.0%} 变化到 {:.0%} —— {}\n"
                "（这是代理指标：统计的是「被你改过的比例」，不是严格意义上的准确率）".format(
                    win, pts[0], pts[-1], trend))

    # ------------------------------------------------------------------ #
    def _export_log(self) -> None:
        fb = self.ctx.repo.feedback
        if not fb:
            self.toast("还没有反馈记录可以导出", "warn")
            return
        p, _ = QFileDialog.getSaveFileName(
            self, "导出反馈日志", str(config.EXPORT_DIR / "反馈日志.xlsx"),
            "Excel (*.xlsx);;CSV (*.csv)")
        if not p:
            return
        import pandas as pd
        rows = []
        for rec in fb:
            rows.append({
                "时间": datetime.fromtimestamp(rec.created_at).strftime("%Y-%m-%d %H:%M:%S"),
                "学科": rec.subject,
                "题型": config.QTYPE_LABELS.get(rec.qtype, rec.qtype),
                "分类维度": config.DIM_LABELS.get(rec.dim, rec.dim),
                "原判标签": rec.from_label_name,
                "修正为": rec.to_label_name,
                "增量特征词": "、".join(rec.tokens),
                "权重变化": self._weight_delta_text(rec),
            })
        df = pd.DataFrame(rows)
        if p.lower().endswith(".csv"):
            df.to_csv(p, index=False, encoding="utf-8-sig")
        else:
            df.to_excel(p, index=False)
        self.toast("反馈日志已导出：{}".format(p), "ok")

    def _reset_weights(self) -> None:
        if QMessageBox.question(self, "重置权重规则",
                                "将把全部特征权重恢复为默认规则（已学到的特征增量保留），确定吗？",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return
        self.ctx.rules.reset()
        self.refresh()
        self.toast("权重规则已恢复默认", "ok")

    def _reset_all(self) -> None:
        if QMessageBox.question(
                self, "清空全部学习成果",
                "将清空反馈日志、全部特征增量与权重改写，回到出厂状态。\n"
                "分类体系（关键词、标准样本）与错题数据不受影响。确定吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.ctx.reset_optimization()
        self.refresh()
        self.toast("已回到出厂状态", "ok")

    def on_show(self) -> None:
        self.refresh()
