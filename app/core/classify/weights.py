# -*- coding: utf-8 -*-
"""多特征融合的差异化权重分配规则（技术路线 4.2.2 第二层）。

设计要点
--------
1. **按 (学科, 题型) 分桶**：高数公式题应该更信公式特征，英语听力题应该更信
   题型与文本特征，物理图形题介于两者之间。这正是"基于错题类型与学科属性
   设置差异化的特征权重分配规则"的落地。
2. **空通道权重再分配**：某条错题若压根没有公式特征，把公式通道的权重按比例
   摊给其余通道，而不是白白扣掉——否则置信度会被系统性拉低，影响"存疑"判定。
3. **在线增量更新**：用户修正一次分类后，用类感知机的更新式把权重朝"更有利于
   正确标签的通道"推一小步。这一步只需要两次评分之差，无需重训任何模型，
   对应"增量更新权重分配规则"。
4. **标签级覆盖**：用户可在界面上为单个标签单独指定权重（零代码自定义）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from app import config

log = logging.getLogger(__name__)

Weights = Dict[str, float]

T, F, Q, S = (config.CHANNEL_TEXT, config.CHANNEL_FORMULA,
              config.CHANNEL_QTYPE, config.CHANNEL_SUBJECT)

# (学科, 题型) -> 权重；"*" 为通配
DEFAULT_RULES: Dict[Tuple[str, str], Weights] = {
    # 公式是数学类错题最强的判别信号
    (config.SUBJECT_MATH, config.QTYPE_FORMULA): {T: 0.35, F: 0.45, Q: 0.10, S: 0.10},
    (config.SUBJECT_LINALG, config.QTYPE_FORMULA): {T: 0.35, F: 0.45, Q: 0.10, S: 0.10},
    (config.SUBJECT_PROB, config.QTYPE_FORMULA): {T: 0.40, F: 0.40, Q: 0.10, S: 0.10},
    # 物理公式题的文字描述同样重要，两者接近对半
    (config.SUBJECT_PHYSICS, config.QTYPE_FORMULA): {T: 0.42, F: 0.38, Q: 0.10, S: 0.10},
    # 图形题的分类信息主要来自 OCR 文本与人工标注，公式权重下调、题型权重上调
    (config.SUBJECT_PHYSICS, config.QTYPE_FIGURE): {T: 0.55, F: 0.15, Q: 0.20, S: 0.10},
    ("*", config.QTYPE_FIGURE): {T: 0.55, F: 0.15, Q: 0.20, S: 0.10},
    # 英语题几乎不含公式，题型（听力/阅读/完形）本身就是强判别特征
    (config.SUBJECT_ENGLISH, config.QTYPE_LISTENING): {T: 0.60, F: 0.02, Q: 0.28, S: 0.10},
    (config.SUBJECT_ENGLISH, config.QTYPE_READING): {T: 0.65, F: 0.02, Q: 0.23, S: 0.10},
    (config.SUBJECT_ENGLISH, config.QTYPE_TEXT): {T: 0.68, F: 0.02, Q: 0.20, S: 0.10},
    (config.SUBJECT_ENGLISH, "*"): {T: 0.65, F: 0.02, Q: 0.23, S: 0.10},
    # 数理类纯文本题干仍以文本为主，但保留公式通道（题干里可能夹带内联公式）
    (config.SUBJECT_MATH, config.QTYPE_TEXT): {T: 0.65, F: 0.15, Q: 0.10, S: 0.10},
    (config.SUBJECT_PHYSICS, config.QTYPE_TEXT): {T: 0.65, F: 0.15, Q: 0.10, S: 0.10},
    # 全局兜底
    ("*", config.QTYPE_TEXT): {T: 0.72, F: 0.08, Q: 0.10, S: 0.10},
    ("*", "*"): {T: 0.70, F: 0.10, Q: 0.10, S: 0.10},
}


def normalize(w: Mapping[str, float]) -> Weights:
    """裁剪到下限并归一化到和为 1。"""
    vals = {c: max(float(w.get(c, 0.0)), config.WEIGHT_MIN) for c in config.CHANNELS}
    s = sum(vals.values())
    if s <= 0:
        return {c: 1.0 / len(config.CHANNELS) for c in config.CHANNELS}
    return {c: v / s for c, v in vals.items()}


def redistribute(w: Mapping[str, float], active: Mapping[str, bool]) -> Weights:
    """把空通道（该题没有对应特征）的权重按比例摊给有效通道。"""
    live = {c: float(w.get(c, 0.0)) for c in config.CHANNELS if active.get(c)}
    if not live:
        return {c: 0.0 for c in config.CHANNELS}
    s = sum(live.values())
    if s <= 0:
        share = 1.0 / len(live)
        return {c: (share if active.get(c) else 0.0) for c in config.CHANNELS}
    return {c: (live[c] / s if c in live else 0.0) for c in config.CHANNELS}


class WeightRules:
    """权重规则表：查询 / 在线更新 / 持久化。"""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or config.WEIGHT_FILE)
        self.rules: Dict[Tuple[str, str], Weights] = {}
        self.hits: Dict[Tuple[str, str], int] = {}   # 各桶被反馈更新的次数
        self.load()

    # ------------------------------------------------------------------ #
    def _key(self, subject: str, qtype: str) -> Tuple[str, str]:
        """按 (精确, 学科通配, 题型通配, 全通配) 顺序回退查找。"""
        for k in ((subject, qtype), (subject, "*"), ("*", qtype), ("*", "*")):
            if k in self.rules or k in DEFAULT_RULES:
                return k
        return ("*", "*")

    def _raw(self, subject: str, qtype: str) -> Weights:
        """该 (学科, 题型) 桶当前生效的原始权重（已学到的规则优先于出厂默认）。"""
        key = self._key(subject, qtype)
        return dict(self.rules.get(key) or DEFAULT_RULES.get(key)
                    or DEFAULT_RULES[("*", "*")])

    def get(self, subject: str, qtype: str,
            override: Optional[Mapping[str, float]] = None) -> Weights:
        """取得该 (学科, 题型) 桶的权重；``override`` 为标签级自定义权重。

        ``override`` 的语义是"在本桶权重之上打补丁"：只覆盖它显式给出的通道，
        没给的通道沿用本桶权重。界面上的滑块一次会给全四个通道，因此等价于
        整体指定；而只给部分通道时（例如只想把公式权重拉高），其余通道仍然
        保留该学科·题型该有的比例，也保留系统已从反馈里学到的调整——
        这比一律退回全局兜底规则更符合用户预期。
        """
        base = self._raw(subject, qtype)
        if override:
            base.update({k: float(v) for k, v in override.items() if k in config.CHANNELS})
        return normalize(base)

    def bucket_key(self, subject: str, qtype: str) -> Tuple[str, str]:
        return self._key(subject, qtype)

    # ------------------------------------------------------------------ #
    def update_from_feedback(self, subject: str, qtype: str,
                             right_channels: Mapping[str, float],
                             wrong_channels: Mapping[str, float],
                             lr: Optional[float] = None) -> Tuple[Weights, Weights]:
        """依据一次人工修正在线更新权重。

        直觉：若在通道 c 上，正确标签的得分高于误判标签，说明该通道当时"说了实话"，
        应当调高其权重；反之调低。更新量正比于两者的得分差：

            Δw_c = lr * (s_c(正确) - s_c(误判))
            w    = normalize(clip(w + Δw, WEIGHT_MIN))

        这是一次 O(通道数)=O(4) 的运算，与语料规模无关，因此可以做到"即时生效、无需重训"。
        """
        import math

        lr = float(config.WEIGHT_LEARNING_RATE if lr is None else lr)
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("Feedback learning rate must be finite and nonnegative")
        before = normalize(self._raw(subject, qtype))

        grad = {}
        for c in config.CHANNELS:
            right = float(right_channels.get(c, 0.0))
            wrong = float(wrong_channels.get(c, 0.0))
            if not math.isfinite(right) or not math.isfinite(wrong):
                raise ValueError("Feedback channel scores must be finite")
            # Nonnegative TF-IDF cosine scores lie in [0, 1]. Clamp numerical
            # roundoff, then use their actual bounded difference. Dividing by
            # max(abs(grad)) would turn a 1e-6 distinction into a full lr step.
            grad[c] = min(1.0, max(0.0, right)) - min(1.0, max(0.0, wrong))
        after = normalize({c: before[c] + lr * grad[c] for c in config.CHANNELS})

        # 更新的是"具体的 (学科, 题型) 桶"，不是通配桶，避免一次修正污染全局
        real_key = (subject or "*", qtype or "*")
        self.rules[real_key] = after
        self.hits[real_key] = self.hits.get(real_key, 0) + 1
        self.save()
        return before, after

    def reset(self) -> None:
        self.rules = {}
        self.hits = {}
        self.save()

    def effective_table(self) -> Dict[Tuple[str, str], Tuple[Weights, int, bool]]:
        """返回 (桶 -> (权重, 反馈次数, 是否被反馈改写过))，供界面可视化。"""
        out: Dict[Tuple[str, str], Tuple[Weights, int, bool]] = {}
        for k, w in DEFAULT_RULES.items():
            out[k] = (normalize(w), 0, False)
        for k, w in self.rules.items():
            out[k] = (normalize(w), self.hits.get(k, 0), True)
        return out

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        try:
            if not self.path.exists():
                return
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.rules = {tuple(k.split("||", 1)): {c: float(v) for c, v in w.items()}
                          for k, w in data.get("rules", {}).items()}
            self.hits = {tuple(k.split("||", 1)): int(v)
                         for k, v in data.get("hits", {}).items()}
        except (OSError, ValueError, TypeError) as exc:
            log.warning("权重规则读取失败，使用默认表：%s", exc)
            self.rules, self.hits = {}, {}

    def save(self) -> None:
        from app.core.storage import write_json
        write_json(self.path, {
            "rules": {"||".join(k): v for k, v in self.rules.items()},
            "hits": {"||".join(k): v for k, v in self.hits.items()},
        })
