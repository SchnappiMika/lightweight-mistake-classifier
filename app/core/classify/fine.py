# -*- coding: utf-8 -*-
"""第二层：多特征融合精匹配（技术路线 4.2.2 第二层）。

传统做法只有一条通道：把整段题干做 TF-IDF，再与标签语料算余弦。
公式题的分类信息藏在符号里、听力题藏在题型里，单通道自然抓不住。

本层把一条错题拆成 4 个**查询视图**，分别对同一个标签画像做余弦：

    text     文本特征词        ← 题干/OCR/转写 的核心实词
    formula  公式知识点特征词  ← "公式-知识点映射词典"的输出
    qtype    题型特征词        ← 公式题/图形题/听力题/阅读题
    subject  学科特征词        ← 高等数学/大学物理/…

再用 weights.py 的 (学科, 题型) 差异化权重做加权求和：

    score(label) = Σ_c  w_c · cos( tfidf(view_c) , tfidf(profile_label) )

四个视图共用一套 TF-IDF 空间与一套标签画像，只是查询侧的 token 集合不同——
这样既保住了经典 TF-IDF + 余弦的稳定性与可解释性，又让不同模态的特征
各自拥有独立的、可被反馈调节的话语权。空视图的权重会被摊给其他通道。
"""
from __future__ import annotations

import time
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from app import config
from app.core.classify import weights as W
from app.core.classify.vector_space import IncrementalTfidfSpace, LabelProfile, cosine
from app.core.models import Candidate, Dimension, FeatureBundle


class FineMatcher:
    """多特征融合精匹配器。"""

    def __init__(self, space: IncrementalTfidfSpace,
                 profiles: Mapping[str, LabelProfile],
                 rules: W.WeightRules) -> None:
        self.space = space
        self.profiles = profiles
        self.rules = rules

    # ------------------------------------------------------------------ #
    def _views(self, fb: FeatureBundle) -> Dict[str, Sequence[str]]:
        return {
            config.CHANNEL_TEXT: fb.text_tokens,
            config.CHANNEL_FORMULA: fb.formula_tokens,
            config.CHANNEL_QTYPE: fb.qtype_tokens,
            config.CHANNEL_SUBJECT: fb.subject_tokens,
        }

    def match(self, fb: FeatureBundle, dim: Dimension,
              candidates: List[Candidate]) -> List[Candidate]:
        """对粗筛候选做精匹配并按综合得分降序返回。"""
        if not candidates:
            return []

        # 候选标签画像矩阵：一次 cosine_similarity 算完全部候选
        count_maps: List[Dict[str, float]] = []
        keep: List[Candidate] = []
        for c in candidates:
            prof = self.profiles.get(c.label_id)
            if prof is None or not prof.is_trained():
                # 画像为空的标签（用户新建但还没填关键词）保留但记 0 分
                c.final_score = 0.0
                c.channel_scores = {ch: 0.0 for ch in config.CHANNELS}
                continue
            count_maps.append(prof.effective_tf())
            keep.append(c)
        if not keep:
            return sorted(candidates, key=lambda x: -x.final_score)

        label_mat = self.space.stack(count_maps)

        views = self._views(fb)
        active = {ch: bool(v) for ch, v in views.items()}
        # 通道相似度矩阵：channel -> 每个候选的 cos 值
        sims: Dict[str, np.ndarray] = {}
        for ch, tokens in views.items():
            if not tokens:
                sims[ch] = np.zeros(len(keep), dtype=np.float64)
                continue
            qv = self.space.transform_tokens(tokens)
            sims[ch] = cosine(qv, label_mat)

        for i, c in enumerate(keep):
            lb = dim.label_by_id(c.label_id)
            override = lb.weight_override if (lb and lb.weight_override) else None
            base_w = self.rules.get(fb.subject, fb.qtype, override)
            # 空通道的权重摊给有效通道，保证置信度可比
            w = W.redistribute(base_w, active)
            c.channel_scores = {ch: float(sims[ch][i]) for ch in config.CHANNELS}
            c.channel_weights = w
            c.final_score = float(sum(w[ch] * c.channel_scores[ch] for ch in config.CHANNELS))

        out = keep + [c for c in candidates if c not in keep]
        out.sort(key=lambda x: (-x.final_score, -x.coarse_score))
        return out

    # ------------------------------------------------------------------ #
    def channel_scores_for(self, fb: FeatureBundle, label_id: str) -> Dict[str, float]:
        """单独计算某个标签的各通道相似度（供反馈优化器计算权重梯度）。"""
        prof = self.profiles.get(label_id)
        if prof is None or not prof.is_trained():
            return {ch: 0.0 for ch in config.CHANNELS}
        mat = self.space.stack([prof.effective_tf()])
        out: Dict[str, float] = {}
        for ch, tokens in self._views(fb).items():
            if not tokens:
                out[ch] = 0.0
                continue
            out[ch] = float(cosine(self.space.transform_tokens(tokens), mat)[0])
        return out
