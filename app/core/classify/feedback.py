# -*- coding: utf-8 -*-
"""分类结果人工反馈的增量优化（技术路线 4.2.3 第 2 点）。

用户在【一键分类与结果修正】页把某条错题从 A 类改判为 B 类之后，本模块做四件事，
**全部是 O(本题词数) 的增量运算，不重扫语料、不重训模型**：

    1. 正反馈：把该题的特征词按增益并入 B 标签画像的 delta 层；
    2. 负反馈：把 A 标签画像中与该题重合的词做小幅衰减（下限 0，绝不抹成负数）；
    3. 词表/df 增量登记：新词进词表，n_docs += 1，idf 随之自适应；
    4. 权重规则在线更新：比较 B 与 A 在四个通道上的得分差，把 (学科, 题型) 桶的
       权重朝"当时说对话的通道"推一小步（见 weights.update_from_feedback）。

之所以能"无需全量重训"，根源在两处设计：
    * TF-IDF 的 idf 只依赖 df 与 n_docs 两个计数器，天然可增量；
    * 标签画像用 base+delta 双层，反馈只写 delta，不与用户手改的词库互相覆盖。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app import config
from app.core.classify.engine import ClassificationEngine
from app.core.models import FeatureBundle, FeedbackRecord, Question
from app.core.preprocess import pipeline

log = logging.getLogger(__name__)


class FeedbackOptimizer:
    """增量反馈优化器。"""

    def __init__(self, engine: ClassificationEngine) -> None:
        self.engine = engine

    # ------------------------------------------------------------------ #
    def apply_correction(self, q: Question, dim_key: str, to_label_id: str,
                         fb: Optional[FeatureBundle] = None) -> Optional[FeedbackRecord]:
        """应用一次人工修正并完成增量优化。

        Returns:
            反馈记录；若无需优化（标签未变 / 特征为空）则返回 None。
        """
        eng = self.engine
        eng.sync_scheme()

        dim = eng.scheme.dim(dim_key)
        if dim is None:
            return None
        to_label = dim.label_by_id(to_label_id)
        if to_label is None:
            return None

        from_label_id = q.labels.get(dim_key, "")
        from_label_name = q.label_names.get(dim_key, "")

        # 标签没变（用户只是把下拉框又选回原值，等于确认"判对了"）：
        # 记下"人工已确认"即可，绝不能借机再往画像里灌一遍词——那不是新信息，
        # 只会把题干里的通用词在已经正确的标签上反复堆积，反而稀释关键词的判别力。
        if from_label_id and from_label_id == to_label_id:
            self._write_back(q, dim_key, to_label_id, to_label.name)
            return None

        if fb is None:
            fb = q.feature_bundle() or pipeline.preprocess(q)
        if dim_key == config.DIM_PRIORITY:
            self._write_back(q, dim_key, to_label_id, to_label.name)
            return None
        fb = eng.matching_features(q, fb, dim_key)
        tokens: List[str] = list(fb.all_tokens())
        if not tokens:
            log.info("错题 %s 无有效特征，跳过增量优化", q.id)
            self._write_back(q, dim_key, to_label_id, to_label.name)
            return None

        # --- 4) 先算权重梯度（必须在画像改动之前算，否则梯度失真） --- #
        w_before: Dict[str, float] = {}
        w_after: Dict[str, float] = {}
        if from_label_id and from_label_id != to_label_id and eng.fine is not None:
            right = eng.fine.channel_scores_for(fb, to_label_id)
            wrong = eng.fine.channel_scores_for(fb, from_label_id)
            w_before, w_after = eng.rules.update_from_feedback(
                fb.subject or q.subject, fb.qtype or q.qtype, right, wrong)

        # --- 1) 正反馈：增量并入正确标签画像 --- #
        prof_to = eng.profiles.get(to_label_id)
        if prof_to is None:
            from app.core.classify.vector_space import LabelProfile
            prof_to = LabelProfile(to_label_id)
            eng.profiles[to_label_id] = prof_to
        prof_to.reinforce(tokens, gain=config.FEEDBACK_POSITIVE_GAIN,
                          idf_of=eng.space.idf_of)

        # --- 2) 负反馈：小幅衰减误判标签 --- #
        if from_label_id and from_label_id != to_label_id:
            prof_from = eng.profiles.get(from_label_id)
            if prof_from is not None:
                prof_from.penalize(tokens, decay=config.FEEDBACK_NEGATIVE_DECAY)

        # --- 2.5) 反哺第一层粗筛：把学到的判别词并入倒排索引 --- #
        # 没有这一步，反馈只能影响精匹配；而冷启动时真正的瓶颈是粗筛召回——
        # 正确标签压根进不了候选集，精匹配再准也没机会翻盘。
        cf = eng.coarse.get(dim_key)
        if cf is not None:
            cf.learn(to_label_id, prof_to.top_delta_terms(config.FEEDBACK_KEYWORD_TOPN))
            if from_label_id and from_label_id != to_label_id:
                prof_from = eng.profiles.get(from_label_id)
                if prof_from is not None:
                    cf.learn(from_label_id, prof_from.top_delta_terms(config.FEEDBACK_KEYWORD_TOPN))

        # --- 3) 词表 / df 增量登记 --- #
        eng.register_feedback_tokens(tokens)
        eng.space.ensure_terms(tokens)
        eng.save_profiles()

        self._write_back(q, dim_key, to_label_id, to_label.name)

        rec = FeedbackRecord(
            question_id=q.id, scheme_id=eng.scheme.id, dim=dim_key,
            from_label_id=from_label_id, from_label_name=from_label_name,
            to_label_id=to_label_id, to_label_name=to_label.name,
            subject=fb.subject or q.subject, qtype=fb.qtype or q.qtype,
            tokens=sorted(set(tokens))[:60],
            weight_before=w_before, weight_after=w_after)
        log.info("增量优化完成：%s [%s] %s -> %s", q.id, dim_key,
                 from_label_name or "(空)", to_label.name)
        return rec

    # ------------------------------------------------------------------ #
    @staticmethod
    def _write_back(q: Question, dim_key: str, label_id: str, label_name: str) -> None:
        import time
        q.labels[dim_key] = label_id
        q.label_names[dim_key] = label_name
        q.scores[dim_key] = 1.0            # 人工确认视为满置信
        q.corrected[dim_key] = True
        q.classified = True
        if dim_key == config.DIM_PRIORITY:
            q.priority = label_name
        q.updated_at = time.time()

    # ------------------------------------------------------------------ #
    def promote_sample(self, q: Question, dim_key: str) -> bool:
        """把一条被修正的错题固化为该标签的"标准错题样本"。

        与 delta 层的区别：样本进的是分类体系（可导出、可分享、可人工编辑），
        属于长期资产；delta 层只是模型侧的增量记忆。
        """
        lid = q.labels.get(dim_key)
        if not lid:
            return False
        dim = self.engine.scheme.dim(dim_key)
        lb = dim.label_by_id(lid) if dim else None
        if lb is None:
            return False
        if dim_key == config.DIM_PRIORITY:
            return False
        if dim_key in (config.DIM_ERROR, config.DIM_DIFFICULTY):
            text = (q.annotation or "").strip()
        else:
            parts = [q.raw_text or "", q.annotation or ""]
            for attr in ("formula_ocr", "audio_transcript"):
                review = getattr(q, attr, None)
                if review is not None and review.confirmed:
                    applied = (review.applied_text or review.draft or review.text).strip()
                    if applied:
                        parts.append("$" + applied + "$" if attr == "formula_ocr" else applied)
            text = "\n".join(dict.fromkeys(part.strip() for part in parts if part.strip()))
        if not text or text in lb.samples:
            return False
        lb.samples.append(text)
        self.engine.scheme.touch()
        self.engine.invalidate()
        return True
