# -*- coding: utf-8 -*-
"""第一层：关键词快速粗筛（技术路线 4.2.2 第一层）。

作用：基于用户搭建的分学科知识点关键词库，在毫秒级把候选标签从几十个缩到 3-5 个，
      让第二层的多特征融合精匹配只需要在很小的候选集上做余弦计算。

两种命中方式互补：
    * **分词精确命中**：题干分词结果直接命中关键词。快，但依赖分词把多字术语切对
      （系统已把关键词动态注入 jieba 词典来保障这一点）。
    * **原文子串命中**：关键词以子串形式出现在清洗后的原文里。慢一点，但能兜住
      分词失误与未登录词，是"召回率"的保险。

**召回率与效率的平衡**（需求"待解决的核心问题 2"）：
    粗筛的致命风险是把正确标签直接筛掉——后面精匹配再准也救不回来。
    因此本实现有两道保险：
      1. 命中数不足 COARSE_MIN_K 时，按标签先验 prior 补齐候选（标记 fallback）；
      2. 标签总数本就 ≤ COARSE_TOP_K 时直接全量进入精匹配，跳过粗筛（无收益还多一层）。
"""
from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Mapping, Tuple

from app import config
from app.core.models import Candidate, Dimension, FeatureBundle

log = logging.getLogger(__name__)

#: 关键词子词在倒排索引中的权重折扣（"杨氏双缝干涉" -> "干涉"）
SUBWORD_DISCOUNT = 0.35


class CoarseFilter:
    """某个维度下的关键词倒排索引与粗筛器。"""

    def __init__(self, dim: Dimension) -> None:
        self.dim = dim
        # 词 -> [(label_id, 关键词权重)]
        self.index: Dict[str, List[Tuple[str, float]]] = {}
        # 需要做子串扫描的多字关键词
        self.substr_keys: List[Tuple[str, str, float]] = []   # (关键词, label_id, 权重)
        self.norm: Dict[str, float] = {}                      # label_id -> 长度归一化因子
        self.label_names: Dict[str, str] = {}
        self.priors: List[Tuple[str, float]] = []
        # 从人工修正学到的词：label_id -> {词: 相对强度}。与用户手写的关键词库
        # 分开存放，这样用户在界面上改关键词触发 build() 重建索引时，
        # 学到的词不会被冲掉，用户也不会看到一堆机器塞进去的词。
        self.learned: Dict[str, Dict[str, float]] = {}
        self._manual_terms: Dict[str, Dict[str, float]] = {}
        self._base_norm: Dict[str, float] = {}
        self._learned_keys: Dict[str, set] = {}
        self.build()

    def build(self) -> None:
        """构建倒排索引。用户改动关键词库后由引擎调用本方法热更新。"""
        from app.core.preprocess import lexicon

        self.index, self.substr_keys, self.norm = {}, [], {}
        self.label_names, self.priors = {}, []
        self._manual_terms, self._base_norm, self._learned_keys = {}, {}, {}
        jieba = lexicon.ensure_jieba()
        stop = lexicon.all_stopwords()

        for lb in self.dim.labels:
            self.label_names[lb.id] = lb.name
            self.priors.append((lb.id, float(lb.prior)))
            kws = [k.strip() for k in lb.keywords if k and k.strip()]
            # 标签名本身也是一个高权重关键词
            names = [lb.name] if lb.name and lb.name not in kws else []
            seen = set()
            for kw in names + kws:
                w = lb.kw_weight(kw) * (1.5 if kw in names else 1.0)
                self._add(kw, lb.id, w, seen)
                if len(kw) >= 3 or (kw.isascii() and len(kw) >= 4):
                    self.substr_keys.append((kw, lb.id, w))
                # 子词扩展：让"干涉条纹"这类局部表述也能召回"杨氏双缝干涉"。
                # 折扣权重保证它只影响召回，不足以让弱相关标签挤掉强相关标签。
                if len(kw) >= 3 and not kw.isascii():
                    for sw in jieba.cut_for_search(kw):
                        sw = sw.strip()
                        if len(sw) >= 2 and sw != kw and sw not in stop:
                            self._add(sw, lb.id, w * SUBWORD_DISCOUNT, seen)
            # 关键词多的标签天然容易命中，做长度归一化避免"大标签"通吃
            self.norm[lb.id] = math.sqrt(len(kws) + len(names) + 1.0)
        self._base_norm = dict(self.norm)
        self._manual_terms = {lid: {} for lid in self.label_names}
        for term, bucket in self.index.items():
            for lid, weight in bucket:
                self._manual_terms[lid][term] = weight
        self.priors.sort(key=lambda x: -x[1])
        # 用户改关键词会触发重建，这里把已学到的词重新贴回去
        for lid, terms in list(self.learned.items()):
            if lid in self.label_names:
                self._apply_learned(lid, terms)
            else:
                del self.learned[lid]

    def learn(self, label_id: str, terms: Mapping[str, float]) -> None:
        """把人工修正学到的判别词并入倒排索引，立即生效。

        这些词权重打折（``COARSE_LEARNED_WEIGHT``），只用于补召回：让正确标签
        能进候选集，至于它能不能胜出，仍旧交给第二层多特征融合精匹配去裁决。
        """
        if label_id not in self.label_names:
            return
        # terms is a complete snapshot of the profile's current top delta, not
        # another batch to append. An empty snapshot removes stale learned hits.
        self._apply_learned(label_id, terms)

    def _apply_learned(self, label_id: str, terms: Mapping[str, float]) -> None:
        if label_id not in self.label_names:      # 标签可能已被用户删掉
            return
        valid = {t: float(v) for t, v in terms.items()
                 if isinstance(t, str) and len(t.strip()) >= 2
                 and math.isfinite(float(v)) and float(v) > 0}
        manual = self._manual_terms.get(label_id, {})
        # Restore the manual contribution before replacing the learned one.
        # Other labels sharing a term are left untouched.
        affected = self._learned_keys.get(label_id, set()) | set(valid)
        peak = max(valid.values(), default=1.0)
        for t in sorted(affected):
            bucket = [(lid, w) for lid, w in self.index.get(t, ()) if lid != label_id]
            weight = manual.get(t, 0.0)
            # 归一到 [0,1] 再打折：学到的词强度只反映相对置信，不能与人工关键词等量齐观
            if t in valid:
                weight = max(weight, config.COARSE_LEARNED_WEIGHT * (valid[t] / peak))
            if weight > 0:
                bucket.append((label_id, weight))
            if bucket:
                self.index[t] = bucket
            else:
                self.index.pop(t, None)
        self.learned[label_id] = valid
        self._learned_keys[label_id] = set(valid)
        # 学到的词也计入长度归一化，避免修正多的标签靠词多通吃
        # Manual/learned overlap is still one indexed feature, not two.
        base = self._base_norm.get(label_id, 1.0)
        self.norm[label_id] = math.sqrt(base * base + len(set(valid) - manual.keys()))

    def _add(self, term: str, label_id: str, weight: float, seen: set) -> None:
        """把 (词 -> 标签) 写入倒排表；同一标签下同词只保留最高权重。"""
        key = (term, label_id)
        if key in seen:
            bucket = self.index.get(term, [])
            for i, (lid, w) in enumerate(bucket):
                if lid == label_id and weight > w:
                    bucket[i] = (lid, weight)
            return
        seen.add(key)
        self.index.setdefault(term, []).append((label_id, weight))

    # ------------------------------------------------------------------ #
    def all_keywords(self) -> List[str]:
        return list(self.index.keys())

    def screen(self, fb: FeatureBundle,
               top_k: int = config.COARSE_TOP_K) -> Tuple[List[Candidate], float, bool]:
        """执行粗筛。

        Returns:
            (候选列表, 耗时ms, 是否走了"全量直通"路径)
        """
        t0 = time.perf_counter()
        total = len(self.dim.labels)

        # 标签数很少时粗筛没有收益，直接全量进精匹配
        if total <= top_k:
            cands = [Candidate(label_id=lb.id, label_name=lb.name, coarse_score=0.0)
                     for lb in self.dim.labels]
            return cands, (time.perf_counter() - t0) * 1000.0, True

        scores: Dict[str, float] = {}
        hits: Dict[str, List[str]] = {}

        # ---- 命中方式 1：分词精确命中 ---- #
        tokens = set(fb.all_tokens()) | set(fb.qtype_tokens) | set(fb.subject_tokens)
        for tok in sorted(tokens):
            for lid, w in self.index.get(tok, ()):
                scores[lid] = scores.get(lid, 0.0) + config.COARSE_KEYWORD_WEIGHT * w
                hits.setdefault(lid, []).append(tok)

        # ---- 命中方式 2：原文子串命中（兜住分词失误 / 未登录词） ---- #
        haystack = " ".join([fb.clean_text or "", fb.latex or "",
                             fb.ocr_text or "", fb.transcript or ""])
        if haystack.strip():
            hay = haystack.lower()
            for kw, lid, w in self.substr_keys:
                if kw in tokens:                 # 已被方式 1 记过分，不重复计
                    continue
                if (kw.lower() if kw.isascii() else kw) in hay:
                    scores[lid] = scores.get(lid, 0.0) + config.COARSE_SUBSTR_WEIGHT * w
                    hits.setdefault(lid, []).append(kw)

        # ---- 长度归一化 + 排序 ---- #
        ranked = sorted(
            ((lid, s / self.norm.get(lid, 1.0)) for lid, s in scores.items()
             if s > config.COARSE_MIN_SCORE),
            key=lambda x: (-x[1], x[0]))[:top_k]

        cands = [Candidate(label_id=lid, label_name=self.label_names.get(lid, lid),
                           coarse_score=s,
                           hit_keywords=sorted(set(hits.get(lid, [])))[:8])
                 for lid, s in ranked]

        # ---- 召回兜底：命中不足时按先验补齐 ---- #
        if len(cands) < min(config.COARSE_MIN_K, total):
            have = {c.label_id for c in cands}
            for lid, prior in self.priors:
                if len(cands) >= min(config.COARSE_MIN_K, total):
                    break
                if lid in have:
                    continue
                cands.append(Candidate(label_id=lid,
                                       label_name=self.label_names.get(lid, lid),
                                       coarse_score=0.0, fallback=True))
        return cands, (time.perf_counter() - t0) * 1000.0, False
