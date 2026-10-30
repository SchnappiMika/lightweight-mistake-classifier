# -*- coding: utf-8 -*-
"""增量式 TF-IDF 向量空间与标签画像。

为什么不直接用 ``TfidfVectorizer.fit_transform`` 走到底？
    需求 4.2.3 要求"用户手动修正分类结果后，无需全量重新计算与训练，
    即可增量更新特征库"。而 ``TfidfVectorizer`` 每次 fit 都要重扫全部语料、
    重建词表与 idf，语料一大就会明显卡顿，与"即时反馈"的交互目标冲突。

因此本模块的做法是：
    * **初始化**用 sklearn 的 ``CountVectorizer`` 一次性统计词表与文档频次 df；
    * **增量**时只做 ``n_docs += 1`` 和 ``df[term] += 1``，复杂度 O(词数)，与语料规模无关；
    * **相似度**仍交给 sklearn 的 ``cosine_similarity`` 计算，保持经典方法的稳定性。
  idf 公式与 sklearn 的 ``smooth_idf=True`` 完全一致：idf = ln((1+n)/(1+df)) + 1。

标签画像 ``LabelProfile`` 采用 base + delta 双层结构：
    base_tf  由分类体系的关键词与标准样本重建（用户编辑体系后刷新）；
    delta_tf 由人工反馈累积（持久化，跨体系编辑保留）。
  有效画像 = clip(base_tf + delta_tf, 0)。
  这样"用户改词库"与"系统学反馈"两条线互不覆盖，也解释了增量优化为何无需重训。
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)


def _identity(doc: Sequence[str]) -> Sequence[str]:
    """CountVectorizer 的 analyzer：输入已是分词结果，原样返回。"""
    return doc


class IncrementalTfidfSpace:
    """可增量扩展的 TF-IDF 向量空间。"""

    def __init__(self, sublinear_tf: bool = True) -> None:
        self.vocab: Dict[str, int] = {}
        self.df: List[float] = []
        self.n_docs: int = 0
        self.sublinear_tf = sublinear_tf
        self._idf_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # 构建 / 增量
    # ------------------------------------------------------------------ #
    def fit(self, docs: Iterable[Sequence[str]]) -> None:
        """用 sklearn CountVectorizer 一次性统计词表与 df（冷启动路径）。"""
        docs = [list(d) for d in docs if d]
        self.vocab, self.df, self.n_docs, self._idf_cache = {}, [], 0, None
        if not docs:
            return
        cv = CountVectorizer(analyzer=_identity, lowercase=False, min_df=1)
        try:
            X = cv.fit_transform(docs)
        except ValueError:               # 全部文档为空
            return
        self.vocab = {str(k): int(v) for k, v in cv.vocabulary_.items()}
        df = np.asarray((X > 0).sum(axis=0), dtype=np.float64).ravel()
        self.df = df.tolist()
        self.n_docs = int(X.shape[0])
        self._idf_cache = None

    def add_document(self, tokens: Sequence[str]) -> None:
        """增量登记一篇文档：O(len(tokens))，不触碰已有语料。"""
        if not tokens:
            return
        self.n_docs += 1
        for t in set(tokens):
            idx = self.vocab.get(t)
            if idx is None:
                idx = len(self.vocab)
                self.vocab[t] = idx
                self.df.append(0.0)
            self.df[idx] += 1.0
        self._idf_cache = None

    def ensure_terms(self, tokens: Iterable[str]) -> None:
        """把新词登记进词表但不计入 df（用于用户新增关键词）。"""
        changed = False
        for t in tokens:
            if t and t not in self.vocab:
                self.vocab[t] = len(self.vocab)
                self.df.append(0.0)
                changed = True
        if changed:
            self._idf_cache = None

    def add_document_statistics(self, n_docs: int, frequencies: Mapping[str, int]) -> None:
        """Merge saved feedback df into a freshly built base space, without texts.

        Counts describe documents, not token occurrences. They are kept apart
        from base statistics so editing a scheme can rebuild its own corpus
        while retaining exactly the feedback contribution.
        """
        if n_docs < 0 or any(count < 0 or count > n_docs for count in frequencies.values()):
            raise ValueError("Invalid feedback document frequencies")
        self.ensure_terms(frequencies)
        self.n_docs += n_docs
        for term, count in frequencies.items():
            self.df[self.vocab[term]] += count
        self._idf_cache = None

    # ------------------------------------------------------------------ #
    # 向量化
    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        return len(self.vocab)

    def idf(self) -> np.ndarray:
        """与 sklearn smooth_idf=True 一致的 idf 向量。"""
        if self._idf_cache is None or len(self._idf_cache) != self.dim:
            df = np.asarray(self.df, dtype=np.float64)
            self._idf_cache = np.log((1.0 + self.n_docs) / (1.0 + df)) + 1.0
        return self._idf_cache

    def idf_of(self, term: str) -> float:
        """单个词的 idf；词表里没有的新词按"只在本文档出现过"估计。

        新词（df 视作 1）会拿到最高的 idf，这正是我们想要的：用户修正里第一次
        出现的专有名词（如"黎曼和"）恰恰是最该被吸收进画像的判别特征。
        """
        i = self.vocab.get(term)
        if i is not None and i < len(self.df):
            return float(self.idf()[i])
        return math.log((1.0 + self.n_docs) / 2.0) + 1.0

    def _tf(self, count: float) -> float:
        """次线性词频。

        注意 ``count`` 不一定是整数：标签画像的 delta 层经过正强化增益与负反馈
        衰减后会累积出小数词频（例如 0.2）。而经典的次线性公式 ``1+log(c)`` 只在
        ``c>=1`` 时才有意义——``1+log(0.2) = -0.61`` 会变成**负权重**，让画像反过来
        排斥含该词的错题，把负反馈的语义整个弄反。

        因此这里只在 ``c>1`` 时做对数压缩，``c<=1`` 时退化为线性。两段在 c=1 处
        连续（同为 1.0）、全程单调不减且非负；对整数词频的结果与 sklearn 的
        ``sublinear_tf=True`` 完全一致。
        """
        if count <= 0:
            return 0.0
        if not self.sublinear_tf:
            return count
        return 1.0 + math.log(count) if count > 1.0 else count

    def transform_counts(self, counts: Mapping[str, float]) -> sp.csr_matrix:
        """把 词 -> 累计词频 映射转为 L2 归一化的 TF-IDF 稀疏行向量。"""
        n = self.dim
        if not counts or n == 0:
            return sp.csr_matrix((1, max(n, 1)), dtype=np.float64)
        idf = self.idf()
        cols: List[int] = []
        vals: List[float] = []
        for term, c in counts.items():
            idx = self.vocab.get(term)
            if idx is None or c <= 0:
                continue
            cols.append(idx)
            vals.append(self._tf(float(c)) * idf[idx])
        if not cols:
            return sp.csr_matrix((1, n), dtype=np.float64)
        v = np.asarray(vals, dtype=np.float64)
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        return sp.csr_matrix((v, (np.zeros(len(cols), dtype=int), np.asarray(cols))),
                             shape=(1, n))

    def transform_tokens(self, tokens: Sequence[str]) -> sp.csr_matrix:
        return self.transform_counts(Counter(tokens))

    def stack(self, count_maps: Sequence[Mapping[str, float]]) -> sp.csr_matrix:
        """把多个画像堆叠为矩阵，便于一次 cosine_similarity 算完全部候选。"""
        if not count_maps:
            return sp.csr_matrix((0, max(self.dim, 1)), dtype=np.float64)
        return sp.vstack([self.transform_counts(c) for c in count_maps], format="csr")

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, object]:
        return {"vocab": self.vocab, "df": self.df, "n_docs": self.n_docs,
                "sublinear_tf": self.sublinear_tf}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "IncrementalTfidfSpace":
        sp_ = cls(sublinear_tf=bool(d.get("sublinear_tf", True)))
        sp_.vocab = {str(k): int(v) for k, v in dict(d.get("vocab", {})).items()}
        sp_.df = [float(x) for x in list(d.get("df", []))]
        sp_.n_docs = int(d.get("n_docs", 0))
        return sp_


def cosine(a: sp.csr_matrix, b: sp.csr_matrix) -> np.ndarray:
    """余弦相似度（直接复用 sklearn 实现，保证与经典方法一致）。"""
    if a.shape[1] == 0 or b.shape[0] == 0 or a.nnz == 0:
        return np.zeros(b.shape[0], dtype=np.float64)
    return cosine_similarity(a, b).ravel()


# --------------------------------------------------------------------------- #
# 标签画像
# --------------------------------------------------------------------------- #
class LabelProfile:
    """单个分类标签的特征画像（base 层 + 反馈增量层）。"""

    __slots__ = ("label_id", "base_tf", "delta_tf", "n_samples", "n_feedback")

    def __init__(self, label_id: str) -> None:
        self.label_id = label_id
        self.base_tf: Counter = Counter()
        self.delta_tf: Counter = Counter()
        self.n_samples = 0
        self.n_feedback = 0

    # ---- base 层：随分类体系重建 ---- #
    def reset_base(self) -> None:
        self.base_tf = Counter()
        self.n_samples = 0

    def add_base(self, tokens: Sequence[str], weight: float = 1.0) -> None:
        for t in tokens:
            self.base_tf[t] += weight

    # ---- delta 层：随人工反馈增量累积 ---- #
    def reinforce(self, tokens: Sequence[str], gain: float = 1.0,
                  idf_of: Optional[Callable[[str], float]] = None) -> None:
        """正反馈：把被判为该标签的错题特征增量并入画像。

        这里有两个刻意的设计，都是为了不让反馈噪声淹没关键词库：

        1. **按 IDF 分配**：一条题干里既有"黎曼和"这种真正指向知识点的词，
           也有"求""计算""判断"这种每个标签都会出现的通用词。若一视同仁地
           各加 1.0，通用词会在所有标签画像里同步堆积，非但不提供判别力，
           还会把画像互相拉近。按 IDF 加权后，通用词几乎分不到质量。
        2. **每次修正注入的总质量固定为 gain**：否则长题干天然比短题干更有
           话语权，而且 delta 层会随修正次数线性膨胀——冷启动时基础画像只有
           两三个关键词，一条题干十几个词各加 1.0，一次修正就能让噪声质量
           压过人工关键词 4 倍，L2 归一化后关键词的贡献被稀释殆尽。

        复杂度仍是 O(本题词数)，与语料规模无关，符合"无需全量重训"的约束。
        """
        counts = Counter(tokens)
        if not counts:
            return
        w = {t: (1.0 + math.log(c) if c > 1 else 1.0) * (idf_of(t) if idf_of else 1.0)
             for t, c in counts.items()}
        total = sum(v for v in w.values() if v > 0)
        if total <= 0:
            return
        for t, v in w.items():
            if v > 0:
                self.delta_tf[t] += gain * v / total
        self.n_feedback += 1

    def penalize(self, tokens: Sequence[str], decay: float = 0.25) -> None:
        """负反馈：按有效重叠词权重分配单次总量预算，不下穿 0。

        decay 是一次修正最多移除的总词频质量，默认 0.25；不是每个词
        各减 25%。这与 reinforce 的固定 gain 总质量同量纲，避免长题或
        大画像产生远超正反馈的惩罚。重复词不增加预算，非重叠词不变。
        """
        decay = float(decay)
        if not math.isfinite(decay) or decay < 0:
            raise ValueError("Feedback decay must be a finite nonnegative mass budget")
        if decay == 0 or not tokens:
            return
        eff = self.effective_tf()
        overlap = {t: eff[t] for t in dict.fromkeys(tokens)
                   if eff.get(t, 0.0) > 0 and math.isfinite(eff[t])}
        total = sum(overlap.values())
        if total <= 0:
            return
        budget = min(decay, total)
        for t, weight in overlap.items():
            self.delta_tf[t] = self.delta_tf.get(t, 0.0) - budget * weight / total
            if self.base_tf.get(t, 0.0) + self.delta_tf[t] < 0:
                self.delta_tf[t] = -self.base_tf.get(t, 0.0)

    def top_delta_terms(self, n: int) -> Dict[str, float]:
        """取从人工修正中学到的最强的 n 个判别词。

        供第一层粗筛消费：粗筛靠关键词倒排索引召回候选，冷启动时每个标签只有
        用户随手敲的两三个关键词，正确标签常常压根进不了候选集——第二层精匹配
        再准也无从谈起。把这里学到的词补进倒排索引，才真正闭合了"改一次、
        下次就找得到"的反馈回路。只取 top-n 是为了不让低价值词撑爆索引。
        """
        pos = [(t, v) for t, v in self.delta_tf.items() if v > 0]
        pos.sort(key=lambda x: -x[1])
        return dict(pos[:max(n, 0)])

    def effective_tf(self) -> Dict[str, float]:
        """有效画像 = clip(base + delta, 0)。"""
        out: Dict[str, float] = {}
        for t, c in self.base_tf.items():
            out[t] = c
        for t, c in self.delta_tf.items():
            out[t] = out.get(t, 0.0) + c
        return {t: c for t, c in out.items() if c > 1e-9}

    def is_trained(self) -> bool:
        return bool(self.base_tf or self.delta_tf)

    def to_dict(self) -> Dict[str, object]:
        # 只持久化 delta 层：base 层由分类体系随时重建，存了反而会与用户编辑冲突
        return {"label_id": self.label_id,
                "delta_tf": dict(self.delta_tf),
                "n_feedback": self.n_feedback}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "LabelProfile":
        p = cls(str(d.get("label_id", "")))
        p.delta_tf = Counter({str(k): float(v) for k, v in dict(d.get("delta_tf", {})).items()})
        p.n_feedback = int(d.get("n_feedback", 0))
        return p
