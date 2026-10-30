# -*- coding: utf-8 -*-
"""多特征融合双层分级分类引擎（技术路线 4.2.2）。

    错题 --预处理--> FeatureBundle
                       │
              ┌────────┴────────┐   对每个启用的分类维度
              │ 第一层 关键词粗筛 │   几十个标签 -> 3~5 个候选   (coarse.py)
              └────────┬────────┘
              ┌────────┴────────┐
              │ 第二层 融合精匹配 │   TF-IDF + 余弦 + 差异化权重 (fine.py)
              └────────┬────────┘
                  最高分标签

引擎同时是"特征库热更新"的责任人：用户在界面上改了关键词/样本/权重后，
``sync_scheme()`` 会重建倒排索引与标签画像 base 层，而由反馈累积的 delta 层
原样保留——这正是"无需重启系统或修改代码即可生效"的实现方式。
"""
from __future__ import annotations

import logging
import dataclasses
import time
from typing import Dict, List, Optional, Sequence

from app import config
from app.core.classify import weights as W
from app.core.classify.coarse import CoarseFilter
from app.core.classify.fine import FineMatcher
from app.core.classify.vector_space import IncrementalTfidfSpace, LabelProfile
from app.core.models import (Candidate, ClassifyResult, DimResult, FeatureBundle,
                             Question, Scheme)
from app.core.preprocess import cleaner, lexicon, pipeline
from app.core.preprocess.text_cn import tokenize_cn
from app.core.preprocess.text_en import tokenize_en
from app.core.storage import read_json, write_json

log = logging.getLogger(__name__)

# 关键词子词扩展的权重折扣（"杨氏双缝干涉" -> "干涉"，提升召回但不喧宾夺主）
SUBWORD_DISCOUNT = 0.35

# Separate from per-scheme label profiles, preserving the existing file format.
_FEEDBACK_STATISTICS_KEY = "__feedback_document_statistics__"


class ClassificationEngine:
    """面向单套分类体系的分级分类引擎。"""

    def __init__(self, scheme: Scheme, rules: Optional[W.WeightRules] = None) -> None:
        self.scheme = scheme
        self.rules = rules or W.WeightRules()
        self.space = IncrementalTfidfSpace()
        self.profiles: Dict[str, LabelProfile] = {}
        self.coarse: Dict[str, CoarseFilter] = {}
        self.fine: Optional[FineMatcher] = None
        self._revision = -1
        self._feedback_n_docs = 0
        self._feedback_df: Dict[str, int] = {}
        # 这里**不建索引**。建一次索引要两三秒（要往 jieba 里灌关键词、
        # 重算全部标签画像），而界面在启动时就会为了显示"XX 个标签 / XX 个关键词"
        # 而创建引擎——若在构造函数里同步建库，主窗口要卡两秒多才出得来，
        # 后台预热线程也就白写了。索引改为首次 classify() 时惰性构建，
        # 或由 warmup 线程在后台提前建好。
        self.load_profiles()

    @property
    def ready(self) -> bool:
        """索引是否已就绪（未就绪时 stats() 里的词表规模等字段为 0）。"""
        return self.fine is not None and self._revision == self.scheme.revision

    # ------------------------------------------------------------------ #
    # 特征库构建与热更新
    # ------------------------------------------------------------------ #
    def _label_tokens(self, label) -> List[str]:
        """把一个标签的关键词 + 标准样本展开为画像词袋。"""
        subject = self.scheme.subject
        toks: List[str] = []
        boost = int(config.LABEL_KEYWORD_BOOST)

        # 1) 关键词本体（含标签名），高权重重复注入
        kws = [k.strip() for k in ([label.name] + list(label.keywords)) if k and k.strip()]
        for kw in kws:
            w = max(1, int(round(boost * label.kw_weight(kw))))
            toks.extend([kw] * w)

        # 2) 关键词子词扩展：让"干涉条纹"这样的表述也能命中"杨氏双缝干涉"
        jieba = lexicon.ensure_jieba()
        stop = lexicon.all_stopwords(subject)
        sub_w = max(1, int(round(boost * SUBWORD_DISCOUNT)))
        for kw in kws:
            if len(kw) < 3:
                continue
            for sw in jieba.cut_for_search(kw):
                sw = sw.strip()
                if len(sw) >= 2 and sw != kw and sw not in stop:
                    toks.extend([sw] * sub_w)

        # 3) 标准错题样本：走与真实错题完全一致的分词流程，保证同分布
        for s in label.samples:
            if not s:
                continue
            sample = Question(subject=subject,
                              qtype=cleaner.guess_qtype(s, subject=subject), raw_text=s)
            toks.extend(pipeline.preprocess(sample).all_tokens())
        return toks

    def sync_scheme(self, force: bool = False) -> None:
        """按当前分类体系重建索引与画像 base 层（反馈 delta 层保留）。"""
        if not force and self._revision == self.scheme.revision and self.fine is not None:
            return
        t0 = time.perf_counter()

        # 注意：这里**刻意不把体系关键词注入 jieba 词典**。
        #
        # 之前是注入的，理由是"用户新加了'格林公式'这种多字关键词，分词器必须
        # 立刻认识它，否则粗筛永远命中不了"。这个理由是错的，代价却很大：
        #
        # * 错在哪：粗筛对未登录关键词有三重兜底——分词精确命中、子词扩展命中
        #   （cut_for_search 拆关键词）、以及原文子串扫描（substr_keys，覆盖
        #   3 字以上关键词）。实测题干里没被切出来的"光程差""条纹间距"照样命中。
        #   精匹配那侧也不需要：标签画像与题目共用同一个分词器，切错但切得一致，
        #   余弦相似度照样匹配。
        # * 代价：jieba 词典是**进程级全局**的，注入不可逆且跨学科互相污染。
        #   实测关键词"分词作状语"被注入后，"现在分词和过去分词作状语"从
        #   现在分词/过去分词/状语 变成 现在分词/分词作状语 —— 长词跨边界把
        #   "过去分词"整个吃掉。更糟的是分类结果会依赖"先同步过哪套体系"，
        #   同一道题换个操作顺序就换个结果，不可复现。
        # * 实测取舍：注入换来 +0.6pp（155 条里 1 条，噪声量级），却把英语题的
        #   核心特征提取准确率从 93.3% 压到 86.7%。不值得。

        self.load_profiles()                      # 先取回历史 delta 层

        docs: List[List[str]] = []
        for d in self.scheme.dims:
            for lb in d.labels:
                toks = self._label_tokens(lb)
                prof = self.profiles.setdefault(lb.id, LabelProfile(lb.id))
                prof.reset_base()
                prof.add_base(toks)
                prof.n_samples = len(lb.samples)
                docs.append(toks)

        # 冷启动：用 sklearn CountVectorizer 一次性建词表与 df
        self.space.fit(docs)
        self.space.add_document_statistics(self._feedback_n_docs, self._feedback_df)
        # 保证反馈中出现过的新词也在词表内
        extra = set()
        for p in self.profiles.values():
            extra.update(p.delta_tf.keys())
        self.space.ensure_terms(extra)

        self.coarse = {d.key: CoarseFilter(d) for d in self.scheme.dims}
        # 把已从人工修正中学到的判别词贴回粗筛索引，让反馈成果在重建后依然生效
        for d in self.scheme.dims:
            cf = self.coarse.get(d.key)
            if cf is None:
                continue
            for lb in d.labels:
                prof = self.profiles.get(lb.id)
                if prof is not None and prof.delta_tf:
                    cf.learn(lb.id, prof.top_delta_terms(config.FEEDBACK_KEYWORD_TOPN))
        self.fine = FineMatcher(self.space, self.profiles, self.rules)
        self._revision = self.scheme.revision
        log.info("特征库同步完成：%d 标签 / %d 词 / %.0fms",
                 len(self.profiles), self.space.dim, (time.perf_counter() - t0) * 1000)

    def invalidate(self) -> None:
        """标记特征库失效（用户编辑体系后调用）。"""
        self._revision = -1

    # ------------------------------------------------------------------ #
    # 分类
    # ------------------------------------------------------------------ #
    def matching_features(self, q: Question, fb: FeatureBundle, dim_key: str) -> FeatureBundle:
        """错因与难度只用用户提供的依据，不能从题干主题推断学生状态。"""
        if dim_key not in (config.DIM_ERROR, config.DIM_DIFFICULTY):
            return fb
        annotation = (q.annotation or "").strip()
        tokens = tokenize_cn(annotation, q.subject, include_ascii=False)
        tokens.extend(tokenize_en(annotation))
        return dataclasses.replace(fb, clean_text=annotation, text_tokens=tokens,
                                   formula_tokens=[], qtype_tokens=[], subject_tokens=[],
                                   latex="", ocr_text="", transcript="")

    def classify(self, q: Question, fb: Optional[FeatureBundle] = None,
                 top_k: Optional[int] = None) -> ClassifyResult:
        """对单条错题在全部启用维度上执行双层分级分类。

        Args:
            top_k: 粗筛保留的候选数；传入极大值即可退化为"单阶段全量匹配"，
                   用于 tools/evaluate.py 中的架构对照实验。
        """
        self.sync_scheme()
        t0 = time.perf_counter()
        if fb is None:
            fb = pipeline.preprocess(q)
        pre_ms = fb.elapsed_ms
        t1 = time.perf_counter()

        k = config.COARSE_TOP_K if top_k is None else top_k
        res = ClassifyResult(question_id=q.id, preprocess_ms=pre_ms)
        for d in self.scheme.dims:
            if not d.enabled or not d.labels:
                continue
            if d.key == config.DIM_PRIORITY:
                selected = d.label_by_name(q.priority) if q.priority else None
                res.dims[d.key] = DimResult(dim=d.key, total=len(d.labels),
                    label_id=selected.id if selected else "",
                    label_name=selected.name if selected else "", score=1.0 if selected else 0.0)
                continue
            dimension_features = self.matching_features(q, fb, d.key)
            if dimension_features.is_empty():
                res.dims[d.key] = DimResult(dim=d.key, total=len(d.labels))
                continue
            cf = self.coarse.get(d.key)
            if cf is None:
                continue
            dim_started = time.perf_counter()
            cands, coarse_ms, direct = cf.screen(dimension_features, k)
            ranked = self.fine.match(dimension_features, d, cands) if self.fine else cands

            dr = DimResult(dim=d.key, candidates=ranked, coarse_ms=coarse_ms,
                           scanned=len(cands), total=len(d.labels))
            if ranked and ranked[0].final_score > 0:
                dr.label_id = ranked[0].label_id
                dr.label_name = ranked[0].label_name
                dr.score = ranked[0].final_score
            else:
                # 精匹配全 0，但粗筛有实打实的关键词命中（非兜底）时，采信粗筛。
                #
                # 这种情况真实存在：题干说"水平渐近线"，而标签关键词库里写的是
                # "渐近线"。粗筛靠原文子串匹配照样命中，但精匹配是按分词后的词
                # 算 TF-IDF 余弦的，两个词形没有任何重叠，相似度就是 0。
                # 此时若坚持"没分数就不分类"，这道题会被整条丢掉——而粗筛
                # 其实已经给出了一个有依据的答案，丢掉它没有任何好处。
                #
                # 置信度按 COARSE_ONLY_CONFIDENCE 折算，保证落在"低置信、
                # 建议复核"区间：这确实只是关键词层面的猜测，得让用户看得见。
                hit = [c for c in ranked if not c.fallback and c.coarse_score > 0]
                if hit:
                    best = max(hit, key=lambda c: c.coarse_score)
                    best.final_score = min(
                        best.coarse_score * config.COARSE_ONLY_CONFIDENCE,
                        config.LOW_CONFIDENCE_THRESHOLD * 0.99)
                    ranked.sort(key=lambda c: -c.final_score)
                    dr.label_id = best.label_id
                    dr.label_name = best.label_name
                    dr.score = best.final_score
            dr.fine_ms = max(0.0, (time.perf_counter() - dim_started) * 1000.0 - coarse_ms)
            res.dims[d.key] = dr
        res.classify_ms = (time.perf_counter() - t1) * 1000.0
        _ = t0
        return res

    def classify_many(self, questions: Sequence[Question],
                      progress: Optional[callable] = None) -> List[ClassifyResult]:
        """批量分类。progress(done, total, question, result)。"""
        self.sync_scheme()
        out: List[ClassifyResult] = []
        total = len(questions)
        for i, q in enumerate(questions, 1):
            r = self.classify(q)
            out.append(r)
            if progress:
                progress(i, total, q, r)
        return out

    @staticmethod
    def apply_result(q: Question, res: ClassifyResult, scheme: Scheme,
                     fb: Optional[FeatureBundle] = None) -> None:
        """把分类结果写回错题记录。"""
        same_scheme = q.scheme_id == scheme.id
        confirmed = {}
        for dimension in scheme.enabled_dims():
            label_id = q.labels.get(dimension.key, "")
            label = dimension.label_by_id(label_id)
            if same_scheme and q.corrected.get(dimension.key) and label is not None:
                confirmed[dimension.key] = label
        q.labels = {}
        q.label_names = {}
        q.scores = {}
        q.corrected = {}
        q.candidates = {}
        q.scheme_id = scheme.id
        q.elapsed_ms = res.total_ms
        for dim_key, label in confirmed.items():
            q.labels[dim_key] = label.id
            q.label_names[dim_key] = label.name
            q.scores[dim_key] = 1.0
            q.corrected[dim_key] = True
        for dim_key, dr in res.dims.items():
            if not dr.label_id:
                continue
            # 已被人工修正过的维度不被自动结果覆盖（尊重用户判断）
            if q.corrected.get(dim_key):
                continue
            q.labels[dim_key] = dr.label_id
            q.label_names[dim_key] = dr.label_name
            q.scores[dim_key] = dr.score
            q.candidates[dim_key] = [c.to_dict() for c in dr.candidates[:config.COARSE_TOP_K]]
            if dim_key == config.DIM_PRIORITY and q.priority == dr.label_name:
                q.corrected[dim_key] = True
        q.classified = bool(q.labels)
        if fb is not None:
            q.feature = fb.to_dict()
        q.updated_at = time.time()

    # ------------------------------------------------------------------ #
    # 画像持久化（只存 delta 层）
    # ------------------------------------------------------------------ #
    def profile_path(self):
        return config.PROFILE_FILE

    def register_feedback_tokens(self, tokens: Sequence[str]) -> None:
        """Register one feedback document in O(unique tokens), without refitting.

        save_profiles persists these counters with label deltas. Rebuilding a
        scheme overlays them on the new base corpus, rather than treating old
        keywords/samples as permanent feedback documents.
        """
        if not tokens:
            return
        self.space.add_document(tokens)
        self._feedback_n_docs += 1
        for term in set(tokens):
            self._feedback_df[term] = self._feedback_df.get(term, 0) + 1

    def load_profiles(self) -> None:
        data = read_json(self.profile_path(), {}) or {}
        statistics = data.get(_FEEDBACK_STATISTICS_KEY, {}).get(self.scheme.id, {})
        try:
            n_docs = int(statistics.get("n_docs", 0))
            frequencies = {str(term): int(count) for term, count in statistics.get("df", {}).items()}
            if n_docs < 0 or any(count < 0 or count > n_docs for count in frequencies.values()):
                raise ValueError("feedback df exceeds document count")
            self._feedback_n_docs, self._feedback_df = n_docs, frequencies
        except (TypeError, ValueError, AttributeError):
            log.warning("反馈文档统计格式无效；保留标签画像，使用基础词频统计")
            self._feedback_n_docs, self._feedback_df = 0, {}
        bucket = data.get(self.scheme.id, {})
        for lid, d in bucket.items():
            p = LabelProfile.from_dict(d)
            old = self.profiles.get(lid)
            if old is not None:
                old.delta_tf = p.delta_tf
                old.n_feedback = p.n_feedback
            else:
                self.profiles[lid] = p

    def save_profiles(self) -> None:
        data = read_json(self.profile_path(), {}) or {}
        data[self.scheme.id] = {lid: p.to_dict() for lid, p in self.profiles.items()
                                if p.delta_tf}
        statistics = data.setdefault(_FEEDBACK_STATISTICS_KEY, {})
        statistics[self.scheme.id] = {"n_docs": self._feedback_n_docs,
                                      "df": dict(self._feedback_df)}
        write_json(self.profile_path(), data)

    def reset_profiles(self) -> None:
        """清空该体系的全部反馈增量（回到出厂状态）。"""
        for p in self.profiles.values():
            p.delta_tf.clear()
            p.n_feedback = 0
        self._feedback_n_docs, self._feedback_df = 0, {}
        self.save_profiles()
        self.invalidate()
        self.sync_scheme(force=True)

    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, object]:
        """引擎状态摘要，供界面展示。"""
        n_labels = sum(len(d.labels) for d in self.scheme.dims)
        n_kw = sum(len(lb.keywords) for d in self.scheme.dims for lb in d.labels)
        n_sample = sum(len(lb.samples) for d in self.scheme.dims for lb in d.labels)
        n_fb = sum(p.n_feedback for p in self.profiles.values())
        return {"labels": n_labels, "keywords": n_kw, "samples": n_sample,
                "vocab": self.space.dim, "docs": self.space.n_docs, "feedback": n_fb}
