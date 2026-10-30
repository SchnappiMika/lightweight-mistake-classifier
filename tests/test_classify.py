# -*- coding: utf-8 -*-
"""双层分级分类模块测试（对应需求 4.4.1 第 2 点）。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import config
from app.core.classify import weights as W
from app.core.classify.engine import ClassificationEngine
from app.core.classify.vector_space import IncrementalTfidfSpace, LabelProfile
from app.core.models import CategoryLabel, Dimension, Question, Scheme

DATASET = Path(__file__).resolve().parent.parent / "samples" / "eval_dataset.json"


def q_of(subject, qtype, text, note=""):
    return Question(subject=subject, qtype=qtype, raw_text=text, annotation=note)


class TestVectorSpace:
    """增量式 TF-IDF 向量空间。"""

    def test_fit_then_incremental_matches_semantics(self):
        sp = IncrementalTfidfSpace()
        sp.fit([["极限", "洛必达"], ["积分", "换元"]])
        assert sp.n_docs == 2 and sp.dim == 4
        sp.add_document(["极限", "夹逼"])
        assert sp.n_docs == 3
        assert sp.df[sp.vocab["极限"]] == 2      # 增量后 df 正确累加
        assert "夹逼" in sp.vocab                # 新词自动进词表

    def test_idf_formula_matches_sklearn(self):
        """idf 必须与 sklearn smooth_idf=True 完全一致，否则无法宣称"经典方法"。"""
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = [["a", "b"], ["b", "c"], ["a", "b", "c"]]
        sp = IncrementalTfidfSpace()
        sp.fit(docs)
        ref = TfidfVectorizer(analyzer=lambda d: d, lowercase=False).fit(docs)
        for term, i in ref.vocabulary_.items():
            assert np.isclose(sp.idf()[sp.vocab[term]], ref.idf_[i])

    def test_vector_l2_normalized(self):
        import numpy as np
        sp = IncrementalTfidfSpace()
        sp.fit([["a", "b", "b"], ["b", "c"]])
        v = sp.transform_tokens(["a", "b", "b"])
        assert np.isclose(np.linalg.norm(v.toarray()), 1.0)

    def test_unknown_tokens_give_zero_vector(self):
        sp = IncrementalTfidfSpace()
        sp.fit([["a"]])
        assert sp.transform_tokens(["zzz"]).nnz == 0

    def test_empty_space_no_crash(self):
        sp = IncrementalTfidfSpace()
        sp.fit([])
        assert sp.transform_tokens(["a"]).shape[0] == 1


class TestLabelProfile:
    """标签画像 base + delta 双层结构。"""

    def test_effective_is_base_plus_delta(self):
        p = LabelProfile("x")
        p.add_base(["a", "a", "b"])
        p.reinforce(["a"], gain=2.0)
        eff = p.effective_tf()
        assert eff["a"] == pytest.approx(4.0) and eff["b"] == pytest.approx(1.0)

    def test_penalize_never_goes_negative(self):
        """负反馈不能把权重压成负数，否则相似度会出现反直觉的负贡献。"""
        p = LabelProfile("x")
        p.add_base(["a"])
        for _ in range(50):
            p.penalize(["a"], decay=0.9)
        assert p.effective_tf().get("a", 0.0) >= 0.0

    def test_only_delta_persisted(self):
        """base 层由分类体系重建，持久化它会与用户手改词库冲突。"""
        p = LabelProfile("x")
        p.add_base(["base_word"])
        p.reinforce(["fb_word"])
        d = p.to_dict()
        assert "base_word" not in d["delta_tf"] and "fb_word" in d["delta_tf"]
        back = LabelProfile.from_dict(d)
        assert back.delta_tf["fb_word"] > 0 and not back.base_tf


class TestWeightRules:
    """差异化特征权重规则。"""

    def test_normalized_to_one(self):
        r = W.WeightRules()
        for subj in config.SUBJECTS:
            for qt in config.QTYPES:
                assert sum(r.get(subj, qt).values()) == pytest.approx(1.0)

    def test_formula_weight_higher_for_math_formula(self):
        """高数公式题必须比高数纯文本题更看重公式通道。"""
        r = W.WeightRules()
        a = r.get(config.SUBJECT_MATH, config.QTYPE_FORMULA)
        b = r.get(config.SUBJECT_MATH, config.QTYPE_TEXT)
        assert a[config.CHANNEL_FORMULA] > b[config.CHANNEL_FORMULA]

    def test_qtype_weight_higher_for_english(self):
        """英语听力题的题型通道应比数学公式题更重要。"""
        r = W.WeightRules()
        a = r.get(config.SUBJECT_ENGLISH, config.QTYPE_LISTENING)
        b = r.get(config.SUBJECT_MATH, config.QTYPE_FORMULA)
        assert a[config.CHANNEL_QTYPE] > b[config.CHANNEL_QTYPE]

    def test_redistribute_empty_channel(self):
        """公式通道为空时，其权重应摊给其他通道，总和仍为 1。"""
        base = {config.CHANNEL_TEXT: 0.4, config.CHANNEL_FORMULA: 0.4,
                config.CHANNEL_QTYPE: 0.1, config.CHANNEL_SUBJECT: 0.1}
        out = W.redistribute(base, {config.CHANNEL_TEXT: True, config.CHANNEL_FORMULA: False,
                                    config.CHANNEL_QTYPE: True, config.CHANNEL_SUBJECT: True})
        assert out[config.CHANNEL_FORMULA] == 0.0
        assert sum(out.values()) == pytest.approx(1.0)
        assert out[config.CHANNEL_TEXT] > base[config.CHANNEL_TEXT]

    def test_label_override_full(self):
        """界面滑块一次给全四个通道，此时覆盖值应按比例原样生效。"""
        r = W.WeightRules()
        w = r.get(config.SUBJECT_MATH, config.QTYPE_TEXT, override={
            config.CHANNEL_FORMULA: 0.9, config.CHANNEL_TEXT: 0.1,
            config.CHANNEL_QTYPE: 0.1, config.CHANNEL_SUBJECT: 0.1})
        assert w[config.CHANNEL_FORMULA] == pytest.approx(0.9 / 1.2, abs=1e-6)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_label_override_partial_patches_bucket(self):
        """只覆盖部分通道时，其余通道沿用本桶权重，而不是退回全局兜底规则。"""
        r = W.WeightRules()
        bucket = r.get(config.SUBJECT_MATH, config.QTYPE_FORMULA)
        w = r.get(config.SUBJECT_MATH, config.QTYPE_FORMULA,
                  override={config.CHANNEL_FORMULA: 0.9})
        # 公式通道被显著拉高
        assert w[config.CHANNEL_FORMULA] > bucket[config.CHANNEL_FORMULA]
        # 其余通道之间的相对比例保持本桶原样（高数公式题 文本:题型 = 0.35:0.10）
        assert (w[config.CHANNEL_TEXT] / w[config.CHANNEL_QTYPE]
                == pytest.approx(bucket[config.CHANNEL_TEXT] / bucket[config.CHANNEL_QTYPE]))
        assert sum(w.values()) == pytest.approx(1.0)

    def test_override_preserves_learned_weights(self):
        """标签级覆盖不应抹掉系统已从人工修正中学到的该桶权重。"""
        r = W.WeightRules()
        r.update_from_feedback(config.SUBJECT_MATH, config.QTYPE_FORMULA,
                               right_channels={config.CHANNEL_FORMULA: 0.9},
                               wrong_channels={config.CHANNEL_TEXT: 0.9})
        learned = r.get(config.SUBJECT_MATH, config.QTYPE_FORMULA)
        w = r.get(config.SUBJECT_MATH, config.QTYPE_FORMULA,
                  override={config.CHANNEL_SUBJECT: 0.3})
        # 只覆盖了学科通道，文本/公式的相对关系应继承"学到之后"的状态
        assert (w[config.CHANNEL_FORMULA] / w[config.CHANNEL_TEXT]
                == pytest.approx(learned[config.CHANNEL_FORMULA] / learned[config.CHANNEL_TEXT]))


class TestCoarse:
    """第一层关键词粗筛。

    注意：引擎的索引是**惰性构建**的（构造时不建，首次 classify 才建，
    否则界面一启动就会在主线程上卡两秒多）。这些测试绕过 classify 直接
    检验粗筛器，因此必须先显式 sync_scheme()。
    """

    def test_narrows_candidates(self, math_engine):
        from app.core.preprocess import pipeline
        math_engine.sync_scheme()
        fb = pipeline.preprocess(q_of(config.SUBJECT_MATH, config.QTYPE_TEXT,
                                      "用比值判别法判断该级数的敛散性"))
        cf = math_engine.coarse[config.DIM_KNOWLEDGE]
        cands, _ms, direct = cf.screen(fb, config.COARSE_TOP_K)
        assert not direct
        assert len(cands) <= config.COARSE_TOP_K
        assert any(c.label_name == "无穷级数" for c in cands)

    def test_recall_fallback_when_no_hit(self, math_engine):
        """一个关键词都没命中时也必须给出候选，否则精匹配无从下手。"""
        from app.core.models import FeatureBundle
        math_engine.sync_scheme()
        fb = FeatureBundle(subject=config.SUBJECT_MATH, qtype=config.QTYPE_TEXT,
                           text_tokens=["紫色", "香蕉"], clean_text="紫色香蕉")
        cands, _ms, _d = math_engine.coarse[config.DIM_KNOWLEDGE].screen(fb)
        assert len(cands) >= config.COARSE_MIN_K
        assert all(c.fallback for c in cands)

    def test_direct_pass_when_few_labels(self):
        """标签数不超过 top_k 时应跳过粗筛直接全量精匹配（粗筛没有收益）。"""
        from app.core.classify.coarse import CoarseFilter
        from app.core.models import FeatureBundle
        d = Dimension(key="k", name="k", labels=[
            CategoryLabel(name="a", keywords=["a"]), CategoryLabel(name="b", keywords=["b"])])
        cands, _ms, direct = CoarseFilter(d).screen(FeatureBundle(), config.COARSE_TOP_K)
        assert direct and len(cands) == 2

    def test_subword_expansion_boosts_recall(self, repo):
        """局部表述"干涉条纹"必须能召回"光的干涉与衍射"。"""
        from app.core.preprocess import pipeline
        eng = ClassificationEngine(repo.scheme_by_subject(config.SUBJECT_PHYSICS))
        eng.sync_scheme()
        fb = pipeline.preprocess(q_of(config.SUBJECT_PHYSICS, config.QTYPE_TEXT,
                                      "干涉条纹的间距怎么算"))
        cands, _ms, _d = eng.coarse[config.DIM_KNOWLEDGE].screen(fb)
        assert any(c.label_name == "光的干涉与衍射" and not c.fallback for c in cands)


class TestEngine:
    """双层分级分类引擎。"""

    @pytest.mark.parametrize("qtype,text,gold", [
        ("formula", r"求极限 $\lim_{x\to0}\frac{\tan x-\sin x}{x^3}$", "函数与极限"),
        ("formula", r"计算定积分 $\int_{0}^{1}xe^{x}\,\mathrm{d}x$", "定积分及其应用"),
        ("text", "用比值判别法判断级数敛散性，讨论绝对收敛还是条件收敛", "无穷级数"),
        ("formula", r"用极坐标计算二重积分 $\iint_D e^{-x^2-y^2}dxdy$", "重积分"),
        ("text", "用格林公式把第二类曲线积分化为二重积分", "曲线曲面积分"),
        ("text", "求二阶常系数线性微分方程的通解，先写特征方程", "微分方程"),
        ("text", "用拉格朗日乘数法求条件极值", "多元函数微分学"),
    ])
    def test_math_classification(self, math_engine, qtype, text, gold):
        r = math_engine.classify(q_of(config.SUBJECT_MATH, qtype, text))
        assert r.dims[config.DIM_KNOWLEDGE].label_name == gold

    def test_all_dims_classified(self, math_engine):
        r = math_engine.classify(q_of(config.SUBJECT_MATH, config.QTYPE_TEXT,
                                      "求极限", "公式的系数记错了，属于低级失误"))
        assert set(r.dims) >= {config.DIM_KNOWLEDGE, config.DIM_ERROR,
                               config.DIM_DIFFICULTY, config.DIM_PRIORITY}
        assert r.dims[config.DIM_ERROR].label_name == "公式记错"

    def test_candidates_carry_explanation(self, math_engine):
        """分类依据必须可解释：候选要带命中关键词与各通道得分/权重。"""
        r = math_engine.classify(q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "求定积分的值"))
        top = r.dims[config.DIM_KNOWLEDGE].candidates[0]
        assert top.channel_scores and top.channel_weights
        assert sum(top.channel_weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_two_stage_reduces_computation(self, math_engine):
        """双层分级的精匹配范围必须小于全量。"""
        q = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "求定积分")
        two = math_engine.classify(q)
        full = math_engine.classify(q, top_k=10 ** 6)
        assert two.dims[config.DIM_KNOWLEDGE].scanned < full.dims[config.DIM_KNOWLEDGE].scanned
        assert two.dims[config.DIM_KNOWLEDGE].label_name == \
            full.dims[config.DIM_KNOWLEDGE].label_name

    def test_empty_question_no_crash(self, math_engine):
        r = math_engine.classify(q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, ""))
        assert r.dims[config.DIM_KNOWLEDGE].score == 0.0

    def test_scheme_edit_hot_reload(self, repo):
        """用户加了标签与关键词后，无需重启即可立刻生效。"""
        sc = repo.scheme_by_subject(config.SUBJECT_MATH)
        eng = ClassificationEngine(sc)
        text = "这道题考查斐波那契数列的通项公式"
        assert eng.classify(q_of(config.SUBJECT_MATH, "text", text)
                            ).dims[config.DIM_KNOWLEDGE].label_name != "斐波那契专题"
        dim = sc.dim(config.DIM_KNOWLEDGE)
        dim.labels.append(CategoryLabel(name="斐波那契专题", dim=config.DIM_KNOWLEDGE,
                                        keywords=["斐波那契数列", "斐波那契", "通项公式"]))
        sc.touch()
        eng.invalidate()
        assert eng.classify(q_of(config.SUBJECT_MATH, "text", text)
                            ).dims[config.DIM_KNOWLEDGE].label_name == "斐波那契专题"

    def test_apply_result_respects_manual_correction(self):
        """已人工修正的维度不应被后续自动分类覆盖。"""
        sc = Scheme(name="t", subject=config.SUBJECT_MATH, dims=[
            Dimension(key=config.DIM_KNOWLEDGE, name="知识点",
                      labels=[CategoryLabel(name="A", keywords=["极限"]),
                              CategoryLabel(name="B", keywords=["积分"])])])
        eng = ClassificationEngine(sc)
        q = q_of(config.SUBJECT_MATH, "text", "求极限")
        q.scheme_id = sc.id
        manual_label = sc.dims[0].labels[1]
        q.labels[config.DIM_KNOWLEDGE] = manual_label.id
        q.label_names[config.DIM_KNOWLEDGE] = manual_label.name
        q.corrected[config.DIM_KNOWLEDGE] = True
        ClassificationEngine.apply_result(q, eng.classify(q), sc)
        assert q.labels[config.DIM_KNOWLEDGE] == manual_label.id
        assert q.label_names[config.DIM_KNOWLEDGE] == manual_label.name


class TestAccuracyOnDataset:
    """基于 155 条人工标注样本的准确率验收（需求指标：纯文本 ≥95%、全类型 ≥90%）。"""

    @pytest.fixture(scope="class")
    def rows(self):
        with open(DATASET, encoding="utf-8") as f:
            return json.load(f)

    def _run(self, repo, rows):
        engines = {s.subject: ClassificationEngine(s) for s in repo.schemes}
        res = []
        for r in rows:
            eng = engines.get(r["学科"])
            if eng is None:
                continue
            out = eng.classify(q_of(r["学科"], r["题型"], r["题干"], r.get("标注", "")))
            dr = out.dims.get(config.DIM_KNOWLEDGE)
            res.append((r, dr.label_name == r["标准知识点"] if dr else False, dr))
        return res

    def test_overall_accuracy(self, repo, rows):
        res = self._run(repo, rows)
        acc = sum(1 for _r, ok, _d in res if ok) / len(res)
        assert acc >= 0.90, "全类型平均准确率 {:.1%} 低于需求指标 90%".format(acc)

    def test_text_accuracy(self, repo, rows):
        res = [x for x in self._run(repo, rows) if x[0]["题型"] == config.QTYPE_TEXT]
        acc = sum(1 for _r, ok, _d in res if ok) / len(res)
        assert acc >= 0.95, "纯文本准确率 {:.1%} 低于需求指标 95%".format(acc)

    def test_formula_accuracy(self, repo, rows):
        res = [x for x in self._run(repo, rows) if x[0]["题型"] == config.QTYPE_FORMULA]
        acc = sum(1 for _r, ok, _d in res if ok) / len(res)
        assert acc >= 0.90, "公式类准确率 {:.1%} 低于需求指标 90%".format(acc)

    def test_coarse_recall(self, repo, rows):
        """粗筛召回率：正确标签必须留在候选集里，否则精匹配再准也白搭。"""
        res = self._run(repo, rows)
        hit = sum(1 for r, _ok, d in res
                  if d and any(c.label_name == r["标准知识点"] for c in d.candidates))
        assert hit / len(res) >= 0.95


class TestNoGlobalPollution:
    """回归锁：同步分类体系不得污染全局分词状态。

    曾经的 bug：引擎同步时把体系里的全部关键词注入 jieba 全局词典，结果
    一条叫"分词作状语"的关键词把"现在分词和过去分词作状语"从
    现在分词/过去分词/状语 变成了 现在分词/分词作状语——"过去分词"被跨边界的
    长词整个吃掉。而 jieba 词典是进程级全局的，这意味着**同一道题的分类结果
    会依赖"你之前打开过哪套体系"**，不可复现。
    """

    def test_tokenize_stable_after_sync(self, repo):
        from app.core.preprocess.text_cn import tokenize_cn

        text = "现在分词和过去分词作状语，逻辑主语判断错"
        before = tokenize_cn(text, config.SUBJECT_ENGLISH)
        for s in repo.schemes:                    # 同步全部体系
            ClassificationEngine(s).sync_scheme()
        after = tokenize_cn(text, config.SUBJECT_ENGLISH)
        assert before == after, (
            "同步分类体系改变了分词结果：{} -> {}".format("/".join(before), "/".join(after)))

    def test_classification_independent_of_sync_order(self, repo):
        """同一道题的分类结果不能因为"先同步了哪套体系"而改变。"""
        q_text = "现在分词和过去分词作状语的区别"
        eng_en = ClassificationEngine(repo.scheme_by_subject(config.SUBJECT_ENGLISH))
        first = eng_en.classify(q_of(config.SUBJECT_ENGLISH, config.QTYPE_READING,
                                     q_text)).dims[config.DIM_KNOWLEDGE].label_name

        # 先把其它学科的体系全同步一遍，再回来分同一道题
        for s in repo.schemes:
            if s.subject != config.SUBJECT_ENGLISH:
                ClassificationEngine(s).sync_scheme()
        again = ClassificationEngine(repo.scheme_by_subject(config.SUBJECT_ENGLISH)).classify(
            q_of(config.SUBJECT_ENGLISH, config.QTYPE_READING,
                 q_text)).dims[config.DIM_KNOWLEDGE].label_name
        assert first == again, "分类结果依赖体系同步顺序：{!r} vs {!r}".format(first, again)

    def test_add_runtime_words_is_forbidden(self):
        """这个口子已经封死，谁想再打开它得先读完那段 docstring。"""
        from app.core.preprocess import lexicon
        with pytest.raises(NotImplementedError):
            lexicon.add_runtime_words(["测试词"])


class TestCoarseOnlyFallback:
    """回归锁：精匹配算不出分数时，必须采信粗筛，不能把题丢着不分。

    曾经的 bug：题干说"水平渐近线"、关键词库写的是"渐近线"，粗筛靠子串匹配
    命中了，但精匹配按分词后的词算余弦——两个词形零重叠，相似度为 0，
    于是整道题被判为空标签、得分 0，用户看到的就是"这题没分类"。
    """

    def test_falls_back_to_coarse_when_fine_scores_zero(self, math_engine):
        r = math_engine.classify(q_of(
            config.SUBJECT_MATH, config.QTYPE_FORMULA,
            r"求 $f(x)=\frac{x^{2}}{x-1}$ 的水平渐近线与铅直渐近线。"))
        dr = r.dims[config.DIM_KNOWLEDGE]
        assert dr.label_name, "精匹配为 0 时把题丢着不分了"
        assert dr.label_name == "微分中值定理与导数应用"

    def test_coarse_fallback_marked_low_confidence(self, math_engine):
        """只有关键词依据的结果必须落在"低置信、建议复核"区间，不能装作很确定。"""
        r = math_engine.classify(q_of(
            config.SUBJECT_MATH, config.QTYPE_FORMULA,
            r"求 $f(x)=\frac{x^{2}}{x-1}$ 的水平渐近线与铅直渐近线。"))
        dr = r.dims[config.DIM_KNOWLEDGE]
        assert 0 < dr.score < config.LOW_CONFIDENCE_THRESHOLD
        assert dr.low_confidence

    def test_no_fallback_when_only_recall_padding(self):
        """粗筛全是兜底补进来的候选时，不该硬凑一个答案出来。"""
        sc = Scheme(name="t", subject=config.SUBJECT_MATH, dims=[
            Dimension(key=config.DIM_KNOWLEDGE, name="知识点", labels=[
                CategoryLabel(id="A", name="甲", keywords=["紫罗兰"]),
                CategoryLabel(id="B", name="乙", keywords=["向日葵"]),
                CategoryLabel(id="C", name="丙", keywords=["康乃馨"]),
                CategoryLabel(id="D", name="丁", keywords=["满天星"]),
                CategoryLabel(id="E", name="戊", keywords=["风信子"]),
                CategoryLabel(id="F", name="己", keywords=["郁金香"])])])
        eng = ClassificationEngine(sc)
        dr = eng.classify(q_of(config.SUBJECT_MATH, config.QTYPE_TEXT,
                               "量子色动力学的渐近自由性")).dims[config.DIM_KNOWLEDGE]
        assert dr.score == 0.0 and not dr.label_name
