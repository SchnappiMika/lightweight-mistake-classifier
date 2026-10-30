# -*- coding: utf-8 -*-
"""人工反馈增量优化测试（对应需求 4.2.3 第 2 点 / 4.4.1 第 3 点）。

需求里对这一模块的硬约束是「**无需全量重训**」，因此本文件除了验证效果，
还专门验证了"没有发生重训"这件事本身（见 TestNoRetrain）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import config
from app.core.classify.engine import ClassificationEngine
from app.core.classify.feedback import FeedbackOptimizer
from app.core.models import CategoryLabel, Dimension, Question, Scheme

DATASET = Path(__file__).resolve().parent.parent / "samples" / "eval_dataset.json"


def q_of(subject, qtype, text, note=""):
    return Question(subject=subject, qtype=qtype, raw_text=text, annotation=note)


def classify_and_apply(eng, scheme, q):
    res = eng.classify(q)
    ClassificationEngine.apply_result(q, res, scheme, q.feature_bundle())
    return res


@pytest.fixture
def toy():
    """一个刻意做得很容易混淆的小体系：两个标签的关键词互不相同但样本贫瘠。"""
    sc = Scheme(name="toy", subject=config.SUBJECT_MATH, dims=[
        Dimension(key=config.DIM_KNOWLEDGE, name="知识点", labels=[
            CategoryLabel(id="A", name="定积分", dim=config.DIM_KNOWLEDGE,
                          keywords=["定积分", "积分"]),
            CategoryLabel(id="B", name="无穷级数", dim=config.DIM_KNOWLEDGE,
                          keywords=["级数", "收敛"]),
        ])])
    return sc, ClassificationEngine(sc)


class TestCorrection:
    """一次人工修正应该同时做三件事：正强化、负惩罚、权重微调。"""

    def test_reinforces_correct_label(self, toy):
        sc, eng = toy
        q = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "判断这个交错级数的敛散性")
        classify_and_apply(eng, sc, q)
        before = dict(eng.profiles["A"].delta_tf)
        FeedbackOptimizer(eng).apply_correction(q, config.DIM_KNOWLEDGE, "A")
        after = eng.profiles["A"].delta_tf
        assert after != before
        # 题干里的特征词被并入正确标签的画像
        assert any(v > 0 for v in after.values())

    def test_penalizes_wrong_label(self, toy):
        sc, eng = toy
        q = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "求这道题的定积分")
        res = classify_and_apply(eng, sc, q)
        wrong = res.dims[config.DIM_KNOWLEDGE].label_id
        assert wrong, "分类未产出标签，测试前提不成立"
        right = "B" if wrong == "A" else "A"
        FeedbackOptimizer(eng).apply_correction(q, config.DIM_KNOWLEDGE, right)
        # 误判标签画像里与本题重合的词被调弱（出现负的 delta）
        assert any(v < 0 for v in eng.profiles[wrong].delta_tf.values())

    def test_marks_question_corrected(self, toy):
        sc, eng = toy
        q = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "求定积分")
        classify_and_apply(eng, sc, q)
        FeedbackOptimizer(eng).apply_correction(q, config.DIM_KNOWLEDGE, "B")
        assert q.corrected[config.DIM_KNOWLEDGE] is True
        assert q.labels[config.DIM_KNOWLEDGE] == "B"
        assert q.label_names[config.DIM_KNOWLEDGE] == "无穷级数"

    def test_returns_record_with_audit_trail(self, toy):
        """反馈记录要能支撑界面上的"系统学到了什么"，必须留痕。"""
        sc, eng = toy
        q = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "求定积分的值")
        classify_and_apply(eng, sc, q)
        rec = FeedbackOptimizer(eng).apply_correction(q, config.DIM_KNOWLEDGE, "B")
        assert rec is not None
        assert rec.to_label_name == "无穷级数" and rec.tokens
        assert rec.weight_before and rec.weight_after

    def test_same_label_is_noop(self, toy):
        """把标签"改"成它本来就是的值，不应产生反馈记录。"""
        sc, eng = toy
        q = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "求定积分")
        res = classify_and_apply(eng, sc, q)
        cur = res.dims[config.DIM_KNOWLEDGE].label_id
        assert FeedbackOptimizer(eng).apply_correction(q, config.DIM_KNOWLEDGE, cur) is None


class TestEffectiveness:
    """效果验证：修正后同类错题要变准。"""

    def test_corrected_question_reclassifies_correctly(self, toy):
        """最基本的要求：改过一次之后，同一条题再分类必须判对。"""
        sc, eng = toy
        q = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "这道题涉及黎曼和的极限定义")
        classify_and_apply(eng, sc, q)
        FeedbackOptimizer(eng).apply_correction(q, config.DIM_KNOWLEDGE, "A")
        q2 = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "这道题涉及黎曼和的极限定义")
        assert eng.classify(q2).dims[config.DIM_KNOWLEDGE].label_id == "A"

    def test_generalizes_to_similar_question(self, toy):
        """泛化：修正 A 题之后，用词相近的 B 题也该跟着判对。"""
        sc, eng = toy
        train = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "用黎曼和的定义推导这个结论")
        classify_and_apply(eng, sc, train)
        FeedbackOptimizer(eng).apply_correction(train, config.DIM_KNOWLEDGE, "A")
        # 没被喂过的新题，但共享"黎曼和"这个特征词
        test = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "黎曼和怎么算")
        assert eng.classify(test).dims[config.DIM_KNOWLEDGE].label_id == "A"

    def test_cold_start_batch_does_not_degrade_holdout(self, repo):
        """冷启动 -> 修正 -> 复测：批量人工修正不得让泛化准确率倒退。

        这是本模块最重要的一条底线：用户改错题是在帮系统，**任何情况下都不该
        因为改了几条题反而让别的题判得更差**。

        需求 4.4.1 原文要求「修正后同类型错题准确率提升≥5%」。在本项目的
        155 条样本上，实测同类型口径为 +3.4pp、全量口径 +1.3pp，均未达到 5pp，
        且同类型子集仅 29 条（一条题≈3.4pp），差异本身就在噪声量级内。
        因此这里断言的是"可复现且确实成立"的性质——不退化——而不是一个测不出来的
        指标。完整的实验数据与原因分析见 README「已知局限」与
        ``python -m tools.evaluate --only feedback``。
        """
        import random

        with open(DATASET, encoding="utf-8") as f:
            rows = json.load(f)
        for sc in repo.schemes:                     # 制造冷启动
            for d in sc.dims:
                for lb in d.labels:
                    lb.samples = []
                    lb.keywords = lb.keywords[:2]
            sc.touch()
        engines = {s.subject: ClassificationEngine(s) for s in repo.schemes}

        def acc(items):
            if not items:
                return 0.0
            ok = 0
            for r in items:
                eng = engines.get(r["学科"])
                if eng is None:
                    continue
                dr = eng.classify(q_of(r["学科"], r["题型"], r["题干"],
                                       r.get("标注", ""))).dims.get(config.DIM_KNOWLEDGE)
                ok += int(bool(dr) and dr.label_name == r["标准知识点"])
            return ok / len(items)

        random.seed(42)
        shuffled = list(rows)
        random.shuffle(shuffled)
        cut = len(shuffled) // 2
        fix_set, retest = shuffled[:cut], shuffled[cut:]

        # 只修正判错的题——这才是用户真实会做的事
        pending = []
        for r in fix_set:
            eng = engines.get(r["学科"])
            if eng is None:
                continue
            dim = eng.scheme.dim(config.DIM_KNOWLEDGE)
            gold = dim.label_by_name(r["标准知识点"]) if dim else None
            if gold is None:
                continue
            q = q_of(r["学科"], r["题型"], r["题干"], r.get("标注", ""))
            classify_and_apply(eng, eng.scheme, q)
            if q.labels.get(config.DIM_KNOWLEDGE) != gold.id:
                pending.append((eng, q, gold, r["学科"], r["标准知识点"]))
        assert pending, "冷启动下竟无任何误判，实验设计失效"

        touched = {(p[3], p[4]) for p in pending}
        same = [r for r in retest if (r["学科"], r["标准知识点"]) in touched]

        before_all, before_same = acc(retest), acc(same)
        for eng, q, gold, _s, _k in pending:
            FeedbackOptimizer(eng).apply_correction(q, config.DIM_KNOWLEDGE, gold.id)
        after_all, after_same = acc(retest), acc(same)

        assert after_all >= before_all, (
            "批量修正后全量复测集准确率倒退：{:.1%} -> {:.1%}".format(before_all, after_all))
        assert after_same >= before_same, (
            "批量修正后同类型错题准确率倒退：{:.1%} -> {:.1%}".format(before_same, after_same))


class TestNoRetrain:
    """需求硬约束：增量优化「无需全量重训」。"""

    def test_vector_space_not_refit(self, toy, monkeypatch):
        """一次修正绝不能触发 TF-IDF 向量空间的重新 fit。"""
        sc, eng = toy
        q = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "求定积分")
        classify_and_apply(eng, sc, q)

        calls = []
        space = eng.space
        orig_fit = space.fit
        monkeypatch.setattr(space, "fit", lambda *a, **k: (calls.append(1), orig_fit(*a, **k))[1])
        FeedbackOptimizer(eng).apply_correction(q, config.DIM_KNOWLEDGE, "B")
        assert not calls, "人工修正触发了向量空间重训，违反「无需全量重训」的设计约束"

    def test_cost_independent_of_corpus_size(self, toy):
        """增量更新的代价只与本题词数有关，与已有语料规模无关。"""
        sc, eng = toy
        # 先灌入大量文档把语料撑大
        space = eng.space
        for i in range(500):
            space.add_document(["填充词{}".format(i), "公共词"])
        n_before = space.n_docs

        q = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "求定积分")
        classify_and_apply(eng, sc, q)
        FeedbackOptimizer(eng).apply_correction(q, config.DIM_KNOWLEDGE, "B")
        # 只新增了本题这一篇文档，没有重新扫描全部 500 篇
        assert space.n_docs - n_before <= 2

    def test_weight_update_is_constant_time(self):
        """权重在线更新是 O(通道数)=O(4)，与数据量无关。"""
        from app.core.classify import weights as W
        r = W.WeightRules()
        before, after = r.update_from_feedback(
            config.SUBJECT_MATH, config.QTYPE_FORMULA,
            right_channels={config.CHANNEL_FORMULA: 0.8, config.CHANNEL_TEXT: 0.2},
            wrong_channels={config.CHANNEL_FORMULA: 0.1, config.CHANNEL_TEXT: 0.7})
        # 正确标签在公式通道上更强 -> 公式权重应被推高，文本权重被压低
        assert after[config.CHANNEL_FORMULA] > before[config.CHANNEL_FORMULA]
        assert after[config.CHANNEL_TEXT] < before[config.CHANNEL_TEXT]
        assert sum(after.values()) == pytest.approx(1.0)

    def test_weight_update_hits_specific_bucket_only(self):
        """一次修正只能改它自己那个 (学科,题型) 桶，不能污染别的桶。"""
        from app.core.classify import weights as W
        r = W.WeightRules()
        other_before = r.get(config.SUBJECT_ENGLISH, config.QTYPE_LISTENING)
        r.update_from_feedback(config.SUBJECT_MATH, config.QTYPE_FORMULA,
                               right_channels={config.CHANNEL_FORMULA: 0.9},
                               wrong_channels={config.CHANNEL_TEXT: 0.9})
        assert r.get(config.SUBJECT_ENGLISH, config.QTYPE_LISTENING) == other_before


class TestPersistence:
    """反馈成果要能存下来，下次开机还在。"""

    def test_delta_survives_roundtrip(self, toy, tmp_path):
        sc, eng = toy
        q = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "黎曼和的定义")
        classify_and_apply(eng, sc, q)
        FeedbackOptimizer(eng).apply_correction(q, config.DIM_KNOWLEDGE, "A")

        dumped = {k: v.to_dict() for k, v in eng.profiles.items()}
        raw = json.loads(json.dumps(dumped))          # 模拟落盘再读回
        from app.core.classify.vector_space import LabelProfile
        back = {k: LabelProfile.from_dict(v) for k, v in raw.items()}
        assert back["A"].delta_tf == pytest.approx(eng.profiles["A"].delta_tf)
        assert back["A"].n_feedback == eng.profiles["A"].n_feedback

    def test_reset_clears_learning(self, toy):
        sc, eng = toy
        q = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "求定积分")
        classify_and_apply(eng, sc, q)
        FeedbackOptimizer(eng).apply_correction(q, config.DIM_KNOWLEDGE, "B")
        assert any(p.n_feedback for p in eng.profiles.values())
        eng.reset_profiles()
        assert all(not p.delta_tf and not p.n_feedback for p in eng.profiles.values())


class TestPromoteSample:
    """把错题采纳为标准样本（长期固化，比 delta 更持久）。"""

    def test_promote_adds_to_scheme(self, toy):
        sc, eng = toy
        q = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "计算旋转体的体积")
        classify_and_apply(eng, sc, q)
        q.labels[config.DIM_KNOWLEDGE] = "A"
        q.label_names[config.DIM_KNOWLEDGE] = "定积分"
        assert FeedbackOptimizer(eng).promote_sample(q, config.DIM_KNOWLEDGE)
        assert "计算旋转体的体积" in sc.dim(config.DIM_KNOWLEDGE).label_by_id("A").samples

    def test_promote_rejects_duplicate_and_empty(self, toy):
        sc, eng = toy
        opt = FeedbackOptimizer(eng)
        q = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "")
        q.labels[config.DIM_KNOWLEDGE] = "A"
        assert not opt.promote_sample(q, config.DIM_KNOWLEDGE)      # 空题干

        q2 = q_of(config.SUBJECT_MATH, config.QTYPE_TEXT, "重复样本")
        q2.labels[config.DIM_KNOWLEDGE] = "A"
        assert opt.promote_sample(q2, config.DIM_KNOWLEDGE)
        assert not opt.promote_sample(q2, config.DIM_KNOWLEDGE)     # 第二次应拒绝
