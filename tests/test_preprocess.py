# -*- coding: utf-8 -*-
"""多模态预处理与特征提取模块测试（对应需求 4.4.1 第 1 点）。"""
from __future__ import annotations

import pytest

from app import config
from app.core.models import Question
from app.core.preprocess import cleaner, lexicon, pipeline
from app.core.preprocess.formula import get_mapper
from app.core.preprocess.text_cn import tokenize_cn
from app.core.preprocess.text_en import tokenize_en


class TestCleaner:
    """文本清洗（4.2.1（1）第 1 步）。"""

    @pytest.mark.parametrize("raw,gone", [
        ("3. 求极限", "3."),
        ("（12）计算定积分", "（12）"),
        ("第 5 题 判断敛散性", "第 5 题"),
        ("[7] 求导数", "[7]"),
        ("一、选择题", "一、"),
    ])
    def test_strip_question_number(self, raw, gone):
        assert gone not in cleaner.clean_text(raw)

    def test_strip_markup_and_junk(self):
        out = cleaner.clean_text("**求极限**  ★★ <br>用洛必达法则")
        assert "**" not in out and "★" not in out and "<br>" not in out
        assert "求极限" in out and "洛必达法则" in out

    def test_extract_latex(self):
        text, fs = cleaner.extract_latex(r"求 $\lim_{x\to0}\frac{\sin x}{x}$ 与 \(a^2+b^2\)")
        assert len(fs) == 2
        assert r"\lim" in fs[0] and "a^2" in fs[1]
        assert "\\lim" not in text     # 公式已被剥离，不会污染中文分词

    def test_looks_like_latex(self):
        assert cleaner.looks_like_latex(r"$\int_0^1 x dx$")
        assert cleaner.looks_like_latex(r"\sum_{n=1}^{\infty}")
        assert not cleaner.looks_like_latex("求这个函数的极限")

    def test_strip_option_bodies(self):
        out = cleaner.strip_option_bodies("The scientists have ___ it. A. invented B. found")
        assert "invented" not in out and "scientists" in out

    def test_guess_qtype(self):
        assert cleaner.guess_qtype(r"$\int x dx$") == config.QTYPE_FORMULA
        assert cleaner.guess_qtype("", "a.mp3") == config.QTYPE_LISTENING
        assert cleaner.guess_qtype("", "受力图.png", config.SUBJECT_PHYSICS) == config.QTYPE_FIGURE
        assert cleaner.guess_qtype("普通题干") == config.QTYPE_TEXT

    def test_guess_subject(self):
        assert cleaner.guess_subject("用洛必达法则求极限") == config.SUBJECT_MATH
        assert cleaner.guess_subject("由动量守恒定律求碰后速度") == config.SUBJECT_PHYSICS
        assert cleaner.guess_subject("求矩阵的特征值与特征向量") == config.SUBJECT_LINALG


class TestChineseTokenize:
    """jieba 分词 + 双层停用词 + 核心实词提取（4.2.1（1）第 2-4 步）。"""

    def test_proper_nouns_not_split(self):
        """专有名词词典必须生效，否则"洛必达法则"会被切碎，关键词永远匹配不上。"""
        for w in ("洛必达法则", "动量守恒定律", "牛顿莱布尼茨公式", "格林公式"):
            assert w in tokenize_cn("这道题要用{}来做".format(w))

    def test_common_stopwords_filtered(self):
        toks = tokenize_cn("这个题目的答案是在于我们对它的理解")
        for w in ("的", "是", "在", "我们", "它"):
            assert w not in toks

    def test_subject_stopwords_filtered(self):
        toks = tokenize_cn("如图所示，下列说法中正确的是", config.SUBJECT_PHYSICS)
        for w in ("如图", "所示", "下列", "说法", "正确"):
            assert w not in toks

    def test_single_char_dropped(self):
        assert all(len(t) >= 2 or t.isascii() for t in tokenize_cn("求这个函数的极限值"))

    def test_keeps_content_words(self):
        toks = tokenize_cn("计算定积分并求旋转体体积")
        assert "定积分" in toks and "旋转体体积" in toks

    def test_empty_input(self):
        assert tokenize_cn("") == [] and tokenize_cn("   ") == []


class TestEnglishTokenize:
    """英文分词 + 词形还原 + 英文停用词（4.2.1（2）英语类）。"""

    def test_stopwords_filtered(self):
        toks = tokenize_en("This is the question that we are going to answer")
        for w in ("this", "is", "the", "that", "we", "are", "question", "answer"):
            assert w not in toks

    def test_lemmatize(self):
        toks = tokenize_en("scientists developed studies running")
        assert "scientist" in toks
        assert "develop" in toks or "developed" not in toks
        assert "study" in toks or "studies" not in toks

    def test_short_words_dropped(self):
        assert all(len(t) >= 3 for t in tokenize_en("a an of go do it be"))


class TestFormulaMapper:
    """公式-知识点映射词典（核心创新点）。"""

    @pytest.mark.parametrize("latex,expect", [
        (r"\lim_{x \to 0}\frac{\sin x}{x}", "极限"),
        (r"\int_{0}^{1} x\,\mathrm{d}x", "定积分"),
        (r"\int x^2 dx", "不定积分"),
        (r"\iint_D f(x,y)\,dxdy", "二重积分"),
        (r"\iiint_\Omega z\,dV", "三重积分"),
        (r"\oint_L P dx + Q dy", "曲线积分"),
        (r"\sum_{n=1}^{\infty} a_n", "无穷级数"),
        (r"\frac{\partial z}{\partial x}", "偏导数"),
        (r"\nabla f", "梯度"),
    ])
    def test_math_rules(self, latex, expect):
        toks, _hits = get_mapper().map_latex(latex, config.SUBJECT_MATH)
        assert expect in toks

    def test_definite_vs_indefinite(self):
        """带上下限判为定积分、不带判为不定积分——这是规则顺序设计的关键。"""
        d, _ = get_mapper().map_latex(r"\int_{0}^{1} x dx", config.SUBJECT_MATH)
        i, _ = get_mapper().map_latex(r"\int x dx", config.SUBJECT_MATH)
        assert "定积分" in d
        assert "不定积分" in i and "定积分" not in i

    @pytest.mark.parametrize("latex,expect", [
        (r"E = mc^{2}", "质能方程"),
        (r"pV = \nu RT", "理想气体状态方程"),
        (r"\lambda = \frac{h}{p}", "德布罗意波"),
        (r"\oint \vec{B} \cdot d\vec{l}", "安培环路定理"),
    ])
    def test_physics_rules(self, latex, expect):
        toks, _ = get_mapper().map_latex(latex, config.SUBJECT_PHYSICS)
        assert expect in toks

    def test_subject_isolation(self):
        """学科隔离：高数规则不应污染物理题的特征。"""
        toks, _ = get_mapper().map_latex(r"E = mc^2", config.SUBJECT_MATH)
        assert "质能方程" not in toks

    def test_weight_repeats_tokens(self):
        """weight 越大，特征词注入次数越多，从而在 TF-IDF 里拿到更高词频。"""
        toks, _ = get_mapper().map_latex(r"\lim_{x \to 0} x", config.SUBJECT_MATH)
        assert toks.count("极限") >= 2

    def test_empty_and_garbage(self):
        assert get_mapper().map_latex("", config.SUBJECT_MATH) == ([], [])
        toks, _ = get_mapper().map_latex("hello world 123", config.SUBJECT_MATH)
        assert toks == []

    def test_rules_loaded(self):
        assert get_mapper().rule_count() > 50


class TestPipeline:
    """预处理调度与融合。"""

    def test_formula_question_fuses_both(self):
        q = Question(subject=config.SUBJECT_MATH, qtype=config.QTYPE_FORMULA,
                     raw_text=r"求极限 $\lim_{x\to0}\frac{\sin x - x}{x^3}$，用洛必达法则")
        fb = pipeline.preprocess(q)
        assert "洛必达法则" in fb.text_tokens        # 文本通道
        assert "极限" in fb.formula_tokens           # 公式通道
        assert fb.latex and fb.qtype_tokens and fb.subject_tokens

    def test_annotation_participates(self):
        q = Question(subject=config.SUBJECT_PHYSICS, qtype=config.QTYPE_FIGURE,
                     annotation="斜面受力分析图，摩擦因数已知，漏画了正压力分量")
        fb = pipeline.preprocess(q)
        assert any("受力" in t or "摩擦" in t for t in fb.text_tokens)

    def test_missing_media_degrades_not_crash(self):
        """坏文件不能让整批分类崩掉。"""
        q = Question(subject=config.SUBJECT_PHYSICS, qtype=config.QTYPE_FIGURE,
                     raw_text="图形题", media_path="/nonexistent/xx.png")
        fb = pipeline.preprocess(q)
        assert any("不存在" in n for n in fb.notes)

    def test_empty_question_warns(self):
        fb = pipeline.preprocess(Question(raw_text="", annotation=""))
        assert fb.is_empty() and fb.notes

    def test_pluggable_units_registered(self):
        from app.core.preprocess import base
        names = {u.name for u in base.all_units()}
        assert {"audio", "figure", "formula", "text_cn", "text_en"} <= names

    def test_engine_report_never_raises(self):
        """引擎未安装时必须优雅降级并给出可读说明，而不是抛异常。"""
        for st in pipeline.engine_report():
            assert isinstance(st.available, bool) and st.detail


class TestFeatureMetrics:
    """需求 4.4.1 第 1 点的四项特征提取指标验收。

    这几条测试锁死的是 ``python -m tools.eval_features`` 的结论。
    关于这些数字的偏差来源（闭卷自测、①的循环性），见该模块的 docstring
    与 README「已知局限」——测试只保证不回归，不代表泛化能力。
    """

    def test_tokenize_error_rate(self):
        from tools.eval_features import eval_tokenize_error_rate
        rate, total, _fails = eval_tokenize_error_rate()
        assert total >= 100
        assert rate <= 0.03, "分词错误率 {:.1%} 超过需求上限 3%".format(rate)

    def test_formula_feature_accuracy(self):
        from tools.eval_features import eval_formula
        acc, n, _f = eval_formula()
        assert n >= 30
        assert acc >= 0.90, "公式特征提取准确率 {:.1%} 低于需求 90%".format(acc)

    def test_text_core_feature_accuracy(self):
        from tools.eval_features import eval_core_features
        acc, n, _f = eval_core_features("text", 0.95)
        assert n >= 25
        assert acc >= 0.95, "纯文本核心特征提取准确率 {:.1%} 低于需求 95%".format(acc)

    def test_figure_core_feature_accuracy(self):
        from tools.eval_features import eval_core_features
        acc, _n, _f = eval_core_features("figure", 0.90)
        assert acc >= 0.90, "图形题核心特征提取准确率 {:.1%} 低于需求 90%".format(acc)

    def test_english_core_feature_accuracy(self):
        from tools.eval_features import eval_core_features
        acc, _n, _f = eval_core_features("english", 0.90)
        assert acc >= 0.90, "英语题核心特征提取准确率 {:.1%} 低于需求 90%".format(acc)

    def test_no_garbage_fragments(self):
        """回归锁：分词不得产出"拉格朗""法求""散性"这类无意义碎片。

        它们曾经真实出现过——专有名词词典缺词导致 jieba 把术语切碎。
        界面会把特征词当作分类依据展示给用户，这种碎片会让人以为系统坏了，
        直接损害"可解释"这个核心卖点。
        """
        cases = [
            ("用拉格朗日乘数法求多元函数在给定约束条件下的条件极值。", config.SUBJECT_MATH,
             ["拉格朗", "法求"]),
            ("用比值判别法判断该正项级数的敛散性。", config.SUBJECT_MATH, ["散性"]),
            ("用凑微分法求该有理函数的原函数。", config.SUBJECT_MATH, ["法求"]),
        ]
        for text, subject, garbage in cases:
            toks = tokenize_cn(text, subject)
            for g in garbage:
                assert g not in toks, "分词产出了无意义碎片 {!r}：{}".format(g, "/".join(toks))
