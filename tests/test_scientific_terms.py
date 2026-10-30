import json

import pytest

from app import config
from app.core.preprocess.scientific_terms import canonical_terms


@pytest.mark.parametrize("subject,text,expected", [
    (config.SUBJECT_MATH, "Differentiate to obtain the second derivative.", ["求导", "二阶导数"]),
    (config.SUBJECT_MATH, "Evaluate the definite integral and the indefinite integral.", ["定积分", "不定积分"]),
    (config.SUBJECT_MATH, "The partial derivatives give a directional derivative.", ["偏导数", "方向导数"]),
    (config.SUBJECT_MATH, "The power series converges absolutely inside its radius of convergence.",
     ["幂级数", "绝对收敛", "收敛半径"]),
    (config.SUBJECT_MATH, "An ordinary differential equation has a general solution.", ["微分方程", "通解"]),
    (config.SUBJECT_MATH, "Use Cauchy's mean value theorem, then integration by parts.",
     ["柯西中值定理", "分部积分法"]),
    (config.SUBJECT_MATH, "The divergence theorem concerns the divergence of a vector field.",
     ["高斯公式", "散度"]),
    (config.SUBJECT_PHYSICS, "The angular velocity and tangential acceleration vary.", ["角速度", "切向加速度"]),
    (config.SUBJECT_PHYSICS, "Newton's second law relates forces to acceleration.", ["牛顿第二定律", "力", "加速度"]),
    (config.SUBJECT_PHYSICS, "The angular momentum and gravitational potential energy are known.",
     ["角动量", "引力势能"]),
    (config.SUBJECT_PHYSICS, "A perfectly inelastic collision conserves momentum.", ["完全非弹性碰撞", "动量"]),
    (config.SUBJECT_PHYSICS, "Calculate the power of the electric circuit at this voltage.", ["功率", "电路", "电势差"]),
    (config.SUBJECT_PHYSICS, "Find the work done by a force over a displacement.", ["功", "力", "位移"]),
    (config.SUBJECT_LINALG, "Find the eigenvalues of the inverse matrix.", ["特征值", "逆矩阵"]),
    (config.SUBJECT_LINALG, "Decide whether vectors are linearly independent.", ["线性无关"]),
    (config.SUBJECT_LINALG, "Apply elementary row operations to the augmented matrix.", ["初等行变换", "增广矩阵"]),
    (config.SUBJECT_PROB, "A standard normal distribution has this probability density function.",
     ["标准正态分布", "概率密度"]),
    (config.SUBJECT_PROB, "Find the conditional probability using Bayes' theorem.", ["条件概率", "贝叶斯公式"]),
    (config.SUBJECT_PROB, "The sample variance and confidence interval are reported.", ["样本方差", "置信区间"]),
])
def test_explicit_scientific_vocabulary_and_longest_phrase(subject, text, expected):
    assert canonical_terms(text, subject) == expected


@pytest.mark.parametrize("text", [
    "homework overpowered workforce powerhouse newtonian acceleratedly",
    "supervelocity powershell _velocity velocity_ velocity123 123velocity",
    "The current topic concerns normal business and a field trip.",
])
def test_substrings_short_ambiguous_words_and_unlisted_inflections_are_not_features(text):
    assert canonical_terms(text, config.SUBJECT_PHYSICS) == []


@pytest.mark.parametrize("text", [
    "We work at a university and discuss political power.",
    "Work out the energy of the particle.",
    "A force is present. We work together and seek power.",
    "Power series approximate the force.",
    "Raise the current to a power of two.",
    "Work and power are ordinary words here.",
])
def test_work_and_power_require_nearby_physical_context_without_known_other_senses(text):
    terms = canonical_terms(text, config.SUBJECT_PHYSICS)
    assert "功" not in terms and "功率" not in terms


def test_physical_context_does_not_leak_across_subjects():
    text = "Work done by a force and electrical power in a circuit."
    assert "功" in canonical_terms(text, config.SUBJECT_PHYSICS)
    assert "功率" in canonical_terms(text, config.SUBJECT_PHYSICS)
    for subject in (config.SUBJECT_MATH, config.SUBJECT_LINALG, config.SUBJECT_PROB,
                    config.SUBJECT_ENGLISH, config.SUBJECT_GENERAL, "unknown", ""):
        assert "功" not in canonical_terms(text, subject)
        assert "功率" not in canonical_terms(text, subject)


def test_generic_quantities_do_not_invent_conservation_laws_or_theorems():
    terms = canonical_terms("Newton measures momentum, kinetic energy, gravity, and electric circuits.",
                            config.SUBJECT_PHYSICS)
    assert terms == ["牛顿", "动量", "动能", "重力", "电路"]
    assert not set(terms) & {"牛顿第二定律", "动量守恒定律", "动能定理", "机械能守恒"}


def test_whitespace_hyphen_case_curly_apostrophe_and_nfkc_variants():
    assert canonical_terms("SECOND–ORDER DERIVATIVES; definite\nintegrals; ＤＥＲＩＶＡＴＩＶＥ", config.SUBJECT_MATH) == [
        "二阶导数", "定积分", "导数"]
    assert canonical_terms("Newton’s second law and centre-of-mass", config.SUBJECT_PHYSICS) == [
        "牛顿第二定律", "质心"]


def test_punctuation_is_not_an_invented_phrase_join():
    assert canonical_terms("definite. integral", config.SUBJECT_MATH) == ["积分"]
    assert canonical_terms("partial, derivative", config.SUBJECT_MATH) == ["导数"]
    assert canonical_terms("electric; circuit", config.SUBJECT_PHYSICS) == []


def test_unique_terms_keep_source_order_without_mutating_text():
    text = "Derivatives, an integral, derivative again; integration."
    before = text
    assert canonical_terms(text, config.SUBJECT_MATH) == ["导数", "积分"]
    assert text == before


@pytest.mark.parametrize("subject", [config.SUBJECT_MATH, config.SUBJECT_PHYSICS,
                                    config.SUBJECT_LINALG, config.SUBJECT_PROB])
def test_chinese_is_not_rewritten(subject):
    assert canonical_terms("求定积分并计算加速度；已知特征值与条件概率。", subject) == []


@pytest.mark.parametrize("text,subject", [(None, config.SUBJECT_MATH), (1, config.SUBJECT_MATH),
                                          ("derivative", None), ("", config.SUBJECT_MATH)])
def test_empty_or_nontext_values_add_nothing(text, subject):
    assert canonical_terms(text, subject) == []


def test_resource_provenance_and_every_explicit_alias_are_consistent():
    data = json.loads((config.RESOURCE_DIR / "scientific_terms.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["source"]["kind"] == "standard_scientific_terminology_mapping"
    assert set(data["subjects"]) == {config.SUBJECT_MATH, config.SUBJECT_PHYSICS,
                                     config.SUBJECT_LINALG, config.SUBJECT_PROB}
    for subject, entries in data["subjects"].items():
        seen = set()
        for entry in entries:
            assert set(entry) <= {"canonical", "aliases", "context_any", "exclude_phrases"}
            assert entry["canonical"] and entry["aliases"]
            for alias in entry["aliases"]:
                assert alias.casefold() not in seen, (subject, alias)
                seen.add(alias.casefold())
                example = alias + (" by force" if entry.get("context_any") else "")
                assert entry["canonical"] in canonical_terms(example, subject), (subject, alias)
