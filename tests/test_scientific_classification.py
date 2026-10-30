"""Explicit scientific vocabulary must survive the complete matching path."""
from copy import deepcopy

from app import config
from app.core.classify.engine import ClassificationEngine
from app.core.models import CategoryLabel, Dimension, Question, Scheme
from app.core.preprocess import pipeline


def test_english_science_preserves_original_and_matches_chinese_custom_keywords():
    topic = CategoryLabel(name="微积分训练", keywords=["导数", "微分"])
    other = CategoryLabel(name="积分复习", keywords=["定积分", "积分"])
    scheme = Scheme(subject=config.SUBJECT_MATH, dims=[Dimension(labels=[topic, other])])
    question = Question(subject=config.SUBJECT_MATH,
                        raw_text="Calculate the derivative of this polynomial.", annotation="保留人工备注")
    before = deepcopy(question.to_dict())
    feature = pipeline.preprocess(question)
    result = ClassificationEngine(scheme).classify(question, feature)
    assert result.dims[config.DIM_KNOWLEDGE].label_id == topic.id
    assert "导数" in feature.text_tokens and "derivative" in feature.text_tokens
    assert question.to_dict() == before
    assert any("术语规范" in note for note in feature.notes)


def test_formula_standard_sample_uses_formula_mapping_not_bare_latex_words():
    topic = CategoryLabel(name="课程甲", samples=[r"$\int_0^1 t\,dt$"])
    other = CategoryLabel(name="课程乙", samples=[r"$\sum_{n=1}^{\infty} \frac{1}{n^2}$"])
    scheme = Scheme(subject=config.SUBJECT_MATH, dims=[Dimension(labels=[topic, other])])
    engine = ClassificationEngine(scheme)
    question = Question(subject=config.SUBJECT_MATH, raw_text="求定积分所围面积")
    result = engine.classify(question)
    assert "定积分" in engine.profiles[topic.id].base_tf
    assert result.dims[config.DIM_KNOWLEDGE].label_id == topic.id


def test_english_standard_sample_and_chinese_input_share_canonical_terms():
    topic = CategoryLabel(name="主题甲", samples=["Calculate the acceleration and velocity of a particle."])
    other = CategoryLabel(name="主题乙", samples=["Calculate the electric field and electric potential."])
    scheme = Scheme(subject=config.SUBJECT_PHYSICS, dims=[Dimension(labels=[topic, other])])
    engine = ClassificationEngine(scheme)
    result = engine.classify(Question(subject=config.SUBJECT_PHYSICS, raw_text="求质点的速度和加速度"))
    assert "加速度" in engine.profiles[topic.id].base_tf
    assert result.dims[config.DIM_KNOWLEDGE].label_id == topic.id


def test_english_language_course_does_not_gain_science_topic_or_student_error():
    question = Question(subject=config.SUBJECT_ENGLISH,
        raw_text="Translate the sentence about the derivative and kinetic energy.")
    feature = pipeline.preprocess(question)
    assert "导数" not in feature.text_tokens and "动能" not in feature.text_tokens
