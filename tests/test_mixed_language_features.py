"""Language channels must not count English prose twice or learn stopwords."""
from collections import Counter
from copy import deepcopy

import pytest

from app import config
from app.core.classify.engine import ClassificationEngine
from app.core.classify.feedback import FeedbackOptimizer
from app.core.models import CategoryLabel, Dimension, Question, Scheme
from app.core.preprocess import cleaner, pipeline
from app.core.preprocess.text_en import tokenize_en


@pytest.mark.parametrize("subject", [config.SUBJECT_PHYSICS, config.SUBJECT_MATH,
                                     config.SUBJECT_GENERAL])
def test_english_prose_is_filtered_and_counted_once(subject):
    text = "The bridge connects two towns. A train takes minutes. What is the speed of the train?"
    question = Question(subject=subject, raw_text=text)
    before = deepcopy(question.to_dict())
    feature = pipeline.preprocess(question)
    english = [token for token in feature.text_tokens if token.isascii()]
    assert Counter(english) == Counter(tokenize_en(cleaner.clean_text(text, keep_options=False)))
    assert "the" not in english and "is" not in english and "of" not in english
    assert "takes" not in english and "connects" not in english
    assert english.count("train") == 2  # Actual repeated words retain TF.
    assert question.to_dict() == before


@pytest.mark.parametrize("raw,annotation", [
    ("Find velocity", ""),
    ("求速度", "The velocity is not constant"),
    ("求导数 derivative", "保留人工备注"),
])
def test_short_and_mixed_language_inputs_keep_english_features(raw, annotation):
    question = Question(subject=config.SUBJECT_PHYSICS, raw_text=raw, annotation=annotation)
    feature = pipeline.preprocess(question)
    english = [token for token in feature.text_tokens if token.isascii()]
    expected = tokenize_en(cleaner.clean_text(raw + " " + annotation, keep_options=False))
    assert Counter(english) == Counter(expected)
    if "not" in annotation:
        assert "not" in english


def test_inline_formula_commands_do_not_become_english_prose():
    question = Question(subject=config.SUBJECT_MATH, qtype=config.QTYPE_FORMULA,
                        raw_text=r"Compute $\frac{\partial f}{\partial x}$.")
    feature = pipeline.preprocess(question)
    assert "frac" not in feature.text_tokens and "partial" not in feature.text_tokens
    assert "偏导数" in feature.formula_tokens


def test_english_annotation_for_science_error_dimension_uses_same_tokenizer(repo):
    engine = ClassificationEngine(repo.scheme_by_subject(config.SUBJECT_PHYSICS))
    question = Question(subject=config.SUBJECT_PHYSICS, annotation="The signs were not checked")
    feature = engine.matching_features(question, pipeline.preprocess(question), config.DIM_ERROR)
    assert feature.text_tokens == tokenize_en(question.annotation)
    assert "not" in feature.text_tokens and "the" not in feature.text_tokens


def test_feedback_never_learns_stopwords_from_english_science_prose(repo):
    left = CategoryLabel(id="left", name="主题甲", keywords=["甲知识点"])
    right = CategoryLabel(id="right", name="主题乙", keywords=["乙知识点"])
    scheme = Scheme(subject=config.SUBJECT_PHYSICS, dims=[Dimension(labels=[left, right])])
    engine = ClassificationEngine(scheme)
    question = Question(subject=config.SUBJECT_PHYSICS,
                        raw_text="The train connects towns. What is the speed of the train?",
                        labels={config.DIM_KNOWLEDGE: left.id})
    feedback = FeedbackOptimizer(engine).apply_correction(question, config.DIM_KNOWLEDGE, right.id)
    assert feedback is not None
    assert "train" in engine.profiles[right.id].delta_tf
    assert not {"the", "is", "of", "what", "connects"} & set(engine.profiles[right.id].delta_tf)
    assert not {"the", "is", "of", "what"} & set(engine.coarse[config.DIM_KNOWLEDGE].learned[right.id])
