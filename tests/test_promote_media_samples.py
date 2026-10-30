from app import config
from app.core.classify.engine import ClassificationEngine
from app.core.classify.feedback import FeedbackOptimizer
from app.core.models import CategoryLabel, Dimension, MediaReviewRecord, Question, Scheme
from app.core.preprocess import pipeline


def test_confirmed_formula_and_audio_can_be_promoted_without_typed_body():
    label = CategoryLabel(id="target", name="积分", dim="course_topic")
    scheme = Scheme(subject=config.SUBJECT_MATH,
                    dims=[Dimension(key="course_topic", labels=[label])])
    optimizer = FeedbackOptimizer(ClassificationEngine(scheme))
    formula = Question(subject=config.SUBJECT_MATH, qtype=config.QTYPE_FORMULA,
                       labels={"course_topic": label.id},
                       formula_ocr=MediaReviewRecord(text="wrong", draft="new draft",
                           applied_text=r"\int_0^1 x dx", confirmed=True))
    assert optimizer.promote_sample(formula, "course_topic")
    assert label.samples == [r"$\int_0^1 x dx$"]
    fb = pipeline.preprocess(Question(subject=config.SUBJECT_MATH, raw_text=label.samples[0]))
    assert "积分" in fb.formula_tokens or "定积分" in fb.formula_tokens
    assert not optimizer.promote_sample(formula, "course_topic")
    audio = Question(labels={"course_topic": label.id},
        audio_transcript=MediaReviewRecord(text="draft text", applied_text="Do not forget the lecture.", confirmed=True))
    assert optimizer.promote_sample(audio, "course_topic")
    assert label.samples[-1] == "Do not forget the lecture."


def test_unconfirmed_recognition_never_becomes_standard_sample():
    label = CategoryLabel(id="target", dim=config.DIM_KNOWLEDGE)
    scheme = Scheme(dims=[Dimension(labels=[label])])
    question = Question(labels={config.DIM_KNOWLEDGE: label.id},
        formula_ocr=MediaReviewRecord(text="unreviewed", draft="unreviewed"),
        audio_transcript=MediaReviewRecord(text="unreviewed", draft="unreviewed"))
    assert not FeedbackOptimizer(ClassificationEngine(scheme)).promote_sample(question, config.DIM_KNOWLEDGE)
    assert not label.samples


def test_error_sample_uses_actual_annotation_and_priority_is_not_trained(math_engine):
    optimizer = FeedbackOptimizer(math_engine)
    error_label = math_engine.scheme.dim(config.DIM_ERROR).labels[0]
    priority_label = math_engine.scheme.dim(config.DIM_PRIORITY).labels[0]
    question = Question(raw_text="求定积分", labels={config.DIM_ERROR: error_label.id,
        config.DIM_PRIORITY: priority_label.id})
    assert not optimizer.promote_sample(question, config.DIM_ERROR)
    question.annotation = "实际是漏看积分上限，审题失误"
    assert optimizer.promote_sample(question, config.DIM_ERROR)
    assert error_label.samples[-1] == question.annotation
    assert not optimizer.promote_sample(question, config.DIM_PRIORITY)
