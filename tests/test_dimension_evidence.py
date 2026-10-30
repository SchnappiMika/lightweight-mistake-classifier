"""The same question can have different actual mistakes and review priorities."""
from app import config
from app.core.classify.engine import ClassificationEngine
from app.core.classify.feedback import FeedbackOptimizer
from app.core.models import Question


def test_question_topic_is_not_evidence_of_student_error(math_engine):
    question = Question(subject=config.SUBJECT_MATH,
                        raw_text="计算定积分，注意化简和单位量纲")
    result = math_engine.classify(question)
    assert result.dims[config.DIM_KNOWLEDGE].label_id
    assert not result.dims[config.DIM_ERROR].label_id
    assert not result.dims[config.DIM_DIFFICULTY].label_id
    assert not result.dims[config.DIM_PRIORITY].label_id


def test_annotation_determines_error_independently_of_problem_topic(math_engine):
    a = Question(subject=config.SUBJECT_MATH, raw_text="计算定积分并化简",
                 annotation="我看错了积分范围，漏看条件，属于审题失误")
    b = Question(subject=config.SUBJECT_MATH, raw_text=a.raw_text,
                 annotation="公式的系数记错了，写错公式")
    ra, rb = math_engine.classify(a), math_engine.classify(b)
    assert ra.dims[config.DIM_ERROR].label_name == "审题失误"
    assert rb.dims[config.DIM_ERROR].label_name == "公式记错"


def test_priority_uses_user_choice_and_does_not_learn_topic_correlation(math_engine):
    question = Question(subject=config.SUBJECT_MATH, raw_text="求极限", priority="高")
    result = math_engine.classify(question)
    ClassificationEngine.apply_result(question, result, math_engine.scheme)
    assert question.label_names[config.DIM_PRIORITY] == "高"
    assert question.corrected[config.DIM_PRIORITY]
    dim = math_engine.scheme.dim(config.DIM_PRIORITY)
    target = dim.label_by_name("低")
    before = dict(math_engine.profiles[target.id].delta_tf)
    assert FeedbackOptimizer(math_engine).apply_correction(question, config.DIM_PRIORITY, target.id) is None
    assert question.priority == "低"
    assert math_engine.profiles[target.id].delta_tf == before


def test_error_feedback_does_not_teach_problem_topic_as_mistake(math_engine):
    question = Question(subject=config.SUBJECT_MATH, raw_text="定积分计算化简与通分")
    ClassificationEngine.apply_result(question, math_engine.classify(question), math_engine.scheme)
    target = math_engine.scheme.dim(config.DIM_ERROR).label_by_name("公式记错")
    before = dict(math_engine.profiles[target.id].delta_tf)
    assert FeedbackOptimizer(math_engine).apply_correction(question, config.DIM_ERROR, target.id) is None
    assert question.corrected[config.DIM_ERROR]
    assert math_engine.profiles[target.id].delta_tf == before


def test_manual_error_stays_confirmed_when_reclassifying(math_engine):
    question = Question(subject=config.SUBJECT_MATH, raw_text="求定积分")
    ClassificationEngine.apply_result(question, math_engine.classify(question), math_engine.scheme)
    target = math_engine.scheme.dim(config.DIM_ERROR).label_by_name("审题失误")
    FeedbackOptimizer(math_engine).apply_correction(question, config.DIM_ERROR, target.id)
    ClassificationEngine.apply_result(question, math_engine.classify(question), math_engine.scheme)
    assert question.labels[config.DIM_ERROR] == target.id
    assert question.corrected[config.DIM_ERROR]
