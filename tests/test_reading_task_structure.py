import pytest

from app import config
from app.core.classify.engine import ClassificationEngine
from app.core.models import Question
from app.core.preprocess import pipeline
from app.core.preprocess.text_en import reading_task_tokens


PASSAGE = ("The town built a new public library last year. Students use its quiet rooms to study after class. "
           "The building also contains a collection of historical photographs and records of local families. "
           "During weekends, visitors can join a guided tour and learn how the railway changed the town. "
           "Admission to these tours is free for local residents. ")


def test_passage_plus_actual_question_contributes_reading_task_without_changing_input(repo):
    text = PASSAGE + "\n\nQuestion: What collection can visitors find in the library?"
    question = Question(subject=config.SUBJECT_ENGLISH, qtype=config.QTYPE_TEXT, raw_text=text)
    feature = pipeline.preprocess(question)
    assert "阅读理解" in feature.qtype_tokens
    engine = ClassificationEngine(repo.scheme_by_subject(config.SUBJECT_ENGLISH))
    assert engine.classify(question, feature).dims[config.DIM_KNOWLEDGE].label_name == "阅读理解"
    assert question.raw_text == text


@pytest.mark.parametrize("text", [PASSAGE, "Question: Which tense is used in this sentence?",
    "Translate the following paragraph into Chinese.\n" + PASSAGE,
    "Write an essay about the town.\n" + PASSAGE,
    "Identify the tense of each sentence.\n" + PASSAGE])
def test_prose_or_other_explicit_task_is_not_inferred_as_reading(text):
    assert reading_task_tokens(text) == []


def test_cloze_and_reading_instructions_are_distinguished():
    assert reading_task_tokens("Cloze test: choose the correct word for each blank.") == ["完形填空"]
    assert reading_task_tokens("Read the following passage and answer the questions.") == ["阅读理解"]


def test_reading_qtype_does_not_automatically_inject_cloze():
    feature = pipeline.preprocess(Question(subject=config.SUBJECT_ENGLISH,
                                          qtype=config.QTYPE_READING, raw_text=PASSAGE))
    assert "完形填空" not in feature.qtype_tokens
