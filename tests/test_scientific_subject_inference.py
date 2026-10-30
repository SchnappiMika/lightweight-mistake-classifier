import pytest

from app import config
from app.core.dataio import importer
from app.core.preprocess.cleaner import guess_subject


@pytest.mark.parametrize("text, expected", [
    ("Calculate the derivative of the polynomial.", config.SUBJECT_MATH),
    ("Determine the electric field produced by this charge.", config.SUBJECT_PHYSICS),
    ("Find the determinant of this matrix.", config.SUBJECT_LINALG),
    ("Calculate the probability density of the random variable.", config.SUBJECT_PROB),
    ("A particle changes its velocity and acceleration during the motion.", config.SUBJECT_PHYSICS),
    ("The current president discussed the power of words and the force of law.", config.SUBJECT_ENGLISH),
    ("Read the following passage and answer the questions about electric fields and acceleration.", config.SUBJECT_ENGLISH),
    ("Translate the following sentence: calculate the derivative of the polynomial.", config.SUBJECT_ENGLISH),
    ("Find the derivative and the electric field.", config.SUBJECT_GENERAL),
    ("求函数极限并计算定积分", config.SUBJECT_MATH),
    ("讨论矩阵行列式", config.SUBJECT_LINALG),
])
def test_conservative_subject_hints(text, expected):
    assert guess_subject(text) == expected


def test_actual_single_and_txt_import_use_science_hints_without_overriding_user(tmp_path):
    text = "Calculate the derivative of this function."
    automatic = importer.import_single(text)
    assert automatic.subject == config.SUBJECT_MATH
    explicit = importer.import_single(text, subject=config.SUBJECT_ENGLISH)
    assert explicit.subject == config.SUBJECT_ENGLISH
    path = tmp_path / "questions.txt"
    path.write_text(text, encoding="utf-8")
    assert importer.import_txt(str(path))[0].subject == config.SUBJECT_MATH
    assert importer.import_txt(str(path), subject=config.SUBJECT_PHYSICS)[0].subject == config.SUBJECT_PHYSICS
