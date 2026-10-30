from app import config
from app.core.models import Question
from app.core.preprocess.formula import FormulaMapper
from app.core.preprocess.pipeline import preprocess


def test_derivative_notation_alone_is_not_differential_equation():
    mapper = FormulaMapper()
    for expression in (r"\frac{dx}{dy}", r"y'", r"\frac{d^{2}y}{dx^{2}}"):
        tokens, _ = mapper.map_latex(expression, config.SUBJECT_MATH)
        assert "导数" in tokens or "高阶导数" in tokens
        assert "微分方程" not in tokens


def test_higher_derivative_from_real_ocr_maps_to_derivative_features():
    q = Question(subject=config.SUBJECT_MATH, qtype=config.QTYPE_FORMULA,
                 raw_text=r"\frac{d x}{d y}\qquad\frac{d^{2}y}{d x^{2}}")
    features = preprocess(q)
    assert "导数" in features.formula_tokens
    assert "高阶导数" in features.formula_tokens
    assert "微分方程" not in features.formula_tokens
