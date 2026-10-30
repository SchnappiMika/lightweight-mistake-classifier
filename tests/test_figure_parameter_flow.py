"""Actual figure preprocessing, persistence and exported file readback."""
import json
from copy import deepcopy

import openpyxl

from app import config
from app.core.classify.engine import ClassificationEngine
from app.core.dataio.exporter import export_excel
from app.core.models import ImageOcrRecord, Question
from app.core.preprocess import figure, pipeline


def test_reviewed_parameters_survive_classification_reload_and_export(repo, tmp_path, monkeypatch):
    question = Question(subject=config.SUBJECT_PHYSICS, qtype=config.QTYPE_FIGURE,
        raw_text="电路中 I₀=1.2×10⁻³ A；R=5Ω；θ=−30°", annotation="人工补充：R=8Ω",
        image_ocr=ImageOcrRecord(text="未校对 R=50Ω", draft="电路中 R=5Ω", confirmed=True))
    before = deepcopy(question.to_dict())
    # Classification must never initialize the legacy in-process model.
    monkeypatch.setattr(figure._engine, "_ensure", lambda: (_ for _ in ()).throw(AssertionError("model loaded")))
    bundle = pipeline.preprocess(question)
    assert question.to_dict() == before
    assert [(p["symbol"], p["value"], p["unit"], p["source"]) for p in bundle.figure_parameters] == [
        ("I₀", "1.2×10⁻³", "A", "reviewed_text"), ("R", "5", "Ω", "reviewed_text"),
        ("θ", "−30", "°", "reviewed_text"), ("R", "8", "Ω", "annotation")]
    assert "50" not in [p["value"] for p in bundle.figure_parameters]
    assert "电流" in bundle.text_tokens and "电阻" in bundle.text_tokens
    scheme = repo.scheme_by_subject(config.SUBJECT_PHYSICS)
    engine = ClassificationEngine(scheme)
    result = engine.classify(question, fb=bundle)
    engine.apply_result(question, result, scheme, bundle)
    repo.questions.append(question)
    repo.save_questions()
    repo.load_all()
    restored = repo.question_by_id(question.id)
    assert restored.feature_bundle().figure_parameters == bundle.figure_parameters
    path = export_excel([restored], scheme, str(tmp_path / "参数.xlsx"))
    rows = list(openpyxl.load_workbook(path)["全部错题"].values)
    saved = dict(zip(rows[0], rows[1]))
    assert json.loads(saved["图形提取参数(JSON)"]) == bundle.figure_parameters


def test_unreviewed_image_uses_only_saved_body_and_annotation(repo, tmp_path, monkeypatch):
    image = tmp_path / "circuit.png"
    image.write_bytes(b"unused by classification")
    question = Question(subject=config.SUBJECT_PHYSICS, qtype=config.QTYPE_FIGURE,
                        media_path=str(image), raw_text="R=2kΩ", annotation="I=−3mA")
    monkeypatch.setattr(figure._engine, "_ensure", lambda: (_ for _ in ()).throw(AssertionError("model loaded")))
    fb = pipeline.preprocess(question)
    assert [p["value"] for p in fb.figure_parameters] == ["2", "−3"]
    assert [p["source"] for p in fb.figure_parameters] == ["question_text", "annotation"]
    assert not any("执行异常" in note for note in fb.notes)
    assert not fb.ocr_text
    assert any("尚未识别" in note for note in fb.notes)
