from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app import config
from app.core.classify.engine import ClassificationEngine
from app.core.dataio import backup, exporter
from app.core.image_review import save_review
from app.core.models import ImageOcrRecord, Question
from app.core.preprocess import figure, formula, image_ocr, pipeline
from app.core.storage import Repository


@pytest.fixture
def image_question(repo, tmp_path):
    path = tmp_path / "原题.png"
    path.write_bytes(b"test-image-content")
    question = Question(raw_text="原题干", subject=config.SUBJECT_MATH,
                        qtype=config.QTYPE_FIGURE, media_path=repo.import_media(str(path)),
                        classified=True, labels={"knowledge": "old"},
                        feature={"clean_text": "旧特征"})
    repo.add_questions([question])
    record = ImageOcrRecord(text="OCR原文", draft="计算定积分", media_hash=image_ocr.file_hash(Path(question.media_path)),
                            lines=[{"text": "OCR原文", "score": 0.7, "box": [[0, 0], [10, 0], [10, 10], [0, 10]]}])
    return question, record


def test_draft_preserves_question_and_classification_across_restart(repo, image_question):
    question, record = image_question
    previous = question.to_dict()
    save_review(repo, question, record)
    loaded = Repository().question_by_id(question.id)
    assert loaded.raw_text == previous["raw_text"] and loaded.classified
    assert loaded.labels == previous["labels"] and loaded.feature == previous["feature"]
    assert loaded.image_ocr.to_dict() == record.to_dict()
    assert not loaded.image_ocr.confirmed
    record.draft = "未保存的外部改动"
    assert question.image_ocr.draft == "计算定积分"


def test_apply_invalidates_old_classification_and_retains_original(repo, image_question):
    question, record = image_question
    record.draft = "  用换元积分法计算定积分  "
    save_review(repo, question, record, apply=True)
    assert question.raw_text == "用换元积分法计算定积分"
    assert not question.classified and not question.labels and question.feature is None
    assert question.image_ocr.text == "OCR原文" and question.image_ocr.confirmed
    assert question.image_ocr.draft == question.raw_text
    assert question.image_ocr_status == "已校对应用"
    record = deepcopy(question.image_ocr)
    record.draft = "未应用的新草稿"
    save_review(repo, question, record)
    assert question.raw_text == "用换元积分法计算定积分"
    assert "待重新应用" in question.image_ocr_status


@pytest.mark.parametrize("problem", ["changed", "missing", "removed", "empty", "save_failure"])
def test_invalid_or_failed_apply_preserves_existing_data(repo, image_question, monkeypatch, problem):
    question, record = image_question
    previous = question.to_dict()
    saved = config.QUESTION_FILE.read_bytes()
    if problem == "changed":
        Path(question.media_path).write_bytes(b"new-image-content")
    elif problem == "missing":
        Path(question.media_path).unlink()
    elif problem == "removed":
        repo.questions.clear()
    elif problem == "empty":
        record.draft = " \n "
    elif problem == "save_failure":
        def fail_save():
            raise OSError("磁盘写入失败")
        monkeypatch.setattr(repo, "save_questions", fail_save)
    with pytest.raises((ValueError, OSError)):
        save_review(repo, question, record, apply=True)
    assert question.to_dict() == previous
    assert config.QUESTION_FILE.read_bytes() == saved


@pytest.mark.parametrize("qtype", [config.QTYPE_FIGURE, config.QTYPE_FORMULA, config.QTYPE_TEXT])
def test_saved_image_review_uses_only_applied_text_without_implicit_ocr(repo, image_question, monkeypatch, qtype):
    question, record = image_question
    question.qtype = qtype
    def forbidden(*args):
        pytest.fail("已进入图片校对流程的题目不应隐式调用旧OCR覆盖人工校对")
    monkeypatch.setattr(figure._engine, "extract", forbidden)
    monkeypatch.setattr(formula._engine, "recognize", forbidden)
    save_review(repo, question, record)
    pending = pipeline.preprocess(question)
    assert "OCR原文" not in pending.clean_text
    assert "计算定积分" not in pending.clean_text
    save_review(repo, question, record, apply=True)
    applied = pipeline.preprocess(question)
    assert "定积分" in applied.clean_text
    assert "OCR原文" not in applied.clean_text


def test_review_classification_export_restart_and_backup_restore(repo, image_question, tmp_path, monkeypatch):
    from tests.test_backup import switch_workspace
    from openpyxl import load_workbook

    question, record = image_question
    save_review(repo, question, record, apply=True)
    scheme = repo.scheme_by_subject(config.SUBJECT_MATH)
    bundle = pipeline.preprocess(question)
    result = ClassificationEngine(scheme).classify(question, bundle)
    ClassificationEngine.apply_result(question, result, scheme, bundle)
    repo.save_questions()
    assert question.classified and question.labels
    path = exporter.export_excel([question], scheme, str(tmp_path / "图片分类.xlsx"))
    workbook = load_workbook(path, read_only=True)
    rows = list(workbook["全部错题"].values)
    row = dict(zip(rows[0], rows[1]))
    assert row["题干"] == record.draft and row["图片识别状态"] == "已校对应用"
    workbook.close()
    original = question.image_ocr.to_dict()
    archive = backup.create_backup(repo, tmp_path / "含识别记录.zip")
    switch_workspace(monkeypatch, tmp_path / "迁移后")
    target = Repository()
    backup.restore_backup(target, archive)
    loaded = Repository().question_by_id(question.id)
    assert loaded.classified and loaded.labels == question.labels
    assert loaded.raw_text == record.draft and loaded.image_ocr.to_dict() == original
    assert image_ocr.file_hash(Path(loaded.media_path)) == record.media_hash
    save_review(target, target.question_by_id(question.id), loaded.image_ocr)


@pytest.fixture
def raster(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    path = tmp_path / "印刷图.png"
    image_module.new("RGB", (120, 80), "white").save(path)
    return path


def test_engine_returns_original_lines_and_hash_without_auto_applying(raster):
    engine = image_ocr.ImageTextEngine()
    engine._model = lambda pixels: SimpleNamespace(
        txts=("  题目  ", "选项A"), scores=(0.7, 0.99), boxes=np.zeros((2, 4, 2)))
    record = engine.recognize(str(raster))
    assert record.text == record.draft == "题目\n选项A"
    assert record.lines[0]["score"] == 0.7 and record.lines[0]["box"] == [[0.0, 0.0]] * 4
    assert record.media_hash == image_ocr.file_hash(raster)
    assert record.elapsed_ms > 0 and not record.confirmed


@pytest.mark.parametrize("problem", ["blank", "corrupt", "missing", "bytes", "pixels"])
def test_unusable_images_raise_readable_errors(raster, monkeypatch, problem):
    engine = image_ocr.ImageTextEngine()
    engine._model = lambda pixels: SimpleNamespace(txts=None, scores=None, boxes=None)
    if problem == "corrupt":
        raster.write_bytes(b"not-an-image")
    elif problem == "missing":
        raster.unlink()
    elif problem == "bytes":
        monkeypatch.setattr(image_ocr, "MAX_IMAGE_BYTES", 1)
    elif problem == "pixels":
        monkeypatch.setattr(image_ocr, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(image_ocr.ImageOcrError, match="文字|读取|不存在|30MB|像素"):
        engine.recognize(str(raster))


def test_failed_local_model_initialization_reports_error_and_can_retry(monkeypatch):
    engine = image_ocr.ImageTextEngine()
    def missing_models(**kwargs):
        raise image_ocr.ImageOcrError("本地模型缺失")
    monkeypatch.setattr(image_ocr, "model_paths", missing_models)
    monkeypatch.setattr(image_ocr.importlib.util, "find_spec", lambda name: object())
    assert engine.status().state == "model_missing"
    with pytest.raises(image_ocr.ImageOcrError, match="本地模型缺失"):
        engine._ensure()
    assert engine._checked and engine._error == "本地模型缺失"
    assert engine.status().state == "failed"


@pytest.mark.parametrize("problem", ["missing", "size", "hash", "valid"])
def test_local_model_files_are_verified_without_import_or_download(tmp_path, monkeypatch, problem):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    path = model_dir / "small.onnx"
    content = b"verified-model"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(image_ocr, "MODEL_FILES", {"Det": (path.name, len(content), digest)})
    monkeypatch.setattr(image_ocr.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(tmp_path / "__init__.py")))
    if problem != "missing":
        path.write_bytes(content if problem == "valid" else (b"x" if problem == "size" else b"x" * len(content)))
    if problem == "valid":
        assert image_ocr.model_paths(verify_hash=True) == {"Det": path}
    else:
        with pytest.raises(image_ocr.ImageOcrError, match="模型"):
            image_ocr.model_paths(verify_hash=True)


def test_isolated_recognition_uses_project_python_and_releases_model(monkeypatch):
    engine = image_ocr.ImageTextEngine()
    calls = []
    record = ImageOcrRecord(text="Printed question", draft="Printed question", media_hash="hash")
    def run_process(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(record.to_dict()), stderr="")
    monkeypatch.setattr(image_ocr.subprocess, "run", run_process)
    monkeypatch.setattr(image_ocr.importlib.util, "find_spec", lambda name: object())
    result = engine.recognize_isolated("question.png")
    command, options = calls[0]
    assert command[0] == image_ocr.sys.executable and "image_ocr_worker" in command[2]
    assert options["env"]["PYTHONUTF8"] == "1" and options["timeout"] == 180
    assert result.to_dict() == record.to_dict() and engine._model is None
    status = engine.status()
    assert status.state == "verified" and status.available
    assert "独立进程" in status.detail


@pytest.mark.parametrize("problem", ["crash", "timeout", "invalid_json", "empty"])
def test_isolated_process_failure_is_readable_and_unlocks_engine(monkeypatch, problem):
    engine = image_ocr.ImageTextEngine()
    def run_process(command, **kwargs):
        if problem == "timeout":
            raise subprocess.TimeoutExpired(command, 180)
        return SimpleNamespace(returncode=3221225477 if problem == "crash" else 0,
                               stderr="", stdout="invalid" if problem == "invalid_json" else "{}")
    monkeypatch.setattr(image_ocr.subprocess, "run", run_process)
    with pytest.raises(image_ocr.ImageOcrError):
        engine.recognize_isolated("question.png")
    assert engine._checked and engine._error and not engine._lock.locked()
    assert not engine._verified
