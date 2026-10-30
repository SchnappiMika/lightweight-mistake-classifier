from __future__ import annotations

import pytest

from app.core.preprocess import audio, base, figure, formula, image_ocr, pipeline, text_en


@pytest.fixture
def passive_engines(monkeypatch):
    monkeypatch.setenv("MC_FORMULA_PYTHON", "__test_formula_runtime_not_installed__")
    monkeypatch.setattr(audio, "_engine", audio.WhisperEngine())
    monkeypatch.setattr(figure, "_engine", figure.PaddleOcrEngine())
    monkeypatch.setattr(formula, "_engine", formula.LatexOcrEngine())
    monkeypatch.setattr(image_ocr, "_engine", image_ocr.ImageTextEngine())
    monkeypatch.setattr(image_ocr, "model_paths", lambda **kwargs: {})
    monkeypatch.setattr(text_en, "_nltk_state", None)
    monkeypatch.setattr(text_en, "_nltk_error", "")

    def forbidden_initialization(*args, **kwargs):
        pytest.fail("状态查询不应初始化引擎或加载语料")

    for engine in (audio._engine, figure._engine, formula._engine, image_ocr._engine):
        monkeypatch.setattr(engine, "_ensure", forbidden_initialization)
    monkeypatch.setattr(text_en, "_try_nltk", forbidden_initialization)
    return audio._engine, figure._engine, formula._engine, image_ocr._engine


def test_report_does_not_load_models_or_mark_installed_packages_ready(passive_engines, monkeypatch):
    monkeypatch.setattr(base.importlib.util, "find_spec", lambda name: object())
    report = pipeline.engine_report()
    assert len(report) == 5
    assert all(status.state == "unchecked" and not status.available for status in report)
    assert all(not engine._checked for engine in passive_engines)
    assert text_en._nltk_state is None
    assert all("不加载" in status.action and "下载" in status.action for status in report)


def test_missing_components_have_next_steps_and_text_fallback(passive_engines, monkeypatch):
    monkeypatch.setattr(base.importlib.util, "find_spec", lambda name: None)
    report = pipeline.engine_report()
    assert all(status.state == "missing" and not status.available for status in report)
    assert all(status.action and status.detail for status in report)
    assert "faster_whisper" in report[0].detail
    assert "内置英文规则" in report[-2].action


def test_paddle_ocr_needs_both_ocr_and_runtime(passive_engines, monkeypatch):
    monkeypatch.setattr(base.importlib.util, "find_spec", lambda name: object() if name == "paddleocr" else None)
    assert figure.FigureUnit().status().state == "missing"


def test_whisper_accepts_either_backend_without_loading(passive_engines, monkeypatch):
    monkeypatch.setattr(base.importlib.util, "find_spec", lambda name: object() if name == "whisper" else None)
    assert audio.AudioUnit().status().state == "unchecked"


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_loading_state_does_not_wait_for_model_lock(passive_engines, index):
    engine = passive_engines[index]
    engine._lock.acquire()
    try:
        status = (audio.AudioUnit(), figure.FigureUnit(), formula.FormulaUnit(), image_ocr._engine)[index].status()
        assert status.state == "loading" and not status.available
    finally:
        engine._lock.release()


def test_loaded_models_are_reported_from_cached_state(passive_engines):
    audio._engine._model = object()
    audio._engine._backend = "faster"
    figure._engine._ocr = object()
    formula._engine._model = object()
    image_ocr._engine._model = object()
    text_en._nltk_state = True
    report = pipeline.engine_report()
    assert all(status.state == "ready" and status.available for status in report)
    assert all("不代表" in status.action for status in report)


def test_failed_initialization_retains_original_error(passive_engines, monkeypatch):
    monkeypatch.setattr(base.importlib.util, "find_spec", lambda name: object())
    for engine in passive_engines:
        engine._checked = True
        engine._error = "模型文件缺失或格式错误 <details>"
    monkeypatch.setattr(text_en, "_nltk_state", False)
    monkeypatch.setattr(text_en, "_nltk_error", "WordNet 语料缺失")
    report = pipeline.engine_report()
    assert all(status.state == "failed" and not status.available for status in report)
    assert "<details>" in report[0].detail
    assert "WordNet" in report[-2].detail


def test_component_lookup_failure_does_not_break_report(passive_engines, monkeypatch):
    def broken_lookup(name):
        raise ValueError("模块安装信息损坏")

    monkeypatch.setattr(base.importlib.util, "find_spec", broken_lookup)
    report = pipeline.engine_report()
    assert len(report) == 5
    assert all(status.state == "check_failed" and "安装信息损坏" in status.detail for status in report)


def test_one_faulty_status_does_not_hide_other_engines(passive_engines, monkeypatch):
    def broken_status():
        raise RuntimeError("状态检查异常")

    monkeypatch.setattr(base.get("audio"), "status", broken_status)
    report = pipeline.engine_report()
    assert len(report) == 5 and report[0].state == "check_failed"
    assert "状态检查异常" in report[0].detail
