"""Separate formula processes must preserve results and failures across runtimes."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from app.core.preprocess import formula_runtime, media_recognition
from app.core.preprocess.image_ocr import ImageOcrError


def test_missing_explicit_formula_interpreter_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv("MC_FORMULA_PYTHON", str(tmp_path / "missing.exe"))
    with pytest.raises(FileNotFoundError, match="MC_FORMULA_PYTHON"):
        formula_runtime.formula_runtime()


def test_frozen_formula_runtime_resolves_alongside_application(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "media_worker.exe"))
    monkeypatch.delenv("MC_FORMULA_PYTHON", raising=False)
    directory = tmp_path / "formula_runtime"
    (directory / "python").mkdir(parents=True)
    (directory / "python" / "python.exe").touch()
    (directory / "formula_recognition_backend.py").touch()
    python, backend = formula_runtime.formula_runtime()
    assert python == directory / "python" / "python.exe"
    assert backend.parent == directory


@pytest.mark.parametrize("error,returncode", [("", 0), ("公式模型校验失败", 1)])
def test_formula_bridge_uses_json_and_preserves_errors(tmp_path, monkeypatch, error, returncode):
    model = tmp_path / "models" / "formula"
    model.mkdir(parents=True)
    (model / "manifest.json").write_text("{}")
    monkeypatch.setenv("MC_MODEL_DIR", str(model.parent))
    monkeypatch.setattr(formula_runtime, "formula_runtime", lambda: (Path("python.exe"), Path("backend.py")))
    def run(command, **kwargs):
        assert command[command.index("--model-dir") + 1] == str(model.resolve())
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        Path(command[command.index("--output") + 1]).write_text(
            json.dumps({"text": r"\frac{dy}{dx}", "error": error}), encoding="utf-8")
        return SimpleNamespace(returncode=returncode, stdout="diagnostic logging", stderr="")
    monkeypatch.setattr(media_recognition.subprocess, "run", run)
    if error:
        with pytest.raises(ImageOcrError, match=error):
            media_recognition.FormulaRecognitionEngine().recognize(tmp_path / "formula.png")
    else:
        assert media_recognition.FormulaRecognitionEngine().recognize(tmp_path / "formula.png") == r"\frac{dy}{dx}"
