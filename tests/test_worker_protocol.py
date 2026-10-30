"""发布版没有控制台时，识别结果与失败原因仍经独立文件传回。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from app.core.models import ImageOcrRecord, MediaReviewRecord
from app.core.preprocess import image_ocr


ROOT = Path(__file__).resolve().parents[1]


def test_image_worker_main_flag_writes_failure_without_opening_gui(tmp_path):
    output = tmp_path / "结果.json"
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), "--image-ocr-worker",
                             str(tmp_path / "missing.png"), str(output)],
                            capture_output=True, timeout=30)
    assert result.returncode == 1
    assert "不存在" in json.loads(output.read_text(encoding="utf-8"))["error"]


def test_audio_worker_main_flag_writes_failure_without_opening_gui(tmp_path):
    output = tmp_path / "结果.json"
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), "--media-recognition-worker",
                             "audio", str(tmp_path / "missing.wav"), str(output)],
                            capture_output=True, timeout=30)
    assert result.returncode == 2
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["error"] and not record["confirmed"]


def test_frozen_image_worker_reads_file_with_empty_stdout(tmp_path, monkeypatch):
    helper = tmp_path / "media_worker.exe"
    helper.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "gui.exe"))
    result_record = ImageOcrRecord(text="测试文字", draft="测试文字", media_hash="hash")
    outputs = []

    def run(command, **kwargs):
        assert command[:2] == [str(helper), "image"]
        assert kwargs["cwd"] is None
        output = Path(command[-1])
        outputs.append(output)
        output.write_text(json.dumps(result_record.to_dict()), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(image_ocr.subprocess, "run", run)
    record = image_ocr.ImageTextEngine().recognize_isolated("sample.png")
    assert record.text == "测试文字"
    assert not outputs[0].exists(), "临时结果文件应在读完后清理"


def test_frozen_image_worker_preserves_file_error(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "gui.exe"))

    def run(command, **kwargs):
        assert "--image-ocr-worker" in command
        Path(command[-1]).write_text(json.dumps({"error": "模型文件校验失败"}), encoding="utf-8")
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(image_ocr.subprocess, "run", run)
    import pytest
    with pytest.raises(image_ocr.ImageOcrError, match="模型文件校验失败"):
        image_ocr.ImageTextEngine().recognize_isolated("sample.png")


def test_frozen_audio_worker_ignores_library_stdout_and_reads_file(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from app.ui import media_review_dialog
    from app.core.preprocess import media_jobs

    helper = tmp_path / "media_worker.exe"
    helper.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "gui.exe"))
    source = tmp_path / "sample.wav"
    source.write_bytes(b"audio original bytes")
    expected = MediaReviewRecord(text="Hello", draft="Hello", media_hash=media_jobs._hash_file(source),
                                 engine="faster-whisper")
    output_files = []

    def popen(command, **kwargs):
        assert command[:2] == [str(helper), "audio"]
        output = Path(command[-1])
        output_files.append(output)
        output.write_text(json.dumps(expected.to_dict()), encoding="utf-8")
        return SimpleNamespace(returncode=0, communicate=lambda **opts: ("third-party log output", ""))

    monkeypatch.setattr(media_jobs.subprocess, "Popen", popen)
    worker = media_review_dialog.MediaReviewWorker("audio", str(source))
    received = []
    worker.completed.connect(lambda record, error: received.append((record, error)))
    worker.run()
    assert received[0][0].text == "Hello" and received[0][1] == ""
    assert not output_files[0].exists()


def test_packaged_model_root_prefers_bundled_weights_but_allows_override(tmp_path, monkeypatch):
    from app.core.preprocess.media_recognition import model_root
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "media_worker.exe"))
    monkeypatch.delenv("MC_MODEL_DIR", raising=False)
    (tmp_path / "models").mkdir()
    assert model_root() == tmp_path / "models"
    monkeypatch.setenv("MC_MODEL_DIR", str(tmp_path / "alternative"))
    assert model_root() == tmp_path / "alternative"


def test_frozen_ocr_selfcheck_checks_bundled_files_without_importing_runtime(tmp_path, monkeypatch):
    helper = tmp_path / "media_worker.exe"
    helper.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "gui.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "_internal"), raising=False)
    folder = tmp_path / "_internal" / "rapidocr" / "models"
    folder.mkdir(parents=True)
    (folder / "fake.onnx").write_bytes(b"abc")
    monkeypatch.setattr(image_ocr, "MODEL_FILES", {"Det": ("fake.onnx", 3, "unused")})
    def unexpected(name):
        raise AssertionError("发布版自检不应加载GUI内没有的OCR运行时")
    monkeypatch.setattr(image_ocr.importlib.util, "find_spec", unexpected)
    status = image_ocr.ImageTextEngine().status()
    assert status.state == "unchecked"
    (folder / "fake.onnx").write_bytes(b"broken")
    assert image_ocr.ImageTextEngine().status().state == "model_missing"
