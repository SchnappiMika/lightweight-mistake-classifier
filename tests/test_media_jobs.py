import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.models import ImageOcrRecord, MediaReviewRecord
from app.core.preprocess import media_jobs


def source_file(tmp_path):
    path = tmp_path / "原始 附件.png"
    path.write_bytes(b"original attachment bytes")
    return path


def fake_process(payload, *, returncode=0, stderr="", observer=None):
    def popen(command, **kwargs):
        if observer:
            observer(command, kwargs)
        output = Path(command[command.index("--options-file") - 1] if "--options-file" in command else command[-1])
        output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return SimpleNamespace(returncode=returncode,
                               communicate=lambda **options: ("non-json third-party logging", stderr))
    return popen


@pytest.mark.parametrize("kind,cls", [("image", ImageOcrRecord), ("formula", MediaReviewRecord),
                                      ("audio", MediaReviewRecord)])
def test_source_recognition_uses_json_file_and_returns_unconfirmed_record(tmp_path, monkeypatch, kind, cls):
    source = source_file(tmp_path)
    contents = source.read_bytes()
    expected_hash = hashlib.sha256(contents).hexdigest()
    payload = {"text": "recognized", "draft": "recognized", "media_hash": expected_hash,
               "confirmed": True, "applied_text": "must not be auto-confirmed"}
    observed = []

    def observe(command, kwargs):
        assert command[:2] == [sys.executable, "-m"]
        module = "image_ocr_worker" if kind == "image" else "media_recognition_worker"
        assert command[2].endswith(module)
        assert kwargs["cwd"] == str(media_jobs.ROOT)
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert "PYTHONHOME" not in kwargs["env"]
        observed.append(Path(command[-1]))

    monkeypatch.setenv("PYTHONHOME", "bad inherited Python environment")
    monkeypatch.setattr(media_jobs.subprocess, "Popen", fake_process(payload, observer=observe))
    record = media_jobs.recognize_media(source, kind)
    assert isinstance(record, cls)
    assert record.text == "recognized" and not record.confirmed
    if isinstance(record, MediaReviewRecord):
        assert record.applied_text == ""
    assert not observed[0].exists()
    assert source.read_bytes() == contents


@pytest.mark.parametrize("kind", ["image", "formula", "audio"])
@pytest.mark.parametrize("has_helper", [True, False])
def test_frozen_command_uses_sibling_worker_or_main_flag(tmp_path, monkeypatch, kind, has_helper):
    source = source_file(tmp_path)
    helper = tmp_path / "media_worker.exe"
    if has_helper:
        helper.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "gui.exe"))
    output = tmp_path / "result.json"
    command = media_jobs._command(kind, source, output)
    if has_helper:
        assert command[:2] == [str(helper), kind]
    else:
        assert command[0] == sys.executable
        assert command[1] == ("--image-ocr-worker" if kind == "image" else "--media-recognition-worker")
        if kind != "image":
            assert command[2] == kind
    assert command[-2:] == [str(source), str(output)]


def test_formula_options_use_unicode_json_file_and_original_hash(tmp_path, monkeypatch):
    source = source_file(tmp_path)
    options = {"crop_box": [40, 40, 550, 250], "invert": True, "resize_mode": "fit"}
    paths = []

    def observe(command, kwargs):
        settings = Path(command[command.index("--options-file") + 1])
        assert json.loads(settings.read_text(encoding="utf-8")) == options
        paths.append(settings)

    payload = {"text": r"S=\int_a^b f(x)dx", "media_hash": media_jobs._hash_file(source)}
    monkeypatch.setattr(media_jobs.subprocess, "Popen", fake_process(payload, observer=observe))
    record = media_jobs.recognize_media(source, "formula", options)
    assert record.media_hash == media_jobs._hash_file(source)
    assert not paths[0].exists()


def test_worker_error_preserves_partial_record_and_reason(tmp_path, monkeypatch):
    source = source_file(tmp_path)
    payload = {"text": "partial", "error": "模型文件校验失败", "media_hash": media_jobs._hash_file(source)}
    monkeypatch.setattr(media_jobs.subprocess, "Popen", fake_process(payload, returncode=2))
    with pytest.raises(media_jobs.MediaJobError, match="模型文件校验失败") as error:
        media_jobs.recognize_media(source, "audio")
    assert error.value.record.text == "partial"
    assert error.value.record.error == "模型文件校验失败"
    assert error.value.returncode == 2


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "result", "media_hash": "wrong hash"}])
def test_empty_or_wrong_attachment_results_are_not_success(tmp_path, monkeypatch, payload):
    source = source_file(tmp_path)
    monkeypatch.setattr(media_jobs.subprocess, "Popen", fake_process(payload))
    with pytest.raises(media_jobs.MediaJobError):
        media_jobs.recognize_media(source, "image")


def test_cancelled_before_start_does_not_spawn_process(tmp_path, monkeypatch):
    source = source_file(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("cancelled task must not spawn a process")
    monkeypatch.setattr(media_jobs.subprocess, "Popen", forbidden)
    with pytest.raises(media_jobs.MediaJobError) as error:
        media_jobs.recognize_media(source, "formula", should_cancel=lambda: True)
    assert error.value.cancelled


def test_timeout_terminates_actual_child_process(tmp_path, monkeypatch):
    source = source_file(tmp_path)
    monkeypatch.setattr(media_jobs, "_command", lambda *args: [sys.executable, "-c", "import time;time.sleep(60)"])
    original_popen = subprocess.Popen
    children = []
    def popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(media_jobs.subprocess, "Popen", popen)
    with pytest.raises(media_jobs.MediaJobError) as error:
        media_jobs.recognize_media(source, "image", timeout=0.1)
    assert error.value.timed_out
    assert source.is_file()
    assert children[0].poll() is not None


def test_actual_worker_failure_preserves_bad_image_reason(tmp_path):
    source = source_file(tmp_path)
    with pytest.raises(media_jobs.MediaJobError) as error:
        media_jobs.recognize_media(source, "image", timeout=30)
    assert "图片无法读取" in str(error.value)
    assert error.value.returncode == 1


def test_client_import_does_not_load_qt_or_inference_runtime():
    code = ("import sys,json; import app.core.preprocess.media_jobs; "
            "print(json.dumps([name for name in ('torch','onnxruntime','rapidocr','PyQt5') if name in sys.modules]))")
    completed = subprocess.run([sys.executable, "-c", code], cwd=str(media_jobs.ROOT),
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []
