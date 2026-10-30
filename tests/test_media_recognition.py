from pathlib import Path
import json
from types import SimpleNamespace

from app.core.models import MediaReviewRecord
from app.core.preprocess import media_recognition
from app.core.preprocess.media_recognition import recognize_isolated


def test_missing_media_returns_error_record(tmp_path):
    record = recognize_isolated(str(tmp_path / "missing.png"), "formula")
    assert isinstance(record, MediaReviewRecord)
    assert record.error
    assert not record.confirmed


def test_worker_record_is_json_compatible(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("x", encoding="utf-8")
    record = recognize_isolated(str(source), "formula")
    assert isinstance(record.to_dict(), dict)
    assert record.media_hash


def test_incomplete_faster_whisper_snapshot_is_rejected(tmp_path, monkeypatch):
    snapshot = tmp_path / "audio" / "models--Systran--faster-whisper-base" / "snapshots" / "x"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    source = tmp_path / "sample.wav"
    source.write_bytes(b"not-audio")
    monkeypatch.setenv("MC_MODEL_DIR", str(tmp_path))
    engine = media_recognition.AudioRecognitionEngine()
    called = []
    monkeypatch.setattr(media_recognition.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr("faster_whisper.WhisperModel", lambda *args, **kwargs: called.append((args, kwargs)))
    record = recognize_isolated(str(source), "audio")
    assert record.error
    assert not called


def test_audio_decode_does_not_patch_av_open(tmp_path, monkeypatch):
    import av
    from app.core.preprocess.media_recognition import _decode_audio
    original = av.open
    source = tmp_path / "empty.wav"
    source.write_bytes(b"not-audio")
    try:
        _decode_audio(source)
    except Exception:
        pass
    assert av.open is original


def test_model_root_follows_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MC_MODEL_DIR", str(tmp_path / "custom-models"))
    assert media_recognition.model_root() == tmp_path / "custom-models"


def test_formula_options_cross_runtime_via_json_file(tmp_path, monkeypatch):
    source = tmp_path / "original.png"
    source.write_bytes(b"original media bytes")
    options = {"crop_box": [20, 30, 530, 240], "invert": True, "resize_mode": "fit"}
    monkeypatch.setattr(media_recognition, "_model_dir", lambda kind: tmp_path)
    monkeypatch.setattr("app.core.preprocess.formula_runtime.formula_runtime",
                        lambda: (tmp_path / "python.exe", tmp_path / "backend.py"))

    def run(command, **kwargs):
        assert command[command.index("--input") + 1] == str(source.resolve())
        options_file = Path(command[command.index("--options-file") + 1])
        assert json.loads(options_file.read_text(encoding="utf-8")) == options
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({"text": "S=integral", "error": ""}), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(media_recognition.subprocess, "run", run)
    monkeypatch.setitem(media_recognition._engines, "formula", media_recognition.FormulaRecognitionEngine())
    record = recognize_isolated(str(source), "formula", options)
    assert record.text == "S=integral"
    assert not record.error
    assert record.media_hash == media_recognition.media_hash(source)


def test_audio_rejects_image_processing_options(tmp_path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"fixture")
    record = recognize_isolated(str(source), "audio", {"invert": True})
    assert "仅适用于公式" in record.error
