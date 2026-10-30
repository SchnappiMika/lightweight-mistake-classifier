from contextlib import nullcontext
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType

from PIL import Image
import pytest

from app.core.preprocess import formula_recognition_backend as backend
from tests.test_formula_backend import model_manifest


@pytest.fixture
def fake_inference(monkeypatch, tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    model_manifest(directory)
    state = {"loads": 0, "rng": 0, "inferences": 0, "fail_once": False}
    torch = ModuleType("torch")
    torch.set_num_threads = lambda value: None
    torch.manual_seed = lambda value: state.update(rng=value)
    torch.get_rng_state = lambda: state["rng"]
    torch.set_rng_state = lambda value: state.update(rng=value)
    torch.inference_mode = nullcontext
    yaml = ModuleType("yaml")
    yaml.safe_load = lambda value: {}
    yaml.safe_dump = json.dumps
    package = ModuleType("pix2tex")
    cli = ModuleType("pix2tex.cli")
    cli.clipboard = type("Clipboard", (), {"copy": staticmethod(lambda *args: None)})()

    class Model:
        def __init__(self, arguments):
            assert Path(arguments.config).is_file()
            state["loads"] += 1
            state["rng"] += 51

        def __call__(self, image):
            state["inferences"] += 1
            if state["fail_once"]:
                state["fail_once"] = False
                state["rng"] += 100
                raise RuntimeError("single inference failed")
            text = "formula-{}-{}x{}".format(state["rng"], *image.size)
            state["rng"] += 1
            return text

    cli.LatexOCR = Model
    package.cli = cli
    for name, module in (("torch", torch), ("yaml", yaml), ("pix2tex", package), ("pix2tex.cli", cli)):
        monkeypatch.setitem(sys.modules, name, module)
    return directory, state


def test_model_and_checksums_reused_but_each_source_rehashed(tmp_path, monkeypatch, fake_inference):
    directory, state = fake_inference
    sources = [tmp_path / "first.png", tmp_path / "second.png"]
    for path, size in zip(sources, [(354, 202), (510, 210)]):
        Image.new("RGB", size, "white").save(path)
    calls = []
    original_sha = backend.sha256

    def track(path):
        calls.append(path)
        return original_sha(path)

    monkeypatch.setattr(backend, "sha256", track)
    recognizer = backend.FormulaRecognizer(directory)
    first = recognizer.recognize(sources[0])
    second = recognizer.recognize(sources[1])
    repeated = recognizer.recognize(sources[0])
    assert not first["error"] and not second["error"] and not repeated["error"]
    assert first["text"] == repeated["text"]
    assert first["text"] != second["text"]
    assert state["loads"] == 1
    for name in backend.REQUIRED:
        assert calls.count(directory / name) == 1
    assert calls.count(sources[0]) == 4
    assert calls.count(sources[1]) == 2
    assert first["confirmed"] is False
    assert first["preprocessing"]["processed_size"] == [336, 192]


def test_invalid_image_or_options_do_not_load_or_poison_model(tmp_path, fake_inference):
    directory, state = fake_inference
    source = tmp_path / "valid.png"
    Image.new("RGB", (64, 32), "white").save(source)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    recognizer = backend.FormulaRecognizer(directory)
    assert recognizer.recognize(source, {"invert": "invalid"})["error"]
    assert recognizer.recognize(broken)["error"]
    assert state["loads"] == 0
    first = recognizer.recognize(source)
    state["fail_once"] = True
    assert recognizer.recognize(source)["error"]
    recovered = recognizer.recognize(source)
    assert recovered["text"] == first["text"] and not recovered["error"]
    assert state["loads"] == 1


def test_source_changed_during_inference_is_rejected(tmp_path, fake_inference, monkeypatch):
    directory, _ = fake_inference
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 32), "white").save(source)
    recognizer = backend.FormulaRecognizer(directory)
    assert not recognizer.recognize(source)["error"]
    original = recognizer._model

    def change(image):
        source.write_bytes(b"changed during recognition")
        return original(image)

    monkeypatch.setattr(recognizer, "_model", change)
    result = recognizer.recognize(source)
    assert result["error"] and not result["text"]


def wait_for(path, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        time.sleep(0.01)
    pytest.fail("service did not publish " + str(path))


def test_service_sequential_requests_errors_and_atomic_responses(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    source = (tmp_path / "formula.png").resolve()
    source.write_bytes(b"fixture")
    calls = []

    class Recognizer:
        def __init__(self, directory):
            calls.append(("init", directory))

        def recognize(self, path, options=None):
            calls.append((path, options))
            return {"text": "result", "error": "", "confirmed": False}

    monkeypatch.setattr(backend, "FormulaRecognizer", Recognizer)
    # Requests must be consumed in sequence, even if a later number arrives first.
    (session / "request-00000002.json").write_text("[1]", encoding="utf-8")
    thread = threading.Thread(target=backend.serve, args=(session, tmp_path), daemon=True)
    thread.start()
    try:
        time.sleep(0.05)
        assert not (session / "response-00000002.json").exists()
        (session / "request-00000001.json").write_text(json.dumps({"input_path": str(source), "options": None}), encoding="utf-8")
        assert wait_for(session / "response-00000001.json")["text"] == "result"
        assert wait_for(session / "response-00000002.json")["error"]
        (session / "request-00000003.json").write_text("invalid json", encoding="utf-8")
        assert wait_for(session / "response-00000003.json")["error"]
        (session / "request-00000004.json").write_text(json.dumps({"input_path": str(source), "options": {"invert": True}}), encoding="utf-8")
        assert wait_for(session / "response-00000004.json")["text"] == "result"
        assert calls == [("init", tmp_path), (source, None), (source, {"invert": True})]
        assert not list(session.glob(".response-*.tmp"))
    finally:
        (session / "stop").touch()
        thread.join(timeout=3)
    assert not thread.is_alive()


@pytest.mark.parametrize("payload", [
    None, [], {}, {"input_path": "relative.png"}, {"input_path": 1},
    {"input_path": ""}, {"input_path": "/tmp/formula.png", "options": []},
    {"input_path": "/tmp/formula.png", "options": {"unknown": True}},
])
def test_request_handler_rejects_invalid_json_fields(tmp_path, payload):
    request = tmp_path / "request-00000001.json"
    if isinstance(payload, dict) and payload.get("input_path") == "/tmp/formula.png":
        payload = dict(payload, input_path=str((tmp_path / "absolute.png").resolve()))
    request.write_text(json.dumps(payload), encoding="utf-8")
    result = backend.handle_request(request, None)
    assert result["error"] and not result["text"] and result["confirmed"] is False


def test_request_directory_and_oversized_json_return_errors(tmp_path, monkeypatch):
    request = tmp_path / "request-00000001.json"
    request.mkdir()
    assert backend.handle_request(request, None)["error"]
    oversized = tmp_path / "request-00000002.json"
    oversized.write_text(" " * 20)
    monkeypatch.setattr(backend, "MAX_REQUEST_BYTES", 10)
    assert backend.handle_request(oversized, None)["error"]


def test_single_request_cli_and_service_cli_remain_separate(tmp_path, monkeypatch):
    source = tmp_path / "formula.png"
    options = tmp_path / "options.json"
    options.write_text('{"invert": true}', encoding="utf-8")
    destination = tmp_path / "result.json"
    calls = []

    def recognize(path, directory, settings):
        calls.append((path, directory, settings))
        return {"text": "formula", "error": "", "confirmed": False}

    monkeypatch.setattr(backend, "recognize", recognize)
    assert backend.main(["--input", str(source), "--output", str(destination),
                         "--model-dir", str(tmp_path), "--options-file", str(options)]) == 0
    assert json.loads(destination.read_text(encoding="utf-8"))["text"] == "formula"
    assert calls == [(source.resolve(), tmp_path.resolve(), {"invert": True})]
    served = []
    monkeypatch.setattr(backend, "serve", lambda session, directory: served.append((session, directory)) or 0)
    assert backend.main(["--serve", str(tmp_path), "--model-dir", str(tmp_path)]) == 0
    assert served == [(tmp_path, tmp_path)]
    with pytest.raises(SystemExit):
        backend.main(["--serve", str(tmp_path), "--model-dir", str(tmp_path), "--input", str(source)])


def test_invalid_service_directory_exits_with_error(tmp_path, capsys):
    assert backend.main(["--serve", str(tmp_path / "missing"), "--model-dir", str(tmp_path)]) == 1
    assert "会话目录" in capsys.readouterr().err
