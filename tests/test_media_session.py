"""Real subprocess tests for a batch that reuses models, not question results."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from app.core.models import ImageOcrRecord
from app.core.preprocess import media_jobs, media_session
from app.core.preprocess.media_jobs import MediaJobError


def pid_alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


SERVER = '''
import hashlib, os, subprocess, sys, time
from pathlib import Path
from app.core.preprocess.media_server import serve
count = 0
def recognize(request):
    global count
    count += 1
    (Path(sys.argv[1]) / 'model_pid').write_text(str(os.getpid()))
    path = Path(request['input_path'])
    text = path.read_text(encoding='utf-8')
    if text == 'bad':
        raise ValueError('individual input failed')
    if text == 'hang':
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        (Path(sys.argv[1]) / 'child_pid').write_text(str(child.pid))
        (Path(sys.argv[1]) / 'started').write_text('yes')
        time.sleep(60)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if text == 'change':
        path.write_text('changed during inference')
    return {'text': text, 'draft': text, 'media_hash': digest,
            'confirmed': True, 'applied_text': 'must be cleared',
            'engine': str(os.getpid()) + ':' + str(count)}
serve('image', Path(sys.argv[1]), recognize)
'''


@pytest.fixture
def fake_server(monkeypatch):
    monkeypatch.setattr(media_session, "server_command",
                        lambda kind, directory: [sys.executable, "-c", SERVER, str(directory)])


def source(tmp_path, text, name="picture.png"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_distinct_inputs_use_one_process_and_leave_no_workers_or_tempfiles(tmp_path, fake_server):
    first, second = source(tmp_path, "one"), source(tmp_path, "two", "second.png")
    with media_session.MediaSession() as session:
        root = Path(session._temporary.name)
        a = session.recognize(first, "audio")
        process = session._servers["audio"].process
        b = session.recognize(second, "audio")
        assert a.text == "one" and b.text == "two"
        assert a.engine.split(":")[0] == b.engine.split(":")[0]
        assert [a.engine.split(":")[1], b.engine.split(":")[1]] == ["1", "2"]
        assert not a.confirmed and not b.confirmed
        assert a.applied_text == b.applied_text == ""
        assert first.read_text() == "one" and second.read_text() == "two"
    assert process.poll() is not None and not root.exists()
    assert not pid_alive(int(a.engine.split(":")[0]))
    with pytest.raises(MediaJobError, match="已经结束"):
        session.recognize(first, "audio")


def test_error_in_one_question_does_not_reload_model_or_skip_next(tmp_path, fake_server):
    with media_session.MediaSession() as session:
        with pytest.raises(MediaJobError, match="individual input failed"):
            session.recognize(source(tmp_path, "bad"), "audio")
        process = session._servers["audio"].process
        model_pid = (session._servers["audio"].directory / "model_pid").read_text()
        good = session.recognize(source(tmp_path, "recovered", "good.png"), "audio")
        assert good.text == "recovered" and good.engine == model_pid + ":2"
        assert process.poll() is None


@pytest.mark.parametrize("cancel", [True, False])
def test_cancel_and_timeout_end_running_process_and_allow_fresh_start(tmp_path, fake_server, cancel):
    with media_session.MediaSession() as session:
        started = time.monotonic()
        def should_cancel():
            return cancel and time.monotonic() - started > 0.5
        with pytest.raises(MediaJobError) as failure:
            session.recognize(source(tmp_path, "hang"), "audio", timeout=5 if cancel else 0.5,
                              should_cancel=should_cancel)
        assert failure.value.cancelled == cancel
        assert failure.value.timed_out != cancel
        assert "audio" not in session._servers
        for marker in Path(session._temporary.name).glob("*/*_pid"):
            assert not pid_alive(int(marker.read_text())), "Cancelled model process or child survived"
        # Process cancellation cannot leave a stale response that satisfies a new request.
        good = session.recognize(source(tmp_path, "new process", "new.png"), "audio")
        assert good.text == "new process" and good.engine.endswith(":1")


def test_original_changed_during_inference_is_rejected(tmp_path, fake_server):
    with media_session.MediaSession() as session:
        with pytest.raises(MediaJobError, match="附件已变化"):
            session.recognize(source(tmp_path, "change"), "audio")


def test_image_kind_uses_one_process_per_file_instead_of_a_reused_server(tmp_path, fake_server, monkeypatch):
    """图片复用模型会越用越慢（2.3s→6.0s→11.4s），因此批量里也必须一图一进程。"""
    calls = []

    def fake_once(path, kind, options=None, **kwargs):
        calls.append((str(path), kind))
        content = Path(path).read_bytes()
        text = content.decode("utf-8")
        return ImageOcrRecord(text=text, draft=text,
                              media_hash=hashlib.sha256(content).hexdigest())

    monkeypatch.setattr(media_jobs, "_recognize_once", fake_once)
    with media_session.MediaSession() as session:
        first = session.recognize(source(tmp_path, "one"), "image")
        second = session.recognize(source(tmp_path, "two", "b.png"), "image")
        assert (first.text, second.text) == ("one", "two")
        assert "image" not in session._servers, "图片不应创建常驻进程"
        assert [kind for _path, kind in calls] == ["image", "image"]
        assert len(calls) == 2


@pytest.mark.parametrize("options", [None, {"crop_box": [1, 2, 40, 50], "invert": True, "resize_mode": "original"}])
def test_direct_formula_service_preserves_processing_settings(tmp_path, fake_server, options):
    with media_session.MediaSession() as session:
        record = session.recognize(source(tmp_path, "formula"), "formula", options=options)
        assert record.recognition_options == (options or {
            "crop_box": None, "invert": False, "resize_mode": "fit"})
        assert not record.confirmed and not record.applied_text


def test_nested_context_reuses_session_and_restores_single_job_mode(tmp_path, fake_server):
    assert media_jobs._ACTIVE_SESSION.get() is None
    with media_jobs.media_session() as outer:
        with media_jobs.media_session() as inner:
            assert inner is outer
            record = media_jobs.recognize_media(source(tmp_path, "nested"), "audio")
        assert record.text == "nested" and not outer._closed
    assert outer._closed and media_jobs._ACTIVE_SESSION.get() is None


def test_main_worker_flag_does_not_launch_gui(tmp_path):
    directory = tmp_path / "server"
    directory.mkdir()
    (directory / "stop").touch()
    result = subprocess.run([sys.executable, str(media_jobs.ROOT / "main.py"),
                             "--media-session-worker", "image", str(directory)],
                            capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
