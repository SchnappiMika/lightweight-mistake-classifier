"""Single-item UI exercises the real subprocess client without inference models."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QMessageBox

from app import config
from app.core.models import MediaReviewRecord, Question
from app.core.preprocess import media_jobs
from app.core.storage import Repository
from app.qt_runtime import configure_qt_plugins
from app.ui.context import AppContext
from app.ui.media_review_dialog import MediaReviewDialog


@pytest.fixture(scope="module")
def qapp():
    configure_qt_plugins()
    return QApplication.instance() or QApplication([])


def until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert predicate(), "The media UI did not reach the expected state."


@pytest.fixture
def subprocess_backend(tmp_path, monkeypatch):
    script = tmp_path / "recognition_stub.py"
    script.write_text(
        "import hashlib,json,sys,time\n"
        "from pathlib import Path\n"
        "script=Path(__file__)\n"
        "source,output=map(Path,sys.argv[1:3])\n"
        "digest=hashlib.sha256(source.read_bytes()).hexdigest()\n"
        "mode=script.with_suffix('.mode').read_text()\n"
        "script.with_suffix('.ready').touch()\n"
        "if mode=='hang': time.sleep(60)\n"
        "if mode=='wait':\n"
        " while not script.with_suffix('.release').exists(): time.sleep(0.01)\n"
        "options=json.loads(Path(sys.argv[-1]).read_text(encoding='utf-8')) if '--options-file' in sys.argv else {}\n"
        "payload={'text':'recognized content','draft':'recognized content','media_hash':digest,"
        "'recognition_options':options,'engine':'test subprocess','confirmed':False}\n"
        "output.write_text(json.dumps(payload),encoding='utf-8')\n",
        encoding="utf-8")
    monkeypatch.setattr(media_jobs, "_command", lambda kind, source, output:
                        [sys.executable, str(script), str(source), str(output)])
    actual_popen = media_jobs.subprocess.Popen
    children = []

    def popen(command, **kwargs):
        process = actual_popen(command, **kwargs)
        if str(script) in command:
            children.append(process)
        return process

    monkeypatch.setattr(media_jobs.subprocess, "Popen", popen)
    environment = SimpleNamespace(
        ready=script.with_suffix(".ready"), release=script.with_suffix(".release"),
        mode=lambda value: script.with_suffix(".mode").write_text(value), children=children)
    yield environment
    for child in children:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)


@pytest.fixture
def dialogs(qapp, repo, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)

    def create(kind="formula", previous=True):
        path = tmp_path / ("原始公式.png" if kind == "formula" else "原始听力.wav")
        if kind == "formula":
            image = QImage(800, 400, QImage.Format_RGB32)
            image.fill(Qt.white)
            assert image.save(str(path))
        else:
            path.write_bytes(b"original audio data")
        ctx = AppContext()
        question = Question(raw_text="原题保持不变", media_path=str(path),
                            qtype=config.QTYPE_FORMULA if kind == "formula" else config.QTYPE_LISTENING)
        if previous:
            record = MediaReviewRecord(text="previous machine result", draft="previous draft",
                                        media_hash=media_jobs._hash_file(path), confirmed=True,
                                        applied_text="previous applied content")
            setattr(question, "formula_ocr" if kind == "formula" else "audio_transcript", record)
        ctx.repo.add_questions([question])
        dialog = MediaReviewDialog(ctx, question, kind)
        opened.append(dialog)
        dialog.show()
        return ctx, question, dialog

    yield create
    for dialog in opened:
        if dialog._worker is not None and dialog._worker.isRunning():
            dialog._worker.requestInterruption()
            dialog._worker.wait(5000)
            qapp.processEvents()
        dialog._dirty = False
        dialog.close()


@pytest.mark.parametrize("kind", ["formula", "audio"])
def test_cancel_ends_actual_process_preserves_record_and_allows_safe_close(dialogs, subprocess_backend, kind):
    subprocess_backend.mode("hang")
    ctx, question, dialog = dialogs(kind)
    before = question.to_dict()
    dialog.btn_recognize.click()
    until(subprocess_backend.ready.exists)
    dialog.close()
    assert dialog.isVisible() and dialog._busy
    dialog.btn_cancel.click()
    until(lambda: not dialog._busy)
    assert subprocess_backend.children[0].poll() is not None
    assert question.to_dict() == before and not ctx.active_task
    assert "取消" in dialog.message.text()
    assert not dialog._worker.isRunning()
    dialog.close()
    assert not dialog.isVisible()


@pytest.mark.parametrize("kind", ["formula", "audio"])
def test_timeout_ends_actual_process_without_replacing_saved_result(dialogs, subprocess_backend, monkeypatch, kind):
    subprocess_backend.mode("hang")
    monkeypatch.setitem(media_jobs.TIMEOUTS, kind, 0.15)
    ctx, question, dialog = dialogs(kind)
    before = question.to_dict()
    dialog.btn_recognize.click()
    until(lambda: not dialog._busy)
    assert subprocess_backend.children[0].poll() is not None
    assert question.to_dict() == before and not ctx.active_task
    assert "超过" in dialog.message.text()
    assert dialog.btn_recognize.isEnabled()


@pytest.mark.parametrize("change", ["bytes", "registered_path"])
def test_changed_attachment_rejects_inflight_result_and_keeps_prior_review(
        dialogs, subprocess_backend, tmp_path, change):
    subprocess_backend.mode("wait")
    ctx, question, dialog = dialogs()
    before_record = question.formula_ocr.to_dict()
    before_editor = dialog.editor.toPlainText()
    dialog.btn_recognize.click()
    until(subprocess_backend.ready.exists)
    if change == "bytes":
        Path(question.media_path).write_bytes(b"attachment changed during recognition")
    else:
        replacement = tmp_path / "replacement.png"
        replacement.write_bytes(b"different registered attachment")
        question.media_path = str(replacement)
    subprocess_backend.release.touch()
    until(lambda: not dialog._busy)
    assert question.formula_ocr.to_dict() == before_record
    assert dialog.record.to_dict() == before_record
    assert dialog.editor.toPlainText() == before_editor
    assert "附件已变化" in dialog.message.text()
    assert not ctx.active_task


def test_formula_options_survive_recognition_draft_save_repository_reload_and_reopen(
        dialogs, subprocess_backend):
    subprocess_backend.mode("success")
    ctx, question, dialog = dialogs(previous=False)
    original = Path(question.media_path).read_bytes()
    options = {"crop_box": [100, 50, 300, 150], "invert": True, "resize_mode": "original"}
    dialog.formula_crop.set_options(options)
    dialog.btn_recognize.click()
    until(lambda: not dialog._busy)
    assert question.formula_ocr is not None
    assert question.formula_ocr.recognition_options == options
    assert not question.formula_ocr.confirmed
    dialog.editor.setPlainText("手动校对后的公式草稿")
    dialog.btn_save.click()
    dialog.close()
    persisted = Repository().question_by_id(question.id)
    assert persisted.formula_ocr.draft == "手动校对后的公式草稿"
    assert persisted.formula_ocr.recognition_options == options
    reopened = MediaReviewDialog(ctx, persisted, "formula")
    try:
        assert reopened.formula_crop.options() == options
        assert reopened.editor.toPlainText() == "手动校对后的公式草稿"
        assert not reopened.record.confirmed
        assert Path(question.media_path).read_bytes() == original
    finally:
        reopened.close()
