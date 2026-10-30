import os
from copy import deepcopy
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication
from app import config
from app.core.models import ImageOcrRecord, MediaReviewRecord, Question
from app.core.preprocess.image_ocr import file_hash
from app.qt_runtime import configure_qt_plugins
from app.ui.context import AppContext
from app.ui.media_batch_dialog import MediaBatchDialog, MediaBatchWorker, media_kind


@pytest.fixture(scope="module")
def app():
    configure_qt_plugins()
    return QApplication.instance() or QApplication([])


@pytest.fixture
def batch(app, repo, tmp_path):
    context = AppContext()
    context.repo = repo
    image = tmp_path / "figure.png"
    audio = tmp_path / "speech.wav"
    image.write_bytes(b"image fixture")
    audio.write_bytes(b"audio fixture")
    questions = [Question(media_path=str(image), qtype=config.QTYPE_FIGURE),
                 Question(media_path=str(audio), qtype=config.QTYPE_LISTENING)]
    repo.questions.extend(questions)
    repo.save_questions()
    dialog = MediaBatchDialog(context, questions)
    yield context, dialog, questions
    dialog.close()


def test_batch_types_and_pending_range(batch):
    _context, dialog, questions = batch
    assert [media_kind(q) for q in questions] == ["image", "audio"]
    captured = []
    dialog._start = captured.extend
    dialog.start_pending()
    assert captured == questions


def test_batch_saves_drafts_without_applying_or_classifying(batch):
    context, dialog, questions = batch
    question = questions[1]
    record = MediaReviewRecord(text="Do not forget", draft="Do not forget",
                               media_hash=file_hash(Path(question.media_path)), engine="test")
    dialog._save_result(question.id, "audio", record, "")
    assert question.audio_transcript.text == "Do not forget"
    assert not question.audio_transcript.confirmed
    assert not question.classified
    context.repo.load_all()
    assert context.repo.question_by_id(question.id).audio_transcript.text == record.text


def test_failed_item_keeps_previous_confirmed_content(batch):
    _context, dialog, questions = batch
    question = questions[1]
    question.audio_transcript = MediaReviewRecord(text="old", draft="old", applied_text="old",
        confirmed=True, media_hash=file_hash(Path(question.media_path)))
    previous = deepcopy(question.audio_transcript)
    dialog._save_result(question.id, "audio", None, "invalid stream")
    assert question.audio_transcript == previous
    assert "invalid stream" in dialog._errors[question.id]


def test_retry_saves_new_draft_preserving_previous_application(batch):
    _context, dialog, questions = batch
    question = questions[1]
    digest = file_hash(Path(question.media_path))
    question.audio_transcript = MediaReviewRecord(text="old", draft="old", applied_text="old",
                                                confirmed=True, media_hash=digest)
    dialog._save_result(question.id, "audio", MediaReviewRecord(text="new", draft="new", media_hash=digest), "")
    assert question.audio_transcript.confirmed
    assert question.audio_transcript.applied_text == "old"
    assert question.audio_transcript.draft == "new"


def test_changed_attachment_is_not_silently_applied(batch):
    _context, dialog, questions = batch
    question = questions[0]
    record = ImageOcrRecord(text="old", draft="old", media_hash=file_hash(Path(question.media_path)))
    Path(question.media_path).write_bytes(b"changed")
    dialog._save_result(question.id, "image", record, "")
    assert question.image_ocr is None
    assert "附件已变化" in dialog._errors[question.id]


def test_batch_worker_continues_after_single_failure(batch, monkeypatch):
    from app.core.preprocess import media_jobs
    _context, _dialog, questions = batch
    def recognize(path, kind, **kwargs):
        if kind == "image":
            raise ValueError("bad image")
        return MediaReviewRecord(text="hello", draft="hello", media_hash=file_hash(Path(path)))
    monkeypatch.setattr(media_jobs, "recognize_media", recognize)
    worker = MediaBatchWorker(questions)
    results = []
    worker.item_done.connect(lambda *args: results.append(args))
    worker.run()
    assert len(results) == 2
    assert results[0][3] == "bad image"
    assert results[1][2].text == "hello"
    assert all(q.image_ocr is None and q.audio_transcript is None for q in questions)


def test_batch_worker_stop_keeps_already_completed_results(batch, monkeypatch):
    from app.core.preprocess import media_jobs
    _context, _dialog, questions = batch
    worker = MediaBatchWorker(questions)
    monkeypatch.setattr(media_jobs, "recognize_media", lambda path, kind, **kwargs: ImageOcrRecord(
        text="draft", draft="draft", media_hash=file_hash(Path(path))))
    results = []
    def receive(*args):
        results.append(args)
        worker.stop()
    worker.item_done.connect(receive)
    worker.run()
    assert len(results) == 1


def test_batch_cancel_interrupts_current_job_without_changing_existing_draft(batch, monkeypatch):
    from app.core.preprocess import media_jobs
    _context, _dialog, questions = batch
    worker = MediaBatchWorker(questions)
    calls = []
    def recognize(path, kind, *, should_cancel, **kwargs):
        calls.append(path)
        assert not should_cancel()
        worker.stop()
        assert should_cancel()
        raise media_jobs.MediaJobError("cancelled", cancelled=True)
    monkeypatch.setattr(media_jobs, "recognize_media", recognize)
    results = []
    worker.item_done.connect(lambda *args: results.append(args))
    worker.run()
    assert len(calls) == 1 and not results
    assert all(q.image_ocr is None and q.audio_transcript is None for q in questions)


def test_batch_formula_retry_keeps_saved_crop_options(batch, monkeypatch):
    from app.core.preprocess import media_jobs
    _context, _dialog, questions = batch
    question = questions[0]
    question.qtype = config.QTYPE_FORMULA
    options = {"crop_box": [10, 20, 300, 150], "invert": True, "resize_mode": "fit"}
    question.formula_ocr = MediaReviewRecord(recognition_options=options)
    captured = []
    def recognize(path, kind, *, options, **kwargs):
        captured.append((kind, options))
        return MediaReviewRecord(text="x", draft="x", recognition_options=options)
    monkeypatch.setattr(media_jobs, "recognize_media", recognize)
    MediaBatchWorker([question]).run()
    assert captured == [("formula", options)]
