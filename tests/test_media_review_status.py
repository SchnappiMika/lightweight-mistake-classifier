"""Review windows describe the actual automatic and manual classification inputs."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication

from app import config
from app.core.models import ImageOcrRecord, MediaReviewRecord, Question
from app.core.preprocess.media_recognition import media_hash
from app.qt_runtime import configure_qt_plugins
from app.ui.image_ocr_dialog import ImageOcrDialog
from app.ui.media_review_dialog import MediaReviewDialog


@pytest.fixture(scope="module")
def qapp():
    configure_qt_plugins()
    return QApplication.instance() or QApplication([])


@pytest.fixture(params=["image", "formula", "audio"])
def review_dialog(request, qapp, repo, tmp_path):
    kind = request.param
    path = tmp_path / ("original.wav" if kind == "audio" else "original.png")
    if kind == "audio":
        path.write_bytes(b"original audio")
    else:
        image = QImage(200, 100, QImage.Format_RGB32)
        image.fill(Qt.white)
        assert image.save(str(path))
    question = Question(raw_text="existing question", media_path=str(path),
                        qtype={"image": config.QTYPE_FIGURE, "formula": config.QTYPE_FORMULA,
                               "audio": config.QTYPE_LISTENING}[kind])
    record_type = ImageOcrRecord if kind == "image" else MediaReviewRecord
    record = record_type(text="original recognition", draft="original recognition",
                         media_hash=media_hash(path))
    field = {"image": "image_ocr", "formula": "formula_ocr", "audio": "audio_transcript"}[kind]
    setattr(question, field, record)
    repo.add_questions([question])
    signal = SimpleNamespace(emit=lambda *_args: None)
    ctx = SimpleNamespace(repo=repo, questions_changed=signal, toast=signal)
    dialog = (ImageOcrDialog(ctx, question) if kind == "image" else
              MediaReviewDialog(ctx, question, kind))
    yield kind, question, record, dialog
    dialog._dirty = False
    dialog.reject()


def classify_with_draft(question, kind):
    question.classified = True
    question.labels = {config.DIM_KNOWLEDGE: "example"}
    question.feature = {"unreviewed_media": [kind]}


def test_opening_an_automatically_classified_draft_shows_pending_review(review_dialog):
    kind, question, record, dialog = review_dialog
    classify_with_draft(question, kind)
    dialog._show_record()
    assert "已使用未校对草稿分类" in dialog.message.text()
    assert "不会参与分类" not in dialog.message.text()
    assert not record.confirmed


def test_editing_used_draft_reports_invalidation_and_keeps_original_input(review_dialog):
    kind, question, record, dialog = review_dialog
    classify_with_draft(question, kind)
    dialog._show_record()
    dialog.editor.setPlainText("new correction")
    dialog.btn_save.click()
    assert not question.classified and not question.labels
    assert "旧分类已清除" in dialog.message.text()
    assert "请重新分类" in dialog.message.text()
    assert question.raw_text == "existing question"
    assert dialog.record.draft == "new correction" and not dialog.record.confirmed


def test_saving_unchanged_used_draft_keeps_classification_and_pending_status(review_dialog):
    kind, question, record, dialog = review_dialog
    classify_with_draft(question, kind)
    dialog._show_record()
    dialog.btn_save.click()
    assert question.classified
    assert "已使用未校对草稿分类" in dialog.message.text()
    assert "旧分类已清除" not in dialog.message.text()


def test_editing_confirmed_draft_explains_prior_confirmed_input_stays_effective(review_dialog):
    kind, question, record, dialog = review_dialog
    record.confirmed = True
    record.applied_text = "existing question"
    record.draft = "existing question"
    classify_with_draft(question, kind)
    dialog.record = type(record).from_dict(record.to_dict())
    dialog._show_record()
    dialog.editor.setPlainText("new unconfirmed draft")
    dialog.btn_save.click()
    assert question.classified
    assert dialog.record.confirmed and dialog.record.applied_text == "existing question"
    assert dialog.record.draft == "new unconfirmed draft"
    assert "当前草稿尚未应用" in dialog.message.text()
    assert "仍用于分类" in dialog.message.text()
    assert "旧分类已清除" not in dialog.message.text()


def test_unused_draft_explains_next_automatic_classification(review_dialog):
    _kind, question, record, dialog = review_dialog
    assert not question.classified and not record.confirmed
    dialog._show_record()
    assert "下次一键分类会使用草稿并标记待复核" in dialog.message.text()


def test_cleared_draft_is_not_replaced_by_original_recognition_on_reopen(review_dialog):
    _kind, question, record, dialog = review_dialog
    dialog.editor.clear()
    dialog.btn_save.click()
    assert dialog.record.draft == "" and dialog.record.text == "original recognition"
    dialog._show_record()
    assert dialog.editor.toPlainText() == ""
    assert "当前草稿为空" in dialog.message.text()
    assert "重新尝试识别" in dialog.message.text()

