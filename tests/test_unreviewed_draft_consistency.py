"""Saved drafts invalidate only classifications that actually used those drafts."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from app import config
from app.core.image_review import save_review
from app.core.models import ImageOcrRecord, MediaReviewRecord, Question
from app.core.preprocess.media_recognition import media_hash, save_media_review
from app.core.storage import Repository


ATTRIBUTES = {"image": "image_ocr", "formula": "formula_ocr", "audio": "audio_transcript"}
CLASSIFICATION_FIELDS = ("labels", "label_names", "scores", "corrected", "candidates",
                         "scheme_id", "feature", "classified", "elapsed_ms", "updated_at")


def save(kind, repo, question, record, apply=False):
    if kind == "image":
        save_review(repo, question, record, apply=apply)
    else:
        save_media_review(repo, question, record, kind, apply=apply)


def classification(question):
    return {key: deepcopy(getattr(question, key)) for key in CLASSIFICATION_FIELDS}


@pytest.fixture(params=["image", "formula", "audio"])
def classified_draft(request, repo, tmp_path):
    kind = request.param
    attachment = tmp_path / ("question.wav" if kind == "audio" else "question.png")
    attachment.write_bytes(b"unchanged original attachment")
    scheme = repo.schemes[0]
    dim = next(dim for dim in scheme.dims if dim.enabled and dim.labels)
    label = dim.labels[0]
    question = Question(raw_text="original question text", subject=scheme.subject,
                        media_path=str(attachment), classified=True, scheme_id=scheme.id,
                        labels={dim.key: label.id}, label_names={dim.key: label.name},
                        scores={dim.key: 0.81}, corrected={dim.key: True},
                        candidates={dim.key: [{"label_id": label.id, "label_name": label.name}]},
                        feature={"clean_text": "original draft", "unreviewed_media": [kind]},
                        elapsed_ms=12.5)
    record_class = ImageOcrRecord if kind == "image" else MediaReviewRecord
    record = record_class(text="original machine output", draft="original draft",
                          media_hash=media_hash(attachment), confirmed=False)
    setattr(question, ATTRIBUTES[kind], record)
    repo.add_questions([question])
    return kind, repo, question


@pytest.mark.parametrize("new_draft", ["edited meaningful content", " \n\t "])
def test_changing_or_removing_classification_input_invalidates_and_persists(classified_draft, new_draft):
    kind, repo, question = classified_draft
    previous_text = question.raw_text
    edited = deepcopy(getattr(question, ATTRIBUTES[kind]))
    edited.draft = new_draft
    save(kind, repo, question, edited)
    assert not question.classified and question.feature is None and not question.scheme_id
    assert not question.labels and not question.scores and not question.corrected and not question.candidates
    assert question.raw_text == previous_text
    stored = getattr(question, ATTRIBUTES[kind])
    assert stored.draft == new_draft and not stored.confirmed
    loaded = Repository().question_by_id(question.id)
    assert not loaded.classified and loaded.feature is None
    assert getattr(loaded, ATTRIBUTES[kind]).draft == new_draft


@pytest.mark.parametrize("same_draft", ["original draft", " \n original draft \n "])
def test_unchanged_effective_draft_keeps_classification_and_saved_text(classified_draft, same_draft):
    kind, repo, question = classified_draft
    before = classification(question)
    edited = deepcopy(getattr(question, ATTRIBUTES[kind]))
    edited.draft = same_draft
    save(kind, repo, question, edited)
    assert classification(question) == before
    loaded = Repository().question_by_id(question.id)
    assert classification(loaded) == before
    assert getattr(loaded, ATTRIBUTES[kind]).draft == same_draft


def test_editing_draft_preserves_classification_using_previously_confirmed_content(classified_draft):
    kind, repo, question = classified_draft
    existing = getattr(question, ATTRIBUTES[kind])
    existing.confirmed = True
    existing.applied_text = "previous confirmed input"
    # Even a stale unreviewed flag must not override a real prior confirmation.
    before = classification(question)
    edited = deepcopy(existing)
    edited.confirmed = False
    edited.applied_text = ""
    edited.draft = "new unconfirmed draft"
    save(kind, repo, question, edited)
    stored = getattr(question, ATTRIBUTES[kind])
    assert classification(question) == before
    assert stored.confirmed and stored.applied_text == "previous confirmed input"
    assert stored.draft == "new unconfirmed draft"


@pytest.mark.parametrize("flags", [[], ["different_media_kind"]])
def test_editing_unused_unconfirmed_draft_does_not_invalidate(classified_draft, flags):
    kind, repo, question = classified_draft
    question.feature["unreviewed_media"] = flags
    before = classification(question)
    edited = deepcopy(getattr(question, ATTRIBUTES[kind]))
    edited.draft = "unused new draft"
    save(kind, repo, question, edited)
    assert classification(question) == before


def test_failed_save_restores_review_and_complete_classification(classified_draft, monkeypatch):
    kind, repo, question = classified_draft
    before = question.to_dict()
    disk_before = config.QUESTION_FILE.read_bytes()
    edited = deepcopy(getattr(question, ATTRIBUTES[kind]))
    edited.draft = "would invalidate if saved"
    incoming_before = edited.to_dict()

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(repo, "save_questions", fail)
    with pytest.raises(OSError, match="disk full"):
        save(kind, repo, question, edited)
    assert question.to_dict() == before
    assert edited.to_dict() == incoming_before
    assert config.QUESTION_FILE.read_bytes() == disk_before
    assert Repository().question_by_id(question.id).to_dict() == before


def test_explicit_apply_still_invalidates_and_confirms(classified_draft):
    kind, repo, question = classified_draft
    question.feature["unreviewed_media"] = []
    edited = deepcopy(getattr(question, ATTRIBUTES[kind]))
    edited.draft = "  explicitly reviewed content  "
    save(kind, repo, question, edited, apply=True)
    stored = getattr(question, ATTRIBUTES[kind])
    assert stored.confirmed and stored.applied_text == "explicitly reviewed content"
    assert not question.classified and question.feature is None
    if kind == "image":
        assert question.raw_text == "explicitly reviewed content"
    else:
        assert question.raw_text == "original question text"
