"""Automatic media preparation preserves user work and rejects stale results."""
from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from app import config
from app.core.models import ImageOcrRecord, MediaReviewRecord, Question
from app.core.preprocess import media_jobs, media_preparation as preparation


FIELDS = {"image": "image_ocr", "formula": "formula_ocr", "audio": "audio_transcript"}


def make_question(tmp_path, kind):
    path = tmp_path / ("题目.wav" if kind == "audio" else "题目.png")
    path.write_bytes(b"original attachment")
    return Question(media_path=str(path), raw_text="用户已有题干", annotation="用户已有标注",
                    qtype=config.QTYPE_FORMULA if kind == "formula" else config.QTYPE_TEXT)


def record_for(question, kind, **values):
    data = {"text": "识别原文", "draft": "识别草稿",
            "media_hash": hashlib.sha256(b"original attachment").hexdigest()}
    data.update(values)
    return (ImageOcrRecord if kind == "image" else MediaReviewRecord)(**data)


@pytest.mark.parametrize("path,qtype,expected", [
    ("", config.QTYPE_FORMULA, ""),
    ("text.txt", config.QTYPE_LISTENING, ""),
    ("IMAGE.PNG", config.QTYPE_FORMULA, "formula"),
    ("IMAGE.PNG", config.QTYPE_LISTENING, "image"),
    ("LISTENING.MP3", config.QTYPE_FORMULA, "audio"),
])
def test_kind_uses_attachment_not_guessed_topic(path, qtype, expected):
    question = Question(media_path=path, qtype=qtype, source="audio")
    assert preparation.media_kind(question) == expected


def test_nonmedia_is_an_unchanged_noop(monkeypatch):
    question = Question(raw_text="普通题目", source="image", qtype=config.QTYPE_FORMULA)
    before = question.to_dict()
    monkeypatch.setattr(media_jobs, "recognize_media", lambda *args, **kwargs: pytest.fail("无附件不应识别"))
    result = preparation.prepare_question_media(question)
    assert result.kind == "" and not result.recognized and not result.needs_review
    assert result.elapsed_ms >= 0
    assert question.to_dict() == before


@pytest.mark.parametrize("kind", ["image", "formula", "audio"])
@pytest.mark.parametrize("confirmed", [False, True])
def test_matching_usable_record_is_reused_without_recognition(tmp_path, monkeypatch, kind, confirmed):
    question = make_question(tmp_path, kind)
    record = record_for(question, kind, confirmed=confirmed)
    setattr(question, FIELDS[kind], record)
    before = question.to_dict()
    monkeypatch.setattr(media_jobs, "recognize_media", lambda *args, **kwargs: pytest.fail("缓存不应重复识别"))
    result = preparation.prepare_question_media(question)
    assert result.kind == kind and not result.recognized
    assert result.needs_review is not confirmed
    assert getattr(question, FIELDS[kind]) is record
    assert question.to_dict() == before


def test_confirmed_applied_content_survives_failed_draft_retry(tmp_path, monkeypatch):
    question = make_question(tmp_path, "formula")
    question.formula_ocr = record_for(question, "formula", text="", draft="", confirmed=True,
                                       applied_text=r"\int x dx", error="后续重试失败")
    monkeypatch.setattr(media_jobs, "recognize_media", lambda *args, **kwargs: pytest.fail("已应用内容有效"))
    result = preparation.prepare_question_media(question)
    assert not result.recognized and not result.needs_review
    assert question.formula_ocr.error == "后续重试失败"


@pytest.mark.parametrize("kind", ["image", "formula", "audio"])
def test_new_record_is_unreviewed_and_only_corresponding_record_changes(tmp_path, monkeypatch, kind):
    question = make_question(tmp_path, kind)
    question.image_ocr = ImageOcrRecord(text="旧图片原文", draft="旧图片草稿", media_hash="stale")
    question.formula_ocr = MediaReviewRecord(text="旧公式", draft="旧公式", media_hash="stale")
    question.audio_transcript = MediaReviewRecord(text="旧音频", draft="旧音频", media_hash="stale")
    prior = question.to_dict()
    output = record_for(question, kind, confirmed=True, applied_text="不能自动标记为人工已应用")
    monkeypatch.setattr(media_jobs, "recognize_media", lambda *args, **kwargs: output)
    result = preparation.prepare_question_media(question)
    assert result.kind == kind and result.recognized and result.needs_review
    assert result.elapsed_ms >= 0
    stored = getattr(question, FIELDS[kind])
    assert not stored.confirmed and stored.applied_text == ""
    assert stored is not output and output.confirmed
    expected = deepcopy(prior)
    expected[FIELDS[kind]] = stored.to_dict()
    assert question.to_dict() == expected


@pytest.mark.parametrize("reason", ["empty", "error", "stale"])
def test_unusable_records_trigger_recognition(tmp_path, monkeypatch, reason):
    question = make_question(tmp_path, "audio")
    previous = record_for(question, "audio")
    if reason == "empty":
        previous.draft = " "
    elif reason == "error":
        previous.error = "之前识别失败"
    else:
        previous.media_hash = "old attachment"
    question.audio_transcript = previous
    calls = []

    def recognize(path, kind, **kwargs):
        calls.append((path, kind))
        return record_for(question, kind)

    monkeypatch.setattr(media_jobs, "recognize_media", recognize)
    assert preparation.prepare_question_media(question).recognized
    assert calls == [(question.media_path, "audio")]


@pytest.mark.parametrize("same_attachment", [False, True])
def test_formula_retry_retains_crop_only_for_same_hash(tmp_path, monkeypatch, same_attachment):
    question = make_question(tmp_path, "formula")
    options = {"crop": [2, 3, 40, 50], "scale": 2, "invert": True}
    previous = record_for(question, "formula", error="重试", recognition_options=deepcopy(options))
    if not same_attachment:
        previous.media_hash = "old attachment"
    question.formula_ocr = previous
    received = []

    def recognize(path, kind, **kwargs):
        received.append(deepcopy(kwargs["options"]))
        if kwargs["options"]:
            kwargs["options"]["crop"][0] = 999
        return record_for(question, kind)

    monkeypatch.setattr(media_jobs, "recognize_media", recognize)
    preparation.prepare_question_media(question)
    assert received == [options if same_attachment else None]
    assert previous.recognition_options == options
    assert question.formula_ocr.recognition_options == (options if same_attachment else {})


@pytest.mark.parametrize("failure", ["error", "cancel", "empty", "wrong_hash", "wrong_type", "unexpected"])
def test_failure_keeps_all_existing_records_and_user_text(tmp_path, monkeypatch, failure):
    question = make_question(tmp_path, "formula")
    question.image_ocr = ImageOcrRecord(text="用户的图片内容")
    question.formula_ocr = record_for(question, "formula", error="旧错误")
    question.audio_transcript = MediaReviewRecord(text="用户的音频内容")
    before = question.to_dict()

    def recognize(*args, **kwargs):
        if failure == "error":
            raise media_jobs.MediaJobError("识别失败", record=record_for(question, "formula"))
        if failure == "cancel":
            raise media_jobs.MediaJobError("取消", cancelled=True)
        if failure == "unexpected":
            raise RuntimeError("引擎异常")
        if failure == "wrong_type":
            return ImageOcrRecord(text="非公式记录")
        return record_for(question, "formula", **({"text": " "} if failure == "empty" else {"media_hash": "wrong"}))

    monkeypatch.setattr(media_jobs, "recognize_media", recognize)
    with pytest.raises(media_jobs.MediaJobError) as caught:
        preparation.prepare_question_media(question)
    assert caught.value.cancelled is (failure == "cancel")
    assert question.to_dict() == before


@pytest.mark.parametrize("change", ["question_path", "file_bytes"])
def test_inflight_attachment_change_rejects_result(tmp_path, monkeypatch, change):
    question = make_question(tmp_path, "image")
    previous = ImageOcrRecord(text="旧文字", media_hash="old")
    question.image_ocr = previous

    def recognize(path, kind, **kwargs):
        output = record_for(question, kind)
        if change == "question_path":
            alternate = tmp_path / "另一张.png"
            alternate.write_bytes(b"original attachment")
            question.media_path = str(alternate)
        else:
            from pathlib import Path
            Path(question.media_path).write_bytes(b"modified")
        return output

    monkeypatch.setattr(media_jobs, "recognize_media", recognize)
    with pytest.raises(media_jobs.MediaJobError, match="附件已变化"):
        preparation.prepare_question_media(question)
    assert question.image_ocr is previous
    assert question.raw_text == "用户已有题干" and question.annotation == "用户已有标注"


def test_cancel_after_job_return_does_not_install_result(tmp_path, monkeypatch):
    question = make_question(tmp_path, "image")
    before = question.to_dict()
    finished = False

    def recognize(*args, **kwargs):
        nonlocal finished
        finished = True
        return record_for(question, "image")

    monkeypatch.setattr(media_jobs, "recognize_media", recognize)
    with pytest.raises(media_jobs.MediaJobError) as caught:
        preparation.prepare_question_media(question, should_cancel=lambda: finished)
    assert caught.value.cancelled
    assert question.to_dict() == before


def test_cancel_before_job_does_not_touch_question(tmp_path, monkeypatch):
    question = make_question(tmp_path, "image")
    before = question.to_dict()
    monkeypatch.setattr(media_jobs, "recognize_media", lambda *args, **kwargs: pytest.fail("已取消"))
    with pytest.raises(media_jobs.MediaJobError) as caught:
        preparation.prepare_question_media(question, should_cancel=lambda: True)
    assert caught.value.cancelled
    assert question.to_dict() == before


def test_path_change_during_cached_hash_validation_is_detected(tmp_path, monkeypatch):
    question = make_question(tmp_path, "image")
    previous = record_for(question, "image")
    question.image_ocr = previous
    original_hash = preparation._hash_file
    calls = 0

    def hash_and_change(path):
        nonlocal calls
        digest = original_hash(path)
        calls += 1
        if calls == 2:
            question.media_path = str(tmp_path / "新的附件.png")
        return digest

    monkeypatch.setattr(preparation, "_hash_file", hash_and_change)
    monkeypatch.setattr(media_jobs, "recognize_media", lambda *args, **kwargs: pytest.fail("复用缓存不应识别"))
    with pytest.raises(media_jobs.MediaJobError, match="附件已变化"):
        preparation.prepare_question_media(question)
    assert question.image_ocr is previous
