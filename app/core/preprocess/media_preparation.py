"""Prepare media drafts for classification without changing the user's text.

Recognition runs through the isolated media job. A usable result is installed
only after rechecking both the question's attachment field and file contents.
The caller decides how to classify and persist the question afterwards.
"""
from __future__ import annotations

import hashlib
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from app import config
from app.core.models import ImageOcrRecord, MediaReviewRecord, Question
from app.core.preprocess import media_jobs


@dataclass(frozen=True)
class MediaPreparationResult:
    kind: str = ""
    recognized: bool = False
    needs_review: bool = False
    elapsed_ms: float = 0.0


_RECORD_FIELDS = {"image": "image_ocr", "formula": "formula_ocr", "audio": "audio_transcript"}


def media_kind(question: Question) -> str:
    """Choose a recognizer from the attachment, without inferring any label."""
    extension = Path(question.media_path).suffix.lower()
    if extension in config.AUDIO_EXTS:
        return "audio"
    if extension in config.IMAGE_EXTS:
        return "formula" if question.qtype == config.QTYPE_FORMULA else "image"
    return ""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _confirmed_content(record) -> bool:
    return bool(record.confirmed and any(_has_text(getattr(record, field, ""))
                                        for field in ("applied_text", "draft", "text")))


def prepare_question_media(question: Question, *,
                           should_cancel: Optional[Callable[[], bool]] = None) -> MediaPreparationResult:
    """Reuse a matching record or recognize one new, explicitly unreviewed draft.

    recognized means this call performed recognition; cache reuse returns False.
    Errors (including cancellation) leave every existing record, raw_text and
    annotation unchanged. Empty/unsupported attachments need no media work.
    """
    started = time.perf_counter()

    def cancelled():
        if should_cancel is not None and should_cancel():
            raise media_jobs.MediaJobError("媒体准备已取消，原题和已有识别结果保持不变。", cancelled=True)

    def result(kind="", recognized=False, needs_review=False):
        return MediaPreparationResult(kind, recognized, needs_review,
                                      (time.perf_counter() - started) * 1000.0)

    cancelled()
    original_path = question.media_path
    kind = media_kind(question)
    if not kind:
        return result()
    try:
        source = Path(original_path).expanduser().resolve()
        original_hash = _hash_file(source)

        def ensure_unchanged():
            if question.media_path != original_path:
                raise media_jobs.MediaJobError("题目附件已变化，请重新开始分类；已有识别记录保持不变。")
            current_hash = _hash_file(source)
            if question.media_path != original_path or current_hash != original_hash:
                raise media_jobs.MediaJobError("题目附件已变化，请重新开始分类；已有识别记录保持不变。")

        cancelled()
        if question.media_path != original_path:
            raise media_jobs.MediaJobError("题目附件已变化，请重新开始分类。")
        field = _RECORD_FIELDS[kind]
        previous = getattr(question, field)
        same_attachment = previous is not None and previous.media_hash == original_hash
        if same_attachment:
            confirmed = _confirmed_content(previous)
            usable_draft = not getattr(previous, "error", "") and _has_text(previous.draft)
            if confirmed or usable_draft:
                cancelled()
                ensure_unchanged()
                return result(kind, needs_review=not confirmed)

        # Crop coordinates belong to file contents, not merely to a path. Pass
        # a copy so a failing recognizer cannot mutate the old record's options.
        options = (deepcopy(previous.recognition_options)
                   if kind == "formula" and same_attachment else None)
        record = media_jobs.recognize_media(original_path, kind, options=deepcopy(options),
                                            should_cancel=should_cancel)
        cancelled()
        ensure_unchanged()
        expected_type = ImageOcrRecord if kind == "image" else MediaReviewRecord
        if (not isinstance(record, expected_type) or record.media_hash != original_hash
                or getattr(record, "error", "") or not _has_text(record.text)):
            raise media_jobs.MediaJobError("媒体识别没有返回与当前附件匹配的有效内容。")
        stored = deepcopy(record)
        stored.confirmed = False
        stored.applied_text = ""
        if not _has_text(stored.draft):
            stored.draft = stored.text
        # Keep options even when a compatible recognizer omits them in its
        # output. Their original object is never shared with a new record.
        if kind == "formula" and options is not None:
            stored.recognition_options = deepcopy(options)
        cancelled()
        ensure_unchanged()
        setattr(question, field, stored)
        return result(kind, recognized=True, needs_review=True)
    except media_jobs.MediaJobError:
        raise
    except Exception as exc:
        raise media_jobs.MediaJobError("媒体准备失败：{}".format(exc)) from exc
