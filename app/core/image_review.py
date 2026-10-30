from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from app import config
from app.core.preprocess import cleaner
from app.core.preprocess.image_ocr import ImageOcrError, file_hash


def save_review(repo, question, record, apply=False):
    if repo.question_by_id(question.id) is not question:
        raise ImageOcrError("当前题目已被删除或重新加载，请重新打开识别窗口。")
    path = Path(question.media_path)
    if not path.is_file() or file_hash(path) != record.media_hash:
        raise ImageOcrError("附件已变化或丢失，不能应用旧识别结果，请重新识别当前图片。")
    if apply and not record.draft.strip():
        raise ImageOcrError("校对文本为空，请填写题干后再应用。")
    previous = deepcopy(question.__dict__)
    try:
        existing = previous.get("image_ocr")
        used_draft_changed = (
            not apply and question.classified and
            "image" in (question.feature or {}).get("unreviewed_media", []) and
            existing is not None and not existing.confirmed and
            existing.draft.strip() != record.draft.strip())
        question.image_ocr = deepcopy(record)
        if not apply and existing is not None:
            if existing.confirmed:
                question.image_ocr.confirmed = True
                question.image_ocr.applied_text = existing.applied_text
        if apply:
            question.raw_text = record.draft.strip()
            question.image_ocr.draft = question.raw_text
            question.image_ocr.confirmed = True
            question.image_ocr.applied_text = question.raw_text
            question.invalidate_classification()
            if question.subject == config.SUBJECT_GENERAL:
                question.subject = cleaner.guess_subject(question.raw_text, "")
        elif used_draft_changed:
            question.invalidate_classification()
        repo.save_questions()
    except BaseException:
        question.__dict__.clear()
        question.__dict__.update(previous)
        raise
