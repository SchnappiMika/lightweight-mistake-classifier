from app.core.models import MediaReviewRecord, Question
from app.core.preprocess.media_recognition import media_hash, save_media_review


class Repo:
    def __init__(self, question):
        self.question = question
        self.saved = 0

    def question_by_id(self, question_id):
        return self.question if question_id == self.question.id else None

    def save_questions(self):
        self.saved += 1


def test_media_review_draft_does_not_confirm(tmp_path):
    media = tmp_path / "formula.png"
    media.write_bytes(b"image")
    question = Question(media_path=str(media))
    repo = Repo(question)
    record = MediaReviewRecord(text="x", draft="x", media_hash=media_hash(media))
    save_media_review(repo, question, record, "formula", apply=False)
    assert question.formula_ocr is not None
    assert not question.formula_ocr.confirmed
    assert repo.saved == 1


def test_media_review_apply_sets_confirmed_content(tmp_path):
    media = tmp_path / "audio.wav"
    media.write_bytes(b"audio")
    question = Question(media_path=str(media))
    repo = Repo(question)
    record = MediaReviewRecord(text="old", draft="confirmed", media_hash=media_hash(media))
    save_media_review(repo, question, record, "audio", apply=True)
    assert question.audio_transcript.confirmed
    assert question.audio_transcript.applied_text == "confirmed"
    assert question.raw_text == ""


def test_new_draft_preserves_previous_confirmation(tmp_path):
    media = tmp_path / "formula.png"
    media.write_bytes(b"image")
    question = Question(media_path=str(media))
    repo = Repo(question)
    first = MediaReviewRecord(text="old", draft="old", media_hash=media_hash(media))
    save_media_review(repo, question, first, "formula", apply=True)
    second = MediaReviewRecord(text="new", draft="new", media_hash=media_hash(media))
    save_media_review(repo, question, second, "formula", apply=False)
    assert question.formula_ocr.confirmed
    assert question.formula_ocr.applied_text == "old"
    assert question.formula_ocr.draft == "new"
