"""Run automatic media classification through the real worker and feature stack.

Only media model inference is mocked. Scheme indexing, preprocessing, matching,
question updates, persistence and exports use their production implementations.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import openpyxl
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from app import config
from app.core.classify.engine import ClassificationEngine
from app.core.dataio import exporter
from app.core.models import CategoryLabel, Dimension, ImageOcrRecord, MediaReviewRecord, Question, Scheme
from app.core.preprocess import media_jobs
from app.core.storage import Repository
from app.qt_runtime import configure_qt_plugins
from app.ui.workers import ClassifyWorker


KINDS = {
    "image": {"suffix": ".png", "subject": config.SUBJECT_PHYSICS, "qtype": config.QTYPE_FIGURE,
              "text": "串联电路中电阻 R = 10 Ω，电流 I = 2 A，根据欧姆定律求电压。",
              "target": "电路计算", "keywords": ["欧姆定律", "电阻", "电流", "电压"],
              "other": "热力学", "other_keywords": ["温度", "理想气体", "热量"],
              "field": "image_ocr", "title": "图片文字"},
    "formula": {"suffix": ".png", "subject": config.SUBJECT_MATH, "qtype": config.QTYPE_FORMULA,
                "text": r"\int_0^1 x^2 dx", "target": "定积分", "keywords": ["定积分", "积分"],
                "other": "无穷级数", "other_keywords": ["级数", "收敛"],
                "field": "formula_ocr", "title": "公式"},
    "audio": {"suffix": ".wav", "subject": config.SUBJECT_ENGLISH, "qtype": config.QTYPE_LISTENING,
              "text": "Passengers are waiting for the train at the railway station.",
              "target": "交通出行", "keywords": ["train", "railway", "station", "passenger"],
              "other": "天气预报", "other_keywords": ["weather", "rain", "cloud"],
              "field": "audio_transcript", "title": "听力音频"},
}


@pytest.fixture(scope="module", autouse=True)
def qapp():
    configure_qt_plugins()
    return QApplication.instance() or QApplication([])


def make_scheme(kind, *, subject=None):
    spec = KINDS[kind]
    return Scheme(name="自动识别集成-" + kind, subject=subject or spec["subject"], dims=[
        Dimension(key=config.DIM_KNOWLEDGE, name="知识点", labels=[
            CategoryLabel(id="target-" + kind, name=spec["target"], dim=config.DIM_KNOWLEDGE,
                          keywords=spec["keywords"]),
            CategoryLabel(id="other-" + kind, name=spec["other"], dim=config.DIM_KNOWLEDGE,
                          keywords=spec["other_keywords"]),
        ])])


def make_question(tmp_path, kind, *, suffix="", subject=None, qtype=None):
    spec = KINDS[kind]
    source = tmp_path / ("原始附件-" + kind + suffix + spec["suffix"])
    # The bytes only identify an attachment. They never reach an image/audio
    # decoder because recognize_media is the sole mocked inference boundary.
    source.write_bytes(("attachment-" + kind + suffix).encode("utf-8"))
    return Question(media_path=str(source), subject=subject or spec["subject"],
                    qtype=qtype or spec["qtype"], raw_text="预录题干保留", annotation="人工批注保留")


def recognized_record(path, kind, text=None):
    cls = ImageOcrRecord if kind == "image" else MediaReviewRecord
    content = text or KINDS[kind]["text"]
    return cls(text=content, draft=content, media_hash=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
               elapsed_ms=321.5, engine="mock inference only")


def run_worker(scheme, questions, engine=None):
    worker = ClassifyWorker(engine or ClassificationEngine(scheme), scheme, questions)
    result = {"finished": [], "done": [], "failed": [], "progress": []}
    worker.finished_all.connect(lambda ok, elapsed, errors: result["finished"].append((ok, elapsed, errors)))
    worker.one_done.connect(lambda question, classification: result["done"].append((question, classification)))
    worker.failed.connect(result["failed"].append)
    worker.progress.connect(lambda done, total, text: result["progress"].append((done, total, text)))
    worker.run()
    assert len(result["finished"]) == 1
    assert not result["failed"]
    return worker, result


def assert_media_features(question, kind):
    feature = question.feature_bundle()
    assert feature is not None
    assert feature.unreviewed_media == [kind]
    assert feature.media_elapsed_ms > 0
    assert feature.total_process_ms >= feature.media_elapsed_ms
    assert any("未校对" in note for note in feature.notes)
    if kind == "image":
        assert "欧姆定律" in feature.text_tokens
        assert feature.ocr_text == KINDS[kind]["text"]
        assert feature.figure_parameters
        assert any(parameter["source"] == "automatic_ocr" for parameter in feature.figure_parameters)
    elif kind == "formula":
        assert feature.latex == KINDS[kind]["text"]
        assert "定积分" in feature.formula_tokens
    else:
        assert feature.transcript == KINDS[kind]["text"]
        assert {"train", "railway", "station", "passenger"} <= set(feature.text_tokens)


@pytest.mark.parametrize("kind", ["image", "formula", "audio"])
def test_worker_recognizes_then_classifies_real_features_and_reuses_draft(repo, tmp_path, monkeypatch, kind):
    scheme = make_scheme(kind)
    question = make_question(tmp_path, kind)
    original = (question.raw_text, question.annotation)
    calls = []

    def recognize(path, actual_kind, **kwargs):
        calls.append((path, actual_kind))
        assert not kwargs["should_cancel"]()
        return recognized_record(path, actual_kind)

    monkeypatch.setattr(media_jobs, "recognize_media", recognize)
    worker, result = run_worker(scheme, [question])
    assert result["finished"][0][0] == 1 and result["finished"][0][2] == []
    assert len(result["done"]) == 1 and result["done"][0][0] is question
    assert question.classified and question.labels[config.DIM_KNOWLEDGE] == "target-" + kind
    assert (question.raw_text, question.annotation) == original
    review = getattr(question, KINDS[kind]["field"])
    assert not review.confirmed and review.applied_text == ""
    assert_media_features(question, kind)
    labels_before = dict(question.labels)
    _, repeated = run_worker(scheme, [question], worker.engine)
    assert repeated["finished"][0][0] == 1 and not repeated["finished"][0][2]
    assert calls == [(question.media_path, kind)]
    assert question.labels == labels_before
    assert question.feature_bundle().unreviewed_media == [kind]
    assert question.feature_bundle().media_elapsed_ms == 0
    assert question.feature_bundle().total_process_ms > 0
    assert (question.raw_text, question.annotation) == original


def test_batch_recognition_failure_keeps_original_item_and_continues(repo, tmp_path, monkeypatch):
    scheme = make_scheme("image")
    failed = make_question(tmp_path, "image", suffix="-bad")
    failed.image_ocr = ImageOcrRecord(text="旧内容保留", media_hash="old attachment")
    before = failed.to_dict()
    good = make_question(tmp_path, "image", suffix="-good")
    calls = []

    def recognize(path, kind, **kwargs):
        calls.append(path)
        if path == failed.media_path:
            raise media_jobs.MediaJobError("图片模型本次失败")
        return recognized_record(path, kind)

    monkeypatch.setattr(media_jobs, "recognize_media", recognize)
    _, result = run_worker(scheme, [failed, good])
    assert result["finished"][0][0] == 1
    assert len(result["finished"][0][2]) == 1
    assert "图片模型本次失败" in result["finished"][0][2][0]
    assert failed.to_dict() == before
    assert good.classified and good.labels[config.DIM_KNOWLEDGE] == "target-image"
    assert calls == [failed.media_path, good.media_path]


def test_cancel_current_media_stops_later_items_without_modifying_them(repo, tmp_path, monkeypatch):
    scheme = make_scheme("audio")
    current = make_question(tmp_path, "audio", suffix="-cancel")
    later = make_question(tmp_path, "audio", suffix="-later")
    before = [question.to_dict() for question in (current, later)]
    worker = ClassifyWorker(ClassificationEngine(scheme), scheme, [current, later])
    calls, finished, done = [], [], []
    worker.finished_all.connect(lambda ok, elapsed, errors: finished.append((ok, errors)))
    worker.one_done.connect(lambda *args: done.append(args))

    def recognize(path, kind, **kwargs):
        calls.append(path)
        worker.stop()
        assert kwargs["should_cancel"]()
        raise media_jobs.MediaJobError("用户取消", cancelled=True)

    monkeypatch.setattr(media_jobs, "recognize_media", recognize)
    worker.run()
    assert calls == [current.media_path]
    assert finished == [(0, [])] and done == []
    assert [question.to_dict() for question in (current, later)] == before


@pytest.mark.parametrize("kind", ["image", "formula", "audio"])
def test_confirmed_applied_content_wins_over_newer_unconfirmed_draft(repo, tmp_path, monkeypatch, kind):
    scheme = make_scheme(kind)
    question = make_question(tmp_path, kind)
    record = recognized_record(question.media_path, kind)
    record.confirmed = True
    record.applied_text = record.text
    record.draft = (r"\sum_{n=1}^{\infty} n" if kind == "formula" else
                    "weather rain cloud" if kind == "audio" else "理想气体温度热量")
    if kind == "image":
        # Applied image text is the user's saved question body.
        question.raw_text = record.applied_text
    setattr(question, KINDS[kind]["field"], record)
    original_text = question.raw_text
    monkeypatch.setattr(media_jobs, "recognize_media", lambda *args, **kwargs: pytest.fail("已确认内容不应重识别"))
    _, result = run_worker(scheme, [question])
    assert result["finished"][0][0] == 1 and not result["finished"][0][2]
    assert question.labels[config.DIM_KNOWLEDGE] == "target-" + kind
    feature = question.feature_bundle()
    assert feature.unreviewed_media == [] and feature.media_elapsed_ms == 0
    actual = feature.latex if kind == "formula" else feature.transcript if kind == "audio" else feature.ocr_text
    assert actual == record.applied_text and actual != record.draft
    assert question.raw_text == original_text
    assert getattr(question, KINDS[kind]["field"]) is record


def test_general_subject_audio_suffix_extracts_transcription_tokens_for_matching(repo, tmp_path, monkeypatch):
    scheme = make_scheme("audio", subject=config.SUBJECT_GENERAL)
    question = make_question(tmp_path, "audio", subject=config.SUBJECT_GENERAL, qtype=config.QTYPE_TEXT)
    question.raw_text = ""
    question.annotation = ""
    monkeypatch.setattr(media_jobs, "recognize_media", lambda path, kind, **kwargs: recognized_record(path, kind))
    _, result = run_worker(scheme, [question])
    assert result["finished"][0][0] == 1 and not result["finished"][0][2]
    assert question.subject == config.SUBJECT_GENERAL and question.qtype == config.QTYPE_TEXT
    assert question.labels[config.DIM_KNOWLEDGE] == "target-audio"
    assert_media_features(question, "audio")
    assert question.raw_text == "" and question.annotation == ""


@pytest.mark.parametrize("kind", ["image", "formula", "audio"])
def test_auto_result_roundtrip_and_real_exports_preserve_unreviewed_flags_and_timings(
        repo, tmp_path, monkeypatch, kind):
    scheme = make_scheme(kind)
    question = make_question(tmp_path, kind)
    monkeypatch.setattr(media_jobs, "recognize_media", lambda path, kind, **kwargs: recognized_record(path, kind))
    _, result = run_worker(scheme, [question])
    assert result["finished"][0][0] == 1
    repo.schemes.append(scheme)
    repo.add_questions([question])
    repo.save_all()
    saved = question.to_dict()
    loaded = Repository().question_by_id(question.id)
    assert loaded.to_dict() == saved
    assert not getattr(loaded, KINDS[kind]["field"]).confirmed
    output = exporter.export_excel([loaded], scheme, str(tmp_path / (kind + ".xlsx")))
    workbook = openpyxl.load_workbook(output)
    rows = list(workbook["全部错题"].iter_rows(values_only=True))
    record = dict(zip(rows[0], rows[1]))
    assert record["题干"] == loaded.raw_text
    assert record["分类所用媒体待校对"] == KINDS[kind]["title"]
    assert record["本次媒体处理耗时(ms)"] == pytest.approx(loaded.feature["media_elapsed_ms"])
    assert record["本次自动处理总耗时(ms)"] == pytest.approx(loaded.feature["total_process_ms"])
    assert record["耗时(ms)"] == round(loaded.elapsed_ms, 1)
    archive, _ = exporter.export_media_archive([loaded], scheme, str(tmp_path / (kind + "-archive")))
    with (Path(archive) / "index.csv").open(encoding="utf-8-sig", newline="") as stream:
        archived_row = next(csv.DictReader(stream))
    assert archived_row["分类所用媒体待校对"] == KINDS[kind]["title"]
    assert float(archived_row["本次媒体处理耗时(ms)"]) == loaded.feature["media_elapsed_ms"]
    assert float(archived_row["本次自动处理总耗时(ms)"]) == loaded.feature["total_process_ms"]
    records = json.loads((Path(archive) / "records.json").read_text(encoding="utf-8"))
    assert records["records"][0]["question"]["feature"] == loaded.feature
    assert records["records"][0]["question"][KINDS[kind]["field"]]["confirmed"] is False
