"""Real classification and review services drive the automatic-media UI states."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys
import time
import traceback
import wave

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QInputDialog, QMessageBox, QPushButton

from app import config
from app.core.models import CategoryLabel, Dimension, MediaReviewRecord, Question, Scheme
from app.core.preprocess import media_jobs
from app.core.preprocess.media_recognition import save_media_review
from app.core.storage import Repository
from app.qt_runtime import configure_qt_plugins
from app.ui.context import AppContext
from app.ui.pages.page_classify import ClassifyPage
from app.ui.pages.page_import import ImportPage
from app.ui.pages.page_export import ExportPage
from app.ui.pages.page_feedback import FeedbackPage
from app.ui.workers import ClassifyWorker


@pytest.fixture(scope="module")
def qapp():
    configure_qt_plugins()
    return QApplication.instance() or QApplication([])


@pytest.fixture
def pages(qapp, repo, monkeypatch):
    ctx = AppContext()
    warnings = []
    def capture_qt_error(error_type, error, trace):
        warnings.append(("Qt callback exception", str(error)))
        traceback.print_exception(error_type, error, trace)
    monkeypatch.setattr(sys, "excepthook", capture_qt_error)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[1:]))
    classify = ClassifyPage(ctx)
    imports = ImportPage(ctx)
    yield ctx, classify, imports, warnings
    imports.shutdown()
    classify.shutdown()
    imports.close()
    classify.close()


def run_classification(ctx, page, warnings):
    page.btn_run.click()
    assert isinstance(page._worker, ClassifyWorker)
    deadline = time.monotonic() + 20
    while ctx.active_task and time.monotonic() < deadline:
        QTest.qWait(10)
    assert not ctx.active_task, "Real ClassifyWorker did not finish."
    assert not page._worker.isRunning()
    assert not warnings, warnings


def select_question(page, question):
    page.table.clearSelection()
    for row in range(page.table.rowCount()):
        if page.table.item(row, 0).data(Qt.UserRole) == question.id:
            page.table.selectRow(row)
            return row
    pytest.fail("Question is missing from the visible list.")


def result_titles(page):
    return [page.tree.topLevelItem(index).text(0) for index in range(page.tree.topLevelItemCount())]


@pytest.mark.parametrize("kind,subject,qtype,recognized,import_state", [
    ("formula", config.SUBJECT_MATH, config.QTYPE_FORMULA, r"\int_0^1 x^2\,dx", "公式待校对"),
    ("audio", config.SUBJECT_ENGLISH, config.QTYPE_LISTENING,
     "Listen to the conversation about a university lecture and choose the correct answer.", "听力转写待校对"),
])
def test_unreviewed_ui_filter_warning_timings_then_confirm_and_reclassify(
        pages, tmp_path, monkeypatch, kind, subject, qtype, recognized, import_state):
    ctx, classify, imports, warnings = pages
    selected_scheme = ctx.repo.scheme_by_subject(subject)
    ctx.set_current_scheme(selected_scheme.id)
    attachment = tmp_path / ("自动公式.png" if kind == "formula" else "自动听力.wav")
    if kind == "formula":
        image = QImage(500, 180, QImage.Format_RGB32)
        image.fill(Qt.white)
        assert image.save(str(attachment))
    else:
        with wave.open(str(attachment), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\0\0" * 160)
    record = MediaReviewRecord(text=recognized, draft=recognized,
                                media_hash=media_jobs._hash_file(attachment))
    question = Question(subject=subject, qtype=qtype, media_path=str(attachment))
    field = "formula_ocr" if kind == "formula" else "audio_transcript"
    setattr(question, field, record)
    ordinary = Question(subject=config.SUBJECT_MATH, raw_text="求定积分的面积")
    ctx.add_questions([question, ordinary])

    def forbidden_inference(*args, **kwargs):
        raise AssertionError("Matching saved drafts and confirmed results must be reused without inference.")

    monkeypatch.setattr(media_jobs, "recognize_media", forbidden_inference)
    run_classification(ctx, classify, warnings)
    assert question.classified and ordinary.classified
    assert question.feature["unreviewed_media"] == [kind]
    assert question.feature["media_elapsed_ms"] == 0
    assert question.feature["total_process_ms"] > 0
    assert question.feature["total_process_ms"] >= question.elapsed_ms
    assert not getattr(question, field).confirmed
    assert not getattr(question, field).applied_text

    row = select_question(classify, question)
    assert classify.table.item(row, 0).text().startswith("【未校对】")
    assert "原始附件" in classify.table.item(row, 0).toolTip()
    assert "媒体识别：未校对" in result_titles(classify)
    performance = classify.lb_perf.text()
    assert "特征分类 {:.1f}ms（不含媒体识别）".format(question.elapsed_ms) in performance
    assert "本次预处理 {:.1f}ms".format(question.feature["elapsed_ms"]) in performance
    assert "本次自动处理 {:.1f}ms".format(question.feature["total_process_ms"]) in performance
    assert "本次媒体识别 0.0ms" in performance
    classify.cb_filter.setCurrentIndex(classify.cb_filter.findData("unreviewed"))
    assert classify.table.rowCount() == 1
    assert classify.table.item(0, 0).data(Qt.UserRole) == question.id
    select_question(imports, question)
    assert import_state in imports.sec_detail.hint.text()
    assert "本次分类使用未校对" in imports.sec_detail.hint.text()

    reviewed = deepcopy(getattr(question, field))
    save_media_review(ctx.repo, question, reviewed, kind, apply=True)
    assert not question.classified and question.feature is None and not question.labels
    assert getattr(question, field).confirmed
    assert getattr(question, field).applied_text == recognized
    # A later draft is distinct from the confirmed content used on reclassifying.
    later_draft = deepcopy(getattr(question, field))
    later_draft.draft = "unconfirmed replacement that must not enter classification"
    save_media_review(ctx.repo, question, later_draft, kind, apply=False)
    ctx.questions_changed.emit()
    assert classify.table.rowCount() == 0
    classify.cb_filter.setCurrentIndex(classify.cb_filter.findData("all"))
    run_classification(ctx, classify, warnings)
    assert question.classified and not question.feature["unreviewed_media"]
    assert question.feature["media_elapsed_ms"] == 0
    feature_field = "latex" if kind == "formula" else "transcript"
    assert question.feature[feature_field] == recognized
    row = select_question(classify, question)
    assert "【未校对】" not in classify.table.item(row, 0).text()
    assert "媒体识别：未校对" not in result_titles(classify)
    classify.cb_filter.setCurrentIndex(classify.cb_filter.findData("unreviewed"))
    assert classify.table.rowCount() == 0
    select_question(imports, question)
    assert import_state not in imports.sec_detail.hint.text()
    assert "已确认" in imports.sec_detail.hint.text()
    persisted = Repository().question_by_id(question.id)
    assert persisted.classified and not persisted.feature["unreviewed_media"]


def test_promote_ui_chooses_actual_custom_dimension_and_respects_cancel(pages, monkeypatch):
    ctx, classify, _imports, warnings = pages
    first = CategoryLabel(name="定积分复习", dim="course_topic", keywords=["定积分", "积分"])
    second = CategoryLabel(name="换元法训练", dim="practice_method", keywords=["换元法", "代换法"])
    selected = Scheme(name="无knowledge维度的自定义体系", subject=config.SUBJECT_MATH,
                      dims=[Dimension(key="course_topic", name="课程主题", labels=[first]),
                            Dimension(key="practice_method", name="练习方法", labels=[second])])
    ctx.repo.add_scheme(selected)
    ctx.set_current_scheme(selected.id)
    question = Question(subject=config.SUBJECT_MATH, raw_text="使用换元法和代换法计算定积分")
    ctx.add_questions([question])
    run_classification(ctx, classify, warnings)
    assert selected.dim(config.DIM_KNOWLEDGE) is None
    assert question.labels == {"course_topic": first.id, "practice_method": second.id}
    select_question(classify, question)
    assert classify.table.item(classify.table.currentRow(), 2).text() == first.name
    assert classify.table.item(classify.table.currentRow(), 3).text() == "{:.2f}".format(
        question.scores["course_topic"])
    choices_seen = []
    response = [1, False]

    def choose(parent, title, prompt, choices, default, editable):
        choices_seen.append(list(choices))
        assert "目标分类维度" in prompt and not editable
        return choices[response[0]], response[1]

    monkeypatch.setattr(QInputDialog, "getItem", choose)
    classify._promote()
    assert not first.samples and not second.samples
    response[:] = [1, True]
    classify._promote()
    assert second.samples == [question.raw_text] and not first.samples
    assert choices_seen[-1] == ["课程主题：定积分复习", "练习方法：换元法训练"]
    persisted = Repository().scheme_by_id(selected.id)
    assert persisted.dim("practice_method").labels[0].samples == [question.raw_text]
    assert not persisted.dim("course_topic").labels[0].samples
    response[:] = [0, True]
    classify._promote()
    assert first.samples == [question.raw_text]


def test_export_filters_and_feedback_refresh_accept_actual_qt_signal_arguments(pages):
    ctx, _classify, _imports, warnings = pages
    selected = ctx.current_scheme()
    dimension = next(dim for dim in selected.dims if dim.enabled and dim.labels)
    label = dimension.labels[0]
    done = Question(subject=selected.subject, raw_text="已分类的积分错题", scheme_id=selected.id,
                    classified=True, labels={dimension.key: label.id},
                    label_names={dimension.key: label.name})
    pending = Question(subject=selected.subject, raw_text="已处理但没有匹配标签", scheme_id=selected.id,
                       feature={"text_tokens": []}, classified=False)
    ctx.add_questions([done, pending])
    exports = ExportPage(ctx)
    feedback = FeedbackPage(ctx)
    try:
        assert exports.tree.topLevelItem(0).text(1) == "1"
        target = next(dim for dim in selected.enabled_dims()
                      if dim.labels and dim.key != exports.cb_dim.currentData())
        exports.cb_dim.setCurrentIndex(exports.cb_dim.findData(target.key))
        root = exports.tree.topLevelItem(0)
        assert any(root.child(index).text(0).startswith(target.name + "-")
                   for index in range(root.childCount()))
        exports.ck_only_done.setChecked(False)
        assert exports.tree.topLevelItem(0).text(1) == "2"
        exports.ck_filter_subject.setChecked(False)
        exports.ck_filter_subject.setChecked(True)
        exports.ck_only_done.setChecked(True)
        assert exports.tree.topLevelItem(0).text(1) == "1"
        export_refresh = next(widget for widget in exports.quick_actions()[1]
                              if isinstance(widget, QPushButton) and widget.text() == "刷新预览")
        export_refresh.click()
        feedback_refresh = next(widget for widget in feedback.quick_actions()[1]
                                if isinstance(widget, QPushButton) and widget.text() == "刷新")
        feedback_refresh.click()
        assert not warnings, warnings
    finally:
        exports.shutdown()
        feedback.shutdown()
        exports.close()
        feedback.close()
