from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from app import config
from app.qt_runtime import configure_qt_plugins
from app.core.classify.engine import ClassificationEngine
from app.core.classify.feedback import FeedbackOptimizer
from app.core.dataio import exporter
from app.core.models import CategoryLabel, Dimension, Question, Scheme
from app.core.preprocess import pipeline
from app.core.storage import Repository, read_json
from app.ui.context import AppContext
from app.ui.main_window import MainWindow
from app.ui.workers import ClassifyWorker
from app.ui.workers import ImportWorker


@pytest.fixture(scope="module")
def qapp():
    configure_qt_plugins()
    return QApplication.instance() or QApplication([])


def wait_image_worker(dialog):
    deadline = time.monotonic() + 5
    while dialog._busy and time.monotonic() < deadline:
        QTest.qWait(10)
    assert not dialog._busy


@pytest.fixture
def image_dialog(window, tmp_path):
    from PyQt5.QtGui import QImage
    from app.ui.image_ocr_dialog import ImageOcrDialog

    path = tmp_path / "待识别题目.png"
    image = QImage(500, 300, QImage.Format_RGB32)
    image.fill(Qt.white)
    assert image.save(str(path))
    question = classified_question(window.ctx)
    question.media_path = window.ctx.repo.import_media(str(path))
    window.ctx.repo.save_questions()
    dialog = ImageOcrDialog(window.ctx, question, window.pages[0])
    dialog.show()
    yield dialog
    if dialog._worker is not None and dialog._worker.isRunning():
        dialog._worker.wait(5000)
        QTest.qWait(30)
    dialog._dirty = False
    dialog.reject()


def fake_image_record(dialog):
    from app.core.models import ImageOcrRecord
    from app.core.preprocess.image_ocr import file_hash

    return ImageOcrRecord(text="计算定积份", draft="计算定积份", elapsed_ms=300,
                          media_hash=file_hash(Path(dialog.question.media_path)),
                          lines=[{"text": "计算定积份", "score": 0.75, "box": []}])


def test_image_dialog_recognizes_as_draft_then_explicitly_applies(image_dialog, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    from app.ui import image_ocr_dialog

    dialog = image_dialog
    before = dialog.question.raw_text
    monkeypatch.setattr(image_ocr_dialog.get_engine(), "recognize_isolated", lambda path: fake_image_record(dialog))
    dialog.btn_recognize.click()
    wait_image_worker(dialog)
    assert dialog.question.raw_text == before and dialog.question.classified
    assert dialog.question.image_ocr.text == "计算定积份" and not dialog.question.image_ocr.confirmed
    assert dialog.editor.toPlainText() == "计算定积份" and "不是准确率" in dialog.original.toPlainText()
    assert not dialog.ctx.active_task and dialog.btn_apply.isEnabled()
    dialog.editor.setPlainText("用换元积分法计算定积分")
    dialog.btn_save.click()
    assert dialog.question.raw_text == before and dialog.question.classified
    assert dialog.question.image_ocr.draft == "用换元积分法计算定积分"
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.No)
    dialog.btn_apply.click()
    assert dialog.question.raw_text == before and dialog.question.classified
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    dialog.btn_apply.click()
    assert dialog.question.raw_text == "用换元积分法计算定积分" and not dialog.question.classified
    assert dialog.question.image_ocr.text == "计算定积份" and dialog.question.image_ocr.confirmed
    assert Repository().question_by_id(dialog.question.id).raw_text == dialog.question.raw_text


@pytest.mark.parametrize("cancel", [False, True])
def test_image_recognition_keeps_ui_responsive_and_preserves_on_failure_or_cancel(image_dialog, monkeypatch, cancel):
    import threading
    from PyQt5.QtCore import QTimer
    from app.ui import image_ocr_dialog

    dialog = image_dialog
    before = dialog.question.to_dict()
    gate = threading.Event()
    def recognize(path):
        if not gate.wait(5):
            raise ValueError("测试等待超时")
        if not cancel:
            raise ValueError("图片损坏，无法识别")
        return fake_image_record(dialog)
    monkeypatch.setattr(image_ocr_dialog.get_engine(), "recognize_isolated", recognize)
    timer_ticks = []
    try:
        dialog.btn_recognize.click()
        QTimer.singleShot(0, lambda: timer_ticks.append(True))
        QTest.qWait(30)
        assert timer_ticks and dialog._busy and dialog.isEnabled()
        assert dialog.ctx.active_task == "图片识别" and dialog.btn_cancel.isEnabled()
        assert not dialog.ctx.begin_task("备份")
        assert not dialog.close() and dialog.isVisible()
        if cancel:
            dialog.btn_cancel.click()
    finally:
        gate.set()
        wait_image_worker(dialog)
    assert dialog.question.to_dict() == before and not dialog.ctx.active_task
    assert ("已取消" if cancel else "图片损坏") in dialog.message.text()
    assert dialog.btn_recognize.isEnabled()


def test_image_dialog_retains_unsaved_text_on_storage_failure_and_close_cancel(image_dialog, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    from app.ui import image_ocr_dialog

    dialog = image_dialog
    monkeypatch.setattr(image_ocr_dialog.get_engine(), "recognize_isolated", lambda path: fake_image_record(dialog))
    dialog._start()
    wait_image_worker(dialog)
    before = dialog.question.to_dict()
    def fail_save():
        raise OSError("磁盘写入失败")
    monkeypatch.setattr(dialog.ctx.repo, "save_questions", fail_save)
    dialog.editor.setPlainText("保留这段校对草稿")
    dialog.btn_save.click()
    assert "磁盘写入失败" in dialog.message.text() and dialog._dirty
    assert dialog.editor.toPlainText() == "保留这段校对草稿" and dialog.question.to_dict() == before
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Cancel)
    assert not dialog.close() and dialog.isVisible()
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Save)
    assert not dialog.close() and dialog.isVisible()
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Discard)
    assert dialog.close() and not dialog.isVisible()
    assert dialog.question.to_dict() == before


@pytest.fixture
def window(qapp, repo):
    window = MainWindow()
    window.ctx.set_setting("guide_done", True)
    window.overlay.hide()
    yield window
    window.close()


def classified_question(ctx):
    scheme = ctx.repo.scheme_by_subject(config.SUBJECT_MATH)
    question = Question(subject=config.SUBJECT_MATH, raw_text="用牛顿莱布尼茨公式计算定积分")
    bundle = pipeline.preprocess(question)
    result = ctx.engine(scheme).classify(question, bundle)
    ClassificationEngine.apply_result(question, result, scheme, bundle)
    ctx.add_questions([question])
    return question


def custom_math_scheme(ctx):
    scheme = Scheme(name="我的高数体系", subject=config.SUBJECT_MATH, dims=[
        Dimension(labels=[CategoryLabel(name="积分专题", keywords=["积分", "定积分"])])])
    ctx.repo.add_scheme(scheme)
    return scheme


@pytest.mark.parametrize("query,expected", [
    ("积分", [0]), ("法向力", [1]), ("听力.mp3", [2]), ("TAYLOR", [3]),
    ("隐藏目录", []), ("不存在", []), (".*", []), ("   ", [0, 1, 2, 3])])
def test_search_matches_question_annotation_or_attachment_name(window, query, expected):
    items = [
        Question(raw_text="求定积分", subject=config.SUBJECT_MATH),
        Question(raw_text="计算摩擦力", annotation="漏画法向力", subject=config.SUBJECT_PHYSICS),
        Question(raw_text="", media_path=r"C:\资料\隐藏目录\英语听力.MP3",
                 subject=config.SUBJECT_ENGLISH, qtype=config.QTYPE_LISTENING),
        Question(raw_text="Taylor expansion", subject=config.SUBJECT_MATH)]
    window.ctx.add_questions(items)
    page = window.pages[0]
    page.ed_search.setText(query)
    assert page._filtered() == [items[index] for index in expected]
    assert page.table.rowCount() == len(expected)


def test_search_combines_with_subject_and_question_type(window):
    items = [Question(raw_text="计算积分", subject=config.SUBJECT_MATH, qtype=config.QTYPE_FORMULA),
             Question(raw_text="计算积分", subject=config.SUBJECT_MATH, qtype=config.QTYPE_TEXT),
             Question(raw_text="计算积分", subject=config.SUBJECT_PHYSICS, qtype=config.QTYPE_FORMULA)]
    window.ctx.add_questions(items)
    page = window.pages[0]
    page.ed_search.setText("积分")
    page.cb_filter_subject.setCurrentIndex(page.cb_filter_subject.findData(config.SUBJECT_MATH))
    page.cb_filter_qtype.setCurrentIndex(page.cb_filter_qtype.findData(config.QTYPE_FORMULA))
    assert page._filtered() == [items[0]] and page.table.rowCount() == 1
    page.ed_search.clear()
    assert page._filtered() == [items[0]]
    page.cb_filter_qtype.setCurrentIndex(0)
    assert page._filtered() == items[:2]


def test_clearing_search_restores_list_without_changing_saved_questions(window):
    window.ctx.add_questions([Question(raw_text="求极限"), Question(raw_text="求导数")])
    original = config.QUESTION_FILE.read_bytes()
    page = window.pages[0]
    page.ed_search.setText("没有结果")
    assert page.table.rowCount() == 0 and page._current is None
    assert not page.ed_text.isEnabled()
    page.btn_clear_search.click()
    assert page.table.rowCount() == 2
    assert config.QUESTION_FILE.read_bytes() == original


def test_editing_search_result_updates_list_without_interrupting_editor(window):
    question = Question(raw_text="求定积分")
    window.ctx.add_questions([question])
    page = window.pages[0]
    page.ed_search.setText("积分")
    page.table.selectRow(0)
    page.ed_text.setPlainText("这道题已修改")
    assert page.table.rowCount() == 0
    assert page._current is question and page.ed_text.isEnabled()
    QTest.keyClicks(page.ed_text, "abc")
    assert page.ed_text.toPlainText() == "abc这道题已修改"
    assert page.ed_text.textCursor().position() == 3
    assert read_json(config.QUESTION_FILE)[0]["raw_text"] == "abc这道题已修改"
    page.ed_search.clear()
    assert page.table.rowCount() == 1
    assert page._current is question


def test_annotation_edit_updates_search_matches(window):
    question = Question(raw_text="求导数", annotation="计算失误")
    window.ctx.add_questions([question])
    page = window.pages[0]
    page.ed_search.setText("计算失误")
    page.table.selectRow(0)
    page.ed_ann.setPlainText("概念理解错误")
    assert page.table.rowCount() == 0
    assert page.ed_ann.toPlainText() == "概念理解错误" and page._current is question
    page.ed_search.setText("概念理解错误")
    assert page.table.rowCount() == 1 and page._current is question


def test_new_question_remains_visible_under_active_search_and_filters(window):
    page = window.pages[0]
    page.cb_filter_subject.setCurrentIndex(page.cb_filter_subject.findData(config.SUBJECT_PHYSICS))
    page.cb_filter_qtype.setCurrentIndex(page.cb_filter_qtype.findData(config.QTYPE_FORMULA))
    page.ed_search.setText("旧题关键词")
    page._add_single()
    assert page.ed_search.text() == "" and page.table.rowCount() == 1
    assert page._current is window.ctx.questions[0]
    assert page._current.subject == config.SUBJECT_PHYSICS
    assert page._current.qtype == config.QTYPE_FORMULA
    assert page.ed_text.isEnabled()


def test_background_task_locks_editing_and_rejects_overlap(window):
    assert window.ctx.begin_task("分类")
    assert not window.stack.isEnabled()
    assert not window.quick.isEnabled()
    assert not window.btn_backup.isEnabled()
    assert not window.ctx.begin_task("导入")
    window.ctx.end_task("导入")
    assert window.ctx.active_task == "分类"
    window.ctx.end_task("分类")
    assert window.stack.isEnabled() and window.btn_backup.isEnabled()


def test_environment_dialog_shows_status_actions_and_escapes_error_details(qapp, monkeypatch):
    from app.core.preprocess.base import EngineStatus
    from app.ui.main_window import EnvDialog

    status = EngineStatus("测试识别组件", False, "模型缺失 <model.bin>", "failed", "准备本地模型后重试")
    monkeypatch.setattr(pipeline, "engine_report", lambda: [status])
    dialog = EnvDialog()
    dialog._load()
    text = dialog.view.toPlainText()
    assert "初始化失败" in text and "<model.bin>" in text
    assert "准备本地模型后重试" in text
    assert "全部功能可用" not in text
    assert "不加载模型、不下载模型" in text
    dialog.close()


def test_warmup_badge_does_not_claim_optional_engines_are_offline_ready(window, monkeypatch):
    from app.core.preprocess.base import EngineStatus

    status = EngineStatus("测试识别组件", False, "等待模型验证", "unchecked")
    monkeypatch.setattr(pipeline, "engine_report", lambda: [status])
    window._on_warm(True, "词典已加载")
    assert "文本就绪" in window.lb_env.text()
    assert "模型/语料未验证" in window.lb_env.toolTip()
    assert "全流程" not in window.statusBar().currentMessage()


def test_import_worker_keeps_successful_files_and_can_retry(repo, tmp_path):
    valid = tmp_path / "有效.txt"
    valid.write_text("计算定积分", encoding="utf-8")
    missing = tmp_path / "稍后补充.txt"
    worker = ImportWorker("txt", [str(valid), str(missing)])
    results = []
    worker.finished_all.connect(lambda items, errors: results.append((items, errors)))
    worker.run()
    items, errors = results[0]
    assert len(items) == 1 and errors and worker.failed_paths == [str(missing)]
    assert repo.add_imported_questions(items) == (1, 0)
    missing.write_text("求解微分方程", encoding="utf-8")
    retry = ImportWorker("txt", worker.failed_paths)
    retry.finished_all.connect(lambda items, errors: results.append((items, errors)))
    retry.run()
    assert results[-1][1] == ""
    assert repo.add_imported_questions(results[-1][0]) == (1, 0)
    assert [item.raw_text for item in repo.questions] == ["计算定积分", "求解微分方程"]


def test_stopped_import_keeps_processed_items_and_lists_pending(repo, tmp_path):
    paths = []
    for number in range(3):
        path = tmp_path / "题目{}.txt".format(number)
        path.write_text("题目{}".format(number), encoding="utf-8")
        paths.append(str(path))
    worker = ImportWorker("txt", paths)
    worker.progress.connect(lambda *args: worker.stop())
    results = []
    worker.finished_all.connect(lambda items, errors: results.append((items, errors)))
    worker.run()
    assert len(results[0][0]) == 1
    assert worker.failed_paths == paths[1:]
    assert "尚未处理 2 个文件" in results[0][1]


def test_backup_dialog_restores_data_and_discards_stale_editors(window, tmp_path):
    from app.ui.backup_dialog import BackupDialog
    from app.core.dataio import backup

    question = classified_question(window.ctx)
    page = window.pages[0]
    page.table.selectRow(0)
    original = question.raw_text
    archive = backup.create_backup(window.ctx.repo, tmp_path / "界面恢复.zip")
    page.ed_text.setPlainText("恢复前修改的内容")
    dialog = BackupDialog(window.ctx, window)
    dialog._start(archive, True)
    assert window.ctx.active_task == "恢复" and not window.stack.isEnabled()
    deadline = time.monotonic() + 20
    while window.ctx.active_task and time.monotonic() < deadline:
        QTest.qWait(10)
    assert not window.ctx.active_task
    assert "恢复完成" in dialog.message.text()
    assert page._current is None
    page.table.selectRow(0)
    assert page._current is window.ctx.repo.question_by_id(question.id)
    assert page.ed_text.toPlainText() == original
    page.ed_text.setPlainText("恢复后再次编辑")
    assert window.ctx.repo.question_by_id(question.id).raw_text == "恢复后再次编辑"
    assert window.stack.isEnabled()
    dialog.close()


def test_classification_save_failure_unlocks_ui_and_reports_unsaved_results(window, monkeypatch):
    window.ctx.add_questions([Question(raw_text="求定积分", subject=config.SUBJECT_MATH)])
    warnings = []

    def fail_save():
        raise OSError("磁盘不可写")

    from PyQt5.QtWidgets import QMessageBox
    with monkeypatch.context() as changes:
        changes.setattr(window.ctx, "save_questions", fail_save)
        changes.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
        window.pages[2]._run_all()
        deadline = time.monotonic() + 20
        while window.ctx.active_task and time.monotonic() < deadline:
            QTest.qWait(10)
    assert not window.ctx.active_task and window.stack.isEnabled()
    assert warnings and "结果尚未保存" in warnings[0]


def test_close_during_import_waits_for_finished_items_to_be_saved(window, tmp_path, monkeypatch):
    import threading
    from app.core.dataio import importer

    source = tmp_path / "题目.txt"
    source.write_text("求导数", encoding="utf-8")
    started = threading.Event()
    original = importer.import_txt

    def slow_import(*args, **kwargs):
        started.set()
        time.sleep(0.1)
        return original(*args, **kwargs)

    monkeypatch.setattr(importer, "import_txt", slow_import)
    window.pages[0]._run_import("txt", [str(source)], "")
    assert started.wait(3)
    assert not window.close()
    deadline = time.monotonic() + 20
    while window.ctx.active_task and time.monotonic() < deadline:
        QTest.qWait(10)
    assert not window.ctx.active_task
    assert read_json(config.QUESTION_FILE)[0]["raw_text"] == "求导数"


def test_keyword_typing_preserves_order_and_cursor(window):
    page = window.pages[1]
    scheme = window.ctx.current_scheme()
    page._select_label_by_id(scheme.dims[0].labels[0].id)
    page.ed_keywords.setPlainText("")
    QTest.keyClicks(page.ed_keywords, "abc")
    QTest.keyClick(page.ed_keywords, Qt.Key_Return)
    QTest.keyClicks(page.ed_keywords, "def")
    assert page.ed_keywords.toPlainText() == "abc\ndef"
    assert page._label.keywords == ["abc", "def"]
    assert page.ed_keywords.textCursor().position() == 7


def test_keyword_edit_keeps_unsubmitted_sample(window):
    page = window.pages[1]
    page._select_label_by_id(window.ctx.current_scheme().dims[0].labels[0].id)
    page.ed_sample.setPlainText("尚未添加的样本草稿")
    QTest.keyClicks(page.ed_keywords, "abc")
    assert page.ed_sample.toPlainText() == "尚未添加的样本草稿"


def test_classify_uses_selected_custom_scheme(window, monkeypatch):
    scheme = custom_math_scheme(window.ctx)
    window.ctx.set_current_scheme(scheme.id)
    page = window.pages[2]
    question = Question(subject=config.SUBJECT_MATH, raw_text="计算定积分")
    monkeypatch.setattr(page, "_next_group", lambda: None)
    page._start([question])
    assert page._queue == [(scheme.id, [question])]
    page.shutdown()


def test_mixed_subjects_keep_appropriate_scheme(window):
    scheme = custom_math_scheme(window.ctx)
    window.ctx.set_current_scheme(scheme.id)
    assert window.ctx.scheme_for_subject(config.SUBJECT_MATH) is scheme
    assert window.ctx.scheme_for_subject(config.SUBJECT_PHYSICS).subject == config.SUBJECT_PHYSICS


@pytest.mark.parametrize("field", ["text", "annotation", "subject", "qtype"])
def test_edit_invalidates_result_without_resetting_input(window, field):
    question = classified_question(window.ctx)
    question.corrected[config.DIM_KNOWLEDGE] = True
    page = window.pages[0]
    page.table.selectRow(0)
    if field == "text":
        page.ed_text.setPlainText("求解微分方程的通解")
        QTest.keyClicks(page.ed_text, "abc")
        assert page.ed_text.toPlainText() == "abc求解微分方程的通解"
    elif field == "annotation":
        page.ed_ann.setPlainText("之前漏掉了初值条件")
    elif field == "subject":
        page.cb_subject.setCurrentText(config.SUBJECT_PHYSICS)
    else:
        page.cb_qtype.setCurrentIndex(page.cb_qtype.findData(config.QTYPE_FORMULA))
    assert not question.classified
    assert question.feature is None
    assert not question.labels and not question.corrected and not question.candidates
    assert not question.scheme_id
    assert any(item.id == question.id for item in window.pages[2]._filtered())
    stored = next(item for item in read_json(config.QUESTION_FILE) if item["id"] == question.id)
    assert not stored["classified"] and stored["feature"] is None


def test_reclassification_cannot_preserve_foreign_manual_label(window):
    question = classified_question(window.ctx)
    question.corrected[config.DIM_KNOWLEDGE] = True
    scheme = custom_math_scheme(window.ctx)
    bundle = pipeline.preprocess(question)
    result = window.ctx.engine(scheme).classify(question, bundle)
    ClassificationEngine.apply_result(question, result, scheme, bundle)
    assert question.labels[config.DIM_KNOWLEDGE] == scheme.dims[0].labels[0].id
    assert not question.corrected


def test_empty_result_remains_pending_and_clears_old_labels(window):
    question = classified_question(window.ctx)
    question.raw_text = ""
    scheme = window.ctx.repo.scheme_by_id(question.scheme_id)
    bundle = pipeline.preprocess(question)
    result = window.ctx.engine(scheme).classify(question, bundle)
    ClassificationEngine.apply_result(question, result, scheme, bundle)
    assert not question.classified
    assert not question.labels and not question.label_names


def test_rename_updates_saved_question_and_export_name(window):
    question = classified_question(window.ctx)
    page = window.pages[1]
    page._select_label_by_id(question.labels[config.DIM_KNOWLEDGE])
    page.ed_name.setText("我的积分章节")
    assert question.label_names[config.DIM_KNOWLEDGE] == "我的积分章节"
    stored = next(item for item in read_json(config.QUESTION_FILE) if item["id"] == question.id)
    assert stored["label_names"][config.DIM_KNOWLEDGE] == "我的积分章节"


def test_deleted_label_marks_affected_question_pending(window):
    question = classified_question(window.ctx)
    scheme = window.ctx.repo.scheme_by_id(question.scheme_id)
    dim = scheme.dim(config.DIM_KNOWLEDGE)
    dim.labels = [label for label in dim.labels if label.id != question.labels[dim.key]]
    window.ctx.notify_scheme_changed(scheme)
    assert not question.classified and not question.labels


def test_deleted_current_scheme_falls_back_and_invalidates_questions(window):
    question = classified_question(window.ctx)
    scheme_id = question.scheme_id
    window.ctx.repo.remove_scheme(scheme_id)
    assert not question.classified and not question.labels
    assert window.ctx.current_scheme() is not None
    assert window.ctx.current_scheme().id != scheme_id


def test_edit_clears_visible_result_and_correction_controls(window):
    classified_question(window.ctx)
    page = window.pages[2]
    page.table.selectRow(0)
    assert page.tree.topLevelItemCount() > 0
    import_page = window.pages[0]
    import_page.table.selectRow(0)
    import_page.ed_text.setPlainText("求微分方程的通解")
    assert page.tree.topLevelItemCount() == 0
    assert not page._fix_boxes


def test_learning_reset_covers_uncached_schemes(qapp, repo):
    ctx = AppContext()
    question = classified_question(ctx)
    scheme = ctx.repo.scheme_by_id(question.scheme_id)
    other = next(label for label in scheme.dim(config.DIM_KNOWLEDGE).labels
                 if label.id != question.labels[config.DIM_KNOWLEDGE])
    FeedbackOptimizer(ctx.engine(scheme)).apply_correction(question, config.DIM_KNOWLEDGE, other.id)
    assert read_json(config.PROFILE_FILE)[scheme.id]
    fresh_ctx = AppContext()
    assert not fresh_ctx._engines
    fresh_ctx.reset_optimization()
    reloaded = AppContext()
    assert not read_json(config.PROFILE_FILE)
    assert reloaded.engine(scheme).stats()["feedback"] == 0


def test_evaluation_preserves_external_workspace(tmp_path):
    workspace = tmp_path / "real_workspace"
    workspace.mkdir()
    sentinel = workspace / "important.txt"
    sentinel.write_text("保留真实数据", encoding="utf-8")
    env = dict(os.environ, MC_WORKSPACE=str(workspace), PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "-m", "tools.evaluate", "--only", "main"],
                            cwd=str(Path(__file__).resolve().parent.parent), env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=90)
    assert result.returncode == 0, result.stderr
    assert sentinel.exists() and sentinel.read_text(encoding="utf-8") == "保留真实数据"
    assert list(workspace.iterdir()) == [sentinel]


def test_empty_question_is_not_counted_as_success(window):
    question = Question(subject=config.SUBJECT_MATH)
    scheme = window.ctx.current_scheme()
    worker = ClassifyWorker(window.ctx.engine(scheme), scheme, [question])
    completions = []
    worker.finished_all.connect(lambda success, elapsed, errors: completions.append((success, errors)))
    worker.run()
    assert completions[0][0] == 0 and len(completions[0][1]) == 1
    assert not question.classified


def test_index_failure_finishes_batch(window, monkeypatch):
    scheme = window.ctx.current_scheme()
    engine = window.ctx.engine(scheme)

    def fail_sync():
        raise ValueError("模拟建库失败")

    monkeypatch.setattr(engine, "sync_scheme", fail_sync)
    worker = ClassifyWorker(engine, scheme, [Question(raw_text="求定积分")])
    completions = []
    worker.finished_all.connect(lambda success, elapsed, errors: completions.append((success, errors)))
    worker.run()
    assert completions[0][0] == 0
    assert "模拟建库失败" in completions[0][1][0]


def test_edit_classify_correct_reload_export_workflow(window, qapp, tmp_path):
    ctx = window.ctx
    scheme = custom_math_scheme(ctx)
    scheme.dims[0].labels.append(CategoryLabel(name="微分方程专题", keywords=["微分方程", "通解"]))
    ctx.notify_scheme_changed(scheme)
    ctx.set_current_scheme(scheme.id)
    question = Question(subject=config.SUBJECT_MATH, raw_text="计算定积分")
    ctx.add_questions([question])
    import_page, classify_page = window.pages[0], window.pages[2]
    import_page.table.selectRow(0)
    import_page.ed_text.setPlainText("求解一阶线性微分方程的通解")
    classify_page._run_all()
    deadline = time.monotonic() + 20
    while not classify_page.progress.isHidden() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert classify_page.progress.isHidden()
    assert question.classified and question.scheme_id == scheme.id
    assert question.label_names[config.DIM_KNOWLEDGE] == "微分方程专题"
    target = scheme.dims[0].labels[0]
    assert ctx.apply_correction(question, config.DIM_KNOWLEDGE, target.id)
    reloaded = Repository()
    saved = reloaded.question_by_id(question.id)
    assert saved.corrected[config.DIM_KNOWLEDGE]
    assert saved.feature["clean_text"] == question.feature["clean_text"]
    path = exporter.export_excel([saved], reloaded.scheme_by_id(scheme.id), str(tmp_path / "复习表.xlsx"))
    from openpyxl import load_workbook
    workbook = load_workbook(path)
    rows = list(workbook["全部错题"].values)
    assert rows[1][rows[0].index("知识点")] == target.name
    assert rows[1][rows[0].index("题干")] == question.raw_text
    workbook.close()
