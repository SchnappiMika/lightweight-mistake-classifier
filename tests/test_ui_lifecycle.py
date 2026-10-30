# -*- coding: utf-8 -*-
"""界面生命周期测试（对应需求 4.4.1 第 3 点「无卡顿、无崩溃、无数据丢失」）。

这些测试需要 PyQt5，在无图形环境下用 offscreen 平台运行。
"""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5", reason="未安装 PyQt5，跳过界面测试")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from app import config  # noqa: E402
from app.qt_runtime import configure_qt_plugins


@pytest.fixture(scope="module")
def qapp():
    configure_qt_plugins()
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def win(qapp, repo, monkeypatch):
    from app.ui import theme
    from app.ui.main_window import MainWindow
    w = MainWindow()
    theme.apply(qapp, theme.current())
    w.ctx.set_setting("guide_done", True)
    w.overlay.hide()
    yield w
    w.close()


class TestStartup:
    def test_main_window_does_not_block_ui_thread(self, qapp, repo):
        """主窗口构造不得在主线程上建特征库。

        建一次索引要两三秒。曾经 ``ClassificationEngine.__init__`` 里直接
        ``sync_scheme()``，而分类页构造时为了显示"XX 个标签"就会创建引擎，
        导致主窗口要卡 2.5 秒才出得来，后台预热线程形同虚设。
        """
        from app.ui.main_window import MainWindow
        t0 = time.perf_counter()
        w = MainWindow()
        ms = (time.perf_counter() - t0) * 1000
        try:
            assert ms < 1500, "主窗口构造耗时 {:.0f}ms，主线程被阻塞".format(ms)
        finally:
            w.close()

    def test_engine_is_lazy(self, repo):
        """构造引擎不该建索引；首次 classify 时才惰性构建。"""
        from app.core.classify.engine import ClassificationEngine
        from app.core.models import Question
        eng = ClassificationEngine(repo.scheme_by_subject(config.SUBJECT_MATH))
        assert not eng.ready, "构造引擎时就建了索引"
        eng.classify(Question(subject=config.SUBJECT_MATH, raw_text="求定积分"))
        assert eng.ready, "首次分类后索引仍未就绪"

    def test_stats_works_before_index_built(self, repo):
        """界面在索引就绪前就要显示体系摘要，这些数字不能依赖索引。"""
        from app.core.classify.engine import ClassificationEngine
        eng = ClassificationEngine(repo.scheme_by_subject(config.SUBJECT_MATH))
        s = eng.stats()
        assert s["labels"] > 0 and s["keywords"] > 0

    def test_all_pages_build(self, win):
        assert win.stack.count() == 5
        for i, p in enumerate(win.pages):
            win._switch(i)
            hint, widgets = p.quick_actions()
            assert widgets, "第 {} 页没有快捷操作按钮".format(i)


class TestShutdown:
    def test_close_during_warmup_does_not_crash(self, qapp, repo):
        """预热线程还在跑时关窗不能崩。

        QThread 在仍运行时被析构，Qt 会直接 abort 掉整个进程。jieba 初始化
        要好几秒，这段窗口期里关窗是很常见的操作。
        """
        from app.ui.main_window import MainWindow
        w = MainWindow()
        w.ctx.set_setting("guide_done", True)
        w.show()
        assert w._w.isRunning() or w._w.isFinished()
        w.close()                                   # 不抛异常、不 abort
        assert not w._w.isRunning(), "关窗后预热线程仍在运行"

    def test_shutdown_stops_all_page_workers(self, win):
        for p in win.pages:
            p.shutdown()
            for wk in p.workers():
                assert wk is None or not wk.isRunning()

    def test_stop_worker_handles_none_and_dead(self):
        from app.ui.workers import stop_worker
        stop_worker(None)                            # 不该抛异常

    def test_close_saves_data(self, win):
        """关窗必须落盘，不能丢数据。"""
        from app.core.dataio import importer
        win.ctx.add_questions([importer.import_single("求定积分的值")])
        win.close()
        assert config.QUESTION_FILE.exists()


class TestThemeAndGuide:
    def test_theme_toggle(self, win):
        from app.ui import theme
        before = theme.current()
        win._toggle_theme()
        assert theme.current() != before
        win._toggle_theme()
        assert theme.current() == before

    def test_guide_steps_cover_all_pages(self, win):
        pages = {s.page for s in win.overlay.steps if s.page is not None}
        assert pages == set(range(5)), "新手引导没有覆盖全部五个功能页"

    def test_guide_runs_through(self, win):
        win.show()                     # 子控件的 isVisible 依赖父窗口可见
        win.overlay.start()
        assert win.overlay.isVisible()
        for _ in range(len(win.overlay.steps) + 1):
            win.overlay.next()
        assert not win.overlay.isVisible()


class TestNavigationRefresh:
    def test_unchanged_navigation_keeps_view_items_and_actions(self, win, monkeypatch):
        from app.core.models import Question
        win.ctx.add_questions([Question(raw_text="计算定积分")])
        for index in range(5):
            win._switch(index)
        import_item = win.pages[0].table.item(0, 0)
        scheme_item = win.pages[1].tree.topLevelItem(0)
        calls = []
        for index, page in enumerate(win.pages):
            original = page.on_show
            def track(index=index, original=original):
                calls.append(index)
                original()
            monkeypatch.setattr(page, "on_show", track)
        cached = {index: tuple(widgets) for index, (_hint, widgets)
                  in win._quick_actions_cache.items()}
        for _ in range(4):
            for index in range(5):
                win._switch(index)
        assert calls == []
        assert win.pages[0].table.item(0, 0) is import_item
        assert win.pages[1].tree.topLevelItem(0) is scheme_item
        for index, widgets in cached.items():
            assert tuple(win._quick_actions_cache[index][1]) == widgets
            assert all(widget.parent() is not None and not widget.isWindow()
                       for widget in widgets)

    def test_changed_questions_refresh_again_but_only_once_on_return(self, win, monkeypatch):
        from app.core.models import Question
        win._switch(2)
        win._switch(0)
        page = win.pages[2]
        calls = []
        original = page.on_show
        def track():
            calls.append(True)
            original()
        monkeypatch.setattr(page, "on_show", track)
        win.ctx.add_questions([Question(raw_text="新的错题")])
        win._switch(2)
        assert page.table.rowCount() == 1
        assert calls == [True]
        win._switch(0)
        win._switch(2)
        assert calls == [True]

    def test_new_renamed_deleted_scheme_updates_export_and_classify(self, win):
        from app.core.models import CategoryLabel, Dimension, Scheme
        export, classify = win.pages[4], win.pages[2]
        win._switch(4)
        win._switch(0)
        scheme = Scheme(name="独立复习体系", subject=config.SUBJECT_MATH,
                        dims=[Dimension(key="custom_review", name="我的章节",
                                        labels=[CategoryLabel(name="章节甲")])])
        win.ctx.repo.add_scheme(scheme)
        win.ctx.set_current_scheme(scheme.id)
        win._switch(4)
        assert export.cb_scheme.currentData() == scheme.id
        assert export.cb_dim.findData("custom_review") >= 0
        assert classify.cb_scheme.currentData() == scheme.id
        scheme.name = "改名后的体系"
        win.ctx.notify_scheme_changed(scheme)
        assert "改名后的体系" in export.cb_scheme.currentText()
        win.ctx.repo.remove_scheme(scheme.id)
        win.ctx.scheme_changed.emit()
        assert export.cb_scheme.findData(scheme.id) == -1
        assert classify.cb_scheme.findData(scheme.id) == -1
        assert export.cb_scheme.currentData() == win.ctx.current_scheme().id

    def test_feedback_and_workspace_reload_invalidate_views(self, win):
        from app.core.models import FeedbackRecord, Question
        win._switch(3)
        win._switch(0)
        win.ctx.repo.add_feedback(FeedbackRecord(to_label_name="实际修正"))
        win.ctx.feedback_changed.emit()
        win._switch(3)
        assert win.pages[3].table.rowCount() == 1
        assert win.pages[3].table.item(0, 4).text() == "实际修正"
        win.ctx.repo.questions = [Question(raw_text="恢复后的题目")]
        win.ctx.workspace_restored()
        win._switch(0)
        assert win.pages[0].table.item(0, 0).text() == "恢复后的题目"

    def test_theme_disposes_every_cached_action_without_top_level_widgets(self, win, qapp):
        from PyQt5 import sip
        from PyQt5.QtCore import QCoreApplication, QEvent
        for _ in range(3):
            for index in range(5):
                win._switch(index)
            old = [widget for _hint, widgets in win._quick_actions_cache.values()
                   for widget in widgets]
            win._toggle_theme()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            assert all(sip.isdeleted(widget) for widget in old)
            assert all(widget.parent() is not None and not widget.isWindow()
                       for _hint, widgets in win._quick_actions_cache.values()
                       for widget in widgets)

    def test_batch_view_updates_restore_state_after_error(self, qapp):
        from PyQt5.QtWidgets import QHeaderView, QTableWidget
        from app.ui.widgets import batch_view_updates
        class View:
            def __init__(self):
                self.table = QTableWidget()
                self.table.setColumnCount(3)
                self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
                self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
                self.table.horizontalHeader().setStretchLastSection(True)
                self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
                self.table.setRowCount(2)
                self.table.setSortingEnabled(True)
            @batch_view_updates("table")
            def fail(self):
                assert not self.table.updatesEnabled()
                assert not self.table.isSortingEnabled()
                assert all(self.table.horizontalHeader().sectionResizeMode(i) == QHeaderView.Fixed
                           for i in range(3))
                assert not self.table.horizontalHeader().stretchLastSection()
                assert self.table.verticalHeader().sectionResizeMode(0) == QHeaderView.Fixed
                raise RuntimeError("模拟插入失败")
        view = View()
        with pytest.raises(RuntimeError, match="模拟插入失败"):
            view.fail()
        assert view.table.updatesEnabled() and view.table.isSortingEnabled()
        assert view.table.horizontalHeader().sectionResizeMode(0) == QHeaderView.Stretch
        assert view.table.horizontalHeader().sectionResizeMode(1) == QHeaderView.ResizeToContents
        assert view.table.horizontalHeader().stretchLastSection()
        assert view.table.verticalHeader().sectionResizeMode(0) == QHeaderView.ResizeToContents
        view.table.deleteLater()

    def test_changed_question_rebuild_preserves_auto_width_and_selection(self, win):
        from PyQt5.QtWidgets import QHeaderView
        from app.core.models import Question
        question = Question(raw_text="原有题干", subject=config.SUBJECT_MATH)
        win.ctx.add_questions([question])
        for page in (win.pages[0], win.pages[2]):
            page.table.selectRow(0)
        question.raw_text = "修改后的题干"
        win.ctx.notify_question_edited(question)
        win._switch(2)
        win._switch(0)
        for page in (win.pages[0], win.pages[2]):
            assert page.table.item(0, 0).text() == "修改后的题干"
            assert page._current is question
            assert page.table.selectionModel().selectedRows()[0].row() == 0
            assert page.table.horizontalHeader().sectionResizeMode(0) == QHeaderView.Stretch
            assert page.table.horizontalHeader().sectionResizeMode(1) == QHeaderView.ResizeToContents
            assert page.table.updatesEnabled()
