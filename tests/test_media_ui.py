# -*- coding: utf-8 -*-
"""公式与听力校对窗口的轻量生命周期检查。"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5", reason="未安装 PyQt5，跳过界面测试")

from PyQt5.QtCore import QPoint, Qt  # noqa: E402
from PyQt5.QtGui import QImage  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from app import config  # noqa: E402
from app.core.models import Question  # noqa: E402
from app.qt_runtime import configure_qt_plugins  # noqa: E402
from app.ui.media_review_dialog import MediaReviewDialog  # noqa: E402


class _Repo:
    def save_questions(self):
        return None


class _Ctx:
    active_task = ""

    def __init__(self):
        self.repo = _Repo()
        self.questions_changed = type("Signal", (), {"emit": lambda *_: None})()
        self.toast = type("Signal", (), {"emit": lambda *_: None})()

    def begin_task(self, _name):
        return True

    def end_task(self, _name):
        return None


@pytest.fixture(scope="module")
def qapp():
    configure_qt_plugins()
    return QApplication.instance() or QApplication([])


def test_media_dialog_builds_without_model(qapp, tmp_path):
    question = Question(qtype=config.QTYPE_FORMULA, media_path=str(tmp_path / "formula.png"))
    dialog = MediaReviewDialog(_Ctx(), question, "formula")
    assert dialog.btn_recognize.isEnabled()
    assert not dialog.btn_apply.isEnabled()
    dialog.close()


def test_audio_dialog_builds_without_model(qapp, tmp_path):
    question = Question(qtype=config.QTYPE_LISTENING, media_path=str(tmp_path / "audio.wav"))
    dialog = MediaReviewDialog(_Ctx(), question, "audio")
    assert dialog.btn_recognize.isEnabled()
    assert not dialog.btn_apply.isEnabled()
    dialog.close()


@pytest.fixture
def formula_path(tmp_path):
    path = tmp_path / "公式原图.png"
    image = QImage(800, 400, QImage.Format_RGB32)
    image.fill(Qt.white)
    assert image.save(str(path))
    return path


def test_formula_selection_uses_original_pixels_at_zoom_and_resets(qapp, formula_path):
    from app.ui.formula_crop_dialog import FormulaCropPanel
    before = formula_path.read_bytes()
    panel = FormulaCropPanel(str(formula_path))
    panel.show()
    panel.zoom.setCurrentIndex(panel.zoom.findData(0.5))
    canvas = panel.canvas
    QTest.mousePress(canvas, Qt.LeftButton, pos=QPoint(50, 25))
    QTest.mouseRelease(canvas, Qt.LeftButton, pos=QPoint(150, 75))
    assert panel.options()["crop_box"] == [100, 50, 300, 150]
    panel.zoom.setCurrentIndex(panel.zoom.findData(2.0))
    assert panel.options()["crop_box"] == [100, 50, 300, 150]
    assert canvas.width() == 1600
    panel.invert.setChecked(True)
    panel.resize_mode.setCurrentIndex(panel.resize_mode.findData("original"))
    assert panel.options() == {"crop_box": [100, 50, 300, 150], "invert": True,
                               "resize_mode": "original"}
    panel.reset.click()
    assert panel.options()["crop_box"] is None
    assert formula_path.read_bytes() == before
    panel.close()


def test_formula_reverse_drag_and_small_click_keep_valid_selection(qapp, formula_path):
    from app.ui.formula_crop_dialog import FormulaCropPanel
    panel = FormulaCropPanel(str(formula_path))
    panel.zoom.setCurrentIndex(panel.zoom.findData(1.0))
    canvas = panel.canvas
    QTest.mousePress(canvas, Qt.LeftButton, pos=QPoint(300, 150))
    QTest.mouseRelease(canvas, Qt.LeftButton, pos=QPoint(100, 50))
    assert panel.options()["crop_box"] == [100, 50, 300, 150]
    QTest.mouseClick(canvas, Qt.LeftButton, pos=QPoint(20, 20))
    assert panel.options()["crop_box"] == [100, 50, 300, 150]
    panel.close()


def test_formula_worker_passes_options_file_without_changing_attachment(qapp, formula_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from app.ui import media_review_dialog
    from app.core.preprocess import media_jobs
    from app.core.models import MediaReviewRecord
    captured = {}
    options = {"crop_box": [100, 50, 300, 150], "invert": True, "resize_mode": "fit"}
    def popen(command, **kwargs):
        captured["options"] = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        assert command[-2] == "--options-file"
        assert str(formula_path) in command
        output = Path(command[command.index(str(formula_path)) + 1])
        output.write_text(json.dumps(MediaReviewRecord(text="x", draft="x",
                          media_hash=media_jobs._hash_file(formula_path)).to_dict()), encoding="utf-8")
        return SimpleNamespace(returncode=0, communicate=lambda **opts: ("library logs", ""))
    monkeypatch.setattr(media_jobs.subprocess, "Popen", popen)
    worker = media_review_dialog.MediaReviewWorker("formula", str(formula_path),
                                                   recognition_options=options)
    result = []
    worker.completed.connect(lambda record, error: result.append((record, error)))
    worker.run()
    assert captured["options"] == options
    assert result[0][0].text == "x" and not result[0][1]


def test_formula_crop_controls_lock_during_recognition(qapp, formula_path):
    question = Question(qtype=config.QTYPE_FORMULA, media_path=str(formula_path))
    dialog = MediaReviewDialog(_Ctx(), question, "formula")
    assert dialog.formula_crop.options() == {"crop_box": None, "invert": False,
                                             "resize_mode": "fit"}
    dialog._set_busy(True)
    assert not dialog.formula_crop.isEnabled()
    dialog._set_busy(False)
    assert dialog.formula_crop.isEnabled()
    dialog.close()


def test_formula_fit_width_follows_dialog_splitter_viewport(qapp, formula_path):
    question = Question(qtype=config.QTYPE_FORMULA, media_path=str(formula_path))
    dialog = MediaReviewDialog(_Ctx(), question, "formula")
    dialog.show()
    qapp.processEvents()
    panel = dialog.formula_crop
    assert panel.canvas.width() <= panel.scroll.viewport().width()
    dialog.resize(720, 540)
    qapp.processEvents()
    assert panel.canvas.width() <= panel.scroll.viewport().width()
    dialog.close()
