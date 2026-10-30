"""Audio preview failures stay visible and do not leak to a new selection."""
from __future__ import annotations

import os
import wave

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QApplication

from app import config
from app.core.models import Question
from app.qt_runtime import configure_qt_plugins
from app.ui.context import AppContext
from app.ui.pages import page_import


class FakePlayer(QObject):
    stateChanged = pyqtSignal(int)
    error = pyqtSignal(int)
    PlayingState = 1
    fail_on_set = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = 0
        self.play_count = self.stop_count = 0

    def state(self):
        return self._state

    def setMedia(self, media):
        self.media = media
        if self.fail_on_set:
            self.error.emit(2)

    def play(self):
        self.play_count += 1
        self._state = self.PlayingState
        self.stateChanged.emit(self._state)

    def stop(self):
        self.stop_count += 1
        self._state = 0
        self.stateChanged.emit(0)

    def errorString(self):
        return "无法解码此音频"


@pytest.fixture(scope="module")
def qapp():
    configure_qt_plugins()
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(qapp, repo, tmp_path, monkeypatch):
    monkeypatch.setattr(page_import, "_HAS_MEDIA", True)
    monkeypatch.setattr(page_import, "QMediaPlayer", FakePlayer, raising=False)
    monkeypatch.setattr(page_import, "QMediaContent", lambda url: url, raising=False)
    path = tmp_path / "听力.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 160)
    widget = page_import.ImportPage(AppContext())
    widget._current = Question(qtype=config.QTYPE_LISTENING, media_path=str(path))
    widget._show_media(widget._current)
    yield widget
    widget.shutdown()
    widget.close()


def test_decoder_failure_stops_player_and_shows_once_per_attempt(page):
    notices = []
    page.ctx.toast.connect(lambda message, kind: notices.append((message, kind)))
    page._toggle_play()
    player = page._player
    assert player.play_count == 1
    player.error.emit(2)
    assert player.state() == 0 and page.btn_play.text() == "播放"
    assert "无法解码此音频" in page.lb_audio.text()
    assert "系统播放器" in page.lb_audio.text()
    player.error.emit(2)
    assert len(notices) == 1
    page._toggle_play()
    assert player.play_count == 2 and "播放失败" not in page.lb_audio.text()
    player.error.emit(2)
    assert len(notices) == 2


def test_selection_change_clears_error_and_ignores_old_player_failure(page, tmp_path):
    page._toggle_play()
    page._player.error.emit(2)
    old_player = page._player
    next_audio = tmp_path / "另一段听力.wav"
    next_audio.write_bytes(b"audio")
    page._current = Question(qtype=config.QTYPE_LISTENING, media_path=str(next_audio))
    page._show_media(page._current)
    assert "播放失败" not in page.lb_audio.text() and next_audio.name in page.lb_audio.text()
    old_player.error.emit(2)
    assert "播放失败" not in page.lb_audio.text()
    page._toggle_play()
    assert page._player is not old_player
    old_player.error.emit(2)
    assert "播放失败" not in page.lb_audio.text()
    assert page._player.state() == FakePlayer.PlayingState


def test_system_player_launch_failure_is_visible(page, monkeypatch):
    notices = []
    page.ctx.toast.connect(lambda message, kind: notices.append((message, kind)))
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: False)
    page._open_external()
    assert "系统播放器未能打开" in page.lb_audio.text()
    assert len(notices) == 1 and notices[0][1] == "warn"


def test_synchronous_media_load_error_does_not_attempt_play(page, monkeypatch):
    monkeypatch.setattr(FakePlayer, "fail_on_set", True)
    page._toggle_play()
    assert page._player.play_count == 0
    assert "播放失败" in page.lb_audio.text()


def test_system_player_remains_available_without_qt_audio_backend(page, monkeypatch):
    monkeypatch.setattr(page_import, "_HAS_MEDIA", False)
    page._show_media(page._current)
    assert not page.btn_play.isEnabled()
    assert page.btn_open_ext.isEnabled()
    assert "系统播放器" in page.btn_play.toolTip()
