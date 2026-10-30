from copy import deepcopy
import json
import os

import pytest

from tools.review_acceptance import ReviewStore, build_window


def rows():
    return [{"id": "one", "group_id": "chapter-a", "raw_text": "求导数", "subject": "高等数学",
             "qtype": "text", "labels": {}, "label_status": "source_derived", "source_url": "https://example.org/one"},
            {"id": "two", "group_id": "chapter-b", "raw_text": "求积分", "subject": "高等数学",
             "qtype": "text", "labels": {}, "label_status": "source_derived", "source_url": "https://example.org/two"}]


def test_only_explicit_review_is_exported_as_human_verified_and_source_is_untouched(tmp_path):
    source = rows()
    before = deepcopy(source)
    store = ReviewStore(source, tmp_path / "reviews.json")
    with pytest.raises(ValueError, match="主动勾选"):
        store.save(source[0], "导数与微分", "reviewer-a")
    with pytest.raises(ValueError):
        store.save(source[0], "导数与微分", "", confirmed=True)
    store.save(source[0], "导数与微分", "reviewer-a", ["导数", "导数"], confirmed=True)
    loaded = ReviewStore(source, store.path)
    assert loaded.current(source[0])["core_features"] == ["导数"]
    path = tmp_path / "reviewed.jsonl"
    assert loaded.export(path) == 1
    output = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert output[0]["label_status"] == "human_verified"
    assert output[0]["labels"] == {"knowledge": "导数与微分"}
    assert output[0]["human_review"]["reviewer"] == "reviewer-a"
    assert output[1]["label_status"] == "source_derived" and output[1]["labels"] == {}
    assert source == before


def test_editing_question_invalidates_old_review(tmp_path):
    source = rows()
    store = ReviewStore(source, tmp_path / "reviews.json")
    store.save(source[0], "导数与微分", "reviewer-a", confirmed=True)
    source[0]["raw_text"] = "改为求积分"
    assert store.current(source[0]) is None
    output = tmp_path / "changed.jsonl"
    assert store.export(output) == 0
    assert json.loads(output.read_text(encoding="utf-8").splitlines()[0])["labels"] == {}


def test_failed_write_keeps_previous_review_in_memory_and_on_disk(tmp_path, monkeypatch):
    import tools.review_acceptance as review
    source = rows()
    store = ReviewStore(source, tmp_path / "reviews.json")
    store.save(source[0], "导数与微分", "reviewer-a", confirmed=True)
    before = store.path.read_bytes()
    def fail(*_args):
        raise OSError("disk full")
    monkeypatch.setattr(review.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        store.save(source[0], "定积分及其应用", "reviewer-b", confirmed=True)
    assert store.path.read_bytes() == before
    assert store.current(source[0])["reviewer"] == "reviewer-a"


def test_visible_confirmation_required_and_progress_is_saved(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from app.qt_runtime import configure_qt_plugins
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from app.core.models import Scheme
    from app import config
    configure_qt_plugins()
    app = QApplication.instance() or QApplication([])
    source = rows()
    store = ReviewStore(source, tmp_path / "reviews.json")
    schemes = [Scheme.from_dict(value) for value in json.loads(
        (config.SCHEME_SEED_DIR / "builtin_schemes.json").read_text(encoding="utf-8"))]
    window = build_window(source, store, schemes, tmp_path / "frozen.jsonl")
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: warnings.append(True))
    window.reviewer.setText("reviewer-a")
    window.labels.setCurrentIndex(window.labels.findData("导数与微分"))
    window.save()
    assert warnings and not store.current(source[0])
    window.confirm.setChecked(True)
    window.save()
    assert store.current(source[0])["explicit_confirmation"]
    assert window.list.currentRow() == 1
    assert "1 / 2" in window.progress.text()
    assert not window.confirm.isChecked()
    window.close()
