"""Exercise consecutive scheme operations through the real page callbacks."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QDialog, QInputDialog, QMessageBox

from app import config
from app.core.models import Dimension, Scheme
from app.core.storage import Repository
from app.qt_runtime import configure_qt_plugins
from app.ui.context import AppContext
from app.ui.pages import page_scheme


@pytest.fixture(scope="module")
def qapp():
    configure_qt_plugins()
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(qapp, repo):
    widget = page_scheme.SchemePage(AppContext())
    yield widget
    widget.close()


def choose_text(monkeypatch, value):
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: (value, True))


def create_scheme(page, monkeypatch):
    scheme = Scheme(name="连续操作新体系", subject=config.SUBJECT_MATH,
                    dims=[Dimension(key=config.DIM_KNOWLEDGE, name="知识点", labels=[])])

    class Dialog:
        def __init__(self, *args):
            pass

        def exec_(self):
            return QDialog.Accepted

        def result_scheme(self, *args):
            return scheme

    monkeypatch.setattr(page_scheme, "NewSchemeDialog", Dialog)
    page._new_scheme()
    return scheme


def test_old_label_then_new_scheme_dimension_and_label_stay_in_new_scheme(page, monkeypatch):
    old = page.ctx.schemes[0]
    old_dimension = old.dims[0]
    old_labels = list(old_dimension.labels)
    page._select_label_by_id(old_labels[0].id)
    scheme = create_scheme(page, monkeypatch)
    assert page._scheme is scheme and page._dim is None and page._label is None
    assert page.ctx.current_scheme() is scheme
    assert page.tree.currentItem().data(0, page_scheme.ROLE_ID) == scheme.id

    choose_text(monkeypatch, "自定义掌握维度")
    page._new_dim()
    dimension = scheme.dims[-1]
    assert page._dim is dimension and page._label is None
    assert page.tree.currentItem().data(0, page_scheme.ROLE_ID) == f"{scheme.id}::{dimension.key}"
    choose_text(monkeypatch, "需要复习")
    page._new_label()
    label = dimension.labels[0]
    assert page._label is label and page.ed_name.text() == "需要复习"
    assert old_dimension.labels == old_labels

    page.ed_name.setText("已掌握")
    page.ed_keywords.setPlainText("掌握\n熟练\n掌握")
    page.ed_sample.setPlainText("已能独立计算该类定积分")
    page._add_sample()
    assert page._label is label and label.keywords == ["掌握", "熟练"]
    assert label.samples == ["已能独立计算该类定积分"]
    saved = Repository().scheme_by_id(scheme.id).dim(dimension.key).label_by_id(label.id)
    assert saved.name == "已掌握" and saved.samples == label.samples


def test_selected_dimension_survives_refresh_and_new_dimension_gets_next_label(page, monkeypatch):
    scheme = page.ctx.schemes[0]
    first = scheme.dims[0]
    page._select_node(f"{scheme.id}::{first.key}")
    page.ctx.scheme_changed.emit()
    assert page._dim is first
    assert page.tree.currentItem().data(0, page_scheme.ROLE_ID) == f"{scheme.id}::{first.key}"
    count = len(first.labels)
    choose_text(monkeypatch, "题目来源")
    page._new_dim()
    added = scheme.dims[-1]
    choose_text(monkeypatch, "期末试卷")
    page._new_label()
    assert page._dim is added and page._label in added.labels
    assert len(first.labels) == count


def test_deleting_label_dimension_scheme_selects_surviving_parent(page, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    scheme = create_scheme(page, monkeypatch)
    choose_text(monkeypatch, "自定义维度")
    page._new_dim()
    dimension = page._dim
    choose_text(monkeypatch, "临时标签")
    page._new_label()
    label = page._label
    page._delete_node()
    assert not dimension.labels and page._label is None and page._dim is dimension
    assert page.tree.currentItem().data(0, page_scheme.ROLE_ID) == f"{scheme.id}::{dimension.key}"
    page._delete_node()
    assert dimension not in scheme.dims and page._dim is None and page._scheme is scheme
    assert page.tree.currentItem().data(0, page_scheme.ROLE_ID) == scheme.id
    page._delete_node()
    assert page.ctx.repo.scheme_by_id(scheme.id) is None
    assert page._scheme is not scheme and page._dim is None and page._label is None
    assert page.ctx.repo.scheme_by_id(page._scheme.id) is page._scheme
    assert not page.ed_name.isEnabled()
    assert label not in [lb for sc in page.ctx.schemes for dim in sc.dims for lb in dim.labels]


def test_reset_rebinds_selected_label_to_replacement_repository_objects(page, monkeypatch):
    original_scheme = page.ctx.schemes[0]
    original_dim = original_scheme.dims[0]
    original_label = original_dim.labels[0]
    page._select_label_by_id(original_label.id)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    page._reset_builtin()
    assert page._scheme is page.ctx.repo.scheme_by_id(original_scheme.id)
    assert page._scheme is not original_scheme
    assert page._dim is page._scheme.dim(original_dim.key) and page._dim is not original_dim
    assert page._label is page._dim.label_by_id(original_label.id)
    assert page._label is not original_label
    page.ed_name.setText("恢复后可继续编辑")
    assert page._label.name == "恢复后可继续编辑"
    assert original_label.name != "恢复后可继续编辑"


def test_clicking_other_scheme_dimension_switches_current_scheme(page, monkeypatch):
    old = page.ctx.schemes[0]
    new = create_scheme(page, monkeypatch)
    node = page.tree.topLevelItem(0).child(0)
    page.tree.setCurrentItem(node)
    assert page._scheme is old and page.ctx.current_scheme() is old
    assert page._dim is old.dims[0]
    assert new is not page._scheme
