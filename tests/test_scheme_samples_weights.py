"""Real scheme-page imports must train selected labels and apply edited weights."""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QFileDialog

from app import config
from app.core.dataio.sample_import import ImportError_, read_samples
from app.core.models import CategoryLabel, Dimension, FeatureBundle, Question, Scheme
from app.core.storage import Repository
from app.qt_runtime import configure_qt_plugins
from app.ui.context import AppContext
from app.ui.pages.page_scheme import SchemePage


@pytest.fixture(scope="module")
def qapp():
    configure_qt_plugins()
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(qapp, repo):
    context = AppContext()
    scheme = Scheme(name="样本与权重", subject=config.SUBJECT_MATH, dims=[
        Dimension(key=config.DIM_KNOWLEDGE, name="知识点", labels=[
            CategoryLabel(id="sample-label", name="目标", dim=config.DIM_KNOWLEDGE, keywords=["unrelated"]),
            CategoryLabel(id="other-label", name="其他", dim=config.DIM_KNOWLEDGE, keywords=["different"]),
        ])])
    context.repo.schemes.append(scheme)
    context.repo.save_schemes()
    context.set_current_scheme(scheme.id)
    widget = SchemePage(context)
    widget._select_node(f"{scheme.id}::{config.DIM_KNOWLEDGE}::sample-label")
    yield widget
    widget.close()


@pytest.mark.parametrize("suffix", [".xlsx", ".csv", ".tsv"])
def test_sample_column_is_selected_without_copying_missing_attachments(tmp_path, monkeypatch, suffix):
    table = pd.DataFrame([{"编号": "1", "标准样本": "计算旋转体体积", "附件": "does-not-exist.png"},
                          {"编号": "2", "标准样本": "计算旋转体体积", "附件": "also-missing.mp3"},
                          {"编号": "3", "标准样本": "求平面区域面积", "附件": ""}])
    source = tmp_path / f"samples{suffix}"
    if suffix == ".xlsx":
        table.to_excel(source, index=False)
    else:
        table.to_csv(source, index=False, sep="\t" if suffix == ".tsv" else ",", encoding="utf-8-sig")
    monkeypatch.setattr(Repository, "import_media", lambda *args: pytest.fail("样本导入不应复制附件"))
    assert read_samples(str(source)) == ["计算旋转体体积", "求平面区域面积"]


def test_unknown_multicolumn_table_is_not_silently_imported_as_ids(tmp_path):
    source = tmp_path / "unknown.csv"
    source.write_text("编号,分类\n1,极限\n2,积分\n", encoding="utf-8")
    with pytest.raises(ImportError_, match="样本文本列"):
        read_samples(str(source))


def test_txt_whitespace_paragraphs_and_numbered_multiline_samples(tmp_path):
    source = tmp_path / "samples.txt"
    source.write_text("# 注释\n计算定积分\n保留解题条件\n  \n求函数极限\n", encoding="gbk")
    assert read_samples(str(source)) == ["计算定积分\n保留解题条件", "求函数极限"]
    source.write_text("1. The family received a letter.\nA. yesterday B. never\n"
                      "2. Choose the suitable route.\nA. driving B. walking", encoding="utf-8")
    samples = read_samples(str(source))
    assert len(samples) == 2
    assert "A. yesterday B. never" in samples[0]
    assert "A. driving B. walking" in samples[1]


def test_page_import_trains_selected_label_saves_and_deduplicates(page, tmp_path, monkeypatch):
    source = tmp_path / "samples.xlsx"
    pd.DataFrame([{"编号": 1, "标准样本": "黎曼和极限定义", "附件": "missing.png"},
                  {"编号": 2, "标准样本": "黎曼和极限定义", "附件": ""}]).to_excel(source, index=False)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(source), ""))
    engine = page.ctx.engine(page._scheme)
    engine.sync_scheme()
    assert "极限" not in engine.profiles[page._label.id].base_tf
    page._import_samples()
    assert page._label.samples == ["黎曼和极限定义"]
    assert not engine.ready
    engine.sync_scheme()
    assert "极限" in engine.profiles[page._label.id].base_tf
    revision = page._scheme.revision
    page._import_samples()
    assert page._scheme.revision == revision
    assert page.list_samples.count() == 1
    assert page._scheme.dims[0].labels[1].samples == []
    saved = Repository().scheme_by_id(page._scheme.id).dims[0].labels[0]
    assert saved.samples == page._label.samples
    assert page.ctx.questions == []


def test_updating_sample_cannot_create_duplicate_training_text(page):
    page._label.samples = ["第一条样本", "第二条样本"]
    page._load_label(page._label)
    page.list_samples.setCurrentRow(1)
    page.ed_sample.setPlainText("第一条样本")
    page._update_sample()
    assert page._label.samples == ["第一条样本", "第二条样本"]


def test_weight_override_display_matches_effective_values_and_hot_updates(page):
    engine = page.ctx.engine(page._scheme)
    engine.sync_scheme()
    label = page._label
    label.weight_override = {config.CHANNEL_FORMULA: .9}
    page._load_label(label)
    expected = engine.rules.get(page._scheme.subject, config.QTYPE_TEXT, label.weight_override)
    for channel, (slider, display) in page.sliders.items():
        assert slider.value() == round(expected[channel] * 100)
        assert display.text() == f"{expected[channel] * 100:.1f}%"
    revision = page._scheme.revision
    page.sliders[config.CHANNEL_TEXT][0].setValue(80)
    assert page._scheme.revision == revision + 1
    assert not engine.ready
    effective = engine.rules.get(page._scheme.subject, config.QTYPE_TEXT, label.weight_override)
    assert sum(effective.values()) == pytest.approx(1)
    for channel, (_, display) in page.sliders.items():
        assert display.text() == f"{effective[channel] * 100:.1f}%"
    fb = FeatureBundle(subject=config.SUBJECT_MATH, qtype=config.QTYPE_TEXT,
                       text_tokens=["unrelated"], formula_tokens=["unrelated"],
                       qtype_tokens=["unrelated"], subject_tokens=["unrelated"])
    result = engine.classify(Question(subject=config.SUBJECT_MATH), fb)
    candidate = next(item for item in result.dims[config.DIM_KNOWLEDGE].candidates if item.label_id == label.id)
    assert candidate.channel_weights == pytest.approx(effective)
    saved = Repository().scheme_by_id(page._scheme.id).dims[0].labels[0]
    assert saved.weight_override == label.weight_override
    revision = page._scheme.revision
    page.cb_override.setChecked(False)
    assert page._scheme.revision == revision + 1
    assert label.weight_override == {}
    assert all(not slider.isEnabled() for slider, _ in page.sliders.values())
    default = engine.rules.get(page._scheme.subject, config.QTYPE_TEXT)
    for channel, (_, display) in page.sliders.items():
        assert display.text() == f"{default[channel] * 100:.1f}%"
