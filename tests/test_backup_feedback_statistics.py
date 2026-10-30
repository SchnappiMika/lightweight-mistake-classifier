from __future__ import annotations

import json

import pytest

from app import config
from app.core.classify.engine import ClassificationEngine
from app.core.classify.feedback import FeedbackOptimizer
from app.core.dataio import backup
from app.core.models import CategoryLabel, Dimension, FeatureBundle, Question, Scheme
from app.core.storage import Repository, read_json, write_json
from tests.test_backup import rewrite_archive, switch_workspace


STATISTICS_KEY = "__feedback_document_statistics__"


def make_scheme(identity):
    return Scheme(id=identity, name=identity, subject=config.SUBJECT_MATH,
                  dims=[Dimension(key=config.DIM_KNOWLEDGE, name="知识点", labels=[
                      CategoryLabel(id=identity + "-first", name="积分",
                                    dim=config.DIM_KNOWLEDGE, keywords=["integral", "shared"]),
                      CategoryLabel(id=identity + "-second", name="导数",
                                    dim=config.DIM_KNOWLEDGE, keywords=["derivative", "shared"]),
                  ])])


def learning_snapshot(engine):
    engine.sync_scheme()
    return {
        "n_docs": engine.space.n_docs,
        "feedback_n_docs": engine._feedback_n_docs,
        "feedback_df": dict(engine._feedback_df),
        "idf": {term: engine.space.idf_of(term) for term in engine.space.vocab},
        "df": {term: engine.space.df[index] for term, index in engine.space.vocab.items()},
        "profiles": {lid: profile.to_dict() for lid, profile in engine.profiles.items()},
    }


def test_feedback_statistics_and_profiles_survive_backup_to_new_workspace(repo, tmp_path, monkeypatch):
    expected = {}
    for identity, documents in (("scheme-one", [["fresh", "fresh", "shared"], ["other", "shared"]]),
                                ("scheme-two", [["unique", "shared"]])):
        scheme = make_scheme(identity)
        repo.add_scheme(scheme)
        engine = ClassificationEngine(scheme)
        for tokens in documents:
            question = Question(subject=config.SUBJECT_MATH, qtype=config.QTYPE_TEXT,
                                raw_text=" ".join(tokens), scheme_id=scheme.id,
                                labels={config.DIM_KNOWLEDGE: identity + "-first"})
            feature = FeatureBundle(subject=config.SUBJECT_MATH, qtype=config.QTYPE_TEXT,
                                    text_tokens=tokens)
            record = FeedbackOptimizer(engine).apply_correction(
                question, config.DIM_KNOWLEDGE, identity + "-second", feature)
            assert record is not None
            repo.add_questions([question])
            repo.add_feedback(record)
        expected[identity] = learning_snapshot(engine)

    profiles = read_json(config.PROFILE_FILE)
    assert profiles[STATISTICS_KEY]["scheme-one"]["n_docs"] == 2
    assert profiles[STATISTICS_KEY]["scheme-one"]["df"]["fresh"] == 1
    archive = backup.create_backup(repo, tmp_path / "学习备份.zip")
    switch_workspace(monkeypatch, tmp_path / "另一台电脑" / "workspace")
    restored = Repository()
    backup.restore_backup(restored, archive)

    assert read_json(config.PROFILE_FILE) == profiles
    assert len(restored.feedback) == 3
    for identity, snapshot in expected.items():
        engine = ClassificationEngine(restored.scheme_by_id(identity))
        assert learning_snapshot(engine) == snapshot


@pytest.mark.parametrize("statistics", [{}, {"retired-scheme": {}},
                                       {"retired-scheme": {"n_docs": 0, "df": {}}},
                                       {"retired-scheme": {"n_docs": 2, "df": {"unused": 0}}}])
def test_empty_default_and_retired_scheme_statistics_remain_compatible(repo, tmp_path, statistics):
    profiles = {STATISTICS_KEY: statistics,
                "legacy-scheme": {"legacy-label": {"delta_tf": {"old": -0.5}, "n_feedback": 1}}}
    write_json(config.PROFILE_FILE, profiles)
    archive = backup.create_backup(repo, tmp_path / "compatible.zip")
    write_json(config.PROFILE_FILE, {})
    backup.restore_backup(repo, archive)
    assert read_json(config.PROFILE_FILE) == profiles


def test_reset_feedback_statistics_can_be_backed_up_and_reloaded(repo, tmp_path):
    scheme = make_scheme("reset-scheme")
    repo.add_scheme(scheme)
    engine = ClassificationEngine(scheme)
    engine.sync_scheme()
    engine.register_feedback_tokens(["temporary"])
    engine.save_profiles()
    engine.reset_profiles()
    expected = learning_snapshot(engine)
    archive = backup.create_backup(repo, tmp_path / "reset.zip")
    backup.restore_backup(repo, archive)
    assert read_json(config.PROFILE_FILE)[STATISTICS_KEY][scheme.id] == {"n_docs": 0, "df": {}}
    assert learning_snapshot(ClassificationEngine(repo.scheme_by_id(scheme.id))) == expected


@pytest.mark.parametrize("statistics", [
    None, [], "invalid", 1,
    {"scheme": None}, {"scheme": []}, {"scheme": "invalid"},
    {"scheme": {"n_docs": -1, "df": {}}},
    {"scheme": {"n_docs": True, "df": {}}},
    {"scheme": {"n_docs": 1.5, "df": {}}},
    {"scheme": {"n_docs": "1", "df": {}}},
    {"scheme": {"n_docs": None, "df": {}}},
    {"scheme": {"n_docs": float("inf"), "df": {}}},
    {"scheme": {"n_docs": 1, "df": []}},
    {"scheme": {"n_docs": 1, "df": None}},
    {"scheme": {"n_docs": 1, "df": {"term": -1}}},
    {"scheme": {"n_docs": 1, "df": {"term": 2}}},
    {"scheme": {"n_docs": 1, "df": {"term": True}}},
    {"scheme": {"n_docs": 1, "df": {"term": 0.5}}},
    {"scheme": {"n_docs": 1, "df": {"term": "1"}}},
    {"scheme": {"n_docs": 1, "df": {"term": float("nan")}}},
    {"scheme": {"df": {"term": 1}}},
])
def test_invalid_feedback_statistics_are_rejected_before_any_live_data_changes(repo, tmp_path, statistics):
    repo.add_questions([Question(raw_text="保留当前题目")])
    write_json(config.PROFILE_FILE, {STATISTICS_KEY: {"safe": {"n_docs": 1, "df": {"valid": 1}}}})
    source = backup.create_backup(repo, tmp_path / "valid.zip")
    damaged = tmp_path / "invalid.zip"

    def corrupt(files):
        profiles = {STATISTICS_KEY: statistics}
        files["model/label_profiles.json"] = json.dumps(profiles).encode("utf-8")

    # Recompute checksums so this exercises semantic validation, rather than ZIP integrity.
    rewrite_archive(source, damaged, corrupt)
    before = {path.relative_to(config.WORKSPACE_DIR): path.read_bytes()
              for path in config.WORKSPACE_DIR.rglob("*") if path.is_file()}
    with pytest.raises(backup.BackupError):
        backup.restore_backup(repo, damaged)
    after = {path.relative_to(config.WORKSPACE_DIR): path.read_bytes()
             for path in config.WORKSPACE_DIR.rglob("*") if path.is_file()}
    assert after == before
    assert repo.questions[0].raw_text == "保留当前题目"
    assert not (config.WORKSPACE_DIR / "backups").exists()
    assert not list(config.WORKSPACE_DIR.glob(".restore-*"))
