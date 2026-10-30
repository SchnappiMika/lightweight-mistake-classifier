"""Learned coarse-index snapshots must be replaceable and reproducible."""
from __future__ import annotations

import copy
import json
import math

import pytest

from app import config
from app.core.classify.coarse import CoarseFilter
from app.core.classify.engine import ClassificationEngine
from app.core.classify.feedback import FeedbackOptimizer
from app.core.classify.vector_space import IncrementalTfidfSpace
from app.core.models import CategoryLabel, Dimension, FeatureBundle, Question, Scheme


def dimension():
    return Dimension(key=config.DIM_KNOWLEDGE, name="知识点", labels=[
        CategoryLabel(id=f"L{i}", name=f"label{i}", dim=config.DIM_KNOWLEDGE,
                      keywords=["manual", "shared"] if i == 0 else [f"keyword{i}", "shared"])
        for i in range(8)
    ])


def snapshot(cf, tokens=("manual", "shared", "fresh", "nextterm")):
    candidates, _, _ = cf.screen(FeatureBundle(text_tokens=list(tokens)), top_k=3)
    return [(c.label_id, c.coarse_score, c.hit_keywords, c.fallback) for c in candidates]


def test_repeated_snapshot_is_idempotent():
    cf = CoarseFilter(dimension())
    cf.learn("L0", {"fresh": 2, "shared": 1})
    before = copy.deepcopy((cf.index, cf.norm, cf.substr_keys, snapshot(cf)))
    for _ in range(12):
        cf.learn("L0", {"fresh": 2, "shared": 1})
    assert (cf.index, cf.norm, cf.substr_keys, snapshot(cf)) == before
    assert len([lid for lid, _ in cf.index["shared"] if lid == "L0"]) == 1


def test_replacement_removes_old_terms_and_preserves_other_labels():
    cf = CoarseFilter(dimension())
    cf.learn("L0", {"fresh": 1})
    cf.learn("L1", {"fresh": 2})
    cf.learn("L0", {"nextterm": 2})
    assert [lid for lid, _ in cf.index["fresh"]] == ["L1"]
    assert [lid for lid, _ in cf.index["nextterm"]] == ["L0"]
    assert cf.norm["L0"] == pytest.approx(math.sqrt(cf._base_norm["L0"] ** 2 + 1))


def test_manual_keyword_survives_learning_and_clear():
    dim = dimension()
    original_keywords = copy.deepcopy([label.keywords for label in dim.labels])
    cf = CoarseFilter(dim)
    before_index = copy.deepcopy(cf.index)
    before_norm = dict(cf.norm)
    cf.learn("L0", {"manual": 3, "fresh": 2})
    assert dict(cf.index["manual"])["L0"] == dict(before_index["manual"])["L0"]
    cf.learn("L0", {})
    assert cf.index == before_index
    assert cf.norm == before_norm
    assert [label.keywords for label in dim.labels] == original_keywords


def test_nonpositive_and_nonfinite_terms_do_not_inflate_norm():
    cf = CoarseFilter(dimension())
    cf.learn("L0", {"fresh": 2, "x": 10, "": 10, "gone": 0,
                    "negative": -1, "badnan": float("nan"), "badinf": float("inf")})
    assert cf.learned["L0"] == {"fresh": 2}
    assert cf.norm["L0"] == pytest.approx(math.sqrt(cf._base_norm["L0"] ** 2 + 1))
    cf.learn("L0", {"fresh": -1})
    assert "fresh" not in cf.index
    assert cf.norm["L0"] == cf._base_norm["L0"]


def test_build_and_reload_preserve_candidates_and_scores():
    dim = dimension()
    cf = CoarseFilter(dim)
    for terms in ({"fresh": 1}, {"fresh": 2, "nextterm": 1}, {"fresh": 4, "nextterm": 3}):
        cf.learn("L0", terms)
        cf.learn("L2", {"nextterm": 1, "shared": .5})
    before = snapshot(cf)
    cf.build()
    assert snapshot(cf) == before
    loaded = CoarseFilter(dim)
    for lid, terms in json.loads(json.dumps(cf.learned)).items():
        loaded.learn(lid, terms)
    assert snapshot(loaded) == before


def test_manual_keyword_edit_uses_fresh_manual_base_and_retains_feedback():
    dim = dimension()
    cf = CoarseFilter(dim)
    cf.learn("L0", {"fresh": 2, "shared": 1})
    dim.labels[0].keywords = ["newmanual"]
    cf.build()
    assert "manual" not in cf.index
    assert dict(cf.index["newmanual"])["L0"] > 0
    expected = CoarseFilter(dim)
    expected.learn("L0", {"fresh": 2, "shared": 1})
    assert snapshot(cf, ("fresh", "shared", "newmanual")) == snapshot(expected, ("fresh", "shared", "newmanual"))
    cf.learn("L0", {})
    assert "L0" not in dict(cf.index["shared"])
    assert dict(cf.index["newmanual"])["L0"] > 0


def test_removed_label_does_not_reappear_from_stale_learned_terms():
    dim = dimension()
    cf = CoarseFilter(dim)
    cf.learn("L0", {"fresh": 2})
    dim.labels.pop(0)
    cf.build()
    assert "fresh" not in cf.index
    assert "L0" not in cf.learned
    cf.learn("L0", {"ghost": 5})
    assert "ghost" not in cf.index


def test_actual_consecutive_feedback_coarse_scores_survive_engine_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROFILE_FILE", tmp_path / "profiles.json")
    monkeypatch.setattr(config, "WEIGHT_FILE", tmp_path / "weights.json")
    scheme = Scheme(name="feedback", subject=config.SUBJECT_MATH, dims=[dimension()])
    engine = ClassificationEngine(scheme)
    engine.sync_scheme()
    fb = FeatureBundle(subject=config.SUBJECT_MATH, qtype=config.QTYPE_TEXT,
                       text_tokens=["fresh", "nextterm", "manual"])
    for _ in range(6):
        question = Question(subject=config.SUBJECT_MATH, qtype=config.QTYPE_TEXT,
                            raw_text="fresh nextterm manual", labels={config.DIM_KNOWLEDGE: "L1"})
        assert FeedbackOptimizer(engine).apply_correction(question, config.DIM_KNOWLEDGE, "L0", fb)
    before = snapshot(engine.coarse[config.DIM_KNOWLEDGE])
    query = Question(subject=config.SUBJECT_MATH, qtype=config.QTYPE_TEXT)

    def fine_scores(eng):
        result = eng.classify(query, fb).dims[config.DIM_KNOWLEDGE]
        return {candidate.label_id: candidate.final_score for candidate in result.candidates}

    fine_before = fine_scores(engine)
    assert engine._feedback_n_docs == 6
    assert engine._feedback_df == {"fresh": 6, "nextterm": 6, "manual": 6}
    documents_before = engine.space.n_docs
    idf_before = {term: engine.space.idf_of(term) for term in engine.space.vocab}
    engine.sync_scheme(force=True)
    assert snapshot(engine.coarse[config.DIM_KNOWLEDGE]) == before
    assert fine_scores(engine) == pytest.approx(fine_before, abs=1e-14)
    assert engine.space.n_docs == documents_before
    reloaded = ClassificationEngine(scheme)
    reloaded.sync_scheme()
    assert snapshot(reloaded.coarse[config.DIM_KNOWLEDGE]) == before
    assert fine_scores(reloaded) == pytest.approx(fine_before, abs=1e-14)
    assert {term: reloaded.space.idf_of(term) for term in reloaded.space.vocab} == idf_before

    # Editing base keywords must not resurrect the old base document statistics.
    scheme.dims[0].labels[0].keywords = ["newkeyword"]
    scheme.touch()
    engine.sync_scheme()
    assert engine.space.n_docs == len(scheme.dims[0].labels) + 6
    assert engine.space.df[engine.space.vocab["manual"]] == 6
    edited_scores = fine_scores(engine)
    loaded_after_edit = ClassificationEngine(scheme)
    loaded_after_edit.sync_scheme()
    assert fine_scores(loaded_after_edit) == pytest.approx(edited_scores, abs=1e-14)
    assert engine._feedback_df == {"fresh": 6, "nextterm": 6, "manual": 6}

    engine.reset_profiles()
    assert engine._feedback_n_docs == 0 and engine._feedback_df == {}
    assert engine.space.n_docs == len(scheme.dims[0].labels)
    assert "fresh" not in engine.space.vocab
    after_reset = ClassificationEngine(scheme)
    after_reset.sync_scheme()
    assert after_reset._feedback_n_docs == 0
    assert "fresh" not in after_reset.space.vocab


def test_feedback_df_counts_documents_not_repeated_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROFILE_FILE", tmp_path / "profiles.json")
    scheme = Scheme(name="df", dims=[dimension()])
    engine = ClassificationEngine(scheme)
    engine.sync_scheme()
    base_count = engine.space.n_docs
    engine.register_feedback_tokens(["fresh", "fresh", "fresh", "shared"])
    engine.register_feedback_tokens(["shared"])
    engine.register_feedback_tokens([])
    engine.save_profiles()
    assert engine._feedback_n_docs == 2
    assert engine._feedback_df == {"fresh": 1, "shared": 2}
    reloaded = ClassificationEngine(scheme)
    reloaded.sync_scheme()
    assert reloaded.space.n_docs == base_count + 2
    assert reloaded.space.df[reloaded.space.vocab["fresh"]] == 1
    assert reloaded.space.df[reloaded.space.vocab["shared"]] == 10


def test_legacy_profiles_and_other_schemes_are_preserved(tmp_path, monkeypatch):
    profile_file = tmp_path / "profiles.json"
    monkeypatch.setattr(config, "PROFILE_FILE", profile_file)
    scheme = Scheme(id="legacy", name="legacy", dims=[dimension()])
    other = {"other-label": {"label_id": "other-label", "delta_tf": {"oldterm": 1}, "n_feedback": 1}}
    profile_file.write_text(json.dumps({"legacy": {"L0": {"label_id": "L0", "delta_tf": {"fresh": 1}}},
                                        "other": other}), encoding="utf-8")
    engine = ClassificationEngine(scheme)
    engine.sync_scheme()
    assert engine.profiles["L0"].delta_tf["fresh"] == 1
    assert engine._feedback_n_docs == 0
    engine.register_feedback_tokens(["newterm"])
    engine.save_profiles()
    data = json.loads(profile_file.read_text(encoding="utf-8"))
    assert data["other"] == other
    assert data["legacy"]["L0"]["delta_tf"] == {"fresh": 1}
    assert data["__feedback_document_statistics__"]["legacy"] == {"n_docs": 1, "df": {"newterm": 1}}


def test_saved_statistics_merge_without_refitting_existing_space():
    space = IncrementalTfidfSpace()
    space.fit([["base", "shared"], ["shared"]])
    space.add_document_statistics(3, {"fresh": 2, "shared": 1})
    assert space.n_docs == 5
    assert space.df[space.vocab["shared"]] == 3
    assert space.df[space.vocab["fresh"]] == 2
    assert space.df[space.vocab["base"]] == 1
    with pytest.raises(ValueError):
        space.add_document_statistics(1, {"invalid": 2})
    assert space.n_docs == 5
