from collections import Counter
import json
import math

import pytest

from app import config
from app.core.classify.engine import ClassificationEngine
from app.core.classify.feedback import FeedbackOptimizer
from app.core.classify.vector_space import LabelProfile
from app.core.classify.weights import WeightRules, normalize
from app.core.models import CategoryLabel, Dimension, FeatureBundle, Question, Scheme
from app.core.storage import read_json


@pytest.mark.parametrize("length", [1, 2, 100, 1000])
def test_negative_update_has_one_budget_for_short_and_long_questions(length):
    profile = LabelProfile("wrong")
    terms = ["term-{}".format(index) for index in range(length)]
    profile.add_base(terms, weight=8)
    before = sum(profile.effective_tf().values())
    profile.penalize(terms)
    removed = before - sum(profile.effective_tf().values())
    assert removed == pytest.approx(0.25, abs=1e-9)
    assert all(profile.effective_tf()[term] >= 0 for term in terms)


def test_penalty_follows_effective_overlap_only_and_ignores_repeated_tokens():
    base = Counter({"strong": 3, "weak": 2, "unrelated": 80})
    delta = Counter({"strong": -1, "weak": -1})
    profiles = [LabelProfile("once"), LabelProfile("repeated")]
    for profile in profiles:
        profile.base_tf = base.copy()
        profile.delta_tf = delta.copy()
    profiles[0].penalize(["strong", "weak", "absent"], decay=0.3)
    profiles[1].penalize(["strong"] * 1000 + ["weak"] * 9 + ["absent"] * 50, decay=0.3)
    assert profiles[0].delta_tf == profiles[1].delta_tf
    assert profiles[0].delta_tf["strong"] == pytest.approx(-1.2)
    assert profiles[0].delta_tf["weak"] == pytest.approx(-1.1)
    assert profiles[0].effective_tf()["unrelated"] == 80
    assert "absent" not in profiles[0].delta_tf
    assert profiles[0].base_tf == base


def test_penalty_clips_at_available_mass_and_repeated_updates_stay_nonnegative():
    profile = LabelProfile("small")
    profile.add_base(["base"], weight=0.03)
    profile.delta_tf["learned"] = 0.02
    profile.penalize(["base", "learned"], decay=0.25)
    assert not profile.effective_tf()
    assert profile.delta_tf["base"] == pytest.approx(-0.03)
    assert profile.delta_tf["learned"] == pytest.approx(0)
    after = dict(profile.delta_tf)
    for _ in range(20):
        profile.penalize(["base", "learned"])
    assert dict(profile.delta_tf) == after
    assert all(profile.base_tf.get(term, 0) + value >= -1e-15
               for term, value in profile.delta_tf.items())


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf")])
def test_invalid_penalty_budget_does_not_change_profile(budget):
    profile = LabelProfile("safe")
    profile.add_base(["term"])
    with pytest.raises(ValueError):
        profile.penalize(["term"], decay=budget)
    assert not profile.delta_tf


def test_zero_or_nonoverlap_penalty_is_noop():
    profile = LabelProfile("safe")
    profile.add_base(["base"])
    profile.penalize(["base"], decay=0)
    profile.penalize(["other"])
    profile.penalize([])
    assert not profile.delta_tf


def test_fixed_positive_and_negative_mass_are_comparable_for_long_document():
    tokens = ["term-{}".format(index) for index in range(500)]
    right, wrong = LabelProfile("right"), LabelProfile("wrong")
    wrong.add_base(tokens, weight=20)
    right.reinforce(tokens, gain=1)
    wrong.penalize(tokens, decay=0.25)
    assert sum(right.delta_tf.values()) == pytest.approx(1)
    assert -sum(wrong.delta_tf.values()) == pytest.approx(0.25)


@pytest.mark.parametrize("epsilon", [1e-3, 1e-6, 1e-10])
def test_weight_step_tracks_actual_small_cosine_difference(tmp_path, epsilon):
    rules = WeightRules(tmp_path / "weights.json")
    before, after = rules.update_from_feedback(
        config.SUBJECT_MATH, config.QTYPE_FORMULA,
        {config.CHANNEL_FORMULA: 0.5 + epsilon, config.CHANNEL_TEXT: 0.5},
        {config.CHANNEL_FORMULA: 0.5, config.CHANNEL_TEXT: 0.5 + epsilon})
    expected_step = config.WEIGHT_LEARNING_RATE * epsilon
    assert after[config.CHANNEL_FORMULA] - before[config.CHANNEL_FORMULA] == pytest.approx(expected_step, abs=1e-15)
    assert before[config.CHANNEL_TEXT] - after[config.CHANNEL_TEXT] == pytest.approx(expected_step, abs=1e-15)
    assert after[config.CHANNEL_QTYPE] == before[config.CHANNEL_QTYPE]
    assert sum(after.values()) == pytest.approx(1)


def test_asymmetric_step_uses_original_lr_then_normalizes(tmp_path):
    rules = WeightRules(tmp_path / "weights.json")
    before, after = rules.update_from_feedback(
        config.SUBJECT_MATH, config.QTYPE_FORMULA,
        {config.CHANNEL_FORMULA: 0.8, config.CHANNEL_TEXT: 0.2},
        {config.CHANNEL_FORMULA: 0.1, config.CHANNEL_TEXT: 0.7})
    expected = normalize({channel: before[channel] + config.WEIGHT_LEARNING_RATE *
                          {config.CHANNEL_FORMULA: 0.7, config.CHANNEL_TEXT: -0.5}.get(channel, 0)
                          for channel in config.CHANNELS})
    assert after == pytest.approx(expected, abs=1e-15)


def test_repeated_tiny_differences_cannot_cause_large_weight_drift(tmp_path):
    rules = WeightRules(tmp_path / "weights.json")
    original = rules.get(config.SUBJECT_MATH, config.QTYPE_FORMULA)
    for _ in range(100):
        rules.update_from_feedback(config.SUBJECT_MATH, config.QTYPE_FORMULA,
                                   {config.CHANNEL_FORMULA: 0.5 + 1e-6, config.CHANNEL_TEXT: 0.5},
                                   {config.CHANNEL_FORMULA: 0.5, config.CHANNEL_TEXT: 0.5 + 1e-6})
    after = rules.get(config.SUBJECT_MATH, config.QTYPE_FORMULA)
    assert max(abs(after[channel] - original[channel]) for channel in config.CHANNELS) < 1e-5
    restored = WeightRules(rules.path)
    assert restored.get(config.SUBJECT_MATH, config.QTYPE_FORMULA) == pytest.approx(after, abs=1e-15)
    assert restored.hits[(config.SUBJECT_MATH, config.QTYPE_FORMULA)] == 100


def test_roundoff_cosines_are_bounded_and_zero_gradient_keeps_default_rules(tmp_path):
    rules = WeightRules(tmp_path / "weights.json")
    pristine = rules.get(config.SUBJECT_ENGLISH, config.QTYPE_LISTENING)
    before, after = rules.update_from_feedback(config.SUBJECT_MATH, config.QTYPE_FORMULA,
                                               {config.CHANNEL_TEXT: 0.25}, {config.CHANNEL_TEXT: 0.25})
    assert before == after
    assert rules.get(config.SUBJECT_ENGLISH, config.QTYPE_LISTENING) == pristine
    before, after = rules.update_from_feedback(config.SUBJECT_MATH, config.QTYPE_FORMULA,
                                               {config.CHANNEL_TEXT: 1 + 1e-12}, {config.CHANNEL_TEXT: -1e-12})
    expected = normalize({channel: before[channel] + (config.WEIGHT_LEARNING_RATE
                          if channel == config.CHANNEL_TEXT else 0) for channel in config.CHANNELS})
    assert after == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_similarity_never_writes_corrupted_weights(tmp_path, invalid):
    path = tmp_path / "weights.json"
    rules = WeightRules(path)
    with pytest.raises(ValueError):
        rules.update_from_feedback(config.SUBJECT_MATH, config.QTYPE_FORMULA,
                                   {config.CHANNEL_TEXT: invalid}, {})
    assert not rules.rules and not path.exists()


def test_real_feedback_negative_delta_and_weights_reload_without_schema_changes(repo):
    scheme = Scheme(name="stability", subject=config.SUBJECT_MATH, dims=[
        Dimension(key=config.DIM_KNOWLEDGE, labels=[
            CategoryLabel(id="first", name="first", dim=config.DIM_KNOWLEDGE,
                          keywords=["overlap-{}".format(index) for index in range(100)]),
            CategoryLabel(id="second", name="second", dim=config.DIM_KNOWLEDGE, keywords=["other"]),
        ])])
    repo.add_scheme(scheme)
    engine = ClassificationEngine(scheme)
    engine.sync_scheme()
    tokens = ["overlap-{}".format(index) for index in range(100)]
    feature = FeatureBundle(subject=config.SUBJECT_MATH, qtype=config.QTYPE_TEXT, text_tokens=tokens)
    question = Question(subject=config.SUBJECT_MATH, qtype=config.QTYPE_TEXT,
                        raw_text=" ".join(tokens), labels={config.DIM_KNOWLEDGE: "first"})
    before_mass = sum(engine.profiles["first"].effective_tf().values())
    record = FeedbackOptimizer(engine).apply_correction(question, config.DIM_KNOWLEDGE, "second", feature)
    assert record is not None
    removed = before_mass - sum(engine.profiles["first"].effective_tf().values())
    assert 0 < removed <= config.FEEDBACK_NEGATIVE_DECAY + 1e-12
    expected_profiles = {key: profile.to_dict() for key, profile in engine.profiles.items()}
    saved = read_json(config.PROFILE_FILE)
    assert any(value < 0 for value in saved[scheme.id]["first"]["delta_tf"].values())
    assert set(saved[scheme.id]["first"]) == {"label_id", "delta_tf", "n_feedback"}
    assert all(math.isfinite(value) for value in record.weight_after.values())
    repo.add_questions([question])
    repo.add_feedback(record)
    restored = ClassificationEngine(scheme)
    restored.sync_scheme()
    assert {key: profile.to_dict() for key, profile in restored.profiles.items()} == expected_profiles
    assert restored.rules.get(question.subject, question.qtype) == pytest.approx(record.weight_after)
    assert json.loads(json.dumps(record.to_dict())) == record.to_dict()


def test_legacy_profile_negative_deltas_remain_compatible_after_penalty():
    payload = {"label_id": "legacy", "delta_tf": {"a": -0.5, "learned": 0.2}, "n_feedback": 2}
    profile = LabelProfile.from_dict(payload)
    profile.add_base(["a", "b"])
    profile.penalize(["a", "learned"], decay=0.25)
    restored = LabelProfile.from_dict(json.loads(json.dumps(profile.to_dict())))
    restored.add_base(["a", "b"])
    assert restored.effective_tf() == profile.effective_tf()
    assert restored.n_feedback == 2
