"""Evaluation protocol tests; synthetic fixtures are not acceptance evidence."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import evaluate_independent as evaluation


def test_import_does_not_change_environment_or_create_workspace(tmp_path):
    workspace = tmp_path / "real-workspace"
    workspace.mkdir()
    sentinel = workspace / "keep.txt"
    sentinel.write_text("user data", encoding="utf-8")
    code = ("import os; before=dict(os.environ); "
            "import tools.evaluate_independent; assert dict(os.environ)==before")
    result = subprocess.run([sys.executable, "-c", code], cwd=evaluation.ROOT,
                            env=dict(os.environ, MC_WORKSPACE=str(workspace)), capture_output=True)
    assert result.returncode == 0, result.stderr
    assert list(workspace.iterdir()) == [sentinel]


def test_failure_is_in_denominator_and_no_valid_gold_means_no_accuracy():
    predictions = [
        {"id": "correct", "label_status": "human_verified", "gold": {"knowledge": {"valid": True, "name": "积分"}},
         "correct": True, "predictions": {"knowledge": {"name": "积分"}}, "error": ""},
        {"id": "failed", "label_status": "human_verified", "gold": {"knowledge": {"valid": True, "name": "积分"}},
         "correct": False, "predictions": {}, "error": "model failed"},
        {"id": "unknown", "label_status": "human_verified", "gold": {"knowledge": {"valid": False, "provided": "未知"}},
         "correct": False, "predictions": {}, "error": "no scheme"},
    ]
    metric = evaluation.summarize(predictions, "human_verified")
    assert metric["denominator"] == 3 and metric["accuracy"] == pytest.approx(1 / 3)
    assert metric["prediction_failures"] == 2 and metric["invalid_gold_rows"] == 1
    assert metric["confusion_matrix"]["积分"]["__FAILED_OR_EMPTY__"] == 1
    assert evaluation.summarize(predictions[2:])["accuracy"] is None
    assert evaluation.summarize([])["accuracy"] is None


def test_explicit_source_mapping_stays_exploratory_and_never_uses_prediction():
    row = {"id": "source-1", "subject": "高等数学", "label_status": "source_derived", "labels": {},
           "source_labels": {"chapter": "2 Limits", "section": "2.2 The Limit of a Function"}}
    assert evaluation.effective_labels(row) == {}
    mapping = {"rules": [{"subject": "高等数学", "source_labels": {"chapter": "2 Limits"},
                          "labels": {"knowledge": "极限"}}]}
    assert evaluation.effective_labels(row, mapping) == {"knowledge": "极限"}
    assert row["labels"] == {} and row["label_status"] == "source_derived"
    conflict = {"rules": mapping["rules"] + [{"source_labels": {"chapter": "2 Limits"},
                                               "labels": {"knowledge": "导数"}}]}
    with pytest.raises(ValueError, match="Conflicting"):
        evaluation.effective_labels(row, conflict)


def test_fixed_group_split_is_order_independent_and_merges_duplicate_contexts():
    rows = [{"id": "a", "group_id": "chapter-a", "raw_text": "Shared question"},
            {"id": "b", "group_id": "chapter-b", "raw_text": " SHARED  question "},
            {"id": "c", "group_id": "chapter-b", "raw_text": "other in same chapter"},
            {"id": "d", "group_id": "chapter-d", "raw_text": "independent question"}]
    for seed in evaluation.FIXED_SEEDS:
        split = evaluation.fixed_group_split(rows, seed)
        assert split == evaluation.fixed_group_split(list(reversed(rows)), seed)
        assert set(split["correction_ids"]).isdisjoint(split["test_ids"])
        assert set(split["correction_ids"] + split["test_ids"]) == {row["id"] for row in rows}
        assert any(set(ids) == {"a", "b", "c"} for ids in split["merged_groups"].values())
    ungrouped = [{"id": "a", "raw_text": "one"}, {"id": "b", "raw_text": "two"}]
    assert evaluation.fixed_group_split(ungrouped, 42)["correction_ids"] == []


def test_feedback_reports_percentage_points_relative_percent_and_both_directions():
    def item(identity, correct):
        return {"id": identity, "label_status": "human_verified", "gold": {"knowledge": {"valid": True, "name": "x"}},
                "correct": correct, "predictions": {"knowledge": {"name": "x" if correct else "y"}}, "error": ""}
    result = evaluation.feedback_comparison([item("one", True), item("two", False)],
                                            [item("one", False), item("two", True)])
    assert result["improved_ids"] == ["two"] and result["regressed_ids"] == ["one"]
    assert result["improvement_percentage_points"] == 0 and result["relative_improvement_percent"] == 0
    result = evaluation.feedback_comparison([item("one", True), item("two", False)],
                                            [item("one", True), item("two", True)])
    assert result["improvement_percentage_points"] == 50 and result["relative_improvement_percent"] == 100


def write_fixture(tmp_path):
    # Deliberately wrong current labels create real correction events in the
    # tool's test, without presenting these fabricated texts as project data.
    schemes = [{"id": "synthetic-test-only", "name": "synthetic protocol test", "subject": "高等数学", "dims": [
        {"key": "knowledge", "name": "知识点", "labels": [
            {"id": "A", "name": "积分", "keywords": ["苹果"]},
            {"id": "B", "name": "级数", "keywords": ["香蕉"]}]}]}]
    rows = [{"id": f"fixture-{i}", "group_id": f"independent-{i}", "subject": "高等数学", "qtype": "text",
             "raw_text": f"苹果 火龙果 {i}", "annotation": "", "label_status": "human_verified",
             "labels": {"knowledge": "B"}, "provenance": "synthetic test fixture, not acceptance evidence"}
            for i in range(8)]
    scheme_path, dataset = tmp_path / "schemes.json", tmp_path / "questions.jsonl"
    scheme_path.write_text(json.dumps(schemes, ensure_ascii=False), encoding="utf-8")
    dataset.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    return dataset, scheme_path


def test_real_engine_feedback_loop_is_isolated_auditable_and_uses_all_seeds(tmp_path, monkeypatch):
    dataset, schemes = write_fixture(tmp_path)
    user_workspace = tmp_path / "user-workspace"
    user_workspace.mkdir()
    sentinel = user_workspace / "keep.txt"
    sentinel.write_text("do not modify", encoding="utf-8")
    monkeypatch.setenv("MC_WORKSPACE", str(user_workspace))
    before_environment = dict(os.environ)
    report = evaluation.run_evaluation(dataset, tmp_path / "report.json", schemes=schemes)
    assert dict(os.environ) == before_environment
    assert list(user_workspace.iterdir()) == [sentinel]
    assert report["dataset"]["sha256"] == evaluation.file_hash(dataset)
    assert report["provenance"]["custom_schemes_sha256"] == evaluation.file_hash(schemes)
    assert report["provenance"]["rules_and_code"]["files"]["app/core/classify/engine.py"]
    assert report["protocol"]["exact_training_sample_overlaps"] == []
    assert len(report["baseline"]["cold_predictions"]) == len(report["baseline"]["warm_predictions"]) == 8
    assert report["baseline"]["by_label_status"]["human_verified"]["summary"]["denominator"] == 8
    trials = report["feedback"]["human_verified"]["trials"]
    assert [trial["allocation"]["seed"] for trial in trials] == list(evaluation.FIXED_SEEDS)
    for trial in trials:
        allocation = trial["allocation"]
        assert set(allocation["correction_ids"]).isdisjoint(allocation["test_ids"])
        assert {item["id"] for item in trial["before_predictions"]} == set(allocation["test_ids"])
        assert {item["id"] for item in trial["after_predictions"]} == set(allocation["test_ids"])
        assert any(change["applied"] for change in trial["corrections"])
        assert all(change["id"] in allocation["correction_ids"] for change in trial["corrections"])
        assert trial["comparison"]["before"]["denominator"] == len(allocation["test_ids"])
        baseline = {item["id"]: item for item in report["baseline"]["warm_predictions"]}
        for item in trial["before_predictions"]:
            assert item["predictions"] == baseline[item["id"]]["predictions"]
        assert trial["state_after"]["feedback_documents_by_subject"]["高等数学"] > 0
    assert report["requirements"]["full_flow_per_question_1_second"]["status"] != "pass"
    assert "persistence" in report["protocol"]["timing_scope"]
    assert report["requirements"]["full_flow_per_question_1_second"]["measurement_scope"].endswith("only")
    assert report["requirements"]["windows_and_macos"]["status"] == "pending"


def test_unreviewed_chapters_do_not_turn_into_accuracy_or_feedback_scores(tmp_path):
    dataset, schemes = write_fixture(tmp_path)
    rows = evaluation.load_rows(dataset)
    for row in rows:
        row.update(label_status="source_derived", labels={}, gold_knowledge=None,
                   source_labels={"chapter": "Source chapter", "section": "Section"})
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    report = evaluation.run_evaluation(dataset, tmp_path / "unlabeled-report.json", schemes=schemes)
    assert report["baseline"]["by_label_status"]["source_derived"]["summary"]["accuracy"] is None
    assert report["baseline"]["by_label_status"]["human_verified"]["summary"]["accuracy"] is None
    assert report["feedback"]["source_derived"]["mean_improvement_percentage_points"] is None
    assert all(trial["status"] == "pending" for trial in report["feedback"]["source_derived"]["trials"])
    assert report["requirements"]["text_accuracy_95_percent"]["status"] == "pending"
    assert report["requirements"]["overall_accuracy_90_percent"]["status"] == "pending"
    assert report["requirements"]["feedback_improvement_5_percent"]["status"] == "pending"
    assert len(report["baseline"]["warm_predictions"]) == len(rows)


def test_missing_scheme_failure_remains_in_real_engine_accuracy_denominator(tmp_path):
    dataset, schemes = write_fixture(tmp_path)
    rows = evaluation.load_rows(dataset)
    rows.append({"id": "missing-scheme", "group_id": "missing", "subject": "未配置课程", "qtype": "text",
                 "raw_text": "无法匹配课程的题目", "label_status": "human_verified", "labels": {"knowledge": "x"}})
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    report = evaluation.run_evaluation(dataset, tmp_path / "failure.json", schemes=schemes, feedback=False)
    summary = report["baseline"]["by_label_status"]["human_verified"]["summary"]
    assert summary["denominator"] == 9
    assert summary["invalid_gold_rows"] == 1
    assert any(item["id"] == "missing-scheme" and "No classification scheme" in item["error"]
               for item in report["baseline"]["warm_predictions"])
