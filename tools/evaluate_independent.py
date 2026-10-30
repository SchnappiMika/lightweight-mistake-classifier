"""Independent, auditable evaluation; importing this module changes no workspace.

Run with ``python -m tools.evaluate_independent INPUT.jsonl --output REPORT.json``.
Inference and feedback run in a child process with a disposable MC_WORKSPACE.
Chapter-derived labels are exploratory and never count as human verification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED_SEEDS = (17, 42, 2026)
STATUSES = ("human_verified", "source_derived", "unlabeled")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        values = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(values, dict):
            values = values.get("questions", values.get("rows"))
    if not isinstance(values, list):
        raise ValueError("Dataset must be a list or JSONL records")
    rows, identities = [], set()
    for original in values:
        if not isinstance(original, dict) or not isinstance(original.get("id"), str) or not original["id"]:
            raise ValueError("Each question needs a nonempty string id")
        row = deepcopy(original)
        if row["id"] in identities:
            raise ValueError("Duplicate question id: " + row["id"])
        identities.add(row["id"])
        if row.get("label_status", "unlabeled") not in STATUSES:
            raise ValueError("Unknown label_status for " + row["id"])
        if not isinstance(row.get("labels", {}), dict):
            raise ValueError("labels must be an object: " + row["id"])
        row["raw_text"] = row.get("raw_text", row.get("text", ""))
        for field in ("raw_text", "annotation", "subject", "qtype"):
            if not isinstance(row.get(field, ""), str):
                raise ValueError("Invalid " + field + " for " + row["id"])
        row.setdefault("label_status", "unlabeled")
        # Unknown context is kept in one group, rather than pretending each row
        # is independent. This deliberately blocks feedback on ungrouped data.
        row.setdefault("group_id", "__ungrouped__")
        if not isinstance(row["group_id"], str) or not row["group_id"]:
            raise ValueError("Invalid group_id for " + row["id"])
        rows.append(row)
    return rows


def effective_labels(row, source_mapping=None):
    labels = dict(row.get("labels") or {})
    if not labels and row.get("gold_knowledge"):
        labels = {"knowledge": row["gold_knowledge"]}
    if row.get("label_status") != "source_derived" or labels or not source_mapping:
        return labels
    matched = []
    for rule in source_mapping.get("rules", []):
        criteria = rule.get("source_labels", {})
        if (criteria and (not rule.get("subject") or rule["subject"] == row.get("subject"))
                and all(row.get("source_labels", {}).get(key) == value for key, value in criteria.items())):
            matched.append(rule.get("labels", {}))
    if len(matched) > 1 and any(value != matched[0] for value in matched[1:]):
        raise ValueError("Conflicting source label mappings for " + row["id"])
    return dict(matched[0]) if matched else {}


def fixed_group_split(rows, seed):
    """Merge explicit groups and exact content duplicates before splitting."""
    parents = {}

    def find(group):
        parents.setdefault(group, group)
        if parents[group] != group:
            parents[group] = find(parents[group])
        return parents[group]

    def union(left, right):
        left, right = find(left), find(right)
        parents[max(left, right)] = min(left, right)

    fingerprints = {}
    for row in rows:
        group = row.get("group_id") or "__ungrouped__"
        find(group)
        content = " ".join(row.get("raw_text", "").lower().split())
        if not content:
            content = "media:" + (row.get("media_sha256") or row.get("media_path") or row["id"])
        if content in fingerprints:
            union(group, fingerprints[content])
        else:
            fingerprints[content] = group
    groups = defaultdict(list)
    for row in rows:
        groups[find(row.get("group_id") or "__ungrouped__")].append(row["id"])
    ordered = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
    cut = len(ordered) // 2
    correction_groups, test_groups = ordered[:cut], ordered[cut:]
    return {"seed": seed, "group_order": ordered,
            "merged_groups": {group: sorted(identities) for group, identities in sorted(groups.items())},
            "correction_groups": correction_groups, "test_groups": test_groups,
            "correction_ids": sorted(identity for group in correction_groups for identity in groups[group]),
            "test_ids": sorted(identity for group in test_groups for identity in groups[group])}


def summarize(predictions, status=None):
    selected = [item for item in predictions if status is None or item["label_status"] == status]
    valid = [item for item in selected if item.get("gold", {}).get("knowledge", {}).get("valid")]
    declared = [item for item in selected if "knowledge" in item.get("gold", {})]
    correct = sum(bool(item.get("correct")) for item in declared)
    confusion = defaultdict(Counter)
    for item in declared:
        gold = item["gold"]["knowledge"].get("name") or str(item["gold"]["knowledge"].get("provided"))
        predicted = item.get("predictions", {}).get("knowledge", {}).get("name") or "__FAILED_OR_EMPTY__"
        confusion[gold][predicted] += 1
    return {"rows": len(selected), "denominator": len(declared), "valid_gold_rows": len(valid),
            "unlabeled_rows": len(selected) - len(declared),
            "invalid_gold_rows": len(declared) - len(valid),
            "prediction_failures": sum(bool(item.get("error")) for item in selected),
            "correct": correct, "accuracy": correct / len(declared) if valid else None,
            "confusion_matrix": {gold: dict(sorted(counts.items())) for gold, counts in sorted(confusion.items())}}


def grouped_summary(predictions, field, status):
    values = sorted({item[field] for item in predictions if item["label_status"] == status})
    return {value: summarize([item for item in predictions if item[field] == value], status) for value in values}


def timing_summary(predictions):
    values = sorted(item["elapsed_ms"] for item in predictions)
    return {"count": len(values), "mean_ms": statistics.mean(values) if values else None,
            "median_ms": statistics.median(values) if values else None,
            "p95_ms": values[max(0, math.ceil(.95 * len(values)) - 1)] if values else None,
            "max_ms": max(values) if values else None,
            "over_1000ms": sum(value > 1000 for value in values),
            "failed_rows": sum(bool(item["error"]) for item in predictions)}


def feedback_comparison(before, after):
    b, a = summarize(before), summarize(after)
    before_by_id = {item["id"]: item for item in before}
    compared = [item for item in after if "knowledge" in item.get("gold", {})]
    change = None if b["accuracy"] is None or a["accuracy"] is None else a["accuracy"] - b["accuracy"]
    return {"before": b, "after": a,
            "improvement_percentage_points": change * 100 if change is not None else None,
            "relative_improvement_percent": change / b["accuracy"] * 100
            if change is not None and b["accuracy"] else None,
            "improved_ids": [item["id"] for item in compared
                             if item.get("correct") and not before_by_id[item["id"]].get("correct")],
            "regressed_ids": [item["id"] for item in compared
                              if not item.get("correct") and before_by_id[item["id"]].get("correct")],
            "unchanged_ids": [item["id"] for item in compared
                              if bool(item.get("correct")) == bool(before_by_id[item["id"]].get("correct"))]}


def _resource_fingerprints():
    files = list((ROOT / "resources").rglob("*.json")) + list((ROOT / "resources").rglob("*.txt"))
    files += list((ROOT / "app/core/classify").glob("*.py"))
    files += list((ROOT / "app/core/preprocess").glob("*.py")) + [ROOT / "app/config.py", Path(__file__)]
    hashes = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in sorted(set(files))}
    combined = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"sha256": combined, "files": hashes}


def _worker_evaluate(args):
    # Delayed imports are critical: no app configuration is loaded in the parent.
    import_started = time.perf_counter()
    from app import config
    from app.core.classify.engine import ClassificationEngine
    from app.core.classify.feedback import FeedbackOptimizer
    from app.core.models import Question, Scheme
    from app.core.preprocess import pipeline
    from app.core.preprocess.media_preparation import prepare_question_media
    from app.core.storage import Repository
    imports_ms = (time.perf_counter() - import_started) * 1000
    dataset = Path(args.dataset).resolve()
    rows = load_rows(dataset)
    mapping = json.loads(Path(args.source_label_map).read_text(encoding="utf-8")) if args.source_label_map else None
    for row in rows:
        row["evaluation_labels"] = effective_labels(row, mapping)
        if row.get("media_path"):
            path = Path(row["media_path"])
            row["media_path"] = str(path if path.is_absolute() else dataset.parent / path)
            try:
                row["media_sha256"] = file_hash(row["media_path"])
            except OSError:
                row["media_sha256"] = None
    if args.schemes:
        scheme_data = json.loads(Path(args.schemes).read_text(encoding="utf-8"))
        if isinstance(scheme_data, dict):
            scheme_data = scheme_data.get("schemes", [scheme_data])
    else:
        scheme_data = [scheme.to_dict() for scheme in Repository().schemes]
    training_texts = defaultdict(list)
    for scheme in scheme_data:
        for dimension in scheme.get("dims", []):
            for label in dimension.get("labels", []):
                for text in label.get("samples", []):
                    training_texts[" ".join(text.lower().split())].append(
                        {"scheme_id": scheme.get("id"), "dimension": dimension.get("key"), "label_id": label.get("id")})
    overlaps = [{"id": row["id"], "training_samples": training_texts[" ".join(row["raw_text"].lower().split())]}
                for row in rows if row["raw_text"].strip()
                and " ".join(row["raw_text"].lower().split()) in training_texts]
    sandbox = Path(os.environ["MC_WORKSPACE"])

    class Evaluation:
        def __init__(self, name):
            folder = sandbox / name
            folder.mkdir(parents=True, exist_ok=True)
            self.profile_path = folder / "profiles.json"
            self.weight_path = folder / "weights.json"
            self.activate()
            self.schemes = [Scheme.from_dict(deepcopy(value)) for value in scheme_data]
            self.engines = {scheme.subject: ClassificationEngine(scheme) for scheme in self.schemes}

        def activate(self):
            # Engine persistence reads config dynamically. Bind every operation
            # to its own trial, even if evaluators are interleaved in future.
            config.PROFILE_FILE = self.profile_path
            config.WEIGHT_FILE = self.weight_path

        def question(self, row):
            return Question(id=row["id"], raw_text=row.get("raw_text", ""),
                            annotation=row.get("annotation", ""), subject=row.get("subject", ""),
                            qtype=row.get("qtype", "text"), media_path=row.get("media_path", ""))

        def predict(self, row):
            self.activate()
            started = time.perf_counter()
            item = {"id": row["id"], "group_id": row["group_id"], "subject": row.get("subject", ""),
                    "qtype": row.get("qtype", "text"), "label_status": row["label_status"],
                    "input": deepcopy(row), "gold": {}, "predictions": {}, "error": "", "correct": False}
            engine = self.engines.get(row.get("subject"))
            for dimension, provided in row["evaluation_labels"].items():
                dim = engine.scheme.dim(dimension) if engine else None
                label = (dim.label_by_id(str(provided)) or dim.label_by_name(str(provided))) if dim else None
                item["gold"][dimension] = {"provided": provided, "valid": label is not None,
                                           "id": label.id if label else None, "name": label.name if label else None}
            question = self.question(row)
            try:
                if engine is None:
                    raise ValueError("No classification scheme for subject: " + row.get("subject", ""))
                prepared = prepare_question_media(question)
                feature = pipeline.preprocess(question, use_unreviewed_media=True)
                result = engine.classify(question, feature)
                for key, dim_result in result.dims.items():
                    item["predictions"][key] = {"id": dim_result.label_id, "name": dim_result.label_name,
                        "score": dim_result.score, "candidates": [candidate.to_dict() for candidate in dim_result.candidates]}
                item["feature"] = feature.to_dict()
                item["media_recognized"] = prepared.recognized
                item["media_kind"] = prepared.kind
                item["media_elapsed_ms"] = prepared.elapsed_ms if prepared.recognized else 0
                ClassificationEngine.apply_result(question, result, engine.scheme, feature)
                item["question_after"] = question.to_dict()
                target = item["gold"].get("knowledge", {})
                item["correct"] = bool(target.get("valid") and
                                         target["id"] == item["predictions"]["knowledge"]["id"])
                if not item["predictions"].get("knowledge", {}).get("id"):
                    raise ValueError("No knowledge prediction")
            except Exception as exc:
                item["error"] = type(exc).__name__ + ": " + str(exc)
            item["elapsed_ms"] = (time.perf_counter() - started) * 1000
            return item, question

    baseline = Evaluation("baseline")
    cold = [baseline.predict(row)[0] for row in rows]
    warm = [baseline.predict(row)[0] for row in rows]
    source_coverage = Counter((row.get("subject", ""), row.get("source_labels", {}).get("chapter", ""),
                               row.get("source_labels", {}).get("section", "")) for row in rows)
    report = {"format": "independent-evaluation-v1", "dataset": {"path": str(dataset), "sha256": file_hash(dataset),
              "rows": len(rows), "label_status_counts": dict(Counter(row["label_status"] for row in rows)),
              "source_label_coverage": [{"subject": s, "chapter": c, "section": sec, "count": n}
                                        for (s, c, sec), n in sorted(source_coverage.items())]},
              "provenance": {"rules_and_code": _resource_fingerprints(),
                "source_label_mapping_sha256": file_hash(args.source_label_map) if args.source_label_map else None,
                "custom_schemes_sha256": file_hash(args.schemes) if args.schemes else None,
                "python": sys.version, "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "jieba")}},
              "protocol": {"seeds_predeclared": list(FIXED_SEEDS), "correction_group_fraction": .5,
                "exact_training_sample_overlaps": overlaps,
                "feedback_selection": "all incorrectly predicted labeled correction items; deterministic id order",
                "source_derived_is_human_gold": False,
                "cold_scope": "fresh evaluator; includes lazy classifier/lexicon initialization in first affected rows",
                "warm_scope": "second pass, classifier indices warm; media jobs still run in separate processes",
                "timing_scope": "processing only: media preparation, feature extraction and classification; excludes input import and repository persistence",
                "process_import_ms": imports_ms,
                "feedback_does_not_promote_samples_or_modify_schemes": True},
              "baseline": {"cold_predictions": cold, "warm_predictions": warm,
                "cold_timing": timing_summary(cold), "warm_timing": timing_summary(warm),
                "by_label_status": {status: {"summary": summarize(warm, status),
                    "by_subject": grouped_summary(warm, "subject", status),
                    "by_qtype": grouped_summary(warm, "qtype", status)} for status in STATUSES}}, "feedback": {}}

    for status in ("human_verified", "source_derived"):
        domain_rows = [row for row in rows if row["label_status"] == status]
        trials = []
        for seed in FIXED_SEEDS:
            allocation = fixed_group_split(domain_rows, seed)
            trial = {"allocation": allocation, "status": "pending", "reason": "", "corrections": []}
            labeled = [row for row in domain_rows if row["evaluation_labels"].get("knowledge")]
            if args.no_feedback or not labeled or not allocation["correction_ids"] or not allocation["test_ids"]:
                trial["reason"] = ("feedback disabled" if args.no_feedback else
                                   "no supplied labels or fewer than two independent groups")
                trials.append(trial)
                continue
            evaluator = Evaluation(f"feedback-{status}-{seed}")
            correction_ids, test_ids = set(allocation["correction_ids"]), set(allocation["test_ids"])
            heldout = sorted((row for row in domain_rows if row["id"] in test_ids), key=lambda row: row["id"])
            before = [evaluator.predict(row)[0] for row in heldout]
            for row in sorted((row for row in domain_rows if row["id"] in correction_ids), key=lambda row: row["id"]):
                prediction, question = evaluator.predict(row)
                change = {"id": row["id"], "prediction_before": prediction, "applied": [], "errors": []}
                # An abstention with valid extracted features can be corrected;
                # model/file failures without features cannot be trained on.
                if prediction.get("feature"):
                    evaluator.activate()
                    engine = evaluator.engines.get(row["subject"])
                    for dimension, gold in prediction["gold"].items():
                        if not gold["valid"] or question.labels.get(dimension) == gold["id"]:
                            continue
                        try:
                            record = FeedbackOptimizer(engine).apply_correction(question, dimension, gold["id"])
                            if record is not None:
                                change["applied"].append(record.to_dict())
                        except Exception as exc:
                            change["errors"].append(type(exc).__name__ + ": " + str(exc))
                trial["corrections"].append(change)
            after = [evaluator.predict(row)[0] for row in heldout]
            corrected_types = {(change["prediction_before"]["subject"], change["prediction_before"]["qtype"])
                               for change in trial["corrections"] if change["applied"]}
            same_type_before = [item for item in before if (item["subject"], item["qtype"]) in corrected_types]
            same_type_after = [item for item in after if (item["subject"], item["qtype"]) in corrected_types]
            trial.update(before_predictions=before, after_predictions=after,
                         comparison=feedback_comparison(before, after),
                         corrected_subject_qtypes=[{"subject": subject, "qtype": qtype}
                                                   for subject, qtype in sorted(corrected_types)],
                         same_type_comparison=feedback_comparison(same_type_before, same_type_after))
            trial["state_after"] = {
                "profile_sha256": file_hash(evaluator.profile_path) if evaluator.profile_path.exists() else None,
                "weights_sha256": file_hash(evaluator.weight_path) if evaluator.weight_path.exists() else None,
                "feedback_documents_by_subject": {subject: engine._feedback_n_docs
                                                   for subject, engine in evaluator.engines.items()}}
            trial["status"] = "measured" if trial["comparison"]["before"]["accuracy"] is not None else "pending"
            trials.append(trial)
        changes = [trial["comparison"]["improvement_percentage_points"] for trial in trials
                   if trial.get("comparison", {}).get("improvement_percentage_points") is not None]
        report["feedback"][status] = {"trials": trials,
            "mean_improvement_percentage_points": statistics.mean(changes) if changes else None,
            "same_type_scope": "unseen test rows sharing (subject,qtype) with at least one applied correction",
            "all_predeclared_seeds_completed": len(changes) == len(FIXED_SEEDS)}

    def accuracy_requirement(items, threshold, required_qtypes=None):
        metric = summarize(items, "human_verified")
        qtypes = {item["qtype"] for item in items if item["label_status"] == "human_verified"
                  and item.get("gold", {}).get("knowledge", {}).get("valid")}
        state = ("pending" if not metric["valid_gold_rows"] else
                 "failed" if metric["invalid_gold_rows"] or metric["accuracy"] < threshold else
                 "pending" if overlaps or (required_qtypes and not set(required_qtypes) <= qtypes) else "pass")
        return {"original_threshold": threshold, "status": state, "evidence": metric,
                "missing_qtypes": sorted(set(required_qtypes or []) - qtypes),
                "scope": "only supplied human-verified independent questions; source-derived rows excluded"}

    times = cold + warm
    processing_over_budget = any(item["elapsed_ms"] > 1000 for item in times)
    human_trials = report["feedback"]["human_verified"]["trials"]
    comparisons = [trial["same_type_comparison"] for trial in human_trials
                   if trial.get("same_type_comparison", {}).get("before", {}).get("accuracy") is not None]
    all_reach_both = len(comparisons) == len(FIXED_SEEDS) and all(
        value["improvement_percentage_points"] >= 5 and value["relative_improvement_percent"] is not None
        and value["relative_improvement_percent"] >= 5 for value in comparisons)
    feedback_state = ("pending" if overlaps else "pass" if all_reach_both else
                      "failed" if len(comparisons) == len(FIXED_SEEDS) else "pending")
    report["requirements"] = {
        "text_accuracy_95_percent": accuracy_requirement([item for item in warm if item["qtype"] == "text"], .95),
        "overall_accuracy_90_percent": accuracy_requirement(warm, .90, config.QTYPES),
        "feedback_improvement_5_percent": {"original_threshold": "5%", "status": feedback_state,
            "interpretation": "reports both percentage points and relative percent; pass requires both >=5 on every predeclared seed",
            "scope": "same-type unseen heldout questions, grouped by corrected (subject,qtype)",
            "source_derived_trials_excluded": True},
        "full_flow_per_question_1_second": {"original_threshold_ms": 1000,
            "status": "failed" if processing_over_budget else "pending",
            "reason": "measured processing is a lower bound; file import and repository persistence are not timed, so this tool cannot establish a full-flow pass",
            "measurement_scope": "media preparation + feature extraction + classification only",
            "processing_over_1000ms": processing_over_budget,
            "cold": timing_summary(cold), "warm": timing_summary(warm)},
        "windows_and_macos": {"status": "pending", "reason": "one runtime cannot establish both operating systems"},
        "student_trial_records": {"status": "pending", "reason": "no independently verified student-trial evidence collected by this tool"}}
    report["overall_acceptance_status"] = "failed" if any(
        requirement["status"] == "failed" for requirement in report["requirements"].values()) else "pending"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def run_evaluation(dataset, output, *, schemes=None, source_label_map=None, feedback=True):
    """Run safely even if app.config was already imported by another caller."""
    command = [sys.executable, "-m", "tools.evaluate_independent", str(Path(dataset).resolve()),
               "--output", str(Path(output).resolve()), "--_worker"]
    if schemes:
        command.extend(["--schemes", str(Path(schemes).resolve())])
    if source_label_map:
        command.extend(["--source-label-map", str(Path(source_label_map).resolve())])
    if not feedback:
        command.append("--no-feedback")
    with tempfile.TemporaryDirectory(prefix="mc_independent_eval_") as temporary:
        environment = dict(os.environ, MC_WORKSPACE=temporary, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MC_INDEPENDENT_EVAL_WORKER="1")
        result = subprocess.run(command, cwd=str(ROOT), env=environment, capture_output=True, text=True, encoding="utf-8")
        if result.returncode:
            raise RuntimeError("Independent evaluation failed: " + (result.stderr or result.stdout)[-6000:])
    return json.loads(Path(output).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--schemes", type=Path, help="Optional fixed custom schemes JSON; never modified")
    parser.add_argument("--source-label-map", type=Path, help="Explicit source_labels-to-labels rules, exploratory only")
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        if os.environ.get("MC_INDEPENDENT_EVAL_WORKER") != "1":
            parser.error("Internal worker must be launched through the isolated parent")
        report = _worker_evaluate(args)
    else:
        report = run_evaluation(args.dataset, args.output, schemes=args.schemes,
                                source_label_map=args.source_label_map, feedback=not args.no_feedback)
    print(json.dumps({"report": str(args.output), "rows": report["dataset"]["rows"],
                      "requirements": {key: value["status"] for key, value in report["requirements"].items()}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
