"""Audit the collected raw sources and keep evaluator inputs free of gold leaks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.collect_independent_corpus import (SOURCES, LICENSE_URL, body_fingerprint,
                                             deduplicate, extract, sha)


ROOT = Path(__file__).resolve().parents[1] / "samples" / "independent_acceptance"


def load_rows():
    return [json.loads(line) for line in (ROOT / "questions.jsonl").read_text(encoding="utf-8").splitlines()]


def test_collected_sources_have_at_least_150_real_unique_numbered_exercises():
    rows = load_rows()
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == manifest["question_count"] >= 150
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["body_sha256"] for row in rows}) == len(rows)
    assert {row["subject"] for row in rows} == {"高等数学", "大学物理"}
    assert manifest["human_gold_count"] == 0
    assert manifest["english_language_course_questions"] == 0
    for row in rows:
        assert row["exercise_number"].isdigit()
        assert row["source_url"].startswith("https://openstax.org/books/")
        assert row["source_url"].endswith("#" + row["source_id"])
        assert row["label_status"] == "source_derived" and row["labels"] == {}
        assert row["gold_knowledge"] is None and row["review_status"] == "pending_human_review"
        assert row["language"] == "en" and not row["annotation"]
        assert row["source_labels"] == {"chapter": row["chapter"], "section": row["section"]}
        assert row["body_sha256"] == body_fingerprint(row["raw_text"])
        assert row["raw_text"] == "\n".join(part for part in (row["shared_instructions"], row["problem_text"]) if part)
        assert row["license"] == "CC BY-NC-SA 4.0" and row["license_url"] == LICENSE_URL
        assert row["authors"] and row["publisher"] and row["retrieved_at"]


def test_raw_sources_hashes_and_offline_reproduction_match_frozen_questions():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    actual = {row["id"]: row for row in load_rows()}
    reproduced = []
    for source in SOURCES:
        book, _subject, _chapter, slug = source
        relative = "sources/" + book + "_" + slug + ".html"
        snapshot = ROOT / relative
        metadata = json.loads(snapshot.with_suffix(".html.json").read_text(encoding="utf-8"))
        assert sha(snapshot.read_bytes()) == metadata["sha256"]
        html = snapshot.read_text(encoding="utf-8")
        assert "creativecommons.org/licenses/by-nc-sa/4.0/" in html
        rows, _exclusions, _answers = extract(html, source, metadata, relative)
        for row in rows:
            assert row["source_id"] in html
            assert row["source_sha256"] == metadata["sha256"]
        reproduced.extend(rows)
    reproduced, _duplicates = deduplicate(reproduced)
    assert {row["id"]: row for row in reproduced} == actual
    for filename, digest in manifest["dataset_files"].items():
        assert sha((ROOT / filename).read_bytes()) == digest


def test_chapter_and_shared_instruction_groups_never_cross_partitions():
    splits = json.loads((ROOT / "splits.json").read_text(encoding="utf-8"))
    rows = load_rows()
    partitions = splits["partitions"]
    assert set(partitions) == {"development", "validation", "holdout"}
    by_id = {identifier: split for split, identifiers in partitions.items() for identifier in identifiers}
    assert len(by_id) == sum(map(len, partitions.values())) == len(rows)
    for key in ("group_id", "instruction_group_id"):
        grouping = {}
        for row in rows:
            grouping.setdefault(row[key], set()).add(by_id[row["id"]])
        assert all(len(values) == 1 for values in grouping.values())


def test_answers_are_separate_links_and_each_input_is_traceable():
    rows = load_rows()
    answers = [json.loads(line) for line in (ROOT / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["id"] for row in rows} == {row["id"] for row in answers}
    assert any(row["answer_url"] for row in answers)
    for row in rows:
        assert not {"answer", "answer_text", "solution", "answer_url"}.intersection(row)
        assert "Solution" not in row["raw_text"]
        assert "Show/Hide" not in row["raw_text"]
    for row in answers:
        assert row["answer_text"] == ""
        assert not row["answer_url"] or row["answer_url"].startswith("https://openstax.org/books/")


def test_formula_transcription_and_shared_instructions_are_structural_without_answer_leak():
    html = '''<title>3.3 Differentiation Rules - Calculus Volume 1 | OpenStax</title>
    <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">License</a>
    <section class="section-exercises"><p>For the following exercises, find the derivative.</p>
    <div data-type="exercise" id="real-source-id"><div data-type="problem">
    <span class="os-number">106</span><div class="os-problem-container"><p><math><semantics>
    <mfrac><msup><mi>x</mi><mn>2</mn></msup><mn>3</mn></mfrac>
    <annotation-xml><mi>duplicate representation must not appear</mi></annotation-xml>
    </semantics></math></p></div></div><details data-type="solution">SECRET ANSWER</details></div></section>'''
    metadata = {"url": "https://openstax.org/books/calculus-volume-1/pages/3-3-differentiation-rules",
                "retrieved_at": "2026-10-29T00:00:00Z", "sha256": sha(html)}
    rows, skipped, _answers = extract(html, SOURCES[1], metadata, "source.html")
    assert len(rows) == 1 and not skipped
    assert rows[0]["problem_text"] == r"$\frac{{x}^{2}}{3}$"
    assert rows[0]["shared_instructions"] == "For the following exercises, find the derivative."
    assert "SECRET" not in rows[0]["raw_text"] and "duplicate" not in rows[0]["raw_text"]


def test_ambiguous_or_figure_dependent_exercises_are_withheld_without_guessing():
    excluded = json.loads((ROOT / "exclusions.json").read_text(encoding="utf-8"))
    reasons = {row["reason"] for row in excluded}
    assert "shared_instruction_scope_requires_human_review" in reasons
    assert "requires_shared_figure" in reasons
    assert "requires_external_or_embedded_figure" in reasons
    assert "requires_referenced_equation_table_or_other_problem" in reasons


def test_missing_source_license_is_rejected():
    with pytest.raises(ValueError, match="license"):
        extract("<title>Unknown</title>", SOURCES[0], {"url": "https://example.org"}, "unknown.html")
