"""Frozen English inputs use original human questions, with separate answers."""
from __future__ import annotations

import json
from pathlib import Path

from tools.collect_independent_english import (LICENSE_URL, MAX_BYTES, SOURCE_URL,
                                               select, sha)


ROOT = Path(__file__).resolve().parents[1] / "samples" / "independent_english"


def records(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines()]


def test_thirty_distinct_original_articles_passages_and_human_questions():
    questions = records("questions.jsonl")
    assert len(questions) == 30
    for key in ("id", "group_id", "article", "passage_sha256", "body_sha256"):
        assert len({row[key] for row in questions}) == 30
    for row in questions:
        assert row["subject"] == "大学英语" and row["language"] == "en"
        assert row["question"].strip() and row["context"].strip()
        assert row["raw_text"] == row["context"] + "\n\nQuestion: " + row["question"]
        assert row["labels"] == {} and row["gold_knowledge"] is None
        assert row["label_status"] == "source_derived"
        assert row["source_labels"] == {"task": "reading_comprehension", "article": row["article"]}
        assert row["license"] == "CC BY-SA 4.0" and row["license_url"] == LICENSE_URL
        assert "not local university examinations" in row["scope"]
        assert row["source_url"] == SOURCE_URL and row["retrieved_at"]


def test_fixed_selection_matches_raw_source_order_and_original_text_exactly():
    source = json.loads((ROOT / "sources/dev-v1.1.json").read_text(encoding="utf-8"))
    chosen = select(source)
    questions = records("questions.jsonl")
    assert [row["article"] for row in questions] == [row["title"] for row in source["data"][:30]]
    assert [row["source_id"] for row in questions] == [row["source_id"] for row in chosen]
    for row in questions:
        locator = row["source_locator"]
        article = source["data"][locator["article_index"]]
        paragraph = article["paragraphs"][locator["paragraph_index"]]
        original_question = paragraph["qas"][locator["question_index"]]
        assert row["context"] == paragraph["context"]
        assert row["question"] == original_question["question"]
        assert row["source_id"] == original_question["id"]
        assert row["passage_sha256"] == sha(paragraph["context"])


def test_answer_records_are_separate_and_all_exact_spans_remain_valid():
    questions = {row["id"]: row for row in records("questions.jsonl")}
    answers = records("answers.jsonl")
    assert {row["id"] for row in answers} == set(questions)
    for row in questions.values():
        assert not {"answer", "answers", "answer_text", "answer_start"}.intersection(row)
    for row in answers:
        context = questions[row["id"]]["context"]
        assert row["status"] == "original_human_answer_spans"
        assert row["answers"]
        for answer in row["answers"]:
            assert context[answer["answer_start"]:answer["answer_start"] + len(answer["text"])] == answer["text"]


def test_dataset_license_comes_from_official_dataset_declaration_not_mit_software_license():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    homepage = (ROOT / "sources/official_homepage.html").read_text(encoding="utf-8")
    software_license = (ROOT / "sources/repository_software_LICENSE.txt").read_text(encoding="utf-8")
    assert "dataset (distributed under" in homepage
    assert "creativecommons.org/licenses/by-sa/4.0/legalcode" in homepage
    assert "MIT License" in software_license
    assert manifest["license"] == "CC BY-SA 4.0"
    assert manifest["human_gold_count"] == 0
    assert manifest["original_human_answer_span_question_count"] == 30
    assert manifest["generated_question_count"] == 0
    assert "not authentic local university examinations" in " ".join(manifest["limitations"])
    for source in manifest["sources"]:
        assert source["bytes"] < MAX_BYTES
        assert sha((ROOT / source["file"]).read_bytes()) == source["sha256"]
    for filename, digest in manifest["dataset_files"].items():
        assert sha((ROOT / filename).read_bytes()) == digest


def test_article_splits_are_complete_and_disjoint():
    splits = json.loads((ROOT / "splits.json").read_text(encoding="utf-8"))
    questions = records("questions.jsonl")
    partitions = splits["partitions"]
    assert {key: len(value) for key, value in partitions.items()} == {
        "development": 10, "validation": 10, "holdout": 10}
    by_id = {identifier: split for split, identifiers in partitions.items() for identifier in identifiers}
    assert len(by_id) == sum(map(len, partitions.values())) == len(questions)
    for row in questions:
        assert splits["group_assignments"][row["group_id"]] == by_id[row["id"]]
