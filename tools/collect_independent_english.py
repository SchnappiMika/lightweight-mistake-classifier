"""Freeze real SQuAD reading-comprehension questions without answer leakage.

Selection is fixed before inference: original file order, first 30 distinct
articles, first nonempty unique passage and its first answerable valid question.
Only source data is CC BY-SA 4.0; the application's license is unaffected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


SOURCE_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
SITE_URL = "https://rajpurkar.github.io/SQuAD-explorer/"
README_URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/README.md"
SOFTWARE_LICENSE_URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/LICENSE"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
MAX_BYTES = 40 * 1024 * 1024
SELECTION = "Original JSON article order; first 30 distinct articles; first nonempty unique paragraph and its first answerable question with exact valid answer spans. No classifier output is consulted."


def sha(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download(url, path):
    meta_path = path.with_suffix(path.suffix + ".json")
    if path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if sha(path.read_bytes()) != metadata["sha256"]:
            raise ValueError("Cached source hash mismatch: " + str(path))
        return metadata
    request = urllib.request.Request(url, headers={"User-Agent": "mistake-classifier-educational-research/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > MAX_BYTES:
            raise ValueError("Source exceeds the authorized 40 MiB text-data limit.")
        data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError("Source exceeds the authorized 40 MiB text-data limit.")
        resolved_url = response.url
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    metadata = {"url": url, "resolved_url": resolved_url,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "bytes": len(data), "sha256": sha(data)}
    write_json(meta_path, metadata)
    return metadata


def select(source, count=30):
    selected, seen_articles, seen_passages = [], set(), set()
    for article_index, article in enumerate(source["data"]):
        title = article.get("title", "")
        if not title or title in seen_articles:
            continue
        candidate = None
        for paragraph_index, paragraph in enumerate(article.get("paragraphs", [])):
            context = paragraph.get("context", "")
            if not context.strip() or sha(context) in seen_passages:
                continue
            for question_index, question in enumerate(paragraph.get("qas", [])):
                prompt, answers = question.get("question", ""), question.get("answers", [])
                if not prompt.strip() or not question.get("id") or not answers:
                    continue
                if all(isinstance(answer.get("answer_start"), int) and answer["answer_start"] >= 0 and
                       isinstance(answer.get("text"), str) and answer["text"] and
                       context[answer["answer_start"]:answer["answer_start"] + len(answer["text"])] == answer["text"]
                       for answer in answers):
                    candidate = {"article_index": article_index, "paragraph_index": paragraph_index,
                                 "question_index": question_index, "article": title,
                                 "context": context, "question": prompt,
                                 "source_id": question["id"], "answers": answers}
                    break
            if candidate:
                break
        if candidate:
            selected.append(candidate)
            seen_articles.add(title)
            seen_passages.add(sha(candidate["context"]))
        if len(selected) == count:
            return selected
    raise ValueError("Insufficient distinct articles with valid human question-answer spans.")


def collect(output, count=30):
    output.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for url, filename in ((SOURCE_URL, "dev-v1.1.json"), (SITE_URL, "official_homepage.html"),
                          (README_URL, "official_README.md"), (SOFTWARE_LICENSE_URL, "repository_software_LICENSE.txt")):
        relative = "sources/" + filename
        snapshots.append(dict(download(url, output / relative), file=relative))
    homepage = (output / "sources/official_homepage.html").read_text(encoding="utf-8")
    if "dataset (distributed under" not in homepage or "creativecommons.org/licenses/by-sa/4.0/legalcode" not in homepage:
        raise ValueError("Official dataset CC BY-SA 4.0 license declaration could not be verified.")
    source = json.loads((output / "sources/dev-v1.1.json").read_text(encoding="utf-8"))
    if source.get("version") != "1.1":
        raise ValueError("Unexpected SQuAD version.")
    inputs, answers = [], []
    for candidate in select(source, count):
        title = candidate["article"]
        identifier = "squad_1_1_" + candidate["source_id"]
        wikipedia_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title, safe="_")
        article_url = SITE_URL + "explore/1.1/dev/" + urllib.parse.quote(title, safe="_") + ".html"
        row = {"id": identifier, "group_id": "squad_1_1:article:" + title,
               "subject": "大学英语", "qtype": "text", "language": "en",
               "raw_text": candidate["context"] + "\n\nQuestion: " + candidate["question"],
               "context": candidate["context"], "question": candidate["question"], "annotation": "",
               "labels": {}, "label_status": "source_derived",
               "source_labels": {"task": "reading_comprehension", "article": title},
               "gold_knowledge": None, "review_status": "pending_human_review",
               "source_url": SOURCE_URL, "source_title": "SQuAD 1.1 development set / " + title,
               "source_id": candidate["source_id"], "article": title,
               "article_url": article_url, "wikipedia_url": wikipedia_url,
               "source_locator": {key: candidate[key] for key in ("article_index", "paragraph_index", "question_index")},
               "source_file": "sources/dev-v1.1.json", "source_sha256": snapshots[0]["sha256"],
               "retrieved_at": snapshots[0]["retrieved_at"], "body_sha256": sha(candidate["context"] + "\n" + candidate["question"]),
               "passage_sha256": sha(candidate["context"]), "license": "CC BY-SA 4.0", "license_url": LICENSE_URL,
               "license_evidence_file": "sources/official_homepage.html",
               "attribution": "Stanford Question Answering Dataset (SQuAD), Rajpurkar et al.; English Wikipedia contributors; " + wikipedia_url,
               "transformations": ["Original passage and original human question unchanged; joined with a Question separator"],
               "scope": "Public English reading-comprehension material; not local university examinations or verified syllabus gold."}
        inputs.append(row)
        answers.append({"id": identifier, "source_id": candidate["source_id"],
                        "answers": candidate["answers"], "status": "original_human_answer_spans",
                        "source_url": SOURCE_URL, "source_sha256": snapshots[0]["sha256"]})
    for filename, rows in (("questions.jsonl", inputs), ("answers.jsonl", answers)):
        (output / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    partitions = {"development": [], "validation": [], "holdout": []}
    assignment = {}
    for index, row in enumerate(inputs):
        split = ("development", "validation", "holdout")[index % 3]
        partitions[split].append(row["id"])
        assignment[row["group_id"]] = split
    write_json(output / "splits.json", {"policy": "Article groups are disjoint; selection precedes classification.",
                                        "group_assignments": assignment, "partitions": partitions})
    manifest = {"schema_version": 1, "question_count": len(inputs), "passage_count": len(inputs),
                "article_count": len(inputs), "subject": "大学英语", "language": "en",
                "selection_policy": SELECTION.replace("30", str(count)),
                "original_dataset": "SQuAD 1.1 development set", "generated_question_count": 0,
                "human_gold_count": 0, "original_human_answer_span_question_count": len(inputs),
                "label_status": "source_derived", "source_labels_meaning": "Original article and dataset reading-comprehension task; not human-verified classifier knowledge labels.",
                "license": "CC BY-SA 4.0", "license_url": LICENSE_URL,
                "license_evidence": {"url": SITE_URL, "file": "sources/official_homepage.html",
                    "declaration": "Download a copy of the dataset (distributed under the CC BY-SA 4.0 license)",
                    "repository_license_distinction": "Repository LICENSE is MIT for explorer software; dataset license is independently stated on the official homepage."},
                "license_scope": "Only this collected SQuAD/Wikipedia reading corpus; not the application's source code or OpenStax corpus.",
                "limitations": ["Public reading-comprehension material, not authentic local university examinations.",
                    "Original human questions and answer spans exist, but there are no human classifier knowledge/error/difficulty labels.",
                    "Wikipedia links identify original article titles; exact historical revision is not specified by this source JSON.",
                    "Raw original source JSON contains answers and must never be used as classifier input; use questions.jsonl only.",
                    "Answers are kept solely in answers.jsonl and raw source; do not train or tune on holdout."],
                "sources": snapshots, "dataset_files": {name: sha((output / name).read_bytes())
                    for name in ("questions.jsonl", "answers.jsonl", "splits.json")}}
    write_json(output / "manifest.json", manifest)
    (output / "LICENSE.txt").write_text(
        "Collected SQuAD dataset / Wikipedia passages: CC BY-SA 4.0.\n" + LICENSE_URL + "\n"
        "SQuAD: Stanford Question Answering Dataset, Rajpurkar et al.; passages: English Wikipedia contributors.\n"
        "Original official data, source URLs, article titles, and attribution are preserved per record.\n"
        "Adaptation only joins the original passage and human question; no generated questions.\n"
        "This applies only to collected text data. It does not change the software license.\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("samples/independent_english"))
    args = parser.parse_args(argv)
    manifest = collect(args.output)
    print(json.dumps({"questions": manifest["question_count"], "articles": manifest["article_count"],
                      "license": manifest["license"], "human_classifier_gold": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
