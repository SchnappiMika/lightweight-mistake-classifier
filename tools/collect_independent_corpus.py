"""Collect numbered OpenStax exercises, preserving sources and separate answers.

Only the collected textbook material is CC BY-NC-SA 4.0. This does not change
the application's license. No generated questions or classification gold labels
are created. MathML is transcribed structurally to inline LaTeX; raw HTML is kept.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from datetime import datetime, timezone
import urllib.parse
import urllib.request


LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
SOURCES = [
    ("calculus-volume-1", "高等数学", "2 Limits", "2-2-the-limit-of-a-function"),
    ("calculus-volume-1", "高等数学", "3 Derivatives", "3-3-differentiation-rules"),
    ("calculus-volume-1", "高等数学", "5 Integration", "5-2-the-definite-integral"),
    ("university-physics-volume-1", "大学物理", "3 Motion Along a Straight Line", "3-problems"),
    ("university-physics-volume-1", "大学物理", "5 Newton's Laws of Motion", "5-problems"),
    ("university-physics-volume-1", "大学物理", "8 Potential Energy and Conservation of Energy", "8-problems"),
]
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "wbr"}


class Node:
    def __init__(self, tag="root", attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def has_class(self, name):
        return name in (self.attrs.get("class") or "").split()

    def find(self, predicate):
        return next((node for node in self.walk() if predicate(node)), None)

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()


class Tree(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


MATH_SYMBOLS = {"∫": r"\int ", "∑": r"\sum ", "∞": r"\infty ", "→": r"\to ",
                "π": r"\pi ", "θ": r"\theta ", "Δ": r"\Delta ", "−": "-", "·": r"\cdot ",
                "×": r"\times ", "≤": r"\le ", "≥": r"\ge ", "≠": r"\ne ", "∈": r"\in "}


def math_text(node):
    if isinstance(node, str):
        return "".join(MATH_SYMBOLS.get(char, char) for char in node.strip())
    if node.tag in {"annotation", "annotation-xml"}:
        return ""
    parts = [math_text(child) for child in node.children]
    parts = [part for part in parts if part]
    if node.tag == "mfrac" and len(parts) == 2:
        return r"\frac{" + parts[0] + "}{" + parts[1] + "}"
    if node.tag in {"msup", "msub", "munder"} and len(parts) == 2:
        operator = "^" if node.tag == "msup" else "_"
        return "{" + parts[0] + "}" + operator + "{" + parts[1] + "}"
    if node.tag == "mover" and len(parts) == 2:
        if parts[1].strip() == r"\to":
            return r"\vec{" + parts[0] + "}"
        if parts[1] == "^":
            return r"\hat{" + parts[0] + "}"
        return r"\overset{" + parts[1] + "}{" + parts[0] + "}"
    if node.tag in {"msubsup", "munderover"} and len(parts) == 3:
        return "{" + parts[0] + "}_{" + parts[1] + "}^{" + parts[2] + "}"
    if node.tag == "msqrt":
        return r"\sqrt{" + "".join(parts) + "}"
    if node.tag == "mroot" and len(parts) == 2:
        return r"\sqrt[" + parts[1] + "]{" + parts[0] + "}"
    if node.tag == "mspace":
        return " "
    if node.tag == "mtr":
        return " & ".join(parts) + r" \\ "
    if node.tag == "mtable":
        return r"\begin{matrix}" + "".join(parts) + r"\end{matrix}"
    return "".join(parts)


def render(node):
    if isinstance(node, str):
        return re.sub(r"\s+", " ", node)
    if node.tag in {"script", "style", "annotation", "annotation-xml", "summary"}:
        return ""
    if node.attrs.get("data-type") in {"solution", "solution-title", "answer"}:
        return ""
    if node.tag == "math":
        return "$" + math_text(node).strip() + "$"
    parts = [render(child) for child in node.children]
    if node.tag == "tr":
        return " | ".join(part.strip() for part in parts if part.strip()) + "\n"
    text = "".join(parts)
    if node.tag in {"p", "li", "div", "section", "h2", "h3", "h4", "table", "ol", "ul"}:
        return "\n" + text.strip() + "\n"
    if node.tag == "br":
        return "\n"
    return text


def text(node):
    return re.sub(r"\n\s*\n+", "\n", render(node)).strip()


def sha(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def body_fingerprint(value):
    return sha(re.sub(r"\s+", " ", value).strip().casefold())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download(url, output, refresh=False):
    metadata = output.with_suffix(output.suffix + ".json")
    if output.exists() and metadata.exists() and not refresh:
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        if sha(output.read_bytes()) != previous["sha256"]:
            raise ValueError("Cached source hash mismatch: " + str(output))
        return previous
    request = urllib.request.Request(url, headers={"User-Agent": "mistake-classifier-noncommercial-course-research/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read()
        final_url = response.url
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    record = {"url": url, "resolved_url": final_url, "sha256": sha(data),
              "retrieved_at": datetime.now(timezone.utc).isoformat(), "bytes": len(data)}
    write_json(metadata, record)
    return record


def dependency_reason(nodes):
    for node in nodes:
        if node.find(lambda child: child.tag in {"img", "svg", "video", "iframe"}):
            return "requires_external_or_embedded_figure"
        if node.find(lambda child: child.tag == "a" and bool(child.attrs.get("data-page-fragment"))):
            return "requires_referenced_equation_table_or_other_problem"
    combined = " ".join(text(node) for node in nodes)
    if re.search(r"\b(?:previous|preceding)\s+(?:exercise|problem|question|table|figure)|\bshown\s+(?:above|below)", combined, re.I):
        return "requires_previous_context"
    if re.search(r"\b(?:following|given)\s+(?:graph|figure|diagram)|\b(?:graph|figure|diagram)\b.{0,100}\bshown\s+here\b", combined, re.I):
        return "requires_shared_figure"
    return ""


def extract(html, source, metadata, source_file):
    if "creativecommons.org/licenses/by-nc-sa/4.0/" not in html:
        raise ValueError("Expected explicit source license not found: " + metadata["url"])
    book, subject, chapter, slug = source
    tree = Tree(html).root
    title_node = tree.find(lambda node: node.tag == "title")
    page_title = text(title_node).split(" | OpenStax")[0]
    authors = [node.attrs.get("content", "") for node in tree.walk()
               if node.tag == "meta" and node.attrs.get("name") == "citation_author"]
    containers = [node for node in tree.walk() if node.has_class("section-exercises") or node.has_class("review-problems")]
    rows, exclusions, answers = [], [], []
    for container in containers:
        heading = container.find(lambda node: node.tag in {"h2", "h3"})
        section = text(heading) if heading else page_title.split(" - ")[0]
        context, following_exercise = [], False
        for child in container.children:
            if not isinstance(child, Node):
                continue
            if child.attrs.get("data-type") != "exercise":
                if child.find(lambda node: node.tag in {"h2", "h3"}):
                    context = []
                elif child.tag in {"p", "div", "figure", "table", "ol", "ul"}:
                    if following_exercise:
                        context = []
                    context.append(child)
                following_exercise = False
                continue
            following_exercise = True
            problem = child.find(lambda node: node.has_class("os-problem-container"))
            number_node = child.find(lambda node: node.has_class("os-number"))
            source_id = child.attrs.get("id", "")
            if child.has_class("unnumbered") or problem is None or number_node is None or not source_id:
                continue
            number = text(number_node).strip().rstrip(".")
            if not number.isdigit():
                continue
            reason = dependency_reason([*context, problem])
            if reason:
                exclusions.append({"source_id": source_id, "source_url": metadata["url"],
                                   "exercise_number": number, "reason": reason})
                continue
            instruction = "\n".join(text(node) for node in context if text(node))
            body = text(problem)
            # OpenStax does not mark the end of every shared-instruction group.
            # Retain clear formula-only members; withhold prose members whose
            # relationship to an inherited instruction would require guessing.
            outside_math = re.sub(r"\$[^$]*\$", "", body, flags=re.S)
            outside_math = outside_math.replace("[T]", "")
            if instruction and re.search(r"[A-Za-z]{2,}", outside_math):
                exclusions.append({"source_id": source_id, "source_url": metadata["url"],
                                   "exercise_number": number, "reason": "shared_instruction_scope_requires_human_review"})
                continue
            combined = "\n".join(part for part in (instruction, body) if part)
            if not body:
                continue
            identifier = "openstax_" + sha(book + "#" + source_id)[:20]
            row = {"id": identifier, "group_id": book + ":chapter-" + chapter.split()[0],
                   "instruction_group_id": sha(metadata["url"] + "#" + instruction)[:20],
                   "subject": subject, "qtype": "formula" if subject == "高等数学" else "text",
                   "language": "en", "raw_text": combined, "problem_text": body,
                   "shared_instructions": instruction, "annotation": "", "labels": {},
                   "label_status": "source_derived", "source_labels": {"chapter": chapter, "section": section},
                   "gold_knowledge": None, "review_status": "pending_human_review",
                   "source_url": metadata["url"] + "#" + source_id, "source_title": page_title,
                   "book": book, "chapter": chapter, "section": section,
                   "exercise_number": number, "source_id": source_id,
                   "license": "CC BY-NC-SA 4.0", "license_url": LICENSE_URL,
                   "authors": authors, "publisher": "OpenStax, Rice University",
                   "attribution": "; ".join([", ".join(authors), "OpenStax, Rice University", page_title, metadata["url"]]),
                   "retrieved_at": metadata["retrieved_at"], "source_file": source_file,
                   "source_sha256": metadata["sha256"], "body_sha256": body_fingerprint(combined),
                   "transformations": ["HTML whitespace normalized", "MathML presentation transcribed structurally to inline LaTeX"],
                   "requires_external_media": False}
            rows.append(row)
            answer_url = ""
            if number_node.tag == "a" and number_node.attrs.get("href"):
                answer_url = urllib.parse.urljoin(metadata["url"], number_node.attrs["href"])
                fragment = number_node.attrs.get("data-page-fragment")
                if fragment:
                    answer_url += "#" + fragment
            answers.append({"id": identifier, "answer_url": answer_url,
                            "answer_text": "", "status": "source_link_only" if answer_url else "not_provided_on_question_page"})
    return rows, exclusions, answers


def deduplicate(rows):
    kept, excluded, seen = [], [], {}
    for row in rows:
        digest = row["body_sha256"]
        if digest in seen:
            excluded.append({"id": row["id"], "duplicate_of": seen[digest], "source_url": row["source_url"]})
        else:
            seen[digest] = row["id"]
            kept.append(row)
    return kept, excluded


def grouped_splits(rows):
    assignments = {}
    for book in sorted({row["book"] for row in rows}):
        groups = sorted({row["group_id"] for row in rows if row["book"] == book})
        for index, group in enumerate(groups):
            assignments[group] = ("development", "validation", "holdout")[index % 3]
    return {"policy": "Whole source chapters stay together; no exercise subpart is a separate item.",
            "group_assignments": assignments,
            "partitions": {split: [row["id"] for row in rows if assignments[row["group_id"]] == split]
                           for split in ("development", "validation", "holdout")}}


def collect(output, refresh=False):
    output.mkdir(parents=True, exist_ok=True)

    def fetch(source):
        book, _subject, _chapter, slug = source
        relative = "sources/" + book + "_" + slug + ".html"
        path = output / relative
        metadata = download("https://openstax.org/books/" + book + "/pages/" + slug, path, refresh)
        rows, exclusions, answers = extract(path.read_text(encoding="utf-8"), source, metadata, relative)
        return rows, exclusions, answers, dict(metadata, file=relative)

    all_rows, exclusions, answers, sources = [], [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for rows, skipped, answer_rows, metadata in pool.map(fetch, SOURCES):
            all_rows.extend(rows)
            exclusions.extend(skipped)
            answers.extend(answer_rows)
            sources.append(metadata)
    rows, duplicates = deduplicate(all_rows)
    if len(rows) < 150:
        raise ValueError("Fewer than 150 independent complete exercises: " + str(len(rows)))
    ids = {row["id"] for row in rows}
    for filename, values in (("questions.jsonl", rows), ("answers.jsonl", [row for row in answers if row["id"] in ids])):
        (output / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")
    write_json(output / "exclusions.json", exclusions)
    write_json(output / "duplicates.json", duplicates)
    write_json(output / "splits.json", grouped_splits(rows))
    manifest = {"schema_version": 1, "collector": "tools/collect_independent_corpus.py", "question_count": len(rows),
                "subjects": {subject: sum(row["subject"] == subject for row in rows) for subject in sorted({row["subject"] for row in rows})},
                "language": "en", "human_gold_count": 0, "english_language_course_questions": 0,
                "label_status": "source_derived", "source_labels_meaning": "Original chapter and section titles only; no mapping to classifier labels.",
                "license": "CC BY-NC-SA 4.0", "license_url": LICENSE_URL,
                "license_scope": "Only collected OpenStax course material and its corpus formatting; not application source code.",
                "permitted_use": "Attributed noncommercial educational research; adapted corpus remains under CC BY-NC-SA 4.0.",
                "limitations": ["No human-verified knowledge, feature, difficulty or error-cause gold labels.",
                    "English-language calculus and physics questions are not college-English language-course questions.",
                    "No translation or generated variants; all original subparts remain one numbered source exercise.",
                    "Figure-dependent and external-reference-dependent exercises are excluded, with reasons retained.",
                    "Prose exercises following shared instructions without a clear scope boundary are withheld for human review.",
                    "Answers are stored separately as official source links where available, not included in classifier input.",
                    "English source wording may not match the application's Chinese keyword dictionaries; do not tune on holdout."],
                "exclusion_count": len(exclusions), "exact_duplicate_count": len(duplicates), "sources": sources,
                "dataset_files": {name: sha((output / name).read_bytes()) for name in ("questions.jsonl", "answers.jsonl", "splits.json")}}
    write_json(output / "manifest.json", manifest)
    (output / "LICENSE.txt").write_text(
        "Collected OpenStax textbook content: Copyright Rice University / OpenStax.\n"
        "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.\n" + LICENSE_URL + "\n"
        "Attribution and original URLs are retained per question and raw source page.\n"
        "This license applies to the collected textbook corpus, not to the application source code.\n"
        "Exercise wording is unchanged apart from whitespace normalization and structural MathML transcription.\n"
        "No endorsement by OpenStax or Rice University is implied.\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("samples/independent_acceptance"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    manifest = collect(args.output, args.refresh)
    print(json.dumps({"questions": manifest["question_count"], "subjects": manifest["subjects"],
                      "human_gold": 0, "license": manifest["license"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
