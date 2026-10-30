"""Visually review collected questions without changing frozen source corpora.

python -m tools.review_acceptance DATASET.jsonl --reviews REVIEWS.json
The exported JSONL can be passed directly to tools.evaluate_independent.
No model predictions or automatically proposed labels are shown to reviewers.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def content_hash(row):
    content = {key: row.get(key, "") for key in
               ("id", "raw_text", "subject", "qtype", "annotation", "source_url", "group_id")}
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".review-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ReviewStore:
    def __init__(self, rows, path):
        self.rows = rows
        self.path = Path(path)
        self.by_id = {row["id"]: row for row in rows}
        if len(self.by_id) != len(rows):
            raise ValueError("题目ID重复，不能记录人工复核。")
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "schema_version": 1, "purpose": "explicit human review, never model-generated gold", "reviews": {}}
        if self.data.get("schema_version") != 1 or not isinstance(self.data.get("reviews"), dict):
            raise ValueError("复核记录格式无效，请保留原文件并检查。")

    def current(self, row):
        entry = self.data["reviews"].get(row["id"])
        if entry and entry.get("content_sha256") == content_hash(row):
            return deepcopy(entry)
        return None

    def save(self, row, label, reviewer, core_features=None, confirmed=False):
        if not confirmed or not reviewer.strip() or not label.strip():
            raise ValueError("需要复核者姓名、明确分类，并主动勾选已逐题核对。")
        if row["id"] not in self.by_id or content_hash(self.by_id[row["id"]]) != content_hash(row):
            raise ValueError("当前题目已变化，请重新载入后复核。")
        features = [value.strip() for value in (core_features or []) if value.strip()]
        entry = {"content_sha256": content_hash(row), "labels": {"knowledge": label.strip()},
                 "reviewer": reviewer.strip(), "reviewed_at": datetime.now(timezone.utc).isoformat(),
                 "core_features": list(dict.fromkeys(features)), "explicit_confirmation": True}
        updated = deepcopy(self.data)
        updated["reviews"][row["id"]] = entry
        atomic_json(self.path, updated)
        self.data = updated

    def export(self, path):
        path = Path(path)
        if path.resolve() == self.path.resolve():
            raise ValueError("导出文件不能覆盖复核记录。")
        rows = []
        for source in self.rows:
            row = deepcopy(source)
            review = self.current(source)
            if review and review.get("explicit_confirmation") and review.get("reviewer") and review.get("labels", {}).get("knowledge"):
                row.update(labels=review["labels"], label_status="human_verified",
                           gold_knowledge=review["labels"]["knowledge"],
                           gold_core_features=review.get("core_features", []),
                           review_status="human_verified", human_review=review)
            else:
                # Stale evidence must never survive as human verification.
                row.update(labels={}, label_status="source_derived", gold_knowledge=None,
                           review_status="pending_human_review")
                row.pop("human_review", None)
            rows.append(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".review-export-", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return sum(row["label_status"] == "human_verified" for row in rows)


def build_window(rows, store, schemes, dataset_path):
    from PyQt5.QtCore import QUrl
    from PyQt5.QtGui import QDesktopServices
    from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
        QLineEdit, QListWidget, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

    class ReviewWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("结题题库人工复核")
            self.resize(1050, 720)
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("只记录你逐题核对的结论。来源题库保持原样；未复核题不会被标为人工真值。"))
            toolbar = QHBoxLayout()
            self.reviewer = QLineEdit()
            self.reviewer.setPlaceholderText("复核者姓名 / 可追溯编号")
            toolbar.addWidget(self.reviewer)
            self.progress = QLabel()
            toolbar.addWidget(self.progress)
            layout.addLayout(toolbar)
            body = QHBoxLayout()
            self.list = QListWidget()
            for row in rows:
                self.list.addItem(row["subject"] + " · " + row["id"])
            body.addWidget(self.list, 1)
            self.text = QPlainTextEdit()
            self.text.setReadOnly(True)
            body.addWidget(self.text, 3)
            layout.addLayout(body, 1)
            self.meta = QLabel()
            self.meta.setWordWrap(True)
            layout.addWidget(self.meta)
            self.labels = QComboBox()
            layout.addWidget(self.labels)
            self.features = QLineEdit()
            self.features.setPlaceholderText("人工核对的核心特征词（可选，以中文逗号或英文逗号分隔）")
            layout.addWidget(self.features)
            self.confirm = QCheckBox("我已阅读原题，确认当前分类标签；此操作不代表自动化验收通过")
            layout.addWidget(self.confirm)
            actions = QHBoxLayout()
            for title, callback in (("保存并下一题", self.save), ("跳过", self.next),
                    ("打开原始来源", self.open_source), ("导出已复核题库", self.export)):
                button = QPushButton(title)
                button.clicked.connect(callback)
                actions.addWidget(button)
            layout.addLayout(actions)
            self.list.currentRowChanged.connect(self.load)
            if rows:
                self.list.setCurrentRow(0)

        def load(self, index):
            if index < 0:
                return
            row = rows[index]
            self.text.setPlainText(row["raw_text"])
            self.meta.setText("{} · {}\n{}".format(row["subject"], row.get("license", ""), row.get("source_url", "")))
            self.labels.clear()
            self.labels.addItem("请选择分类标签", "")
            for scheme in schemes:
                if scheme.subject == row["subject"]:
                    dimension = scheme.dim("knowledge")
                    for label in (dimension.labels if dimension else []):
                        self.labels.addItem(scheme.name + " / " + label.name, label.name)
            existing = store.current(row)
            self.confirm.setChecked(False)
            self.features.clear()
            if existing:
                self.labels.setCurrentIndex(max(0, self.labels.findData(existing["labels"]["knowledge"])))
                self.features.setText("，".join(existing.get("core_features", [])))
                self.reviewer.setText(existing.get("reviewer", ""))
            verified = sum(store.current(item) is not None for item in rows)
            self.progress.setText("已复核 {} / {}".format(verified, len(rows)))

        def save(self):
            import re
            try:
                row = rows[self.list.currentRow()]
                store.save(row, self.labels.currentData() or "", self.reviewer.text(),
                           re.split(r"[,，]", self.features.text()), self.confirm.isChecked())
            except Exception as exc:
                QMessageBox.warning(self, "尚未保存", str(exc))
                return
            self.next()

        def next(self):
            target = min(self.list.currentRow() + 1, len(rows) - 1)
            self.list.setCurrentRow(target)
            self.load(target)

        def open_source(self):
            url = rows[self.list.currentRow()].get("source_url", "")
            if url.startswith(("https://", "http://")):
                QDesktopServices.openUrl(QUrl(url))

        def export(self):
            destination, _ = QFileDialog.getSaveFileName(self, "导出已复核题库", str(store.path.with_suffix(".jsonl")), "JSONL (*.jsonl)")
            if destination:
                try:
                    protected = dataset_path if isinstance(dataset_path, list) else [dataset_path]
                    if Path(destination).resolve() in {path.resolve() for path in protected}:
                        raise ValueError("请另存到新文件，保留冻结来源题库。")
                    count = store.export(destination)
                    QMessageBox.information(self, "导出完成", "共 {} 题已有复核者确认；其他题保留待复核状态。".format(count))
                except Exception as exc:
                    QMessageBox.warning(self, "导出失败", str(exc))

    return ReviewWindow()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="*", help="默认载入已采集的高数、物理与英语220题")
    parser.add_argument("--reviews", type=Path, default=ROOT / "samples/human_reviews/reviews.json")
    args = parser.parse_args(argv)
    datasets = args.dataset or [ROOT / "samples/independent_acceptance/questions.jsonl",
                               ROOT / "samples/independent_english/questions.jsonl"]
    if args.reviews.resolve() in {path.resolve() for path in datasets}:
        parser.error("复核文件不能覆盖来源题库")
    from tools.evaluate_independent import load_rows
    from app.core.models import Scheme
    from app import config
    from app.qt_runtime import configure_qt_plugins
    from PyQt5.QtWidgets import QApplication
    rows = [row for dataset in datasets for row in load_rows(dataset)]
    store = ReviewStore(rows, args.reviews)
    schemes = [Scheme.from_dict(value) for value in json.loads(
        (config.SCHEME_SEED_DIR / "builtin_schemes.json").read_text(encoding="utf-8"))]
    configure_qt_plugins()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = build_window(rows, store, schemes, datasets)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
