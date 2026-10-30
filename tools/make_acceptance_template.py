# -*- coding: utf-8 -*-
"""生成真实媒体验收记录模板，不填充或伪造测试结果。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "sample_id", "source_description", "permission_scope", "subject", "qtype",
    "input_path", "gold_knowledge", "gold_core_features", "gold_error_cause",
    "gold_difficulty", "expected_transcript_or_latex", "ocr_or_transcript_text",
    "confirmed_text", "predicted_knowledge", "recognition_status", "failure_reason",
    "input_seconds", "recognition_ms", "classification_ms", "save_export_ms",
    "total_automatic_ms", "reviewer", "review_date", "notes",
]


def make_template(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index in range(1, rows + 1):
            writer.writerow({"sample_id": "sample_{:03d}".format(index)})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成真实媒体验收 CSV 模板")
    parser.add_argument("output", help="输出 CSV 路径")
    parser.add_argument("--rows", type=int, default=30, help="空白记录数")
    args = parser.parse_args(argv)
    if args.rows < 1:
        parser.error("--rows 必须大于 0")
    make_template(Path(args.output), args.rows)
    print("已生成空白验收模板：{}（{} 条）".format(args.output, args.rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
