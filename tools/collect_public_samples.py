# -*- coding: utf-8 -*-
"""下载少量可追溯的公开媒体验收素材。

素材来自 Wikimedia Commons，清单保留原页面、直链、作者和许可证。
自动采集不会填写课程知识点真值；运行后必须由人工复核标签，才能进入封存测试集。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path


SAMPLES = [
    {
        "id": "formula_center_of_mass",
        "title": "File:Center of mass using integral.png",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/2b/Center_of_mass_using_integral.png",
        "source_url": "https://commons.wikimedia.org/wiki/File:Center_of_mass_using_integral.png",
        "license": "CC BY-SA 4.0", "creator": "Wikimedia Commons contributor",
        "subject": "高等数学", "qtype": "formula", "review_status": "待人工复核",
    },
    {
        "id": "formula_leibniz_notation",
        "title": "File:LeibnizNotation.png",
        "url": "https://upload.wikimedia.org/wikipedia/commons/6/61/LeibnizNotation.png",
        "source_url": "https://commons.wikimedia.org/wiki/File:LeibnizNotation.png",
        "license": "CC BY-SA 4.0", "creator": "Wikimedia Commons contributor",
        "subject": "高等数学", "qtype": "formula", "review_status": "待人工复核",
    },
    {
        "id": "figure_intro_physics_42",
        "title": "File:Introductory Physics fig 4.2.png",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/fc/Introductory_Physics_fig_4.2.png",
        "source_url": "https://commons.wikimedia.org/wiki/File:Introductory_Physics_fig_4.2.png",
        "license": "CC BY-SA 3.0", "creator": "David J. Raymond",
        "subject": "大学物理", "qtype": "figure", "review_status": "待人工复核",
    },
    {
        "id": "figure_reflection_boundary",
        "title": "File:Eight Lectures on Theoretical Physics illo085.png",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/5f/Eight_Lectures_on_Theoretical_Physics_illo085.png",
        "source_url": "https://commons.wikimedia.org/wiki/File:Eight_Lectures_on_Theoretical_Physics_illo085.png",
        "license": "Public domain", "creator": "Public domain source",
        "subject": "大学物理", "qtype": "figure", "review_status": "待人工复核",
    },
    {
        "id": "audio_dunkley_snippet",
        "title": "File:Audio snippet of J. R. Dunkley.wav",
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/9b/Audio_snippet_of_J._R._Dunkley.wav",
        "source_url": "https://commons.wikimedia.org/wiki/File:Audio_snippet_of_J._R._Dunkley.wav",
        "license": "CC BY-SA 4.0", "creator": "Feuras",
        "subject": "大学英语", "qtype": "listening", "review_status": "待人工复核",
    },
]


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "sample"


def download(url: str, path: Path) -> str:
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url + ("?download=1" if "?" not in url else "&download=1"),
                headers={"User-Agent": "mistake-classifier/1.0 (public acceptance collector)"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            path.write_bytes(payload)
            return hashlib.sha256(payload).hexdigest()
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise last_error


def collect(output_dir: Path) -> Path:
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in SAMPLES:
        suffix = Path(sample["url"].split("?", 1)[0]).suffix.lower() or ".bin"
        path = media_dir / (safe_filename(sample["id"]) + suffix)
        try:
            digest = download(sample["url"], path)
            status = "downloaded"
            error = ""
        except Exception as exc:
            digest = ""
            status = "failed"
            error = str(exc)
        row = dict(sample)
        row.update({"local_path": str(path), "sha256": digest, "download_status": status,
                    "error": error, "gold_knowledge": "", "gold_core_features": "",
                    "permission_scope": "公开教学验证候选；提交前需按许可证复核"})
        rows.append(row)
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_dir / "acceptance_records.csv"
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="下载带许可证清单的公开验收素材")
    parser.add_argument("output", nargs="?", default="samples/public_acceptance")
    args = parser.parse_args(argv)
    manifest = collect(Path(args.output))
    print("公开素材清单已生成：{}".format(manifest))
    print("知识点和核心特征仍为空，必须人工复核后才能作为封存测试集。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
