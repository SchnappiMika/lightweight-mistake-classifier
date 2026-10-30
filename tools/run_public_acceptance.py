# -*- coding: utf-8 -*-
"""对公开素材执行本地公式、图中文字识别与听力转写，结果待人工复核。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def audio_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as stream:
        return {
            "duration_seconds": stream.getnframes() / float(stream.getframerate() or 1),
            "sample_rate": stream.getframerate(),
            "channels": stream.getnchannels(),
        }


def run(manifest_path: Path) -> list[dict]:
    from app.core.preprocess.image_ocr import get_engine
    from app.core.preprocess.media_recognition import recognize_isolated

    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for item in items:
        path = Path(item["local_path"])
        if not path.is_absolute() and not path.is_file():
            path = ROOT / path
        row = {"id": item["id"], "subject": item["subject"], "qtype": item["qtype"],
               "source_url": item["source_url"], "license": item["license"],
               "local_path": str(path), "status": "skipped", "error": "",
               "review_status": "待人工复核"}
        if item.get("download_status") != "downloaded" or not path.is_file():
            row["status"] = "missing_source"
            row["error"] = item.get("error") or "本地素材不存在"
            results.append(row)
            continue
        started = time.perf_counter()
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            row["sha256"] = digest
            if digest != item.get("sha256"):
                row.update({"status": "source_changed", "error": "素材哈希与采集清单不一致，请复核原文件。"})
                results.append(row)
                continue
            if item["qtype"] == "formula":
                record = recognize_isolated(str(path), "formula")
                row.update({"status": "recognized" if not record.error else "failed",
                            "text": record.text, "latex": record.text,
                            "engine": record.engine, "elapsed_ms": record.elapsed_ms,
                            "error": record.error})
            elif item["qtype"] == "figure":
                record = get_engine().recognize_isolated(str(path))
                row.update({"status": "recognized", "text": record.text,
                            "engine": "RapidOCR", "line_count": len(record.lines),
                            "elapsed_ms": record.elapsed_ms})
            elif item["qtype"] == "listening":
                info = audio_info(path)
                record = recognize_isolated(str(path), "audio")
                row.update(info)
                row.update({"status": "recognized" if not record.error else "failed",
                            "engine": record.engine,
                            "transcript": record.text, "elapsed_ms": record.elapsed_ms,
                            "error": record.error})
        except Exception as exc:
            row.update({"status": "failed", "elapsed_ms": (time.perf_counter() - started) * 1000,
                        "error": str(exc)})
        row["wall_elapsed_ms"] = (time.perf_counter() - started) * 1000
        results.append(row)
    return results


def write_report(output_dir: Path, results: list[dict]) -> None:
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 公开素材本地验收记录",
        "",
        "本记录由 `python tools/run_public_acceptance.py` 生成。公开素材来自 Wikimedia Commons，原页面、许可证和本地文件哈希见同目录 `manifest.json`。知识点和核心特征尚未人工复核，因此本记录不能作为分类准确率结论。",
        "",
        "| 编号 | 题型 | 许可证 | 状态 | 识别耗时/时长 | 调用总耗时 | 结果摘要 |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in results:
        if row["qtype"] == "listening":
            metric = "{:.2f}s 音频 / {:.1f}ms".format(row.get("duration_seconds", 0), row.get("elapsed_ms", 0))
            summary = row.get("error") or row.get("transcript", "")[:80]
        elif row["qtype"] == "formula":
            metric = "{:.1f}ms / LaTeX".format(row.get("elapsed_ms", 0))
            summary = row.get("text", "")[:150] or row.get("error", "")
        else:
            metric = "{:.1f}ms / {} 行".format(row.get("elapsed_ms", 0), row.get("line_count", 0))
            summary = row.get("text", "")[:100].replace("\n", " | ") or row.get("error", "")
        wall = "{:.1f}ms".format(row["wall_elapsed_ms"]) if "wall_elapsed_ms" in row else "未记录"
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            row["id"], row["qtype"], row["license"], row["status"], metric, wall,
            summary.replace("|", "/").replace("\n", " ").replace("\r", " ")))
    audio_rows = [row for row in results if row["qtype"] == "listening"]
    audio_successes = sum(row["status"] == "recognized" for row in audio_rows)
    lines.extend([
        "",
        "## 解释",
        "",
        "- 公式行使用独立 pix2tex 本地模型输出 LaTeX；图形行使用 RapidOCR 提取文字，不等于图形关系理解。所有结果仍需对照原图人工校对，成功返回不代表转写准确。",
        "- 本次音频转写成功 {}/{} 条；成功结果来自本地模型，仍需人工复核。失败记录保留原始错误，不将解码或推理失败统称为模型缺失。".format(audio_successes, len(audio_rows)),
        "- 识别耗时为引擎记录；调用总耗时含素材校验、加载、解码及子进程开销，不含导入、人工校对、分类和保存，不能当作每题全流程耗时达标证据。",
        "- 下一步需要人工复核课程知识点和核心特征，再将合格记录复制到封存测试集；未经复核的公开素材不能直接用于申报书准确率。",
    ])
    (output_dir / "验收记录.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="运行公开素材本地验收")
    parser.add_argument("directory", nargs="?", default="samples/public_acceptance")
    args = parser.parse_args(argv)
    directory = Path(args.directory)
    results = run(directory / "manifest.json")
    write_report(directory, results)
    print("已生成：{} 和 {}".format(directory / "results.json", directory / "验收记录.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
