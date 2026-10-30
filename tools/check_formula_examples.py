from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="记录实际 pix2tex 公式图片识别结果")
    parser.add_argument("--python", default=str(ROOT / ".venv-formula/Scripts/python.exe"))
    parser.add_argument("--model-dir", default=str(Path.home() / ".mistake_classifier/models/formula"))
    parser.add_argument("--output", default=str(ROOT / "docs/公式识别验收记录.json"))
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / "samples/public_acceptance/manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in manifest:
        if item["qtype"] != "formula":
            continue
        result_path = ROOT / "samples/public_acceptance" / (item["id"] + "_pix2tex.json")
        input_path = Path(item["local_path"])
        if not input_path.is_absolute():
            input_path = ROOT / input_path
        command = [args.python, str(ROOT / "app/core/preprocess/formula_recognition_backend.py"),
                   "--input", str(input_path), "--output", str(result_path), "--model-dir", args.model_dir]
        result_path.unlink(missing_ok=True)
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, timeout=240)
            row = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {
                "error": completed.stderr.decode("utf-8", errors="replace")[-2000:] or "识别进程未返回记录",
                "elapsed_ms": 0.0}
            returncode = completed.returncode
        except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
            row = {"error": str(exc), "elapsed_ms": 0.0}
            returncode = -1
        row.update(id=item["id"], source_url=item["source_url"], license=item["license"],
                   input_path=str(input_path), source_sha256=item["sha256"],
                   subprocess_wall_ms=(time.perf_counter() - started) * 1000,
                   returncode=returncode)
        rows.append(row)
    derived = []
    for suffix, processing in (("integral_crop", "原图裁剪(40,40,550,250)并反色"),
                               ("integral_crop_50pct", "同一裁剪图缩放至50%")):
        path = ROOT / "samples/public_acceptance" / ("formula_center_of_mass_" + suffix + "_pix2tex.json")
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            record.update(source_id="formula_center_of_mass", preprocessing=processing,
                          independent_sample=False, correctness_verified=False)
            derived.append(record)
    model_manifest = json.loads((Path(args.model_dir) / "manifest.json").read_text(encoding="utf-8"))
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "python_executable": args.python,
              "model_manifest": model_manifest, "network": "backend 在导入模型前阻断 socket connect/create_connection",
              "timing_scope": "每张独立工作进程；含模型哈希、模块导入、模型加载和推理；不含人工校对、分类、存档。",
              "accuracy_evaluated": False, "results": rows, "derived_examples": derived}
    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 公式识别验收记录", "", "日期：2026-10-29。真实识别使用独立 Python 3.12.13、pix2tex 0.1.4、CPU PyTorch 2.5.1。两个官方检查点为 v0.0.1，来源与 SHA-256 见同名 JSON。运行期间阻断 socket 连接，未通过 RapidOCR 替代公式识别。", "", "| 图片 | 原始 LaTeX 调用 | 引擎耗时 | 校对状态 |", "|---|---|---:|---|"]
    for row in rows:
        lines.append("| {} | {} | {:.2f} 秒 | 待人工复核 |".format(row["id"], "返回 LaTeX" if not row["error"] else row["error"], row["elapsed_ms"] / 1000))
    lines.extend(["", "原图中 Leibniz 符号是两个独立导数式；另图同时含三行积分式与质心图形，不属于单个裁剪公式。完整原图的输出仅证明本地模型调用返回，不证明结构与数学含义正确。质心全图输出明显不可靠，应裁剪单条公式、校对后再使用。", "", "准确率尚未评估，未声称达到公式特征提取 90% 或全流程一秒。计时包括每张新进程中的加载和推理，也不是用户操作到分类保存的完整流程。", "", "安全回归验证了缺模型提前返回、哈希损坏拒绝、禁止联网及 socket 恢复。识别结果保持未确认，需人工核对后明确应用。"])
    if derived:
        lines.extend(["", "裁剪和反色的派生样例另存 JSON，不增加独立样本数；输出与尺寸有关，仍需人工核对，不能据此保证裁剪一定改善结果。"])
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return 1 if any(row["error"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
