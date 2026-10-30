"""Compare independent jobs with model reuse on the same ordered inputs.

Each mode uses a fresh sandbox and new question records. Timers include import,
recognition, feature extraction, classification and saving; failures stay counted.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def summarize(rows, batch_ms):
    times = [row["total_ms"] for row in rows]
    ordered = sorted(times)
    return {"attempted": len(rows), "recognized": sum(row.get("recognized", False) for row in rows),
            "classified": sum(row.get("classified", False) for row in rows),
            "failures": sum(bool(row.get("error")) for row in rows),
            "mean_item_ms": statistics.fmean(times) if times else None,
            "p95_item_ms": ordered[min(len(ordered)-1, int(len(ordered)*0.95))] if ordered else None,
            "batch_ms": batch_ms, "batch_ms_per_item": batch_ms / len(rows) if rows else None,
            "one_second_requirement_met": bool(rows) and not any(row.get("error") for row in rows)
                and max(times) <= 1000 and batch_ms / len(rows) <= 1000}


def worker(manifest, mode, repeats, model_dir):
    expected = os.environ.get("MC_BATCH_BENCHMARK_WORKER")
    if not expected or expected != os.environ.get("MC_WORKSPACE"):
        raise ValueError("性能worker必须在专用临时工作区运行。")
    from app.core.preprocess.media_jobs import media_session
    from app.core.preprocess.media_preparation import prepare_question_media, media_kind
    from app.core.dataio.importer import import_media
    from app.core.classify.engine import ClassificationEngine
    from app.core.preprocess import pipeline
    from app.core.storage import Repository
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    repo = Repository()
    engines = {}
    rows = []
    started_batch = time.perf_counter()
    with media_session() if mode == "reuse_models" else nullcontext():
        for repetition in range(repeats):
            for entry in entries:
                started = time.perf_counter()
                row = {"id": entry["id"], "repetition": repetition, "recognized": False,
                       "classified": False, "error": "", "input": entry}
                try:
                    source = ROOT / entry["local_path"]
                    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                    if entry.get("sha256") and entry["sha256"] != original_hash:
                        raise ValueError("公开素材hash与清单不符")
                    question = import_media([str(source)], subject=entry["subject"], qtype=entry["qtype"])[0]
                    prepared = prepare_question_media(question)
                    kind = media_kind(question)
                    record = getattr(question, {"image": "image_ocr", "formula": "formula_ocr",
                                                "audio": "audio_transcript"}[kind])
                    row.update(recognized=prepared.recognized, recognition_text=record.text,
                               recognition_ms=prepared.elapsed_ms, confirmed=record.confirmed)
                    scheme = repo.scheme_by_subject(question.subject)
                    if scheme is None:
                        raise ValueError("当前学科没有分类体系")
                    engine = engines.setdefault(scheme.id, ClassificationEngine(scheme))
                    feature = pipeline.preprocess(question, use_unreviewed_media=True)
                    result = engine.classify(question, feature)
                    engine.apply_result(question, result, scheme, feature)
                    repo.questions.append(question)
                    repo.save_questions()
                    row.update(classified=question.classified, labels=question.label_names,
                               unreviewed_media=feature.unreviewed_media,
                               source_unchanged=hashlib.sha256(source.read_bytes()).hexdigest() == original_hash)
                    if not question.classified:
                        row["error"] = "识别已返回，但未匹配有效分类标签"
                except Exception as exc:
                    row["error"] = type(exc).__name__ + ": " + str(exc)
                row["total_ms"] = (time.perf_counter() - started) * 1000
                rows.append(row)
    elapsed = (time.perf_counter() - started_batch) * 1000
    return {"mode": mode, "results": rows, "summary": summarize(rows, elapsed),
            "models": str(model_dir), "new_question_per_attempt": True,
            "timing_scope": "每题含附件导入复制、识别、预处理、分类与保存；批次含模型退出；不含人工思考或应用启动"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "samples/public_acceptance/manifest.json"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", default=str(ROOT / "samples/media_batch_performance.json"))
    parser.add_argument("--model-dir")
    parser.add_argument("--mode", choices=("all", "single_jobs", "reuse_models"), default="all")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("repeats必须至少为1")
    models = Path(args.model_dir or os.environ.get("MC_MODEL_DIR") or
                  str(Path(os.environ.get("MC_WORKSPACE", str(Path.home()/".mistake_classifier"))) / "models")).resolve()
    if args.worker:
        result = worker(Path(args.manifest), args.mode, args.repeats, models)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "platform": platform.platform(), "source_manifest": str(Path(args.manifest).resolve()),
              "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
              "limitations": "新模型进程不等于操作系统磁盘冷态；公开素材未构成独立准确率验收；原1秒指标不变", "modes": []}
    for mode in (("single_jobs", "reuse_models") if args.mode == "all" else (args.mode,)):
        with tempfile.TemporaryDirectory(prefix="mc_batch_benchmark_") as sandbox:
            env = os.environ.copy()
            env.update(MC_WORKSPACE=sandbox, MC_BATCH_BENCHMARK_WORKER=sandbox,
                       MC_MODEL_DIR=str(models), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--mode", mode,
                "--manifest", str(Path(args.manifest).resolve()), "--repeats", str(args.repeats),
                "--model-dir", str(models)]
            completed = subprocess.run(command, env=env, cwd=str(ROOT), capture_output=True,
                                       text=True, encoding="utf-8", errors="replace", timeout=3600)
            if completed.returncode:
                raise RuntimeError(completed.stderr[-3000:] or "性能worker失败")
            report["modes"].append(json.loads(completed.stdout))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
