"""Measure classification and public media calls in disposable child workspaces."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MEDIA_SCOPE = (
    "公开素材校验和识别调用；子进程总耗时另含解释器与模块加载。"
    "不含完整导入、分类、保存、重启、导出和人工校对，不能证明全流程一秒达标。"
)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[index]


def summarize(values: list[float], attempted: Optional[int] = None,
              failures: Optional[int] = None) -> dict:
    result = {
        "count": len(values), "mean_ms": statistics.fmean(values) if values else 0.0,
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values) if values else 0.0, "max_ms": max(values) if values else 0.0,
    }
    if attempted is not None:
        result.update(attempted=attempted, successful=max(0, attempted - (failures or 0)),
                      failures=failures or 0)
    return result


def _environment() -> dict:
    result = {
        "platform": platform.platform(), "python": sys.version,
        "machine": platform.machine(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "load_average": None,
        "system_cpu_percent": None, "system_cpu_percent_window_seconds": None,
        "load_limitations": "运行负载快照不能保证机器空闲；请结束构建等竞争任务后测量。",
    }
    try:
        result["load_average"] = list(os.getloadavg())
    except (AttributeError, OSError):
        pass
    try:
        import psutil
        result["system_cpu_percent"] = psutil.cpu_percent(interval=0.1)
        result["system_cpu_percent_window_seconds"] = 0.1
        result["memory_available_bytes"] = psutil.virtual_memory().available
        result["process_count"] = len(psutil.pids())
    except (ImportError, OSError):
        result["load_limitations"] += " psutil 不可用时系统 CPU 占用和内存不记录。"
    result["package_versions"] = {}
    for name in ("jieba", "numpy", "scipy", "scikit-learn", "rapidocr", "onnxruntime",
                 "faster-whisper", "av", "psutil"):
        try:
            result["package_versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["package_versions"][name] = None
    return result


def _text_worker(samples: int) -> dict:
    import_started = time.perf_counter()
    from app import config
    from app.core.classify.engine import ClassificationEngine
    from app.core.models import Question, Scheme
    from app.core.preprocess import pipeline

    module_import_ms = (time.perf_counter() - import_started) * 1000.0
    initialization_started = time.perf_counter()
    seed_path = config.SCHEME_SEED_DIR / "builtin_schemes.json"
    seeds = json.loads(seed_path.read_text(encoding="utf-8"))
    scheme = next(Scheme.from_dict(item) for item in seeds
                  if item.get("subject") == config.SUBJECT_MATH)
    engine = ClassificationEngine(scheme)
    question = Question(subject=config.SUBJECT_MATH,
                        raw_text="求函数在区间上的定积分并判断收敛性")
    initialization_ms = (time.perf_counter() - initialization_started) * 1000.0

    def classify_once() -> dict:
        started = time.perf_counter()
        row = {"status": "failed", "error": ""}
        try:
            bundle = pipeline.preprocess(question)
            result = engine.classify(question, bundle)
            knowledge = result.dims.get(config.DIM_KNOWLEDGE)
            if knowledge is None or not knowledge.label_id:
                raise ValueError("知识点分类未返回标签")
            row["status"] = "classified"
        except Exception as exc:
            row["error"] = str(exc)
        row["wall_elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return row

    cold = classify_once()
    rows = [classify_once() for _ in range(samples)]
    result = summarize([row["wall_elapsed_ms"] for row in rows], attempted=samples,
                       failures=sum(row["status"] != "classified" for row in rows))
    result.update(
        module_import_ms=module_import_ms, initialization_ms=initialization_ms,
        cold_first_call=cold, warm_calls=rows,
        cold_total_inside_worker_ms=module_import_ms + initialization_ms + cold["wall_elapsed_ms"],
        builtin_scheme_id=scheme.id,
        builtin_scheme_sha256=hashlib.sha256(seed_path.read_bytes()).hexdigest(),
        question=question.raw_text,
        initialization_scope="模块导入单列；初始化为读取内置体系与构造引擎，首次调用含分词和惰性建库。",
        timing_scope="热态为同一题预处理+分类，包括失败耗时；不含导入、保存、重启、导出或人工操作。",
        accuracy_claim=False,
    )
    return result


def _media_worker(manifest: Path) -> dict:
    started = time.perf_counter()
    from tools.run_public_acceptance import run
    import_ms = (time.perf_counter() - started) * 1000.0
    items = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "results": run(manifest), "manifest_count": len(items), "module_import_ms": import_ms,
        "phase_note": "新媒体子进程为进程冷态，不代表磁盘缓存冷态；后续模型复用由识别引擎决定，未分解模型加载与推理。",
    }


def _run_worker(args: list[str], workspace: Path,
                model_dir: Optional[Path] = None) -> dict:
    workspace.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="worker_", dir=str(workspace)) as directory:
        isolated = Path(directory).resolve()
        environment = os.environ.copy()
        environment.update(MC_WORKSPACE=str(isolated), MC_PERF_ISOLATED_WORKER=str(isolated),
                           PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if model_dir is not None:
            environment["MC_MODEL_DIR"] = str(model_dir.resolve())
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "tools.measure_performance"] + args,
                cwd=str(ROOT), env=environment, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=900,
                **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if not lines:
                raise ValueError(completed.stderr[-2000:] or "性能子进程无输出")
            payload = json.loads(lines[-1])
            if not isinstance(payload, dict):
                raise ValueError("性能子进程输出格式错误")
            if completed.returncode:
                payload["error"] = payload.get("error") or completed.stderr[-2000:] or "性能子进程失败"
            payload["returncode"] = completed.returncode
        except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
            payload = {"error": str(exc)}
        payload["subprocess_wall_ms"] = (time.perf_counter() - started) * 1000.0
        payload["workspace_was_empty_at_start"] = True
        payload["reused_model_directory"] = str(model_dir.resolve()) if model_dir else None
    payload["workspace_deleted_after_run"] = not isolated.exists()
    return payload


def text_classification(samples: int, workspace: Path) -> dict:
    return _run_worker(["--worker-text", str(max(1, samples))], workspace)


def media_measure(directory: Path, workspace: Path,
                  model_dir: Optional[Path] = None) -> dict:
    manifest = (directory / "manifest.json").resolve()
    try:
        items = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise ValueError("素材清单必须是列表")
    except (OSError, ValueError) as exc:
        return {"manifest": str(manifest), "error": str(exc), "recognized": 0, "total": 0,
                "failures": 0, "per_item": summarize([], attempted=0, failures=0),
                "results": [], "timing_scope": MEDIA_SCOPE}
    payload = _run_worker(["--worker-media", str(manifest)], workspace, model_dir)
    rows = payload.get("results", [])
    if payload.get("error"):
        rows = [{"id": item.get("id", str(index)), "status": "worker_failed",
                 "error": payload["error"]} for index, item in enumerate(items)]
    recognized = sum(row.get("status") == "recognized" for row in rows)
    values = [float(row["wall_elapsed_ms"]) for row in rows
              if row.get("wall_elapsed_ms") is not None]
    total = len(items)
    batch_wall_ms = payload.get("subprocess_wall_ms", 0.0)
    result = {
        "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "recognized": recognized, "total": total, "failures": total - recognized,
        "per_item": summarize(values, attempted=total, failures=total - recognized),
        "items_without_wall_timing": max(0, total - len(values)),
        "batch_wall_ms": batch_wall_ms,
        "batch_wall_ms_per_manifest_item": batch_wall_ms / total if total else None,
        "timing_scope": MEDIA_SCOPE,
        "per_item_timing_note": "逐条耗时摘要只使用实际计时，缺失不填零；成功率及批次平均分母包含失败和缺失素材。",
        "results": rows,
    }
    result.update({key: value for key, value in payload.items() if key != "results"})
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="在隔离工作区记录分类与公开媒体调用性能")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--public-dir", default=str(ROOT / "samples/public_acceptance"))
    parser.add_argument("--model-dir", help="复用已有本地模型，默认 MC_MODEL_DIR 或原工作区/models；不下载模型")
    parser.add_argument("--skip-media", action="store_true")
    parser.add_argument("--load-note", default="未提供；不能保证测量时无其他竞争任务")
    parser.add_argument("--output", default=str(ROOT / "samples/performance_report_isolated.json"))
    parser.add_argument("--worker-text", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-media", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_text is not None or args.worker_media:
        expected = os.environ.get("MC_PERF_ISOLATED_WORKER")
        if not expected or os.environ.get("MC_WORKSPACE") != expected:
            parser.error("内部 worker 只能由隔离性能工具启动")
        before = _environment()
        cpu_started = time.process_time()
        started = time.perf_counter()
        try:
            payload = _text_worker(max(1, args.worker_text)) if args.worker_text is not None else _media_worker(Path(args.worker_media))
        except Exception as exc:
            payload = {"error": str(exc)}
        payload.update(worker_wall_ms=(time.perf_counter() - started) * 1000.0,
                       worker_cpu_ms=(time.process_time() - cpu_started) * 1000.0,
                       environment_before=before, environment_after=_environment(),
                       workspace=str(Path(expected).resolve()))
        print(json.dumps(payload, ensure_ascii=False))
        return 1 if payload.get("error") else 0
    historical = ROOT / "samples/performance_report.json"
    output = Path(args.output).resolve()
    if output == historical.resolve():
        parser.error("保留历史 performance_report.json；请指定新的报告路径")
    original_workspace = Path(os.environ.get("MC_WORKSPACE", str(Path.home() / ".mistake_classifier")))
    model_dir = Path(args.model_dir or os.environ.get("MC_MODEL_DIR") or str(original_workspace / "models"))
    before = _environment()
    with tempfile.TemporaryDirectory(prefix="mc_perf_") as temp_dir:
        workspace = Path(temp_dir)
        text_result = text_classification(args.samples, workspace)
        media_result = {"status": "not_run", "reason": "--skip-media"} if args.skip_media else media_measure(Path(args.public_dir), workspace, model_dir)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment_before": before, "environment_after": _environment(), "load_note": args.load_note,
        "workspace_isolated": True, "workspace_deleted_after_run": not workspace.exists(),
        "dataset": "固定内置高数体系与单条文本重复测量，不估计分类泛化准确率",
        "text_classification": text_result, "public_media": media_result,
        "historical_reports": [{"path": str(historical), "status": "historical_non_baseline",
                                "reason": "旧记录可能与发布构建并行，负载未隔离；保留原文件，不用作性能基准。"}],
        "interpretation": {
            "scope_warning": MEDIA_SCOPE,
            "cold_definition": "新解释器和空工作区；不清除操作系统文件缓存，不代表机器重启冷态。",
            "model_reuse": "MC_MODEL_DIR 指定已有本地模型，运行期不准备或下载模型；不复用用户题库、设置或学习数据。",
            "load_warning": "即使工作区隔离，也应在构建和其他高负载任务停止后重测；本工具不清理系统缓存或控制其他进程。",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("性能报告已生成：{}".format(output))
    failed = (text_result.get("error") or text_result.get("failures") or
              text_result.get("cold_first_call", {}).get("error") or
              media_result.get("error") or media_result.get("failures"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
