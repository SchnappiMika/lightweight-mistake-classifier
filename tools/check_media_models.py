# -*- coding: utf-8 -*-
"""只读检查公式和听力本地模型准备状态，不安装或下载任何文件。"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config


def _audio_snapshot(directory: Path) -> Optional[Path]:
    for snapshot in [directory, *directory.glob("models--Systran--faster-whisper-base/snapshots/*")]:
        required = (snapshot / "config.json", snapshot / "model.bin", snapshot / "tokenizer.json")
        if all(path.is_file() and path.stat().st_size > 0 for path in required):
            return snapshot
    return None


def check_kind(root: Path, kind: str) -> dict:
    directory = root / kind
    files = sorted(path.name for path in directory.iterdir()) if directory.is_dir() else []
    module = "pix2tex" if kind == "formula" else "faster_whisper"
    alternate = "" if kind == "formula" else "whisper"
    snapshot = _audio_snapshot(directory) if kind == "audio" else None
    runtime = ""
    if kind == "formula":
        from app.core.preprocess.formula_runtime import formula_runtime
        try:
            python, _backend = formula_runtime()
            runtime = str(python)
        except (OSError, ValueError):
            pass
        required = ("weights.pth", "image_resizer.pth", "config.yaml", "tokenizer.json", "manifest.json")
        ready = bool(runtime) and all((directory / name).is_file() and (directory / name).stat().st_size for name in required)
    else:
        ready = bool(snapshot) and importlib.util.find_spec("faster_whisper") is not None
    return {"kind": kind, "directory": str(directory), "files": files,
            "module": module, "module_installed": importlib.util.find_spec(module) is not None,
            "alternate_module": alternate,
            "alternate_installed": bool(alternate and importlib.util.find_spec(alternate)),
            "snapshot": str(snapshot) if snapshot else "", "independent_runtime": runtime,
            "ready": ready}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="只读检查公式/音频模型目录和组件")
    parser.add_argument("--model-dir", default=os.environ.get("MC_MODEL_DIR", ""),
                        help="模型根目录，默认使用 MC_MODEL_DIR 或工作区 models")
    args = parser.parse_args(argv)
    root = Path(args.model_dir or (config.WORKSPACE_DIR / "models"))
    results = [check_kind(root, kind) for kind in ("formula", "audio")]
    for result in results:
        print("{}: {}".format(result["kind"], "ready" if result["ready"] else "not ready"))
        print("  directory: {}".format(result["directory"]))
        print("  files: {}".format(", ".join(result["files"]) or "(none)"))
        if result.get("snapshot"):
            print("  snapshot: {}".format(result["snapshot"]))
        if result.get("independent_runtime"):
            print("  independent_runtime: {}".format(result["independent_runtime"]))
        print("  components: {}={} {}={}".format(
            result["module"], result["module_installed"], result["alternate_module"],
            result["alternate_installed"]))
    print("只读检查不下载或加载模型；就绪仅指组件路径和必要文件存在，真实识别及公式模型哈希仍需执行验证。")
    return 0 if all(item["ready"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
