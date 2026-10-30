# -*- coding: utf-8 -*-
"""离线交付包与结题证据的静态核验工具。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List


REQUIRED_RELATIVE = ("main.py", "app", "resources", "README.md", "requirements.txt")
EXCLUDED_DIRS = {".venv", ".venv-formula", ".formula-setup", ".git", "__pycache__", ".pytest_cache", ".startup-check",
                 "build", "dist", "deliverables", "node_modules"}


def source_files(root: Path):
    """跳过运行环境、现场数据与构建结果，避免把它们打入源码清单。"""
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = sorted(name for name in names if name not in EXCLUDED_DIRS
                          and not (Path(directory) / name).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink() or name == "启动日志.txt" or name.endswith((".log", ".pyc", ".pyo")):
                continue
            yield path


def missing_paths(root: Path, required: Iterable[str] = REQUIRED_RELATIVE) -> List[str]:
    """返回发布目录中缺失的必需路径。"""
    return [item for item in required if not (root / item).exists()]


def build_manifest(root: Path) -> dict:
    """生成不含用户数据的可复现清单。"""
    files = []
    for path in source_files(root):
        files.append(str(path.relative_to(root)).replace("\\", "/"))
    return {
        "root": str(root.resolve()), "python": sys.version,
        "required_paths": list(REQUIRED_RELATIVE), "file_count": len(files), "files": files,
        "claims": {"real_media_recognition_verified": False,
                   "target_device_verified": False, "student_trial_verified": False,
                   "school_acceptance_confirmed": False},
    }


def binary_manifest(root: Path) -> dict:
    required = ("错题自动分类.exe", "media_worker.exe", "_internal", "release-info.json")
    missing = missing_paths(root, required)
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            entries.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                            "sha256": digest.hexdigest()})
    return {"kind": "windows_onedir", "missing": missing, "files": entries,
            "file_count": len(entries), "total_bytes": sum(row["bytes"] for row in entries),
            "verification": "static_only", "clean_machine_verified": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="核验源码交付目录并生成 manifest")
    parser.add_argument("root", nargs="?", default=".", help="源码或发布目录")
    parser.add_argument("--manifest", help="将 JSON 清单写入此路径")
    parser.add_argument("--binary", action="store_true", help="检查目录式 Windows 发布包，附大小与 SHA-256")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    manifest = binary_manifest(root) if args.binary else build_manifest(root)
    missing = manifest["missing"] if args.binary else missing_paths(root)
    if args.manifest:
        output = Path(args.manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if missing:
        print("缺少必需路径：{}".format(", ".join(missing)))
        return 1
    print("发布目录静态核验通过：{}（{} 个文件）".format(root, manifest["file_count"]))
    print("真实媒体、目标设备、学生试用和学校验收仍需独立证据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
