"""为目录发布包复制已准备的模型、依赖许可和源码；不联网或读取题库。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_models(target: Path, source: Path) -> None:
    from app.core.preprocess.media_recognition import _faster_whisper_path
    snapshot = _faster_whisper_path(source / "audio")
    audio = target / "models" / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt", "preprocessor_config.json"):
        original = snapshot / name
        if not original.is_file():
            continue
        destination = audio / name
        shutil.copy2(original, destination)
        digest = sha256(destination)
        if digest != sha256(original):
            raise OSError("模型复制校验失败：{}".format(name))
        files.append({"path": "audio/" + name, "bytes": destination.stat().st_size, "sha256": digest})
    manifest = {"model": "Systran/faster-whisper-base", "revision": snapshot.name,
                "source": "https://huggingface.co/Systran/faster-whisper-base",
                "upstream": "https://github.com/openai/whisper", "files": files,
                "operation": "copied_existing_local_files_no_download"}
    (target / "models" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_licenses(target: Path) -> None:
    directory = target / "third-party-licenses"
    directory.mkdir(parents=True, exist_ok=True)
    distributions = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "unknown")
        record = {"name": name, "version": distribution.version, "license_files": []}
        for file in distribution.files or []:
            if not any(marker in file.name.casefold() for marker in ("license", "copying", "notice")):
                continue
            if not any(part.endswith(".dist-info") for part in file.parts):
                continue
            original = Path(distribution.locate_file(file))
            if original.is_file():
                destination = directory / name / Path(*file.parts[1:])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, destination)
                record["license_files"].append(destination.relative_to(target).as_posix())
        distributions.append(record)
    (directory / "installed-components.json").write_text(
        json.dumps({"scope": "构建环境的已安装依赖；包含开发工具，非逐一判定为发布程序所需组件。",
                    "components": sorted(distributions, key=lambda row: row["name"].lower())},
                   ensure_ascii=False, indent=2), encoding="utf-8")


def stage_formula_runtime(target: Path, interpreter: Path, model_dir: Path) -> None:
    """Copy a complete independent CPython runtime, not a nonportable venv launcher."""
    from app.core.preprocess.formula_recognition_backend import validate_models
    formula = model_dir / "formula"
    model_manifest = validate_models(formula)
    probe = subprocess.run([str(interpreter.resolve()), "-c",
                            "import json,sys,sysconfig; print(json.dumps({'base':sys.base_prefix,'site':sysconfig.get_path('purelib')}))"],
                           capture_output=True, text=True, encoding="utf-8", check=True)
    locations = json.loads(probe.stdout)
    base, packages = Path(locations["base"]), Path(locations["site"])
    if not (base / "python.exe").is_file() or not (packages / "pix2tex").is_dir():
        raise ValueError("公式环境必须为已安装 pix2tex 的独立 Windows CPython。")
    runtime = target / "formula_runtime"
    destination = runtime / "python"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(base, destination, dirs_exist_ok=True, ignore=ignore)
    shutil.copytree(packages, destination / "Lib" / "site-packages", dirs_exist_ok=True, ignore=ignore)
    shutil.copy2(ROOT / "app/core/preprocess/formula_recognition_backend.py", runtime)
    formula_target = target / "models" / "formula"
    formula_target.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", *(entry["filename"] for entry in model_manifest["files"])):
        if Path(name).name != name:
            raise ValueError("公式模型清单文件名不合法：{}".format(name))
        original = formula / name
        if original.is_file():
            shutil.copy2(original, formula_target / name)
    validate_models(formula_target)
    (runtime / "runtime-info.json").write_text(json.dumps({
        "layout": "independent_cpython_with_site_packages",
        "source_interpreter": str(interpreter.resolve()),
        "python_base": str(base), "site_packages": str(packages),
        "runtime_network": "disabled_in_formula_backend",
        "models": "../models/formula", "clean_machine_verified": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_source(target: Path) -> Path:
    from tools.validate_release import source_files
    archive = target / "对应源代码.zip"
    allowed_roots = {"app", "resources", "samples", "docs", "tests", "tools"}
    allowed_files = {"main.py", "README.md", "LICENSE", "开始使用.md", "启动软件.bat",
                     "安装图片识别.bat", "mistake_classifier.spec", ".gitignore"}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        entries = []
        for path in source_files(ROOT):
            relative = path.relative_to(ROOT)
            if (relative.parts[0] not in allowed_roots and relative.name not in allowed_files
                    and not relative.name.startswith("requirements")):
                continue
            if relative.parts[0] == "samples" and relative.name == "performance_report.json":
                continue
            output.write(path, relative.as_posix())
            entries.append({"path": relative.as_posix(), "sha256": sha256(path)})
        output.writestr("source-manifest.json", json.dumps(entries, ensure_ascii=False, indent=2))
    return archive


def stage_evidence(target: Path) -> None:
    """Keep README links and reproducible public examples usable outside the repo."""
    shutil.copytree(ROOT / "docs", target / "docs", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "samples" / "public_acceptance",
                    target / "samples" / "public_acceptance", dirs_exist_ok=True)
    for name in ("performance_report_isolated.json",):
        source = ROOT / "samples" / name
        if source.is_file():
            shutil.copy2(source, target / "samples" / name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("--model-dir", type=Path, help="已有模型根目录；不传则不复制模型")
    parser.add_argument("--formula-python", type=Path, help="可选公式独立环境解释器；复制完整运行时及已校验本地公式模型")
    args = parser.parse_args(argv)
    target = args.target.resolve()
    if not (target / "错题自动分类.exe").is_file() or not (target / "media_worker.exe").is_file():
        parser.error("目标必须是同时含主程序和 media_worker.exe 的已构建发布目录")
    if args.model_dir:
        stage_models(target, args.model_dir)
    if args.formula_python:
        if not args.model_dir:
            parser.error("--formula-python 同时需要 --model-dir")
        stage_formula_runtime(target, args.formula_python, args.model_dir)
    stage_licenses(target)
    for name in ("README.md", "LICENSE", "开始使用.md"):
        shutil.copy2(ROOT / name, target / name)
    stage_evidence(target)
    source = stage_source(target)
    print("发布资料已准备：{}；源码 SHA-256：{}".format(target, sha256(source)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
