from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RELEASE = "v0.0.1"
MODEL_URL = "https://github.com/lukas-blecher/LaTeX-OCR/releases/download/{}/{}"
MODEL_SIZES = {"weights.pth": 102113875, "image_resizer.pth": 19441973}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(root: Path, allow_download: bool) -> dict:
    destination = root / "formula"
    distribution = importlib.metadata.distribution("pix2tex")
    if distribution.version != "0.1.4":
        raise ValueError("需要固定 pix2tex==0.1.4，当前为 {}".format(distribution.version))
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for filename, expected_size in MODEL_SIZES.items():
        target = destination / filename
        url = MODEL_URL.format(RELEASE, filename)
        if not target.is_file() or target.stat().st_size != expected_size:
            if not allow_download:
                raise ValueError("模型未准备，需显式 --download：{}".format(target))
            partial = target.with_suffix(target.suffix + ".part")
            request = urllib.request.Request(url, headers={"User-Agent": "MistakeClassifier-model-setup/1.0"})
            print("Downloading {}".format(filename), flush=True)
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            if partial.stat().st_size != expected_size:
                raise ValueError("下载文件大小不正确：{}".format(partial))
            partial.replace(target)
        entries.append({"filename": filename, "source_url": url, "release": RELEASE,
                        "bytes": target.stat().st_size, "sha256": sha256(target)})
    resources = {"config.yaml": "pix2tex/model/settings/config.yaml",
                 "tokenizer.json": "pix2tex/model/dataset/tokenizer.json",
                 "LICENSE.pix2tex.txt": "pix2tex-0.1.4.dist-info/LICENSE"}
    for filename, relative in resources.items():
        source = Path(distribution.locate_file(relative))
        if not source.is_file():
            raise ValueError("安装包缺少资源：{}".format(source))
        target = destination / filename
        shutil.copyfile(source, target)
        entries.append({"filename": filename, "source": "pix2tex==0.1.4 wheel/" + relative,
                        "bytes": target.stat().st_size, "sha256": sha256(target)})
    manifest = {
        "prepared_at": datetime.now(timezone.utc).isoformat(), "engine": "pix2tex",
        "package_version": distribution.version, "checkpoint_release": RELEASE,
        "upstream_repository": "https://github.com/lukas-blecher/LaTeX-OCR",
        "code_license": "MIT", "license_file": "LICENSE.pix2tex.txt",
        "checkpoint_license_note": "官方仓库在 MIT 许可下发布；v0.0.1 release 资产未另附独立模型许可证，保留仓库许可及来源，公开再分发前复核权利范围。",
        "hash_note": "SHA-256 是本次实际下载的固定指纹；上游未提供独立签名校验值。",
        "python": sys.version, "files": entries,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="显式下载 pix2tex 官方本地模型并记录来源校验")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model-dir", default=os.environ.get("MC_MODEL_DIR") or
                        str(Path(os.environ.get("MC_WORKSPACE", str(Path.home() / ".mistake_classifier"))) / "models"))
    args = parser.parse_args(argv)
    prepare(Path(args.model_dir).expanduser(), args.download)
    print("公式模型准备完成：{}".format(Path(args.model_dir) / "formula"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
