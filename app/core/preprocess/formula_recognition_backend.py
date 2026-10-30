from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import tempfile
import time
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

REQUIRED = ("weights.pth", "image_resizer.pth", "config.yaml", "tokenizer.json")
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
# The bundled pix2tex 0.1.4 checkpoint's config.yaml declares these input limits.
# Keep supported-size images intact and let the model's learned resizer choose
# the glyph scale; an extra 105px height cap loses fraction/superscript detail.
MODEL_MAX_WIDTH = 672
MODEL_MAX_HEIGHT = 192
MAX_REQUEST_BYTES = 64 * 1024


def normalize_options(options=None) -> dict:
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("公式图片处理选项必须是对象。")
    unknown = set(options) - {"crop_box", "invert", "resize_mode"}
    if unknown:
        raise ValueError("不支持的公式图片选项：{}".format(", ".join(sorted(unknown))))
    crop = options.get("crop_box")
    if crop is not None:
        if (not isinstance(crop, (list, tuple)) or len(crop) != 4 or
                any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or int(value) != value for value in crop)):
            raise ValueError("选区必须是四个原图整数像素坐标。")
        crop = [int(value) for value in crop]
        if crop[0] < 0 or crop[1] < 0 or crop[2] <= crop[0] or crop[3] <= crop[1]:
            raise ValueError("公式选区为空或超出原图，请重新框选。")
    invert = options.get("invert", False)
    if not isinstance(invert, bool):
        raise ValueError("反色选项必须是布尔值。")
    resize = options.get("resize_mode", "fit")
    if resize not in ("fit", "original"):
        raise ValueError("缩放模式必须是 fit 或 original。")
    return {"crop_box": crop, "invert": invert, "resize_mode": resize}


def prepare_image(input_path: Path, options=None):
    from PIL import Image, ImageOps
    options = normalize_options(options)
    if not input_path.is_file():
        raise ValueError("公式图片不存在：{}".format(input_path))
    if input_path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("公式图片超过30MB，请缩小后重试。")
    try:
        with Image.open(input_path) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("公式图片超过2500万像素，请缩小后重试。")
            width, height = source.size
            crop = options["crop_box"]
            if crop is not None and (crop[2] > width or crop[3] > height):
                raise ValueError("公式选区超出原图边界，请重新框选。")
            image = source.convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            image = Image.alpha_composite(background, image).convert("RGB")
            if crop is not None:
                image = image.crop(tuple(crop))
            if image.width < 2 or image.height < 2:
                raise ValueError("公式选区过小，请框选完整公式。")
            if options["invert"]:
                image = ImageOps.invert(image)
            before = image.size
            if options["resize_mode"] == "fit":
                ratio = min(1.0, MODEL_MAX_WIDTH / image.width,
                            MODEL_MAX_HEIGHT / image.height)
                if ratio < 1.0:
                    image = image.resize((max(2, round(image.width * ratio)),
                                          max(2, round(image.height * ratio))), Image.Resampling.LANCZOS)
            metadata = {"original_size": [width, height], "selected_size": list(before),
                        "processed_size": list(image.size), "options": options}
            return image, metadata
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("公式图片损坏或格式不支持：{}".format(exc)) from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_models(directory: Path) -> dict:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("公式模型清单缺失，请先准备公式识别模型。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {item["filename"]: item for item in manifest.get("files", [])}
    for name in REQUIRED:
        path = directory / name
        expected = entries.get(name, {})
        if not path.is_file() or path.stat().st_size != expected.get("bytes"):
            raise ValueError("公式模型文件缺失或大小异常：{}".format(name))
        if sha256(path) != expected.get("sha256"):
            raise ValueError("公式模型校验失败：{}".format(name))
    return manifest


@contextmanager
def offline_network():
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create = socket.create_connection
    def forbidden(*args, **kwargs):
        raise RuntimeError("公式识别运行期禁止联网；请在准备阶段下载完整本地模型。")
    socket.socket.connect = forbidden
    socket.socket.connect_ex = forbidden
    socket.create_connection = forbidden
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.create_connection = original_create


def _new_record() -> dict:
    return {"text": "", "draft": "", "media_hash": "", "engine": "pix2tex 0.1.4 / CPU",
            "elapsed_ms": 0.0, "confirmed": False, "created_at": time.time(),
            "error": "", "applied_text": "", "review_status": "待人工复核"}


class FormulaRecognizer:
    """One sequential worker's lazy model, with independent per-image results."""

    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        self._model = None
        self._torch = None
        self._rng_state = None

    def _load_model(self):
        # Only an unloaded worker validates large model files and constructs the
        # network. Per-request input hashes and image validation remain separate.
        directory = self.directory
        validate_models(directory)
        import torch
        import yaml
        from pix2tex.cli import LatexOCR
        import pix2tex.cli as cli
        torch.set_num_threads(2)
        torch.manual_seed(42)
        settings = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        settings["tokenizer"] = str((directory / "tokenizer.json").resolve())
        with tempfile.TemporaryDirectory(prefix="mc_formula_") as temporary:
            config_file = Path(temporary) / "config.yaml"
            config_file.write_text(yaml.safe_dump(settings), encoding="utf-8")
            arguments = SimpleNamespace(config=str(config_file), checkpoint=str((directory / "weights.pth").resolve()),
                                        no_cuda=True, no_resize=False)
            model = LatexOCR(arguments)
        cli.clipboard.copy = lambda *args, **kwargs: None
        # Match the original one-shot seed *after* model construction consumed
        # random values. Resetting this state also makes request order irrelevant.
        self._rng_state = torch.get_rng_state()
        self._torch = torch
        self._model = model

    def recognize(self, input_path: Path, options=None) -> dict:
        started = time.perf_counter()
        record = _new_record()
        try:
            input_path = Path(input_path)
            if not input_path.is_file():
                raise ValueError("公式图片不存在：{}".format(input_path))
            record["media_hash"] = sha256(input_path)
            prepared, preprocessing = prepare_image(input_path, options)
            record["preprocessing"] = preprocessing
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                              NO_ALBUMENTATIONS_UPDATE="1", CUDA_VISIBLE_DEVICES="")
            with offline_network():
                if self._model is None:
                    self._load_model()
                self._torch.set_rng_state(self._rng_state)
                with self._torch.inference_mode():
                    text = str(self._model(prepared)).strip()
                if not text:
                    raise ValueError("公式图片未返回 LaTeX，请裁剪单个清晰公式后重试。")
            if sha256(input_path) != record["media_hash"]:
                raise ValueError("公式附件在识别期间已变化，请重新识别。")
            record.update(text=text, draft=text)
        except Exception as exc:
            record["error"] = "{}: {}".format(type(exc).__name__, exc)
        record["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return record


def recognize(input_path: Path, directory: Path, options=None) -> dict:
    """Retain the original independent, single-request Python entry point."""
    return FormulaRecognizer(directory).recognize(input_path, options)


def handle_request(request: Path, recognizer: FormulaRecognizer) -> dict:
    """Read only the documented JSON fields; malformed jobs get normal errors."""
    started = time.perf_counter()
    try:
        if request.is_symlink() or not request.is_file():
            raise ValueError("公式请求必须是普通 JSON 文件。")
        if request.stat().st_size > MAX_REQUEST_BYTES:
            raise ValueError("公式请求 JSON 过大。")
        payload = json.loads(request.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) - {"input_path", "options"}:
            raise ValueError("公式请求必须是包含 input_path 和 options 的 JSON 对象。")
        path = payload.get("input_path")
        if not isinstance(path, str) or not path or not Path(path).is_absolute():
            raise ValueError("公式请求 input_path 必须是绝对路径。")
        options = payload.get("options")
        normalize_options(options)
        return recognizer.recognize(Path(path), options)
    except Exception as exc:
        record = _new_record()
        record["error"] = "{}: {}".format(type(exc).__name__, exc)
        record["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return record


def _write_response(path: Path, record: dict):
    descriptor, temporary = tempfile.mkstemp(prefix=".response-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def serve(session_dir: Path, directory: Path) -> int:
    """Consume request-NNNNNNNN.json in order until the client's stop file."""
    session_dir = Path(session_dir).absolute()
    if session_dir.is_symlink() or not session_dir.is_dir():
        raise ValueError("公式会话目录不存在或不是普通目录。")
    recognizer = FormulaRecognizer(directory)
    sequence = 1
    while not (session_dir / "stop").exists():
        request = session_dir / "request-{:08d}.json".format(sequence)
        if not request.exists() and not request.is_symlink():
            time.sleep(0.025)
            continue
        response = session_dir / "response-{:08d}.json".format(sequence)
        _write_response(response, handle_request(request, recognizer))
        sequence += 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="pix2tex 本地离线公式识别工作进程")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--serve", help="私有会话目录：按序读取请求并复用公式模型")
    parser.add_argument("--model-dir", required=True, help="包含 weights.pth 等文件的 formula 目录")
    parser.add_argument("--options-file", help="公式选区、反色和缩放选项 JSON 文件")
    args = parser.parse_args(argv)
    if args.serve:
        if args.input or args.output or args.options_file:
            parser.error("--serve 不能与单次请求参数混用")
        try:
            return serve(Path(args.serve), Path(args.model_dir))
        except (OSError, ValueError) as exc:
            print("公式服务启动或通信失败：{}".format(exc), file=sys.stderr)
            return 1
    if not args.input or not args.output:
        parser.error("单次识别必须指定 --input 和 --output")
    try:
        options = json.loads(Path(args.options_file).read_text(encoding="utf-8")) if args.options_file else None
    except (OSError, ValueError) as exc:
        options = None
        record = {"text": "", "draft": "", "error": "无法读取识别选项：{}".format(exc),
                  "media_hash": "", "confirmed": False}
    else:
        record = recognize(Path(args.input).resolve(), Path(args.model_dir).resolve(), options)
    Path(args.output).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if record["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
