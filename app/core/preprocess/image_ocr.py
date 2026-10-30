from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from app.core.models import ImageOcrRecord
from app.core.preprocess.base import EngineStatus, passive_engine_status

MODEL_FILES = {
    "Det": ("PP-OCRv6_det_small.onnx", 9929594,
            "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"),
    "Cls": ("ch_ppocr_mobile_v2.0_cls_mobile.onnx", 585532,
            "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"),
    "Rec": ("PP-OCRv6_rec_small.onnx", 21234383,
            "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"),
}
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


class ImageOcrError(ValueError):
    pass


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_paths(verify_hash: bool = False):
    if getattr(sys, "frozen", False):
        folder = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent / "_internal")) / "rapidocr" / "models"
    else:
        specification = importlib.util.find_spec("rapidocr")
        if specification is None or not specification.origin:
            raise ImageOcrError("未安装文字识别组件，请先运行项目中的「安装图片识别.bat」。")
        folder = Path(specification.origin).parent / "models"
    paths = {}
    for kind, (name, size, digest) in MODEL_FILES.items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != size:
            raise ImageOcrError("本地模型缺失或大小不正确：{}。请重新安装图片识别组件。".format(path))
        if verify_hash and file_hash(path) != digest:
            raise ImageOcrError("本地模型校验失败：{}。请重新安装图片识别组件。".format(path))
        paths[kind] = path
    return paths


class ImageTextEngine:
    def __init__(self):
        self._model = None
        self._checked = False
        self._error = ""
        self._verified = False
        self._lock = threading.Lock()

    def status(self):
        if getattr(sys, "frozen", False) and (Path(sys.executable).parent / "media_worker.exe").is_file():
            if self._lock.locked():
                return EngineStatus("RapidOCR 图片文字识别", False, "独立工作进程正在识别。", "loading")
            if self._error:
                return EngineStatus("RapidOCR 图片文字识别", False, self._error, "failed", "检查发布包与附件后重试。")
            try:
                model_paths()
            except (ImageOcrError, OSError) as exc:
                return EngineStatus("RapidOCR 图片文字识别", False, str(exc), "model_missing", "请重新解压完整发布包。")
            return EngineStatus("RapidOCR 图片文字识别", self._verified,
                                "发布包包含独立识别程序和三个本地模型；实际识别在独立进程执行。",
                                "verified" if self._verified else "unchecked",
                                "请在导入页选择图片进行识别与人工校对。")
        status = passive_engine_status(
            "RapidOCR 图片文字识别", (("rapidocr", "onnxruntime", "PIL", "cv2"),),
            checked=self._checked, loaded=self._model is not None or self._verified,
            loading=self._lock.locked(), error=self._error,
            fallback="使用「安装图片识别.bat」准备组件，随后在导入页点击「识别与校对」。",
            ready_detail=("本次运行已通过本地模型识别。界面使用独立进程进行 CPU 识别，"
                          "任务结束后释放模型，下次识别重新加载。"))
        if status.state == "ready" and self._verified:
            status.state = "verified"
            status.action = "请逐行校对识别结果。识别成功不代表复杂公式、手写或图形含义已验证。"
        if status.state == "unchecked":
            try:
                model_paths()
            except (ImageOcrError, OSError) as exc:
                return EngineStatus(status.name, False, str(exc), "model_missing",
                                    "重新运行「安装图片识别.bat」，模型检查不会触发下载。")
            status.detail += "本地三个模型文件及大小已检查；完整校验和加载在实际识别时执行。"
        return status

    def recognize_isolated(self, image_path: str) -> ImageOcrRecord:
        with self._lock:
            self._checked = True
            self._verified = False
            try:
                options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
                environment = os.environ.copy()
                environment["PYTHONUTF8"] = "1"
                environment["PYTHONIOENCODING"] = "utf-8"
                with tempfile.TemporaryDirectory(prefix="mc_ocr_") as temporary:
                    output = Path(temporary) / "result.json"
                    frozen = getattr(sys, "frozen", False)
                    if frozen:
                        helper = Path(sys.executable).parent / "media_worker.exe"
                        command = ([str(helper), "image"] if helper.is_file() else
                                   [sys.executable, "--image-ocr-worker"])
                        command.extend([str(Path(image_path).resolve()), str(output)])
                    else:
                        command = [sys.executable, "-m", "app.core.preprocess.image_ocr_worker",
                                   str(Path(image_path).resolve())]
                    result = subprocess.run(
                        command, cwd=None if frozen else str(Path(__file__).resolve().parents[3]),
                        env=environment, capture_output=True, text=True, encoding="utf-8",
                        errors="replace", timeout=180, **options)
                    payload = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else None
                if result.returncode:
                    detail = (payload or {}).get("error") or (result.stderr or "").strip() or "识别进程异常退出，代码 {}".format(result.returncode)
                    raise ImageOcrError(detail[-2000:])
                if frozen and payload is None:
                    raise ImageOcrError("识别进程未生成结果文件，请检查发布包中的 media_worker.exe。")
                record = ImageOcrRecord.from_dict(payload if payload is not None else json.loads(result.stdout))
                if not record.text.strip() or not record.media_hash:
                    raise ImageOcrError("识别进程未返回有效文字，请重试。")
                self._verified = True
                self._error = ""
                return record
            except subprocess.TimeoutExpired as exc:
                self._error = "图片识别超过三分钟，已结束本次进程。请裁剪或缩小图片后重试。"
                raise ImageOcrError(self._error) from exc
            except Exception as exc:
                self._verified = False
                self._error = str(exc)
                raise ImageOcrError("本地图片识别失败：{}".format(exc)) from exc

    def _ensure(self):
        if self._model is not None:
            return
        self._checked = True
        try:
            paths = model_paths(verify_hash=True)
            from rapidocr import RapidOCR

            parameters = {"{}.model_path".format(kind): str(path) for kind, path in paths.items()}
            parameters.update({"Global.log_level": "error", "Global.text_score": 0.3,
                               "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                               "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                               "EngineConfig.onnxruntime.use_cuda": False,
                               "EngineConfig.onnxruntime.use_dml": False})
            model = RapidOCR(params=parameters)
            model._load_det_model()
            model._load_cls_model()
            model._load_rec_model()
            self._model = model
            self._error = ""
        except Exception as exc:
            self._error = str(exc)
            raise ImageOcrError("文字识别模型初始化失败：{}。请检查本地组件后重试。".format(exc)) from exc

    def recognize(self, image_path: str) -> ImageOcrRecord:
        started = time.perf_counter()
        path = Path(image_path)
        if not path.is_file():
            raise ImageOcrError("图片文件不存在：{}".format(path))
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ImageOcrError("图片超过 30MB，请裁剪或缩小后重试。")
        try:
            import numpy as np
            from PIL import Image, ImageOps

            content = path.read_bytes()
            with Image.open(io.BytesIO(content)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ImageOcrError("图片超过 2500 万像素，请裁剪或缩小后重试。")
                image = ImageOps.exif_transpose(image).convert("RGB")
                pixels = np.asarray(image)[:, :, ::-1].copy()
        except ImportError as exc:
            raise ImageOcrError("图片识别组件未准备，请运行「安装图片识别.bat」。") from exc
        except ImageOcrError:
            raise
        except Exception as exc:
            raise ImageOcrError("图片无法读取，请检查格式或重新截图：{}".format(exc)) from exc
        with self._lock:
            try:
                self._ensure()
                result = self._model(pixels)
            except Exception as exc:
                self._error = str(exc)
                raise ImageOcrError(str(exc)) from exc
        lines = []
        texts = result.txts if result.txts is not None else ()
        scores = result.scores if result.scores is not None else ()
        for index, text in enumerate(texts):
            if not text.strip():
                continue
            lines.append({"text": text.strip(), "score": float(scores[index]),
                          "box": result.boxes[index].tolist() if result.boxes is not None else []})
        if not lines:
            raise ImageOcrError("未识别到可用文字。请裁剪题目区域、提高图片清晰度后重试；也可手动录入。")
        text = "\n".join(line["text"] for line in lines)
        return ImageOcrRecord(text=text, draft=text, lines=lines, media_hash=hashlib.sha256(content).hexdigest(),
                              elapsed_ms=(time.perf_counter() - started) * 1000)


_engine = ImageTextEngine()


def get_engine() -> ImageTextEngine:
    return _engine
