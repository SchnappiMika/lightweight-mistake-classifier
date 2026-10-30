from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from copy import deepcopy
from pathlib import Path
from typing import Optional

from app import config
from app.core.models import MediaReviewRecord, Question
from app.core.preprocess.image_ocr import ImageOcrError


def media_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_root() -> Path:
    override = os.environ.get("MC_MODEL_DIR")
    if override:
        return Path(override)
    if getattr(sys, "frozen", False):
        bundled = Path(sys.executable).parent / "models"
        if bundled.is_dir():
            return bundled
    return config.WORKSPACE_DIR / "models"


def _model_dir(kind: str) -> Path:
    path = model_root() / ("formula" if kind == "formula" else "audio")
    if not path.is_dir() or not any(path.iterdir()):
        raise ImageOcrError("{} 本地模型目录为空：{}。请先准备模型后再识别。".format(kind, path))
    return path


def _faster_whisper_path(directory: Path) -> Path:
    candidates = [directory]
    candidates.extend(directory.glob("models--Systran--faster-whisper-base/snapshots/*"))
    for candidate in candidates:
        required = (candidate / "config.json", candidate / "model.bin", candidate / "tokenizer.json")
        if all(path.is_file() and path.stat().st_size > 0 for path in required):
            return candidate
    raise ImageOcrError("faster-whisper base 模型文件不完整：{}".format(directory))


def _decode_audio(path: Path):
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise ImageOcrError("音频解码需要 av 和 numpy。") from exc
    chunks = []
    raw_chunks = []
    source_rate = 16000
    with av.open(str(path), mode="r") as container:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            raise ImageOcrError("音频文件没有可用音轨。")
        source_rate = int(stream.rate or 16000)
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(stream):
            raw = frame.to_ndarray()
            if raw.ndim == 2:
                raw = raw.mean(axis=0)
            raw_chunks.append(raw)
            converted = resampler.resample(frame)
            if not isinstance(converted, list):
                converted = [converted]
            chunks.extend(item.to_ndarray() for item in converted if item is not None)
        flushed = resampler.resample(None)
        if not isinstance(flushed, list):
            flushed = [flushed]
        chunks.extend(item.to_ndarray() for item in flushed if item is not None)
    if chunks:
        samples = np.concatenate(chunks, axis=1)[0]
        return samples.astype(np.float32) / 32768.0
    if not raw_chunks:
        raise ImageOcrError("音频文件没有可解码的采样数据。")
    raw_array = np.concatenate(raw_chunks)
    raw_dtype = raw_array.dtype
    raw = raw_array.astype(np.float32)
    if raw.size == 1:
        return raw
    if np.issubdtype(raw_dtype, np.integer) or float(max(abs(raw.min()), abs(raw.max()), 0.0)) > 1.0:
        limit = float(max(abs(raw.min()), abs(raw.max()), 1.0))
        raw /= limit
    source_time = np.linspace(0.0, raw.size / float(source_rate), num=raw.size, endpoint=False)
    target_size = max(1, int(round(raw.size * 16000.0 / float(source_rate))))
    target_time = np.linspace(0.0, raw.size / float(source_rate), num=target_size, endpoint=False)
    return np.interp(target_time, source_time, raw).astype(np.float32)


def _duration(path: Path) -> float:
    if path.suffix.lower() != ".wav":
        return 0.0
    try:
        with wave.open(str(path), "rb") as stream:
            return float(stream.getnframes()) / float(stream.getframerate() or 1)
    except (OSError, wave.Error, EOFError):
        return 0.0


class FormulaRecognitionEngine:
    def __init__(self) -> None:
        self._model = None
        self._error = ""

    def recognize(self, path: Path, options=None) -> str:
        directory = _model_dir("formula")
        try:
            from app.core.preprocess.formula_runtime import formula_runtime
            python, backend = formula_runtime()
            environment = os.environ.copy()
            environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            for name in ("PYTHONHOME", "PYTHONPATH"):
                environment.pop(name, None)
            process_options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            with tempfile.TemporaryDirectory(prefix="mc_formula_") as temporary:
                output = Path(temporary) / "result.json"
                command = [str(python), str(backend), "--input", str(path.resolve()),
                           "--output", str(output), "--model-dir", str(directory.resolve())]
                if options is not None:
                    options_file = Path(temporary) / "options.json"
                    options_file.write_text(json.dumps(options, ensure_ascii=False), encoding="utf-8")
                    command.extend(["--options-file", str(options_file)])
                result = subprocess.run(
                    command,
                    env=environment, cwd=str(python.resolve().parent),
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=540, **process_options)
                if not output.is_file():
                    raise ImageOcrError("公式识别进程没有返回结果（退出码 {}）：{}".format(
                        result.returncode, (result.stderr or "")[-1500:]))
                record = json.loads(output.read_text(encoding="utf-8"))
                if record.get("error"):
                    raise ImageOcrError(record["error"])
                if result.returncode:
                    raise ImageOcrError("公式识别进程异常退出：{}".format(result.returncode))
                return str(record.get("text", "")).strip()
        except Exception as exc:
            self._error = str(exc)
            raise ImageOcrError("公式图片识别失败：{}".format(exc)) from exc

    def recognize_isolated(self, path: str, options=None) -> MediaReviewRecord:
        return recognize_isolated(path, "formula", options)


class AudioRecognitionEngine:
    def __init__(self) -> None:
        self._model = None
        self._backend = ""
        self._error = ""

    def recognize(self, path: Path) -> str:
        directory = _model_dir("audio")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            if importlib.util.find_spec("faster_whisper") is not None:
                from faster_whisper import WhisperModel
                model_path = _faster_whisper_path(directory)
                if self._model is None:
                    self._model = WhisperModel(str(model_path), device="cpu", compute_type="int8",
                                              local_files_only=True)
                    self._backend = "faster-whisper"
                samples = _decode_audio(path)
                segments, _info = self._model.transcribe(samples, language="en",
                                                         beam_size=1, vad_filter=True)
                return " ".join(segment.text.strip() for segment in segments).strip()
            if importlib.util.find_spec("whisper") is not None:
                import whisper
                raise ImageOcrError("openai-whisper 未纳入本项目离线音频运行路径，请准备 faster-whisper base。")
        except Exception as exc:
            self._error = str(exc)
            raise ImageOcrError("听力音频转写失败：{}".format(exc)) from exc
        raise ImageOcrError("未安装 faster-whisper 或 whisper，无法执行听力转写。")

    def recognize_isolated(self, path: str) -> MediaReviewRecord:
        return recognize_isolated(path, "audio")


_engines = {"formula": FormulaRecognitionEngine(), "audio": AudioRecognitionEngine()}


def get_engine(kind: str):
    if kind not in _engines:
        raise ValueError("不支持的媒体识别类型：{}".format(kind))
    return _engines[kind]


def recognize_isolated(path: str, kind: str, options=None) -> MediaReviewRecord:
    started = time.perf_counter()
    source = Path(path)
    record = MediaReviewRecord(engine=kind)
    try:
        if kind == "formula":
            from app.core.preprocess.formula_recognition_backend import normalize_options
            record.recognition_options = normalize_options(options)
        if not source.is_file():
            raise ImageOcrError("媒体文件不存在：{}".format(source))
        digest = media_hash(source)
        if options is not None and kind != "formula":
            raise ImageOcrError("图片处理选项仅适用于公式识别。")
        text = (get_engine(kind).recognize(source, options=options) if kind == "formula" and options is not None
                else get_engine(kind).recognize(source))
        if not text:
            raise ImageOcrError("未识别到可用内容，请校对后手动录入。")
        record.text = text
        record.draft = text
        record.media_hash = digest
        record.engine = getattr(get_engine(kind), "_backend", "pix2tex") or "pix2tex"
        record.duration_seconds = _duration(source) if kind == "audio" else 0.0
    except Exception as exc:
        record.error = str(exc)
        record.media_hash = media_hash(source) if source.is_file() else ""
        record.duration_seconds = _duration(source) if kind == "audio" and source.is_file() else 0.0
    record.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return record


def save_media_review(repo, question: Question, record: MediaReviewRecord,
                      kind: str, apply: bool = False) -> None:
    if kind not in ("formula", "audio"):
        raise ValueError("不支持的媒体识别类型：{}".format(kind))
    if repo.question_by_id(question.id) is not question:
        raise ImageOcrError("当前题目已被删除或重新加载，请重新打开识别窗口。")
    path = Path(question.media_path)
    if not path.is_file() or media_hash(path) != record.media_hash:
        raise ImageOcrError("附件已变化或丢失，不能应用旧识别结果，请重新识别当前媒体。")
    if apply and not record.draft.strip():
        raise ImageOcrError("校对内容为空，请填写后再应用。")
    previous = deepcopy(question.__dict__)
    try:
        stored = deepcopy(record)
        field_name = "formula_ocr" if kind == "formula" else "audio_transcript"
        existing = getattr(question, field_name, None)
        used_draft_changed = (
            not apply and question.classified and
            kind in (question.feature or {}).get("unreviewed_media", []) and
            existing is not None and not existing.confirmed and
            existing.draft.strip() != stored.draft.strip())
        if not apply and existing is not None and existing.confirmed:
            stored.confirmed = True
            stored.applied_text = existing.applied_text or existing.draft or existing.text
        if apply:
            stored.confirmed = True
            stored.applied_text = stored.draft.strip()
        if kind == "formula":
            question.formula_ocr = stored
        else:
            question.audio_transcript = stored
        if apply or used_draft_changed:
            question.invalidate_classification()
        repo.save_questions()
    except BaseException:
        question.__dict__.clear()
        question.__dict__.update(previous)
        raise
