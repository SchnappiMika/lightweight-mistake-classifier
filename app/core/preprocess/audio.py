# -*- coding: utf-8 -*-
"""英语听力音频预处理单元（技术路线 4.2.1（2）英语类错题）。

链路：听力音频 --Whisper 轻量化模型(tiny/base, CPU 离线)--> 英文转写文本
      --> 交给 text_en 单元提取核心场景词、知识点词与问题特征词

引擎优先级（都不可用则降级为"仅用户标注"）：
    1. faster-whisper —— CTranslate2 后端，CPU 上比原版快 3-5 倍，体积更小，
                         对"普通笔记本流畅运行"这一硬指标最友好，故列为首选；
    2. openai-whisper —— 官方实现，兼容性好；
    3. 降级       —— 提示用户在标注区手动补充听力关键信息。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from app import config
from app.core.models import FeatureBundle, Question
from app.core.preprocess.base import BasePreprocessor, EngineStatus, passive_engine_status, register

log = logging.getLogger(__name__)

# 轻量化优先：tiny 约 39M 参数，base 约 74M，均可在 CPU 上实时转写
DEFAULT_MODEL = "base"


class WhisperEngine:
    """Whisper 封装（惰性加载 + 双后端 + 优雅降级）。"""

    def __init__(self, model_size: str = DEFAULT_MODEL) -> None:
        self.model_size = model_size
        self._backend = ""          # "faster" / "openai" / ""
        self._model = None
        self._checked = False
        self._error = ""
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        self._ensure()
        return self._model is not None

    @property
    def backend(self) -> str:
        self._ensure()
        return self._backend

    @property
    def error(self) -> str:
        self._ensure()
        return self._error

    def _ensure(self) -> None:
        if self._checked:
            return
        with self._lock:
            if self._checked:
                return
            self._checked = True
            try:
                from faster_whisper import WhisperModel    # noqa: WPS433
                self._model = WhisperModel(self.model_size, device="cpu",
                                           compute_type="int8")
                self._backend = "faster"
                log.info("faster-whisper(%s) 加载成功", self.model_size)
                return
            except Exception as exc:                       # noqa: BLE001
                self._error = str(exc)
            try:
                import whisper                             # noqa: WPS433
                self._model = whisper.load_model(self.model_size)
                self._backend = "openai"
                log.info("openai-whisper(%s) 加载成功", self.model_size)
                return
            except Exception as exc:                       # noqa: BLE001
                self._error = "{} | {}".format(self._error, exc)
                self._model = None
                log.info("Whisper 不可用，听力题降级为人工标注：%s", self._error)

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        self._ensure()
        if self._model is None:
            return ""
        try:
            if self._backend == "faster":
                segments, _info = self._model.transcribe(audio_path, language=language,
                                                         beam_size=1, vad_filter=True)
                return " ".join(seg.text.strip() for seg in segments).strip()
            result = self._model.transcribe(audio_path, language=language, fp16=False)
            return str(result.get("text", "")).strip()
        except Exception as exc:                           # noqa: BLE001
            log.warning("音频转写失败 %s: %s", audio_path, exc)
            return ""


_engine = WhisperEngine()


def get_engine() -> WhisperEngine:
    return _engine


@register
class AudioUnit(BasePreprocessor):
    """英语听力题单元。"""

    name = "audio"
    display_name = "听力音频离线转写"
    qtypes = frozenset({config.QTYPE_LISTENING})

    def accept(self, q: Question) -> bool:
        return q.qtype in self.qtypes or bool(q.media_path and Path(q.media_path).suffix.lower() in config.AUDIO_EXTS)

    def process(self, q: Question, fb: FeatureBundle) -> None:
        if not q.media_path or Path(q.media_path).suffix.lower() not in config.AUDIO_EXTS:
            fb.notes.append("听力题未关联音频文件，特征仅来自题干与标注")
            return
        if not Path(q.media_path).exists():
            fb.notes.append("音频文件不存在：{}".format(q.media_path))
            return

        review = getattr(q, "audio_transcript", None)
        if review is None or (not review.confirmed and "audio" not in fb.unreviewed_media):
            fb.notes.append("听力转写尚未确认，未参与分类，请先校对音频文字")
            fb.pipeline.append(self.display_name + "（待确认）")
            return
        text = review.draft if "audio" in fb.unreviewed_media else (review.applied_text or review.draft or review.text)
        if text:
            fb.transcript = (fb.transcript + " " + text).strip()
            fb.notes.append("已使用{}听力转写 {} 字".format(
                "未校对的自动" if "audio" in fb.unreviewed_media else "用户确认的", len(text)))
        fb.pipeline.append(self.display_name + ("（未校对）" if "audio" in fb.unreviewed_media else "（已确认）"))

    def status(self) -> EngineStatus:
        return passive_engine_status(
            "Whisper 听力转写", (("faster_whisper",), ("whisper",)),
            checked=_engine._checked, loaded=_engine._model is not None,
            loading=_engine._lock.locked(), error=_engine._error,
            fallback="听力题可先手动填写题干和标注。",
            ready_detail="已加载 {} 后端 / {} 模型。".format(_engine._backend, _engine.model_size))
