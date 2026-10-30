# -*- coding: utf-8 -*-
"""融合图形题已保存的题干与标注，保留参数原值和来源。

图片识别由独立媒体任务完成；本单元不加载模型或改写用户内容。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional

from app import config
from app.core.models import FeatureBundle, Question
from app.core.preprocess.base import BasePreprocessor, EngineStatus, passive_engine_status, register
from app.core.preprocess.figure_parameters import extract_parameters, parameter_tokens

log = logging.getLogger(__name__)


class PaddleOcrEngine:
    """PaddleOCR 轻量版封装（惰性加载 + 优雅降级）。"""

    def __init__(self) -> None:
        self._ocr = None
        self._checked = False
        self._error = ""
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        self._ensure()
        return self._ocr is not None

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
                from paddleocr import PaddleOCR       # noqa: WPS433
                # 轻量版：mobile 模型 + CPU + 关闭方向分类以降低耗时
                try:
                    self._ocr = PaddleOCR(use_angle_cls=False, lang="ch",
                                          use_gpu=False, show_log=False)
                except TypeError:
                    # 兼容 PaddleOCR 3.x 的新参数签名
                    self._ocr = PaddleOCR(lang="ch")
                log.info("PaddleOCR 加载成功")
            except Exception as exc:                  # noqa: BLE001
                self._ocr = None
                self._error = str(exc)
                log.info("PaddleOCR 不可用，图形题降级为人工标注：%s", exc)

    def extract(self, image_path: str) -> List[str]:
        """返回图内识别出的文本行（按置信度过滤）。"""
        self._ensure()
        if self._ocr is None:
            return []
        try:
            try:
                res = self._ocr.ocr(image_path, cls=False)
            except TypeError:
                res = self._ocr.ocr(image_path)
            return self._flatten(res)
        except Exception as exc:                      # noqa: BLE001
            log.warning("图形 OCR 失败 %s: %s", image_path, exc)
            return []

    @staticmethod
    def _flatten(res: object, min_conf: float = 0.6) -> List[str]:
        """兼容 PaddleOCR 2.x / 3.x 的多种返回结构。"""
        lines: List[str] = []

        def walk(node: object) -> None:
            if isinstance(node, (list, tuple)):
                # 2.x: [[box, (text, conf)], ...]
                if (len(node) == 2 and isinstance(node[1], (list, tuple))
                        and len(node[1]) == 2 and isinstance(node[1][0], str)):
                    text, conf = node[1]
                    if float(conf) >= min_conf and text.strip():
                        lines.append(text.strip())
                    return
                for x in node:
                    walk(x)
            elif isinstance(node, dict):
                # 3.x: {"rec_texts": [...], "rec_scores": [...]}
                texts = node.get("rec_texts")
                scores = node.get("rec_scores")
                if isinstance(texts, list):
                    for i, t in enumerate(texts):
                        c = scores[i] if isinstance(scores, list) and i < len(scores) else 1.0
                        if float(c) >= min_conf and str(t).strip():
                            lines.append(str(t).strip())

        walk(res)
        return lines


_engine = PaddleOcrEngine()


@register
class FigureUnit(BasePreprocessor):
    """图形类错题单元。

    注册顺序在 text_cn 之前（见 preprocess/__init__.py 的导入顺序），
    使 OCR 结果能被后续文本单元一起分词。
    """

    name = "figure"
    display_name = "图形 OCR 与标注融合"
    qtypes = frozenset({config.QTYPE_FIGURE, config.QTYPE_FORMULA})

    def accept(self, q: Question) -> bool:
        return q.qtype in self.qtypes or bool(q.media_path and Path(q.media_path).suffix.lower() in config.IMAGE_EXTS)

    def process(self, q: Question, fb: FeatureBundle) -> None:
        source = "reviewed_text" if q.image_ocr is not None and q.image_ocr.confirmed else "question_text"
        fb.figure_parameters.extend(extract_parameters(q.raw_text or "", source))
        if "image" in fb.unreviewed_media and q.image_ocr is not None:
            fb.figure_parameters.extend(extract_parameters(q.image_ocr.draft, "automatic_ocr"))
            q.raw_text = "\n".join(dict.fromkeys(text.strip()
                for text in (q.raw_text or "", q.image_ocr.draft) if text.strip()))
        fb.figure_parameters.extend(extract_parameters(q.annotation or "", "annotation"))
        fb.text_tokens.extend(parameter_tokens(fb.figure_parameters))
        fb.pipeline.append("{}（参数 {} 项 / 标注 {} 字）".format(
            self.display_name, len(fb.figure_parameters), len(q.annotation or "")))
        if q.image_ocr is not None:
            if q.image_ocr.confirmed:
                fb.ocr_text = q.raw_text
                fb.notes.append("使用已校对的图片题干，不重复识别或追加原始 OCR 文字")
            elif "image" in fb.unreviewed_media:
                fb.notes.append("图形参数来自未校对的自动识别文字，原值保留供复核")
            else:
                fb.notes.append("图片识别结果尚未应用，分类仅使用已保存的题干和标注")
            return
        if not q.media_path or Path(q.media_path).suffix.lower() not in config.IMAGE_EXTS:
            return
        if not Path(q.media_path).exists():
            fb.notes.append("图片文件不存在：{}".format(q.media_path))
            return

        fb.notes.append("图片尚未识别，当前使用已保存题干与标注；可使用批量媒体识别与校对")

    def status(self) -> EngineStatus:
        return passive_engine_status(
            "PaddleOCR 图形文本提取", (("paddleocr", "paddle"),),
            checked=_engine._checked, loaded=_engine._ocr is not None,
            loading=_engine._lock.locked(), error=_engine._error,
            fallback="图片题可先手动填写题干和图中关键信息。")
