# -*- coding: utf-8 -*-
"""公式类错题预处理单元（技术路线 4.2.1（2）公式类错题）。

核心链路：
    公式图片 --pix2tex(LaTeX-OCR, CPU 离线)--> LaTeX 源码
    题干内联 $...$                        --> LaTeX 源码
    LaTeX 源码 --公式-知识点映射词典--> 知识点特征词（如 \\int_a^b -> 定积分）
    知识点特征词 + 题干文本特征 --> 完整特征向量

这是本项目突破"传统纯文本方法无法适配公式题"的关键：把无法分词的数学符号
结构，翻译成能进入 TF-IDF / 余弦相似度计算的中文特征词。

pix2tex 在独立 Python 运行环境内执行。未准备时：
* 题干里带 $...$ 的公式仍能正常映射（大多数手录错题属于这种情况）；
* 公式图片保留原图，提示准备识别组件或手动校对，不冒用文字OCR结果。
"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app import config
from app.core.models import FeatureBundle, Question
from app.core.preprocess import cleaner
from app.core.preprocess.base import BasePreprocessor, EngineStatus, passive_engine_status, register

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 公式-知识点映射词典
# --------------------------------------------------------------------------- #
class FormulaMapper:
    """把 LaTeX 源码映射为知识点特征词。

    词典为 resources/formula_map.json（纯数据，用户可自行扩展新学科规则）。
    规则按顺序匹配，越靠前越具体；命中后按 weight 决定特征词重复注入次数，
    从而在后续 TF-IDF 中获得更高词频权重。
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or config.FORMULA_MAP_FILE)
        self.rules: List[Tuple[str, "re.Pattern", List[str], float]] = []
        self.load()

    def load(self) -> None:
        self.rules = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            log.warning("公式映射词典加载失败 %s: %s", self.path, exc)
            return
        for r in data.get("rules", []):
            try:
                pat = re.compile(r["pattern"])
            except re.error as exc:
                log.warning("跳过非法正则 %r: %s", r.get("pattern"), exc)
                continue
            self.rules.append((r.get("subject", "*"), pat,
                               list(r.get("tokens", [])), float(r.get("weight", 1.0))))
        log.info("公式映射词典载入 %d 条规则", len(self.rules))

    def map_latex(self, latex: str, subject: str = "") -> Tuple[List[str], List[str]]:
        """返回（知识点特征词列表, 命中的规则说明）。"""
        if not latex:
            return [], []
        tokens: List[str] = []
        hits: List[str] = []
        for subj, pat, toks, weight in self.rules:
            if subj != "*" and subject and subj != subject:
                continue
            if not pat.search(latex):
                continue
            times = max(1, int(round(weight)))
            for t in toks:
                tokens.extend([t] * times)
            hits.append("{}→{}".format(pat.pattern, "/".join(toks)))
        return tokens, hits

    def rule_count(self) -> int:
        return len(self.rules)


_mapper: Optional[FormulaMapper] = None
_mapper_lock = threading.Lock()


def get_mapper() -> FormulaMapper:
    global _mapper
    if _mapper is None:
        with _mapper_lock:
            if _mapper is None:
                _mapper = FormulaMapper()
    return _mapper


def reload_mapper() -> None:
    """用户编辑词典后热重载。"""
    get_mapper().load()


# --------------------------------------------------------------------------- #
# pix2tex（LaTeX-OCR）引擎：惰性加载 + 优雅降级
# --------------------------------------------------------------------------- #
class LatexOcrEngine:
    """pix2tex 封装。首次使用时才加载模型，未安装则返回 None。"""

    def __init__(self) -> None:
        self._model = None
        self._checked = False
        self._error = ""
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        self._ensure()
        return self._model is not None

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
                from pix2tex.cli import LatexOCR      # noqa: WPS433
                self._model = LatexOCR()              # 默认 CPU 推理
                log.info("pix2tex 模型加载成功")
            except Exception as exc:                  # noqa: BLE001
                self._model = None
                self._error = str(exc)
                log.info("pix2tex 不可用，公式图片识别将降级：%s", exc)

    def recognize(self, image_path: str) -> str:
        self._ensure()
        if self._model is None:
            return ""
        try:
            from PIL import Image                     # noqa: WPS433
            with Image.open(image_path) as img:
                return str(self._model(img.convert("RGB")))
        except Exception as exc:                      # noqa: BLE001
            log.warning("公式识别失败 %s: %s", image_path, exc)
            return ""


_engine = LatexOcrEngine()


# --------------------------------------------------------------------------- #
# 预处理单元
# --------------------------------------------------------------------------- #
_RE_TEX_TOKEN = re.compile(r"\\[A-Za-z]+")


@register
class FormulaUnit(BasePreprocessor):
    """公式类错题单元（高数 / 物理 / 线代 / 概率）。"""

    name = "formula"
    display_name = "公式识别与知识点映射"
    qtypes = frozenset({config.QTYPE_FORMULA})

    def accept(self, q: Question) -> bool:
        # 公式题必走；其它题型若题干里含内联 LaTeX 也顺带处理
        if q.qtype in self.qtypes:
            return True
        return cleaner.looks_like_latex(q.raw_text or "")

    def process(self, q: Question, fb: FeatureBundle) -> None:
        latex_parts: List[str] = []

        # 来源 1：题干内联公式（$...$、\(...\)、\[...\]）
        _stripped, inline = cleaner.extract_latex(q.raw_text or "")
        latex_parts.extend(inline)

        # 来源 2：无 $ 包裹但含裸 LaTeX 命令的题干
        if not inline and _RE_TEX_TOKEN.search(q.raw_text or ""):
            latex_parts.append(q.raw_text or "")

        # 来源 3：仅使用已确认的公式图片校对结果，分类过程不隐式加载模型。
        review = getattr(q, "formula_ocr", None)
        if review is not None:
            if "formula" in fb.unreviewed_media:
                latex_parts.append(review.draft)
                fb.notes.append("公式图片使用未校对的自动识别 LaTeX")
            elif review.confirmed and (review.applied_text or review.draft or review.text):
                latex_parts.append(review.applied_text or review.draft or review.text)
                fb.notes.append("公式图片已使用用户确认的 LaTeX")
            elif q.media_path and Path(q.media_path).suffix.lower() in config.IMAGE_EXTS:
                fb.notes.append("公式图片识别结果尚未确认，未参与分类")
        elif q.media_path and Path(q.media_path).suffix.lower() in config.IMAGE_EXTS:
            fb.notes.append("公式图片尚未识别，请先完成公式校对")

        latex = " ".join(p for p in latex_parts if p).strip()
        if not latex:
            fb.pipeline.append(self.display_name + "（无公式）")
            return

        fb.latex = (fb.latex + " " + latex).strip()
        tokens, hits = get_mapper().map_latex(latex, q.subject)
        fb.formula_tokens.extend(tokens)
        if hits:
            fb.notes.append("公式映射命中 {} 条规则".format(len(hits)))
        fb.pipeline.append("{}（{} 条规则命中）".format(self.display_name, len(hits)))

    def status(self) -> EngineStatus:
        status = passive_engine_status(
            "pix2tex 公式识别 (LaTeX-OCR)", (("pix2tex",),),
            checked=_engine._checked, loaded=_engine._model is not None,
            loading=_engine._lock.locked(), error=_engine._error,
            fallback="手动输入的内联 LaTeX 公式仍可参与知识点映射。")
        if status.state != "missing":
            return status
        try:
            from app.core.preprocess.formula_runtime import formula_runtime
            from app.core.preprocess.media_recognition import model_root
            python, _backend = formula_runtime()
        except (OSError, ValueError):
            return status
        directory = model_root() / "formula"
        required = ("weights.pth", "image_resizer.pth", "config.yaml", "tokenizer.json", "manifest.json")
        complete = all((directory / name).is_file() and (directory / name).stat().st_size for name in required)
        return EngineStatus(status.name, False,
                            "独立公式识别环境：{}；{}。".format(python, "已找到模型，尚需实际图片验证" if complete else "模型文件未齐备"),
                            "unchecked" if complete else "model_missing",
                            "状态检查不加载或下载模型；请在公式校对窗口执行本地识别，识别时会校验模型文件。")
