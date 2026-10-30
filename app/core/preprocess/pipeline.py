# -*- coding: utf-8 -*-
"""多模态预处理调度器。

职责：类型分流 -> 依次调用可插拔单元 -> 融合为统一的 FeatureBundle。

单元执行顺序（有依赖关系，不可随意调换）：
    audio   -> 产出 transcript，供 text_en 消费
    figure  -> 产出图内 OCR 文本，并入题干供后续分词
    formula -> 抽取 LaTeX 并映射为知识点特征词
    text_cn -> 中文分词 / 双层停用词 / 核心实词
    text_en -> 英文分词 / 词形还原 / 英文停用词

调度器在 Question 的**副本**上工作，各单元对 raw_text 的补充不会污染原始记录。
"""
from __future__ import annotations

import dataclasses
import logging
import time
from typing import List, Optional

from app import config
from app.core.models import FeatureBundle, Question
from app.core.preprocess import base

# 导入即注册：顺序决定执行顺序
from app.core.preprocess import audio as _audio      # noqa: F401
from app.core.preprocess import figure as _figure    # noqa: F401
from app.core.preprocess import formula as _formula  # noqa: F401
from app.core.preprocess import text_cn as _text_cn  # noqa: F401
from app.core.preprocess import text_en as _text_en  # noqa: F401

log = logging.getLogger(__name__)


def preprocess(q: Question, *, use_unreviewed_media: bool = False) -> FeatureBundle:
    """对单条错题执行完整的多模态预处理，返回标准化特征包。

    任何单元内部异常都会被捕获并记入 ``fb.notes``，保证批量分类过程
    不会因为一张坏图 / 一个坏音频而整体崩溃（对应"无崩溃、无数据丢失"）。
    """
    t0 = time.perf_counter()
    work = dataclasses.replace(q)     # 副本：允许单元把 OCR 文本并入题干
    fb = FeatureBundle(question_id=q.id, subject=q.subject, qtype=q.qtype)
    if use_unreviewed_media:
        from app.core.preprocess.media_preparation import media_kind
        kind = media_kind(q)
        attribute = {"image": "image_ocr", "formula": "formula_ocr", "audio": "audio_transcript"}.get(kind)
        record = getattr(q, attribute, None) if attribute else None
        if record is not None and not record.confirmed and record.draft.strip():
            fb.unreviewed_media.append(kind)
            fb.notes.append("本次分类使用自动识别的未校对内容，请对照原始附件复核")
            if kind == "image":
                fb.ocr_text = record.draft.strip()

    for unit in base.all_units():
        try:
            if unit.accept(work):
                unit.process(work, fb)
        except Exception as exc:      # noqa: BLE001
            log.exception("预处理单元 %s 执行失败", unit.name)
            fb.notes.append("{} 执行异常已跳过：{}".format(unit.display_name, exc))

    # Explicit bilingual vocabulary bridges scientific source wording to the
    # user's Chinese keywords. Original question text and English tokens stay.
    from app.core.preprocess.scientific_terms import canonical_terms
    scientific = canonical_terms("\n".join(part for part in (
        work.raw_text, work.annotation, fb.transcript) if part), work.subject)
    if scientific:
        fb.text_tokens.extend(scientific)
        fb.notes.append("英文科学术语规范词：" + "、".join(scientific))

    # 去重但保留词频（TF-IDF 需要词频，因此只做顺序稳定化，不做集合去重）
    fb.text_tokens = [t for t in fb.text_tokens if t]
    fb.formula_tokens = [t for t in fb.formula_tokens if t]

    if fb.is_empty():
        fb.notes.append("未提取到任何有效特征，建议补充题干文本或人工标注")

    fb.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return fb


def preprocess_many(questions: List[Question],
                    progress: Optional[callable] = None) -> List[FeatureBundle]:
    """批量预处理。``progress(done, total, question)`` 用于界面进度回调。"""
    out: List[FeatureBundle] = []
    total = len(questions)
    for i, q in enumerate(questions, 1):
        out.append(preprocess(q))
        if progress:
            progress(i, total, q)
    return out


def normalize_question(q: Question) -> Question:
    """导入阶段的标准化：自动补全学科与题型。"""
    if not q.subject or q.subject == config.SUBJECT_GENERAL:
        guessed = _cleaner_guess_subject(q)
        if guessed != config.SUBJECT_GENERAL:
            q.subject = guessed
    if not q.qtype:
        q.qtype = config.QTYPE_TEXT
    return q


def _cleaner_guess_subject(q: Question) -> str:
    from app.core.preprocess import cleaner
    return cleaner.guess_subject(q.raw_text or "", q.media_path or "")


def engine_report():
    """外部引擎自检报告（供界面【环境自检】展示）。"""
    from app.core.preprocess.image_ocr import get_engine
    try:
        status = get_engine().status()
    except Exception as exc:
        status = base.EngineStatus("RapidOCR 图片文字识别", False, str(exc), "check_failed",
                                   "检查本地图片识别组件后刷新状态。")
    report = base.engine_report()
    import sys
    from pathlib import Path
    if getattr(sys, "frozen", False) and (Path(sys.executable).parent / "media_worker.exe").is_file():
        from app.core.preprocess.media_recognition import model_root, _faster_whisper_path
        for index, row in enumerate(report):
            if row.name == "Whisper 听力转写":
                try:
                    directory = _faster_whisper_path(model_root() / "audio")
                    report[index] = base.EngineStatus(row.name, False,
                        "发布包使用独立转写程序；已找到本地模型：{}。尚需实际音频验证。".format(directory),
                        "unchecked", "在导入页执行听力转写并校对，不需要在电脑安装 Python。")
                except (ValueError, OSError) as exc:
                    report[index] = base.EngineStatus(row.name, False, str(exc), "model_missing",
                        "请按发布说明准备本地模型或用 MC_MODEL_DIR 指定已有模型；不会自动下载。")
    return report + [status]


def warmup() -> None:
    """预热分词和公式映射词典，减少首次分类延迟。"""
    from app.core.preprocess import formula, lexicon
    lexicon.ensure_jieba()
    formula.get_mapper()
