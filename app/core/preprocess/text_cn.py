# -*- coding: utf-8 -*-
"""中文文本预处理单元（技术路线 4.2.1（1））。

四步标准化流程：
    1. 文本清洗          -> cleaner.clean_text
    2. jieba 精准分词     -> 挂载分学科专有名词词典
    3. 双层停用词过滤     -> 通用停用词库 + 学科专属停用词库
    4. 核心分类特征提取   -> 保留名词 / 动词 / 形容词等实词

同时产出题型特征词与学科特征词，供分级分类层的多通道融合使用。
"""
from __future__ import annotations

from typing import List, Set

from app import config
from app.core.models import FeatureBundle, Question
from app.core.preprocess import cleaner, lexicon
from app.core.preprocess.base import BasePreprocessor, register

# 保留的词性前缀：n* 名词、v* 动词、a* 形容词
_KEEP_PREFIX = ("n", "v", "a")
# 额外保留的精确词性：eng 英文、i 成语、l 习用语、j 简称、s 处所、b 区别词
_KEEP_EXACT = {"eng", "i", "l", "j", "s", "b"}
# 明确丢弃的词性：数词、量词、时间词、方位词、标点等
_DROP_EXACT = {"m", "mq", "q", "t", "f", "x", "w", "u", "p", "c", "d", "r", "y", "e", "o", "z"}

# 题型特征词：让"题型"成为一个可参与相似度计算的通道
QTYPE_TOKENS = {
    config.QTYPE_TEXT: ["纯文本题", "文字题"],
    config.QTYPE_FORMULA: ["公式题", "计算题", "推导题", "算式"],
    config.QTYPE_FIGURE: ["图形题", "读图题", "受力分析图", "示意图"],
    config.QTYPE_LISTENING: ["听力题", "音频题", "听力理解"],
    config.QTYPE_READING: ["阅读题", "篇章题"],
}

# 学科特征词
SUBJECT_TOKENS = {
    config.SUBJECT_MATH: ["高等数学", "高数", "微积分"],
    config.SUBJECT_PHYSICS: ["大学物理", "普通物理"],
    config.SUBJECT_ENGLISH: ["大学英语", "英语"],
    config.SUBJECT_LINALG: ["线性代数", "线代"],
    config.SUBJECT_PROB: ["概率论与数理统计", "概率论", "数理统计"],
    config.SUBJECT_GENERAL: ["通用"],
}


def tokenize_cn(text: str, subject: str = "", *, include_ascii: bool = True) -> List[str]:
    """中文分词 + 双层停用词过滤 + 核心实词提取。"""
    if not text:
        return []
    jieba = lexicon.ensure_jieba()
    import jieba.posseg as pseg  # noqa: WPS433 局部导入，随 jieba 一起惰性加载

    stop: Set[str] = lexicon.all_stopwords(subject)
    protected: Set[str] = lexicon.userdict_words()
    out: List[str] = []

    for word, flag in pseg.cut(text, HMM=True):
        w = word.strip()
        if not w:
            continue
        # The fused pipeline routes Latin words through the English tokenizer
        # exactly once, with its own stopwords and morphology. Keep the public
        # standalone tokenizer's historical behavior for other callers.
        if not include_ascii and w.isascii():
            continue
        # 专有名词词典里的词无条件保留（避免被词性规则误杀）
        if w in protected:
            if w not in stop:
                out.append(w)
            continue
        if w in stop:                       # 第 3 步：双层停用词过滤
            continue
        if len(w) < 2 and not w.isascii():  # 单字中文判别力低，直接丢弃
            continue
        if flag in _DROP_EXACT:
            continue
        # 第 4 步：核心实词提取（名词 / 动词 / 形容词 + 英文等）
        if flag.startswith(_KEEP_PREFIX) or flag in _KEEP_EXACT:
            if w.isascii():
                w = w.lower()
                if len(w) < 2 or w in stop:
                    continue
            out.append(w)
    _ = jieba  # 保持引用，说明依赖来源
    return out


@register
class ChineseTextUnit(BasePreprocessor):
    """通用中文文本单元：对所有题型都执行，是全流程的基础兜底。"""

    name = "text_cn"
    display_name = "中文文本预处理"
    qtypes = frozenset()   # 空 = 全题型均执行

    def process(self, q: Question, fb: FeatureBundle) -> None:
        # 英语题的正文交给英文单元，这里只处理中文批注
        is_en = q.subject == config.SUBJECT_ENGLISH or q.qtype in (
            config.QTYPE_LISTENING, config.QTYPE_READING)

        raw = q.raw_text or ""
        # 抽走内联 LaTeX，避免公式符号污染中文分词（公式交给 formula 单元）
        text_wo_formula, _formulas = cleaner.extract_latex(raw)

        parts: List[str] = []
        if not is_en:
            parts.append(text_wo_formula)
        if q.annotation:
            parts.append(q.annotation)      # 用户标注始终参与中文特征

        merged = " ".join(p for p in parts if p)
        keep_options = q.qtype not in (config.QTYPE_READING,)
        clean = cleaner.clean_text(merged, keep_options=keep_options)
        if not keep_options:
            clean = cleaner.strip_option_bodies(clean)

        if clean:
            fb.clean_text = (fb.clean_text + " " + clean).strip()
            fb.text_tokens.extend(tokenize_cn(clean, q.subject, include_ascii=False))

        # 题型 / 学科特征通道
        fb.qtype_tokens.extend(QTYPE_TOKENS.get(q.qtype, []))
        fb.subject_tokens.extend(SUBJECT_TOKENS.get(q.subject, [q.subject]))
        fb.pipeline.append(self.display_name)
