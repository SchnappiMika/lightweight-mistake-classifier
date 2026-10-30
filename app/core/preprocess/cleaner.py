# -*- coding: utf-8 -*-
"""通用文本清洗（技术路线 4.2.1（1）第 1 步）。

职责：
1. 去除题号、换行、格式标记、冗余标点与特殊符号，统一文本格式；
2. 剥离并保留内联 LaTeX 片段（``$...$`` / ``\\(...\\)`` / ``\\[...\\]``），
   交给公式预处理单元，避免公式符号污染中文分词；
3. 依据文本与文件后缀自动推断学科与题型，减少用户手工选择成本。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from app import config

# 题号：1. / 1、/ （1） / (1) / 第 3 题 / [12] / 一、
_RE_QNUM = re.compile(
    r"^\s*(?:第\s*[0-9一二三四五六七八九十百]+\s*[题小]?\s*[.、．:：]?"
    r"|[（(]\s*\d{1,3}\s*[)）]"
    r"|\[\s*\d{1,3}\s*\]"
    r"|[0-9]{1,3}\s*[.、．)）]"
    r"|[一二三四五六七八九十]+\s*[、.．])\s*"
)
# 常见格式标记
_RE_MARKUP = re.compile(r"</?[a-zA-Z][^>]{0,40}>|\*{1,3}|_{2,}|~{2,}|`{1,3}|\|{2,}")
# 连续空白
_RE_WS = re.compile(r"[ \t\u3000\xa0]+")
_RE_NEWLINE = re.compile(r"[\r\n]+")
# 需要剔除的特殊符号（保留中英文、数字、基本标点、公式常用符号）
_RE_JUNK = re.compile(
    r"[^\u4e00-\u9fffA-Za-z0-9"
    r"，。？！；：、,.?!;:'\"“”‘’()（）\[\]【】<>《》%°±×÷=+\-*/^_{}$\\\s]"
)
# 内联公式
_RE_LATEX = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.S)
# 裸 LaTeX 命令（未用 $ 包裹时的兜底信号）
_RE_TEX_CMD = re.compile(r"\\(?:lim|int|iint|iiint|oint|frac|sum|prod|sqrt|partial|nabla|"
                         r"alpha|beta|gamma|lambda|omega|mu|varepsilon|epsilon|theta|pi|phi|"
                         r"infty|cdot|times|vec|hat|bar|begin|left|right|mathrm|operatorname|"
                         r"sin|cos|tan|ln|log|leq|geq|neq|approx|sim|Delta|Phi|hbar)\b")
# 选项行：A. xxx  B) xxx
_RE_OPTION = re.compile(r"(?:^|\s)([A-DＡ-Ｄ])\s*[.、．)）]\s*")


def extract_latex(text: str) -> Tuple[str, List[str]]:
    """把内联 LaTeX 片段抽出来，返回（去公式后的文本, 公式列表）。"""
    formulas: List[str] = []

    def _sub(m: "re.Match") -> str:
        frag = next((g for g in m.groups() if g), "")
        frag = frag.strip()
        if frag:
            formulas.append(frag)
        return " "

    stripped = _RE_LATEX.sub(_sub, text or "")
    return stripped, formulas


def looks_like_latex(text: str) -> bool:
    """判断文本是否含有 LaTeX 公式特征（用于自动推断题型）。"""
    if not text:
        return False
    if "$" in text and text.count("$") >= 2:
        return True
    return bool(_RE_TEX_CMD.search(text))


def clean_text(text: str, keep_options: bool = True) -> str:
    """标准化清洗：去题号 / 格式标记 / 特殊符号 / 冗余空白。

    Args:
        keep_options: 为 False 时剔除选项标号（英语完形、阅读的干扰内容）。
    """
    if not text:
        return ""
    lines = _RE_NEWLINE.split(text)
    out: List[str] = []
    for ln in lines:
        ln = _RE_QNUM.sub("", ln)
        ln = _RE_MARKUP.sub(" ", ln)
        if not keep_options:
            ln = _RE_OPTION.sub(" ", ln)
        ln = _RE_JUNK.sub(" ", ln)
        ln = _RE_WS.sub(" ", ln).strip()
        if ln:
            out.append(ln)
    return " ".join(out).strip()


def strip_option_bodies(text: str) -> str:
    """剔除选择题的选项正文，只保留题干。

    英语完形填空 / 阅读的四个选项往往互相干扰（近义词堆叠），
    对分类特征提取是噪声，需求 4.2.1 明确要求"剔除冗余选项"。
    """
    if not text:
        return ""
    # A single "A." may be a person's initial or a passage heading. Only an
    # ordered group starting A, B is evidence of options. Keep physical lines:
    # flattening first would destroy the boundary after the final option.
    markers = list(_RE_OPTION.finditer(text))
    spans = []
    question_start = re.compile(
        r"(?:^|\s)(?:\d{1,3}\s*[.、．)）]|第\s*\d+\s*题)(?!\d)\s*(?=\S)|"
        r"\n\s*(?:passage\s+[a-z0-9]+|questions?\s+\d+)", re.I)

    def marker_letter(match):
        return chr(ord(match.group(1)) - 0xFEE0) if ord(match.group(1)) > 127 else match.group(1)

    index = 0
    while index + 1 < len(markers):
        first, second = markers[index:index + 2]
        if marker_letter(first) != "A" or marker_letter(second) != "B":
            index += 1
            continue
        prefix = text[text.rfind("\n", 0, first.start()) + 1:first.start()]
        if re.search(r"\b(?:passage|section|part|appendix|article)\s*$", prefix, re.I):
            index += 1
            continue
        between = text[first.end():second.start()]
        # Do not join an A in one question to a B much later in the document.
        if question_start.search(between) or len(between.splitlines()) > 2:
            index += 1
            continue
        group = [first, second]
        cursor = index + 2
        while cursor < len(markers) and marker_letter(markers[cursor]) == chr(ord("A") + len(group)):
            between = text[group[-1].end():markers[cursor].start()]
            if question_start.search(between) or len(between.splitlines()) > 2:
                break
            group.append(markers[cursor])
            cursor += 1
        # Remove only lines containing confirmed choices. An unmarked next
        # paragraph is retained, even if it could be an option continuation.
        for choice_index, choice in enumerate(group):
            start = choice.start()
            # The matcher may consume the previous newline as whitespace.
            while start < choice.end() and text[start].isspace():
                start += 1
            line_end = text.find("\n", choice.end())
            end = len(text) if line_end < 0 else line_end
            if choice_index + 1 < len(group):
                end = min(end, group[choice_index + 1].start())
            boundary = question_start.search(text, choice.end(), end)
            if boundary:
                end = boundary.start()
            spans.append((start, end))
        index = cursor
    if not spans:
        return text
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + " " + text[end:]
    return text.strip()


def guess_qtype(text: str, media_path: str = "", subject: str = "") -> str:
    """依据内容与媒体后缀推断题型。"""
    ext = Path(media_path).suffix.lower() if media_path else ""
    if ext in config.AUDIO_EXTS:
        return config.QTYPE_LISTENING
    if ext in config.IMAGE_EXTS:
        # 图片既可能是公式截图也可能是受力分析图；用文件名与学科做弱先验
        name = Path(media_path).stem.lower()
        if any(k in name for k in ("formula", "eq", "公式", "算式")):
            return config.QTYPE_FORMULA
        if any(k in name for k in ("fig", "图", "graph", "circuit", "电路", "受力")):
            return config.QTYPE_FIGURE
        return config.QTYPE_FORMULA if subject in (
            config.SUBJECT_MATH, config.SUBJECT_LINALG, config.SUBJECT_PROB) else config.QTYPE_FIGURE
    if looks_like_latex(text):
        return config.QTYPE_FORMULA
    if subject == config.SUBJECT_ENGLISH:
        return config.QTYPE_READING
    return config.QTYPE_TEXT


# 学科弱先验关键词（仅在用户未指定学科时用于自动推断）
_SUBJECT_HINTS = {
    config.SUBJECT_MATH: ("极限", "导数", "积分", "级数", "微分方程", "洛必达", "泰勒",
                          "偏导", "重积分", "收敛"),
    config.SUBJECT_PHYSICS: ("动量", "电场", "磁场", "转动惯量", "振动", "波长", "热力学",
                             "光栅", "相对论", "光电效应", "刚体", "电动势"),
    config.SUBJECT_LINALG: ("行列式", "矩阵", "特征值", "向量组", "二次型", "线性无关", "秩"),
    config.SUBJECT_PROB: ("概率", "分布", "期望", "方差", "置信区间", "假设检验",
                          "似然", "随机变量"),
    config.SUBJECT_ENGLISH: ("从句", "虚拟语气", "时态", "完形", "听力", "词汇辨析", "非谓语"),
}


def guess_subject(text: str, media_path: str = "") -> str:
    """Use explicit science terms before the English-prose fallback.

    This remains a weak import hint; a user's selected subject takes priority
    in the importer. Ambiguous cross-subject evidence stays unspecified.
    """
    t = text or ""
    best, best_hit = config.SUBJECT_GENERAL, 0
    for subj, hints in _SUBJECT_HINTS.items():
        hit = sum(1 for h in hints if h in t)
        if hit > best_hit:
            best, best_hit = subj, hit
    if best_hit > 0:
        return best
    instruction = t[:220].casefold()
    if re.search(r"\b(?:translate|translation|reading comprehension|"
                 r"read the (?:following )?passage|correct the grammar|"
                 r"identify the tense|write (?:an? )?(?:essay|composition|letter))\b", instruction):
        return config.SUBJECT_ENGLISH
    from app.core.preprocess.scientific_terms import canonical_terms
    action = bool(re.search(r"\b(?:calculate|compute|evaluate|determine|find|differentiate|integrate)\b", instruction))
    unmistakable = {
        config.SUBJECT_MATH: {"导数", "定积分", "不定积分", "偏导数", "微分方程", "洛必达法则"},
        config.SUBJECT_PHYSICS: {"电场", "电场强度", "磁场", "磁感应强度", "动量", "角动量", "电动势", "动能", "加速度"},
        config.SUBJECT_LINALG: {"矩阵", "行列式", "特征值", "特征向量", "线性无关"},
        config.SUBJECT_PROB: {"条件概率", "随机变量", "概率密度", "二项分布", "正态分布", "置信区间"},
    }
    # Common English meanings of these words provide no scientific context.
    ambiguous = {"功", "功率", "力", "电流", "极限", "势能", "正态分布"}
    evidence = {}
    for subject, decisive in unmistakable.items():
        terms = set(canonical_terms(t, subject))
        distinct = terms - ambiguous
        if len(distinct) >= 2 or (action and distinct & decisive):
            evidence[subject] = len(distinct)
    if evidence:
        highest = max(evidence.values())
        leaders = [subject for subject, count in evidence.items() if count == highest]
        return leaders[0] if len(leaders) == 1 else config.SUBJECT_GENERAL
    letters = sum(1 for c in t if c.isascii() and c.isalpha())
    cjk = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    if letters > 20 and letters > cjk * 2:
        return config.SUBJECT_ENGLISH
    return config.SUBJECT_GENERAL
