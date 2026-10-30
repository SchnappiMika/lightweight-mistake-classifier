# -*- coding: utf-8 -*-
"""多模态数据输入层（技术路线 4.1 第一层）。

支持三种导入模式，覆盖"大学生错题来源分散、格式不统一"的真实场景：
    1. 单条手动录入
    2. 批量文件导入：TXT（空行分隔 / 每行一条自动判别）、Excel（列名智能匹配）
    3. 媒体本地导入：公式图片 / 工科图形题图片 / 英语听力音频（可批量选目录）

所有导入都会经过 ``normalize`` 自动补全学科与题型，用户无需提前手工分类，
这正是"无需用户提前手动转换格式、拆分题干"的入口层设计。
"""
from __future__ import annotations

import logging
import io
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from app import config
from app.core.models import Question
from app.core.preprocess import cleaner

log = logging.getLogger(__name__)


class ImportError_(Exception):
    """导入失败（文件缺失 / 格式非法 / 编码不可识别）。"""


# 列名智能匹配：用户的 Excel 表头五花八门，这里做同义词映射
COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "raw_text": ("题干", "题目", "错题", "内容", "题目内容", "题干内容", "text",
                 "question", "content", "题面"),
    "subject": ("学科", "科目", "课程", "subject", "course"),
    "qtype": ("题型", "类型", "题目类型", "qtype", "type"),
    "annotation": ("标注", "备注", "错因", "笔记", "错误原因", "note", "annotation",
                   "remark", "分析"),
    "media_path": ("媒体路径", "图片", "音频", "附件", "文件", "路径", "media",
                   "image", "audio", "path"),
    "priority": ("复习优先级", "优先级", "priority"),
}

# 题型列的中文/英文双向识别
QTYPE_ALIASES: Dict[str, str] = {
    "text": config.QTYPE_TEXT, "纯文本": config.QTYPE_TEXT, "文本": config.QTYPE_TEXT,
    "纯文本题干": config.QTYPE_TEXT, "文字题": config.QTYPE_TEXT,
    "formula": config.QTYPE_FORMULA, "公式": config.QTYPE_FORMULA,
    "公式类": config.QTYPE_FORMULA, "公式题": config.QTYPE_FORMULA,
    "计算题": config.QTYPE_FORMULA,
    "figure": config.QTYPE_FIGURE, "图形": config.QTYPE_FIGURE,
    "图形类": config.QTYPE_FIGURE, "图形题": config.QTYPE_FIGURE, "读图题": config.QTYPE_FIGURE,
    "listening": config.QTYPE_LISTENING, "听力": config.QTYPE_LISTENING,
    "英语听力": config.QTYPE_LISTENING, "听力题": config.QTYPE_LISTENING,
    "reading": config.QTYPE_READING, "阅读": config.QTYPE_READING,
    "阅读题": config.QTYPE_READING, "完形填空": config.QTYPE_READING,
    "英语阅读/完形": config.QTYPE_READING,
}

_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb18030", "big5", "utf-16")


def _read_text_file(path: Path) -> str:
    """按常见中文编码依次尝试读取（Windows 记事本默认 GBK，是高频坑）。"""
    last: Optional[Exception] = None
    for enc in _ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError) as exc:
            last = exc
        except OSError as exc:
            raise ImportError_("文件读取失败：{}".format(exc)) from exc
    raise ImportError_("无法识别文件编码（已尝试 {}）：{}".format("/".join(_ENCODINGS), last))


def normalize(q: Question) -> Question:
    """补全学科与题型；已由用户显式指定的不覆盖。"""
    if not q.subject or q.subject == config.SUBJECT_GENERAL:
        q.subject = cleaner.guess_subject(q.raw_text or "", q.media_path or "")
    if not q.qtype:
        q.qtype = cleaner.guess_qtype(q.raw_text or "", q.media_path or "", q.subject)
    return q


# --------------------------------------------------------------------------- #
# 单条录入
# --------------------------------------------------------------------------- #
def import_single(text: str, subject: str = "", qtype: str = "",
                  annotation: str = "", media_path: str = "") -> Question:
    if media_path:
        from app.core.storage import Repository
        media_path = Repository.import_media(media_path)
    q = Question(raw_text=(text or "").strip(), subject=subject or config.SUBJECT_GENERAL,
                 qtype=qtype, annotation=(annotation or "").strip(),
                 media_path=media_path, source="manual")
    return normalize(q)


# --------------------------------------------------------------------------- #
# TXT 批量导入
# --------------------------------------------------------------------------- #
def import_txt(path: str, subject: str = "", qtype: str = "") -> List[Question]:
    """导入 TXT。

    自动判别两种常见格式：
        * 段落式：错题之间用空行分隔（推荐，可容纳多行题干与选项）；
        * 行式：一行一条错题。
    以 ``#`` 开头的行视为注释。
    """
    p = Path(path)
    if not p.exists():
        raise ImportError_("文件不存在：{}".format(path))
    raw = _read_text_file(p)
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("#")]
    body = "\n".join(lines)

    blocks: List[str]
    if "\n\n" in body.replace("\r", ""):
        blocks = [b.strip() for b in body.replace("\r", "").split("\n\n")]
    else:
        blocks = [b.strip() for b in body.splitlines()]
    blocks = [b for b in blocks if b]
    if not blocks:
        raise ImportError_("文件中没有解析到任何错题内容")

    out: List[Question] = []
    for b in blocks:
        q = Question(raw_text=b, subject=subject or config.SUBJECT_GENERAL,
                     qtype=qtype, source="txt")
        out.append(normalize(q))
    log.info("TXT 导入 %d 条：%s", len(out), path)
    return out


# --------------------------------------------------------------------------- #
# Excel 批量导入
# --------------------------------------------------------------------------- #
def _match_columns(columns: Sequence[str]) -> Dict[str, str]:
    """把用户表头映射到内部字段名。"""
    mapping: Dict[str, str] = {}
    lowered = {str(c): str(c).strip().lower() for c in columns}
    for field, aliases in COLUMN_ALIASES.items():
        for col, low in lowered.items():
            if col in mapping.values():
                continue
            if low in [a.lower() for a in aliases]:
                mapping[field] = col
                break
    # 没匹配到题干列时，退化为使用第一列
    if "raw_text" not in mapping and columns:
        mapping["raw_text"] = str(columns[0])
    return mapping


def import_excel(path: str, subject: str = "", qtype: str = "",
                 sheet: Optional[str] = None, errors: Optional[List[str]] = None) -> List[Question]:
    """导入 Excel / CSV，自动匹配列名。"""
    p = Path(path)
    if not p.exists():
        raise ImportError_("文件不存在：{}".format(path))
    if p.suffix.lower() == ".xls":
        raise ImportError_("暂不支持旧版 .xls，请在 Excel 中另存为 .xlsx 后导入")
    try:
        import pandas as pd
    except ImportError as exc:                     # pragma: no cover
        raise ImportError_("缺少 pandas 依赖，无法导入 Excel") from exc

    try:
        if p.suffix.lower() in (".csv", ".tsv"):
            sep = "\t" if p.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(io.StringIO(_read_text_file(p)), sep=sep, dtype=str, keep_default_na=False)
        else:
            df = pd.read_excel(p, sheet_name=sheet or 0, dtype=str)
            df = df.fillna("")
    except Exception as exc:                       # noqa: BLE001
        raise ImportError_("Excel 解析失败：{}".format(exc)) from exc

    if df.empty:
        raise ImportError_("表格为空")
    mapping = _match_columns(list(df.columns))
    if "raw_text" not in mapping:
        raise ImportError_("未找到题干列，请把题干列命名为「题干」或「题目」")

    out: List[Question] = []
    for row_number, (_, row) in enumerate(df.iterrows(), 2):
        text = str(row.get(mapping["raw_text"], "")).strip()
        media = str(row.get(mapping.get("media_path", ""), "")).strip() \
            if "media_path" in mapping else ""
        if not text and not media:
            continue
        if media:
            from app.core.storage import Repository
            attachment = Path(media)
            if not attachment.is_absolute():
                attachment = p.resolve().parent / attachment
            try:
                media = Repository.import_media(str(attachment))
            except OSError as exc:
                message = "第 {} 行附件导入失败：{}".format(row_number, exc)
                if errors is None:
                    raise ImportError_(message) from exc
                errors.append(message)
                continue
        qt = str(row.get(mapping.get("qtype", ""), "")).strip() if "qtype" in mapping else ""
        q = Question(
            raw_text=text,
            subject=(str(row.get(mapping.get("subject", ""), "")).strip()
                     if "subject" in mapping else "") or subject or config.SUBJECT_GENERAL,
            qtype=QTYPE_ALIASES.get(qt.lower(), QTYPE_ALIASES.get(qt, qtype)),
            annotation=(str(row.get(mapping.get("annotation", ""), "")).strip()
                        if "annotation" in mapping else ""),
            media_path=media,
            priority=(str(row.get(mapping.get("priority", ""), "")).strip()
                      if "priority" in mapping else ""),
            source="excel")
        out.append(normalize(q))
    if not out and not errors:
        raise ImportError_("表格中没有解析到任何有效错题")
    log.info("Excel 导入 %d 条（列映射 %s）：%s", len(out), mapping, path)
    return out


# --------------------------------------------------------------------------- #
# 媒体导入
# --------------------------------------------------------------------------- #
def import_media(paths: Sequence[str], subject: str = "", qtype: str = "",
                 copy_to_workspace: bool = True) -> List[Question]:
    """导入图片 / 音频。文件默认复制进工作区，避免原文件被移动后预览失效。"""
    from app.core.storage import Repository

    out: List[Question] = []
    for p in paths:
        pp = Path(p)
        ext = pp.suffix.lower()
        if ext not in config.IMAGE_EXTS and ext not in config.AUDIO_EXTS:
            raise ImportError_("不支持的附件类型：{}".format(p))
        if not pp.is_file():
            raise ImportError_("附件不存在：{}".format(p))
        dst = Repository.import_media(str(pp)) if copy_to_workspace else str(pp)
        q = Question(raw_text="", media_path=dst,
                     subject=subject or config.SUBJECT_GENERAL,
                     qtype=qtype, source="image" if ext in config.IMAGE_EXTS else "audio")
        out.append(normalize(q))
    log.info("媒体导入 %d 条", len(out))
    return out


def scan_dir(folder: str) -> Tuple[List[str], List[str]]:
    """扫描目录，返回（图片列表, 音频列表）。"""
    root = Path(folder)
    imgs: List[str] = []
    auds: List[str] = []
    if not root.exists():
        return imgs, auds
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in config.IMAGE_EXTS:
            imgs.append(str(p))
        elif ext in config.AUDIO_EXTS:
            auds.append(str(p))
    return imgs, auds
