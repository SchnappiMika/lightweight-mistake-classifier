# -*- coding: utf-8 -*-
"""词典管理：分学科可插拔专有名词词典 + "通用 + 学科专属"双层停用词库。

对应技术路线 4.2.1（2）(3)：
* 专有名词词典避免"洛必达法则""动量守恒定律"被 jieba 错误切分；
* 双层停用词库先滤掉助词/介词/语气词，再滤掉理工科题干套话（"如图所示""下列说法"）。

词典全部为 resources/ 下的纯文本，用户可直接增删，无需改代码。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set

from app import config

log = logging.getLogger(__name__)

# 学科 -> 专有名词词典文件名
SUBJECT_DICT_FILES = {
    config.SUBJECT_MATH: "math.txt",
    config.SUBJECT_PHYSICS: "physics.txt",
    config.SUBJECT_ENGLISH: "english.txt",
    config.SUBJECT_LINALG: "linalg.txt",
    config.SUBJECT_PROB: "prob.txt",
}

_lock = threading.Lock()
_jieba_ready = False
_stopwords_cache: Dict[str, Set[str]] = {}
_userdict_words: Set[str] = set()


# --------------------------------------------------------------------------- #
# 停用词
# --------------------------------------------------------------------------- #
def _load_wordfile(path: Path) -> Set[str]:
    words: Set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w and not w.startswith("#"):
                    words.add(w)
    except OSError as exc:
        log.warning("停用词文件读取失败 %s: %s", path, exc)
    return words


def common_stopwords() -> Set[str]:
    """第一层：通用停用词（助词、介词、连词、语气词、代词）。"""
    if "common" not in _stopwords_cache:
        _stopwords_cache["common"] = _load_wordfile(config.STOPWORD_DIR / "common.txt")
    return _stopwords_cache["common"]


def subject_stopwords(subject: Optional[str] = None) -> Set[str]:
    """第二层：学科专属停用词（题干套话）。

    ``subject_common.txt`` 为全理工科共用；若存在 ``subject_<学科>.txt``
    则叠加该学科私有停用词（用户可自行新增该文件扩展）。
    """
    key = "subject::{}".format(subject or "")
    if key not in _stopwords_cache:
        words = _load_wordfile(config.STOPWORD_DIR / "subject_common.txt")
        if subject:
            p = config.STOPWORD_DIR / "subject_{}.txt".format(subject)
            if p.exists():
                words |= _load_wordfile(p)
        _stopwords_cache[key] = words
    return _stopwords_cache[key]


def all_stopwords(subject: Optional[str] = None) -> Set[str]:
    """双层停用词合集。"""
    return common_stopwords() | subject_stopwords(subject)


def is_stopword(word: str, subject: Optional[str] = None) -> bool:
    return word in all_stopwords(subject)


# --------------------------------------------------------------------------- #
# jieba 专有名词词典
# --------------------------------------------------------------------------- #
def ensure_jieba() -> "object":
    """惰性初始化 jieba 并挂载全部分学科专有名词词典（进程内只做一次）。

    jieba 首次加载约 0.5-1s，放在这里做惰性初始化，避免拖慢界面启动；
    分类线程首次调用时完成，之后单条耗时可稳定控制在毫秒级。
    """
    global _jieba_ready
    import jieba                      # 局部导入：便于纯 UI 场景更快启动
    import jieba.posseg  # noqa: F401  确保词性标注子模块被加载

    if _jieba_ready:
        return jieba
    with _lock:
        if _jieba_ready:
            return jieba
        jieba.setLogLevel(logging.WARNING)
        for subject, fname in SUBJECT_DICT_FILES.items():
            p = config.USERDICT_DIR / fname
            if not p.exists():
                continue
            try:
                jieba.load_userdict(str(p))
                _userdict_words.update(_parse_dict_words(p))
            except Exception as exc:            # noqa: BLE001 词典损坏不应导致崩溃
                log.warning("加载专有名词词典失败 %s: %s", p, exc)
        _jieba_ready = True
        log.info("jieba 初始化完成，专有名词 %d 条", len(_userdict_words))
    return jieba


def _parse_dict_words(path: Path) -> Set[str]:
    out: Set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                out.add(line.split()[0])
    except OSError:
        pass
    return out


def userdict_words() -> Set[str]:
    """已挂载的专有名词集合（用于保护它们不被词性过滤误删）。"""
    ensure_jieba()
    return _userdict_words


def add_runtime_words(words: List[str]) -> None:
    """（已废弃，不要再用）把词动态注入 jieba 全局词典。

    保留这个函数只是为了留下一句警告：**不要这么做**。

    jieba 的词典是进程级全局的，注入不可逆，而且会跨学科互相污染。本项目
    曾在同步分类体系时把用户的关键词全量注入进来，结果一条叫"分词作状语"的
    关键词把"现在分词和过去分词作状语"从 ``现在分词/过去分词/状语`` 变成了
    ``现在分词/分词作状语`` —— "过去分词"被跨边界的长词整个吃掉。更糟的是
    分类结果会依赖"你先打开过哪套体系"，完全不可复现。

    正确做法：让粗筛的子串匹配与子词扩展去兜底未登录关键词（见 coarse.py），
    精匹配那一侧本来就不需要——标签画像与题目共用同一个分词器。
    """
    raise NotImplementedError(
        "add_runtime_words 已废弃：注入 jieba 全局词典会造成跨学科污染且不可复现，"
        "未登录关键词由粗筛的子串匹配与子词扩展兜底。详见本函数 docstring。")


def reset_cache() -> None:
    """清空停用词缓存（用户编辑词典文件后调用）。"""
    _stopwords_cache.clear()
