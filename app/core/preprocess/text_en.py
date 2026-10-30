# -*- coding: utf-8 -*-
"""英语错题专属预处理单元（技术路线 4.2.1（2）英语类错题）。

流程：英文分词 -> 保守词形归一化 -> 英文专属停用词过滤
      -> 阅读/完形剔除冗余选项 -> 提取核心场景词与知识点特征词

NLTK 的 punkt / wordnet 语料需要联网下载，与"全流程离线"冲突，因此：
* 优先使用 NLTK（若本地语料已就绪，可由 tools/prepare_nltk.py 预置到 resources/）；
* 语料缺失时使用内置正则分词器、常见词形表和保守规则，功能不中断。
不做 Porter 式词干截断；未知的 -ing/-ed/-ly 词保留原词，避免生成不存在的词。
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Set

from app import config
from app.core.models import FeatureBundle, Question
from app.core.preprocess import cleaner
from app.core.preprocess.base import BasePreprocessor, EngineStatus, passive_engine_status, register

log = logging.getLogger(__name__)

_RE_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?(?:-[A-Za-z]+)*")

# 英文专属停用词（内置，保证离线可用）
EN_STOPWORDS: Set[str] = set("""
a an the this that these those it its it's he she they them his her their our your my mine
i we you me us am is are was were be been being do does did doing done have has had having
will would shall should can could may might must ought need dare
and or but so because if then than as of in on at by for with about into over under
from to up down out off again further once here there when where why how all any both each
few more most other some such only own same too very just now also however
what which who whom whose while during before after above below between through against
one two three four five ten
say says said get gets got make makes made go goes went take takes took come comes came
thing things way ways lot lots kind sort etc eg ie please
""".split())

# 题干套话（英语题）
EN_SUBJECT_STOP: Set[str] = set("""
question questions choose choice choices following passage text answer answers correct
best complete completion blank blanks fill according sentence sentences underlined word
words section part directions read listen listening conversation dialogue exercise
option options mark paper exam test paragraph line lines below above
""".split())

# 不规则动词/名词的最小还原表（离线降级路径用）
_IRREGULAR = {
    "was": "be", "were": "be", "been": "be", "am": "be", "is": "be", "are": "be",
    "went": "go", "gone": "go", "did": "do", "done": "do", "had": "have", "has": "have",
    "made": "make", "took": "take", "taken": "take", "given": "give", "gave": "give",
    "written": "write", "wrote": "write", "spoken": "speak", "spoke": "speak",
    "children": "child", "men": "man", "women": "woman", "feet": "foot", "teeth": "tooth",
    "mice": "mouse", "people": "person", "data": "datum", "analyses": "analysis",
    "bought": "buy", "brought": "bring", "thought": "think", "taught": "teach",
    "caught": "catch", "found": "find", "felt": "feel", "left": "leave",
    "kept": "keep", "slept": "sleep", "met": "meet", "paid": "pay",
    "ran": "run", "began": "begin", "begun": "begin", "drank": "drink",
    "drunk": "drink", "drove": "drive", "driven": "drive", "ate": "eat",
    "eaten": "eat", "fell": "fall", "fallen": "fall", "flew": "fly",
    "flown": "fly", "forgot": "forget", "forgotten": "forget", "grew": "grow",
    "grown": "grow", "knew": "know", "known": "know", "rode": "ride",
    "ridden": "ride", "rose": "rise", "risen": "rise", "saw": "see",
    "seen": "see", "sent": "send", "sold": "sell", "stood": "stand",
    "understood": "understand", "won": "win", "lost": "lose",
    "geese": "goose", "oxen": "ox", "criteria": "criterion",
    "phenomena": "phenomenon", "theses": "thesis", "crises": "crisis",
    "indices": "index", "matrices": "matrix", "bacteria": "bacterium",
    "leaves": "leaf", "knives": "knife", "lives": "life", "wives": "wife",
    "wolves": "wolf", "shelves": "shelf", "halves": "half",
    "buses": "bus", "gases": "gas", "movies": "movie", "cookies": "cookie",
    "ties": "tie", "pies": "pie", "lies": "lie", "dies": "die",
    "heroes": "hero", "tomatoes": "tomato", "potatoes": "potato",
    "echoes": "echo", "quizzes": "quiz", "calories": "calorie",
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
}

# Base forms that can safely generate regular inflections. This is a finite
# offline lexicon, not a claim to infer the lemma of every English word.
_REGULAR_VERBS = set("""
accept achieve act add admit agree allow answer appear apply argue arrive ask avoid
believe belong borrow call care carry cause change check clean close collect compare
complete consider contain continue cook copy count create dance decide deliver depend
describe design develop die discuss divide dress drive drop earn educate employ enjoy
enter escape examine exist expect explain express face fail fill finish fix follow
force form gain graduate guess happen hate help hope identify improve include increase
influence intend interest introduce invent invite join jump kick kill kiss learn like
listen live look love manage marry measure miss move name need notice offer open order
organize pass perform plan play point practice practise prefer prepare prevent produce
promise protect provide pull push rain raise reach realize receive reduce refer refuse
relate release remain remember remove repeat reply report require research return review
save seem serve share shop show sign smile solve sound start stay stop study succeed
suggest support surprise talk taste test thank tie travel try turn unite use value vary
visit wait walk want wash watch waste welcome wish wonder work worry write
""".split())
_DOUBLE_VERBS = set("admit drop plan prefer refer shop stop".split())
_REGULAR_FORMS = {}
for _base in _REGULAR_VERBS:
    if _base.endswith("e"):
        _past = _base + "d"
        _progressive = _base[:-1] + "ing" if not _base.endswith("ee") else _base + "ing"
        if _base.endswith("ie"):
            _progressive = _base[:-2] + "ying"
    elif _base.endswith("y") and _base[-2] not in "aeiou":
        _past, _progressive = _base[:-1] + "ied", _base + "ing"
    elif _base in _DOUBLE_VERBS:
        _past, _progressive = _base + _base[-1] + "ed", _base + _base[-1] + "ing"
    else:
        _past, _progressive = _base + "ed", _base + "ing"
    if _base not in _IRREGULAR.values():
        _REGULAR_FORMS[_past] = _base
    _REGULAR_FORMS[_progressive] = _base
_REGULAR_FORMS.update({"running": "run", "swimming": "swim", "sitting": "sit",
                       "getting": "get", "beginning": "begin", "lying": "lie",
                       "travelling": "travel", "travelled": "travel"})
_UNCHANGED = set("""
news series species means analysis basis crisis thesis diagnosis synthesis hypothesis
physics mathematics economics statistics politics ethics business success process access
address glass class grass mass loss stress cross status campus virus focus bonus census
always sometimes perhaps besides nevertheless nonetheless whereas clothes headquarters
""".split())
_NEGATION = {"no", "nor", "not", "never", "neither", "without"}
_CONTRACTIONS = {"can't": "can not", "cannot": "can not", "won't": "will not",
                 "shan't": "shall not", "ain't": "be not"}

_nltk_state: Optional[bool] = None
_lemmatizer = None
_nltk_error = ""


def _try_nltk() -> bool:
    """探测 NLTK 及其语料是否本地可用（只探测一次）。"""
    global _nltk_state, _lemmatizer, _nltk_error
    if _nltk_state is not None:
        return _nltk_state
    try:
        import nltk                                        # noqa: WPS433
        from nltk.stem import WordNetLemmatizer

        # 允许把语料随程序打包到 resources/nltk_data 实现完全离线
        local = config.RESOURCE_DIR / "nltk_data"
        if local.exists() and str(local) not in nltk.data.path:
            nltk.data.path.insert(0, str(local))
        nltk.data.find("corpora/wordnet.zip")
        _lemmatizer = WordNetLemmatizer()
        _lemmatizer.lemmatize("tests")      # 触发真实加载，失败则抛异常
        _nltk_state = True
    except Exception as exc:                                # noqa: BLE001
        _nltk_error = str(exc)
        log.info("NLTK 语料不可用，英文预处理降级为内置规则：%s", exc)
        _nltk_state = False
    return _nltk_state


def _rule_lemmatize(w: str) -> str:
    """有限词形归一化；不把删除任意后缀得到的 stem 当作 lemma。"""
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if w in _REGULAR_FORMS:
        return _REGULAR_FORMS[w]
    if w in _UNCHANGED or w in _REGULAR_VERBS:
        return w
    if len(w) > 4 and w.endswith("ies") and w[-4] not in "aeiou":
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("sses", "ches", "shes", "xes", "zzes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is", "ous", "ics")):
        return w[:-1]
    return w


def tokenize_en(text: str) -> List[str]:
    """英文分词 + 词形还原 + 双层英文停用词过滤。"""
    if not text:
        return []
    normalized = text.lower().translate(str.maketrans({"’": "'", "‘": "'", "ʼ": "'"}))
    normalized = re.sub(r"\b(?:can't|cannot|won't|shan't|ain't)\b",
                        lambda match: _CONTRACTIONS[match.group()], normalized)
    normalized = re.sub(r"\b([a-z]+)n't\b", r"\1 not", normalized)
    words = [w.removesuffix("'s").removesuffix("'") for w in _RE_WORD.findall(normalized)]
    out: List[str] = []
    use_nltk = _try_nltk()
    for w in words:
        if w in _NEGATION:
            out.append(w)
            continue
        if len(w) < 3 or w in EN_STOPWORDS or w in EN_SUBJECT_STOP:
            continue
        base = _rule_lemmatize(w)
        # Optional WordNet is used only for words our finite rules left intact.
        # Never run an unrestricted verb pass on nouns/adjectives (e.g. meeting).
        if base == w and w not in _UNCHANGED and use_nltk and _lemmatizer is not None:
            base = _lemmatizer.lemmatize(w, pos="n")
        if len(base) < 3 or base in EN_STOPWORDS or base in EN_SUBJECT_STOP:
            continue
        out.append(base)
    return out


def reading_task_tokens(text: str) -> List[str]:
    """Identify explicit task instructions or a passage followed by a question.

    Grammar, translation and writing exercises can contain long passages too;
    their explicit instructions take precedence. Passage content alone is not
    evidence that an exercise asks for reading comprehension.
    """
    if not text:
        return []
    normalized = text.casefold()
    instruction = normalized[:220]
    if re.search(r"\b(?:translate|translation|write\s+(?:an?\s+)?(?:essay|composition|letter)|"
                 r"identify\s+(?:the\s+)?(?:tense|clause)|correct\s+the\s+(?:grammar|sentence))\b|翻译|写作", instruction):
        return []
    if re.search(r"\bcloze(?:\s+test)?\b|完形填空", instruction):
        return ["完形填空"]
    if re.search(r"\breading comprehension\b|阅读理解|"
                 r"\bread\s+(?:the\s+)?(?:following\s+)?(?:passage|text)\b", instruction):
        return ["阅读理解"]
    question = re.search(r"(?:^|\n)\s*(?:questions?\s*\d*|问题|题目)\s*[:：]\s*", text, re.I)
    if question:
        passage = text[:question.start()]
        prompt = text[question.end():]
        if len(re.findall(r"\b[A-Za-z]+\b", passage)) >= 50 and re.search(r"[?？]", prompt):
            return ["阅读理解"]
    return []


@register
class EnglishTextUnit(BasePreprocessor):
    """英语听力 / 阅读 / 完形专属单元。"""

    name = "text_en"
    display_name = "英文文本预处理"
    qtypes = frozenset({config.QTYPE_READING, config.QTYPE_LISTENING, config.QTYPE_TEXT})

    def accept(self, q: Question) -> bool:
        if q.subject == config.SUBJECT_ENGLISH or q.qtype == config.QTYPE_LISTENING:
            return True
        from pathlib import Path
        if q.media_path and Path(q.media_path).suffix.lower() in config.AUDIO_EXTS:
            return True
        # Short prompts and English annotations need the same filtering as long
        # passages. Inline mathematical commands belong to the formula unit.
        text, _ = cleaner.extract_latex(" ".join((q.raw_text or "", q.annotation or "")))
        return any(len(word) >= 3 for word in _RE_WORD.findall(text))

    def process(self, q: Question, fb: FeatureBundle) -> None:
        text = q.raw_text or ""
        if q.subject == config.SUBJECT_ENGLISH and q.qtype != config.QTYPE_LISTENING:
            task = reading_task_tokens(text)
            if task:
                fb.qtype_tokens.extend(task)
                fb.notes.append("英文任务结构：" + "、".join(task))
        # 听力题正文优先用 Whisper 转写结果（由 audio 单元先行填充）
        if fb.transcript:
            text = (text + " " + fb.transcript).strip()

        text = " ".join(part for part in (text, q.annotation or "") if part)
        text, _ = cleaner.extract_latex(text)

        if q.qtype in (config.QTYPE_READING,):
            text = cleaner.strip_option_bodies(text)   # 剔除冗余选项

        clean = cleaner.clean_text(text, keep_options=False)
        if clean and clean not in fb.clean_text:
            fb.clean_text = (fb.clean_text + " " + clean).strip()
        tokens = tokenize_en(clean)
        if tokens:
            fb.text_tokens.extend(tokens)
        fb.pipeline.append(self.display_name +
                           ("（离线词形规则 + WordNet 名词）" if _try_nltk() else "（离线词形规则）"))

    def status(self) -> EngineStatus:
        return passive_engine_status(
            "NLTK 英文词形还原", (("nltk",),),
            checked=_nltk_state is not None, loaded=_nltk_state is True,
            error=_nltk_error, ready_detail="本次运行已成功加载 WordNet 词形还原。",
            fallback="未就绪时使用内置英文规则；其效果与 WordNet 可能不同。")
