"""Translate explicit English science terms into canonical Chinese features.

This lexicon supplies vocabulary, never class ids or inferred answers. Callers
retain the source text and English tokens alongside these additional terms.
"""
from __future__ import annotations

from functools import lru_cache
import json
import re
import unicodedata

from app import config


_WORD_EDGE = r"[A-Za-z0-9_]"
_SENTENCE_EDGE = re.compile(r"[.!?;。！？；\n]")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().replace("’", "'").replace("‘", "'")


def _phrase_pattern(phrase: str):
    # Space and hyphen are interchangeable only *inside* a listed phrase.
    # Do not join through commas, sentence ends, or arbitrary intervening words.
    words = re.split(r"[\s\-‐‑–—]+", _normalize(phrase).strip())
    body = r"[\s\-‐‑–—]+".join(re.escape(word) for word in words)
    return re.compile(r"(?<!" + _WORD_EDGE + r")" + body + r"(?!" + _WORD_EDGE + r")")


@lru_cache(maxsize=1)
def _lexicon():
    path = config.RESOURCE_DIR / "scientific_terms.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("不支持的科学术语词典版本")
    subjects = {}
    for subject, entries in data["subjects"].items():
        compiled = []
        for entry in entries:
            canonical = entry["canonical"]
            if not isinstance(canonical, str) or not canonical.strip():
                raise ValueError("科学术语规范词为空")
            contexts = tuple(_phrase_pattern(word) for word in entry.get("context_any", []))
            exclusions = tuple(_phrase_pattern(word) for word in entry.get("exclude_phrases", []))
            for alias in entry["aliases"]:
                if not isinstance(alias, str) or not alias.strip():
                    raise ValueError("科学术语英文别名为空")
                compiled.append((_phrase_pattern(alias), canonical, contexts, exclusions))
        subjects[subject] = tuple(compiled)
    return subjects


def _local_sentence(text: str, start: int, end: int) -> str:
    # A nearby anchor in this sentence is required for ambiguous words such as
    # work/power. A physical noun in a different sentence must not qualify them.
    left = text[max(0, start - 120):start]
    right = text[end:min(len(text), end + 120)]
    left = _SENTENCE_EDGE.split(left)[-1]
    right = _SENTENCE_EDGE.split(right)[0]
    return left + text[start:end] + right


def canonical_terms(text: str, subject: str) -> list[str]:
    """Return unique explicit terms in source order, using longest matches first.

    Only a known science subject enables its own dictionary. Inflection aliases
    are listed explicitly, so ordinary words, substrings and unlisted senses
    cannot silently become scientific evidence. Chinese input is not rewritten.
    """
    if not isinstance(text, str) or not text or not isinstance(subject, str):
        return []
    entries = _lexicon().get(subject, ())
    if not entries:
        return []
    normalized = _normalize(text)
    matches = []
    for pattern, canonical, contexts, exclusions in entries:
        for match in pattern.finditer(normalized):
            start, end = match.span()
            if contexts or exclusions:
                sentence = _local_sentence(normalized, start, end)
                if contexts and not any(anchor.search(sentence) for anchor in contexts):
                    continue
                if any(exclusion.search(sentence) for exclusion in exclusions):
                    continue
            matches.append((start, end, canonical))
    # Select globally longest spans first; partial/indefinite integrals and
    # angular momentum must not also inject the less specific nested term.
    accepted = []
    for candidate in sorted(matches, key=lambda item: (-(item[1] - item[0]), item[0], item[2])):
        start, end, _ = candidate
        if any(start < old_end and old_start < end for old_start, old_end, _ in accepted):
            continue
        accepted.append(candidate)
    terms = []
    seen = set()
    for _, _, canonical in sorted(accepted):
        if canonical not in seen:
            seen.add(canonical)
            terms.append(canonical)
    return terms
