# -*- coding: utf-8 -*-
import os, json
from .keyword_matcher import KeywordMatcher
from .tfidf_matcher import TfidfMatcher


class Classifier:
    def __init__(self, dict_dir=None):
        if dict_dir is None:
            import os
            dict_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "knowledge_base", "data", "dictionaries"
            )
        self.dict_dir = dict_dir
        self.keyword_matcher = KeywordMatcher(dict_dir)
        self.tfidf_matcher = TfidfMatcher()
        self._build_index()

    def _build_index(self):
        cats, subj_map = {}, {}
        for fn in os.listdir(self.dict_dir):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(self.dict_dir, fn), encoding="utf-8") as f:
                data = json.load(f)
            for c in data["categories"]:
                cats[c["name"]] = c
                subj_map[c["name"]] = data["subject"]
        self.tfidf_matcher.build_index(cats, subj_map)

    def classify(self, words, subject=None, top_k=3):
        kw_candidates = self.keyword_matcher.match(words, top_k=5)
        detected_subject = subject
        if not detected_subject and kw_candidates:
            top_cat = kw_candidates[0][0]
            for fn in os.listdir(self.dict_dir):
                if not fn.endswith(".json"):
                    continue
                with open(os.path.join(self.dict_dir, fn), encoding="utf-8") as f:
                    data = json.load(f)
                for c in data["categories"]:
                    if c["name"] == top_cat:
                        detected_subject = data["subject"]
                        break
        tfidf_results = self.tfidf_matcher.match_by_subject(words, detected_subject, top_k=top_k)
        if not tfidf_results:
            tfidf_results = self.tfidf_matcher.match(words, top_k=top_k)
        final = tfidf_results[0] if tfidf_results else ("N/A", 0.0)
        return {
            "keyword_candidates": kw_candidates,
            "final_result": final,
            "all_scores": tfidf_results,
            "detected_subject": detected_subject or "unknown"
        }

    def classify_text(self, text, subject=None, top_k=3):
        from ..preprocessing.text_cleaner import TextCleaner
        from ..preprocessing.tokenizer import Tokenizer
        from ..preprocessing.stopwords import StopwordsFilter
        cleaner = TextCleaner()
        tok = Tokenizer()
        sw = StopwordsFilter()
        cleaned = cleaner.clean(text)
        words = tok.segment(cleaned)
        core_words = tok.extract_core_words(words)
        filtered = sw.filter(core_words, subject)
        return self.classify(filtered, subject, top_k)

    def get_subjects(self):
        subjects = set()
        for fn in os.listdir(self.dict_dir):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(self.dict_dir, fn), encoding="utf-8") as f:
                subjects.add(json.load(f)["subject"])
        return sorted(subjects)

    def get_categories(self, subject=None):
        result = []
        for fn in os.listdir(self.dict_dir):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(self.dict_dir, fn), encoding="utf-8") as f:
                data = json.load(f)
            if subject and data["subject"] != subject:
                continue
            for c in data["categories"]:
                result.append({
                    "name": c["name"],
                    "subject": data["subject"],
                    "keywords": c.get("keywords", []),
                })
        return result