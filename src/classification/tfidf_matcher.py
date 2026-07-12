import os, json, pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfMatcher:
    def __init__(self, kb_dir=None):
        self.vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b"
        )
        self.category_texts = {}
        self.category_vectors = None
        self.is_fitted = False
        self.category_names = []
        self.subject_map = {}

    def build_index(self, categories, subject_map=None):
        self.category_texts = {}
        self.subject_map = subject_map or {}
        for cat_name, cat_info in categories.items():
            keywords = cat_info.get("keywords", [])
            sample = cat_info.get("sample", "")
            combined = " ".join(keywords)
            if sample:
                combined += " " + sample
            self.category_texts[cat_name] = combined
        if not self.category_texts:
            return
        self.category_names = list(self.category_texts.keys())
        texts = list(self.category_texts.values())
        self.category_vectors = self.vectorizer.fit_transform(texts)
        self.is_fitted = True

    def match(self, query_words, top_k=3):
        if not query_words or not self.is_fitted:
            return []
        query_text = " ".join(query_words)
        query_vector = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vector, self.category_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.category_names[idx], round(float(similarities[idx]), 4)))
        return results

    def match_by_subject(self, query_words, subject, top_k=3):
        if not query_words or not self.is_fitted or not subject:
            return self.match(query_words, top_k)
        filtered = [n for n in self.category_names if self.subject_map.get(n) == subject]
        if not filtered:
            return self.match(query_words, top_k)
        query_text = " ".join(query_words)
        query_vector = self.vectorizer.transform([query_text])
        indices = [self.category_names.index(n) for n in filtered]
        vectors = self.category_vectors[indices]
        sims = cosine_similarity(query_vector, vectors)[0]
        top = sorted(zip(filtered, sims), key=lambda x: x[1], reverse=True)[:top_k]
        return [(n, round(float(s), 4)) for n, s in top if s > 0]

    def add_sample(self, cat_name, sample_text):
        if cat_name in self.category_texts:
            self.category_texts[cat_name] += " " + sample_text
        else:
            self.category_texts[cat_name] = sample_text
        self.build_index({k: {"keywords": [], "sample": v} for k, v in self.category_texts.items()})
