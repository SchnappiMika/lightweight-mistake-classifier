"""
关键词快速粗筛模块

第一层分类：基于关键词匹配快速筛选出3-5个高相关候选项。
"""
import json
import os
from pathlib import Path
from typing import List, Tuple


class KeywordMatcher:
    """关键词粗筛器"""

    def __init__(self, kb_dir: str = None):
        self.categories = {}  # {分类名: {keywords: [...], weight: float}}

        if kb_dir is None:
            kb_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "knowledge_base", "data", "dictionaries"
            )
        self._load_categories(kb_dir)

    def _load_categories(self, kb_dir: str):
        """加载分类关键词库"""
        kb_path = Path(kb_dir)
        if kb_path.exists():
            for json_file in kb_path.glob("*.json"):
                data = json.loads(json_file.read_text(encoding="utf-8"))
                subject = json_file.stem
                for cat in data.get("categories", []):
                    cat_name = cat.get("name", "")
                    keywords = cat.get("keywords", [])
                    weight = cat.get("weight", 1.0)
                    self.categories[cat_name] = {
                        "keywords": [k.lower() for k in keywords],
                        "weight": weight,
                        "subject": subject
                    }

    def add_category(self, name: str, keywords: list, weight: float = 1.0, subject: str = "通用"):
        """动态添加分类"""
        self.categories[name] = {
            "keywords": [k.lower() for k in keywords],
            "weight": weight,
            "subject": subject
        }

    def match(self, words: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        关键词粗筛匹配

        Args:
            words: 预处理后的核心特征词列表
            top_k: 返回前k个候选分类

        Returns:
            [(分类名, 匹配得分), ...]
        """
        if not words or not self.categories:
            return []

        word_set = {w.lower() for w in words}
        scores = {}

        for cat_name, cat_info in self.categories.items():
            keywords = cat_info["keywords"]
            weight = cat_info["weight"]

            # 统计匹配到的关键词数量
            matches = sum(1 for kw in keywords if kw in word_set)
            if matches > 0:
                # 得分 = 匹配数 / 总关键词数 * 权重
                score = (matches / max(len(keywords), 1)) * weight
                scores[cat_name] = round(score, 4)

        # 按得分降序排序，返回前 top_k 个
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]
