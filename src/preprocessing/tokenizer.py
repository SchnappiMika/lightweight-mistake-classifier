"""
分词处理模块

基于 jieba 分词的错题文本分词，支持分学科专有名词词典。
"""
import jieba
import os
from pathlib import Path


class Tokenizer:
    """错题文本分词器"""

    _initialized = False

    @classmethod
    def initialize(cls, dict_dir: str = None):
        """初始化分词器，加载分学科专有名词词典"""
        if cls._initialized:
            return

        if dict_dir is None:
            dict_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "knowledge_base", "data", "dictionaries"
            )

        # 加载分学科词典（如果存在）
        dict_dir_path = Path(dict_dir)
        if dict_dir_path.exists():
            for dict_file in dict_dir_path.glob("*.txt"):
                jieba.load_userdict(str(dict_file))
                print(f"  已加载词典: {dict_file.name}")

        cls._initialized = True
        print("  分词器初始化完成")

    @staticmethod
    def segment(text: str) -> list:
        """
        对文本进行分词

        Args:
            text: 清洗后的文本

        Returns:
            分词结果列表
        """
        if not text:
            return []
        return list(jieba.cut(text, cut_all=False))

    @staticmethod
    def extract_core_words(words: list, pos_tags: set = None) -> list:
        """
        提取核心实词（过滤虚词，保留名词、动词等）

        Args:
            words: 分词结果列表
            pos_tags: 需要保留的词性集合（预留，当前暂不依赖词性标注）

        Returns:
            核心词汇列表
        """
        if not words:
            return []

        # 简单启发式过滤：长度小于2的单个字/词通常没有分类价值
        # 但不包括数学符号如 "sin", "cos", "lim" 等
        math_terms = {"sin", "cos", "tan", "lim", "log", "ln", "dx", "dy",
                      "∑", "∫", "π", "α", "β", "γ", "θ", "λ", "μ", "σ", "ω"}
        result = []
        for w in words:
            w = w.strip()
            if not w:
                continue
            if w in math_terms:
                result.append(w)
            elif len(w) >= 2:
                result.append(w)
        return result
