"""
文本清洗模块

负责对原始错题文本进行标准化清洗，去除无关内容。
"""
import re


class TextCleaner:
    """文本清洗器"""

    @staticmethod
    def clean(text: str) -> str:
        """
        对文本执行标准化清洗

        Args:
            text: 原始错题文本

        Returns:
            清洗后的纯净文本
        """
        if not text or not isinstance(text, str):
            return ""

        text = text.strip()

        # 去除题干序号标记
        text = re.sub(r'^[\d]+[.、)．）]', '', text)
        text = re.sub(r'^[（(]\d+[）)]', '', text)
        text = re.sub(r'^[①②③④⑤⑥⑦⑧⑨⑩]', '', text)

        # 统一括号类型
        text = text.replace("（", "(").replace("）", ")")
        text = text.replace("【", "[").replace("】", "]")
        text = text.replace("《", "<").replace("》", ">")

        # 去除特殊符号（保留中文、英文、数字、基本数学符号和标点）
        text = re.sub(r'[^\w\s\u4e00-\u9fff+\-*/=<>()\[\]{},.;:!?\'"%°]', '', text)

        # 合并连续空白
        text = re.sub(r'\s+', ' ', text)

        # 去除选项标记
        text = re.sub(r'\b[A-D][.、)．）]\s*', '', text)

        return text.strip()
