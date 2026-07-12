"""
停用词过滤模块

双层停用词过滤体系：通用停用词 + 学科专属停用词。
"""
import os
from pathlib import Path


class StopwordsFilter:
    """停用词过滤器"""

    def __init__(self, stopwords_dir: str = None):
        self.general_stopwords = set()
        self.subject_stopwords = {}

        if stopwords_dir is None:
            stopwords_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "knowledge_base", "data", "stopwords"
            )

        self._load_stopwords(stopwords_dir)

    def _load_stopwords(self, stopwords_dir: str):
        """加载停用词文件"""
        stopwords_path = Path(stopwords_dir)
        if not stopwords_path.exists():
            print("  警告: 停用词目录不存在，使用内置默认停用词")
            self._load_default_stopwords()
            return

        for file_path in stopwords_path.glob("*.txt"):
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            words = {line.strip() for line in content.splitlines() if line.strip()}

            if "通用" in file_path.stem or "general" in file_path.stem:
                self.general_stopwords.update(words)
                print(f"  已加载通用停用词: {len(words)} 个")
            else:
                subject = file_path.stem
                self.subject_stopwords[subject] = words
                print(f"  已加载学科停用词 [{subject}]: {len(words)} 个")

        if not self.general_stopwords:
            self._load_default_stopwords()

    def _load_default_stopwords(self):
        """加载内置默认停用词"""
        self.general_stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "它", "们", "那", "这个", "那个", "什么", "怎么", "如何", "为",
            "吗", "啊", "呢", "吧", "呀", "哦", "嗯", "哈哈", "呵呵",
            "能", "可以", "应该", "会", "可能", "能够", "得", "地",
            "与", "或", "而", "但", "且", "若", "如", "因", "所", "被",
            "把", "从", "对", "向", "让", "以", "于", "则", "之", "其",
            "中", "等", "某", "每", "各", "该", "另外", "此外", "同时",
            "通过", "进行", "以及", "为了", "因为", "所以", "如果",
            "虽然", "但是", "而且", "并且", "或者", "只是", "不过",
            "然后", "接着", "最后", "首先", "其次", "例如", "比如",
            "即", "使", "将", "已", "已", "仍", "仍", "可", "须",
            "作为", "关于", "按照", "根据", "基于", "针对",
            "第", "个", "种", "些", "些", "点", "种", "次", "回", "遍",
        }

    def filter(self, words: list, subject: str = None) -> list:
        """
        过滤停用词

        Args:
            words: 分词后的词汇列表
            subject: 学科名称（用于加载学科专属停用词）

        Returns:
            过滤后的词汇列表
        """
        if not words:
            return []

        result = []
        for w in words:
            w = w.strip().lower()
            if not w:
                continue
            if w in self.general_stopwords:
                continue
            if subject and subject in self.subject_stopwords:
                if w in self.subject_stopwords[subject]:
                    continue
            result.append(w)
        return result
