\"\"\"
阶段2完整测试：纯文本分类引擎全流水线验证
\"\"\"
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.tokenizer import Tokenizer
from src.preprocessing.stopwords import StopwordsFilter
from src.classification.keyword_matcher import KeywordMatcher
from src.classification.tfidf_matcher import TfidfMatcher


def load_sample_questions(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["questions"]


def load_category_dicts(path: str) -> dict:
    \"\"\"加载所有分类词典\"\"\"
    categories = {}
    dict_dir = os.path.dirname(path)
    for fname in os.listdir(dict_dir):
        if fname.endswith('.json'):
            fpath = os.path.join(dict_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for cat in data["categories"]:
                categories[cat["name"]] = cat
    return categories


def main():
    print(\"=\" * 60)
    print(\"阶段2：纯文本分类引擎全流水线验证\")
    print(\"=\" * 60)

    # 路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dict_dir = os.path.join(base_dir, \"src\", \"knowledge_base\", \"data\", \"dictionaries\")
    sample_path = os.path.join(base_dir, \"data\", \"samples\", \"sample_questions.json\")

    # ---- 1. 初始化所有模块 ----
    print(\"\\n[1/5] 初始化各模块...\")
    cleaner = TextCleaner()
    Tokenizer.initialize(dict_dir)
    tokenizer = Tokenizer()
    stopwords = StopwordsFilter()
    keyword_matcher = KeywordMatcher()
    print(\"  初始化完成\")

    # ---- 2. 加载分类词典到 TF-IDF 匹配器 ----
    print(\"\\n[2/5] 构建 TF-IDF 分类索引...\")
    categories = load_category_dicts(dict_dir)
    tfidf_matcher = TfidfMatcher()
    tfidf_matcher.build_index(categories)
    print(f\"  已建立 {len(categories)} 个分类的索引\")

    # ---- 3. 加载样本数据 ----
    print(\"\\n[3/5] 加载测试样本...\")
    questions = load_sample_questions(sample_path)
    print(f\"  共 {len(questions)} 条测试样本\")

    # ---- 4. 全流水线测试 ----
    print(\"\\n[4/5] 执行全流水线测试...\")
    results = []
    correct_keyword = 0
    correct_tfidf = 0

    for i, q in enumerate(questions):
        text = q[\"text\"]
        subject = q.get(\"subject\", \"\")
        expected = q[\"expected\"]

        # 清洗
        cleaned = cleaner.clean(text)

        # 分词
        words = tokenizer.segment(cleaned)
        core_words = tokenizer.extract_core_words(words)

        # 停用词过滤
        filtered = stopwords.filter(core_words, subject)

        # 关键词粗筛
        keyword_top = keyword_matcher.match(filtered, top_k=5)
        keyword_pred = keyword_top[0][0] if keyword_top else \"未匹配\"
        kw_correct = 1 if keyword_pred == expected else 0
        correct_keyword += kw_correct

        # TF-IDF 精匹配
        tfidf_top = tfidf_matcher.match(filtered, top_k=3)
        tfidf_pred = tfidf_top[0][0] if tfidf_top else \"未匹配\"
        tf_correct = 1 if tfidf_pred == expected else 0
        correct_tfidf += tf_correct

        results.append({
            \"text\": text[:30] + \"...\" if len(text) > 30 else text,
            \"expected\": expected,
            \"keyword_pred\": keyword_pred,
            \"tfidf_pred\": tfidf_pred,
            \"kw_ok\": \"✓\" if kw_correct else \"✗\",
            \"tf_ok\": \"✓\" if tf_correct else \"✗\"
        })

        tag = \"✓\" if (kw_correct or tf_correct) else \"✗\"
        print(f\"  [{i+1:2d}] {tag} 期望={expected:10s} | 关键词={keyword_pred:10s} | TF-IDF={tfidf_pred:10s}\")

    # ---- 5. 统计结果 ----
    print(\"\\n[5/5] 统计结果\")
    total = len(questions)
    kw_acc = correct_keyword / total * 100
    tf_acc = correct_tfidf / total * 100
    print(f\"\\n{'='*60}\")
    print(f\"关键词粗筛准确率: {correct_keyword}/{total} = {kw_acc:.1f}%\")
    print(f\"TF-IDF精匹配准确率: {correct_tfidf}/{total} = {tf_acc:.1f}%\")
    print(f\"综合准确率: {(correct_keyword + correct_tfidf)/(2*total)*100:.1f}%\")
    print(f\"{'='*60}\")

    # 列出错误详情
    failed = [r for r in results if r[\"kw_ok\"] == \"✗\" and r[\"tf_ok\"] == \"✗\"]
    if failed:
        print(\"\\n错误详情：\")
        for r in failed:
            print(f\"  原题: {r['text']}\")
            print(f\"  期望: {r['expected']}\")
            print(f\"  关键词预测: {r['keyword_pred']}\")
            print(f\"  TF-IDF预测: {r['tfidf_pred']}\")
            print()

    if kw_acc >= 50 and tf_acc >= 50:
        print(\"\\n阶段2基础流水线验证通过！\")
    else:
        print(\"\\n准确率偏低，需要优化词典和权重参数\")


if __name__ == \"__main__\":
    main()
