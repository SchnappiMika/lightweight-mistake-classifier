# -*- coding: utf-8 -*-
"""预处理与特征提取模块的指标评测（对应需求 4.4.1 第 1 点）。

需求对这一模块提了四条**特征提取**层面的指标（注意：不是分类准确率，
是分类之前那一步）：

    ① 纯文本：核心分类特征提取准确率 ≥95%，分词错误率 ≤3%
    ② 公式类：公式特征提取准确率 ≥90%
    ③ 图形类：图形题核心特征提取准确率 ≥90%
    ④ 英语类：英语题核心特征提取准确率 ≥90%

用法：
    python -m tools.eval_features            # 全部四项
    python -m tools.eval_features -v         # 附带失败明细

各指标的操作化定义见下方各函数的 docstring —— 需求只给了指标名，
没给测法，所以怎么测必须写清楚，否则数字没有意义。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MC_WORKSPACE", "/tmp/mc_feateval")

from app.core.models import Question                       # noqa: E402
from app.core.preprocess import pipeline  # noqa: E402
from app.core.preprocess.formula import get_mapper         # noqa: E402
from app.core.preprocess.text_cn import tokenize_cn        # noqa: E402

FEATURE_SET = ROOT / "samples" / "feature_eval.json"

# --------------------------------------------------------------------------- #
# ① 分词错误率
# --------------------------------------------------------------------------- #
def eval_tokenize_error_rate(verbose: bool = False) -> Tuple[float, int, List[str]]:
    """分词错误率（需求：≤3%）。

    操作化定义：对 ``feature_eval.json`` 里**按学科语义标注**的核心术语，
    把每条错题的题干与标注拼起来跑完整分词，检查该术语是否被作为一个
    完整的词切出来。切碎了算一次错误。

        分词错误率 = 被切碎的(术语, 语境)数 / 总数

    **这个指标怎么测比测出什么更重要，两点必须说清楚：**

    1. **不能拿专有名词词典里的词条当测试集**。词典里的词当然切得对——
       那是词典的定义，测出来必然 0%，是循环论证。本项目最初就是这么测的，
       得到 0.00% 的漂亮数字，实际上什么也没验证。换成按语义独立标注的
       术语来测，真实错误率是 31.4%，而且切出了"拉格朗""法求""散性"
       这样的垃圾碎片。

    2. **本函数当前的数字同样有偏**：那 31.4% 暴露的缺词已经补进词典了，
       再拿同一批术语来测，只能确认"修复落地了"，**不能**估计未见术语的
       错误率。要无偏估计，需要一批没参与过词典修订的独立标注术语。

    另外，实测表明本架构对这个指标其实不敏感（见 README「已知局限」）：
    题目与标签画像共用同一个分词器，切错但切得一致，余弦相似度照样匹配；
    粗筛还有原文子串匹配兜底。保留词典优化的主要理由是**可解释性**——
    界面会把特征词作为分类依据展示给用户，"拉格朗""法求"这种碎片会让人
    以为系统坏了。
    """
    data = _load()
    fails: List[str] = []
    total = 0
    for kind in ("text", "figure", "english"):
        for item in data[kind]:
            ctx = "{} {}".format(item.get("text", ""), item.get("annotation", "")).strip()
            toks = set(tokenize_cn(ctx, item["subject"]))
            for w in item["expect"]:
                total += 1
                if w not in toks:
                    fails.append("{}｜{}｜{}".format(item["subject"], w, ctx[:30]))
    rate = len(fails) / total if total else 0.0
    print("① 分词错误率        {:.2%}   ({} / {} 个术语×语境)  需求 ≤3%  {}".format(
        rate, len(fails), total, "✅" if rate <= 0.03 else "❌"))
    if verbose and fails:
        for f in fails[:20]:
            print("      ✗ {}".format(f))
    return rate, total, fails


# --------------------------------------------------------------------------- #
# ② 公式特征提取准确率
# --------------------------------------------------------------------------- #
def eval_formula(verbose: bool = False) -> Tuple[float, int, List[str]]:
    """公式特征提取准确率（需求：≥90%）。

    操作化定义：对每条人工标注的 (LaTeX, 学科, 期望知识点特征词)，
    跑"公式-知识点映射词典"，检查期望的特征词是否出现在映射结果里。

        准确率 = 命中期望特征词的公式数 / 公式总数

    注意这测的是**映射词典**这一环，不含 pix2tex 的图片识别——
    那是可选依赖，容器里没装，也不该由它的装没装来决定本项目的指标。
    真实场景里，用户在题干里用 $...$ 手写公式走的正是这条路径。
    """
    data = _load()["formula"]
    ok, fails = 0, []
    for item in data:
        toks, _hits = get_mapper().map_latex(item["latex"], item["subject"])
        want = item["expect"]
        # expect 是"可接受同义词集合"，命中任意一个即算正确；空集表示
        # 该公式本就没有判别结构（如 y=x^{sin x}），不产出特征词是对的。
        if not want or any(w in toks for w in want):
            ok += 1
        else:
            fails.append("{}｜{}｜期望 {} 实得 {}".format(
                item["subject"], item["latex"][:38], want, sorted(set(toks))[:6]))
    acc = ok / len(data) if data else 0.0
    print("② 公式特征提取准确率 {:.1%}   ({} / {} 条)              需求 ≥90%  {}".format(
        acc, ok, len(data), "✅" if acc >= 0.90 else "❌"))
    if verbose:
        for f in fails:
            print("      ✗ {}".format(f))
    return acc, len(data), fails


# --------------------------------------------------------------------------- #
# ③④ 核心分类特征提取准确率（纯文本 / 图形 / 英语）
# --------------------------------------------------------------------------- #
def eval_core_features(kind: str, threshold: float,
                       verbose: bool = False) -> Tuple[float, int, List[str]]:
    """核心分类特征提取准确率。

    操作化定义：对每条人工标注了"核心特征词"的错题，跑完整预处理流程，
    检查这些核心词是否都出现在提取出的特征里（``all_tokens()``）。

        准确率 = 核心词被完整提取出来的错题数 / 错题总数

    为什么用"全部命中"而不是"部分命中"：核心特征词是这道题能被分对的
    **充要信息**，漏掉任何一个都可能让它被分到别处去。标准定得严一点，
    数字才有参考价值。

    图形题与听力题在容器里没有装 PaddleOCR / Whisper，走的是"人工标注"
    降级路径——这恰恰是绝大多数用户的真实状态（那两个引擎上 GB），
    所以这么测是贴合实际的。
    """
    data = _load()[kind]
    ok, fails = 0, []
    for item in data:
        q = Question(subject=item["subject"], qtype=item["qtype"],
                     raw_text=item.get("text", ""),
                     annotation=item.get("annotation", ""))
        fb = pipeline.preprocess(q)
        got = set(fb.all_tokens())
        miss = [w for w in item["expect"] if w not in got]
        if not miss:
            ok += 1
        else:
            fails.append("{}｜{}｜漏 {}".format(
                item["subject"], (item.get("text") or item.get("annotation", ""))[:34], miss))
    acc = ok / len(data) if data else 0.0
    name = {"text": "③ 纯文本核心特征提取", "figure": "④ 图形题核心特征提取",
            "english": "⑤ 英语题核心特征提取"}[kind]
    print("{} {:.1%}   ({} / {} 条)              需求 ≥{:.0%}  {}".format(
        name, acc, ok, len(data), threshold, "✅" if acc >= threshold else "❌"))
    if verbose:
        for f in fails:
            print("      ✗ {}".format(f))
    return acc, len(data), fails


def _load() -> Dict[str, List[dict]]:
    with open(FEATURE_SET, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", "-v", action="store_true", help="打印失败明细")
    args = ap.parse_args()

    pipeline.warmup()
    print("=" * 78)
    print("预处理与特征提取模块指标评测（需求 4.4.1 第 1 点）")
    print("=" * 78)
    eval_tokenize_error_rate(args.verbose)
    eval_formula(args.verbose)
    eval_core_features("text", 0.95, args.verbose)
    eval_core_features("figure", 0.90, args.verbose)
    eval_core_features("english", 0.90, args.verbose)
    print("=" * 78)
    print("这些数字怎么读（很重要）：")
    print("  · 标注集与本项目同源（samples/feature_eval.json），属于闭卷自测，")
    print("    适合看回归与相对趋势，不宜当作独立泛化能力的承诺。")
    print("  · ① 分词错误率的 0% 尤其要打折看：它测的正是当初暴露出 31.4% 错误率、")
    print("    并据此补进词典的那批术语。它只能确认修复落地了，不能估计未见术语的错误率。")
    print("  · 另需说明：实测本架构对分词错误率并不敏感——题目与标签画像共用同一个")
    print("    分词器，切错但切得一致，余弦相似度照样匹配；粗筛还有原文子串匹配兜底。")
    print("    详见 README「已知局限」。")


if __name__ == "__main__":
    main()
