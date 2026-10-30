# -*- coding: utf-8 -*-
"""分类准确率 / 粗筛召回率 / 单条耗时评测（对应需求 4.4.1 第 2 点）。

用法：
    python -m tools.evaluate                 # 全量评测
    python -m tools.evaluate --subject 高等数学
    python -m tools.evaluate --feedback      # 附带增量优化前后的对比实验
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_EVAL_WORKSPACE = tempfile.TemporaryDirectory(prefix="mc_eval_")
os.environ["MC_WORKSPACE"] = _EVAL_WORKSPACE.name

from app import config                                    # noqa: E402
from app.core.classify.engine import ClassificationEngine  # noqa: E402
from app.core.classify.feedback import FeedbackOptimizer   # noqa: E402
from app.core.models import Question                       # noqa: E402
from app.core.preprocess import pipeline                   # noqa: E402
from app.core.storage import Repository                    # noqa: E402
from app.core.classify import weights as W                  # noqa: E402

DATASET = ROOT / "samples" / "eval_dataset.json"


def load_rows(subject: str = "") -> List[dict]:
    with open(DATASET, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return [r for r in rows if not subject or r["学科"] == subject]


def build_engines(repo: Repository) -> Dict[str, ClassificationEngine]:
    return {s.subject: ClassificationEngine(s) for s in repo.schemes
            if s.subject != config.SUBJECT_GENERAL}


def run(rows: List[dict], engines: Dict[str, ClassificationEngine], verbose: bool = False):
    ok = 0
    times: List[float] = []
    by_subject: Dict[str, List[int]] = defaultdict(list)
    by_qtype: Dict[str, List[int]] = defaultdict(list)
    recall_hit = 0
    wrong: List[tuple] = []

    for r in rows:
        eng = engines.get(r["学科"])
        if eng is None:
            continue
        q = Question(subject=r["学科"], qtype=r["题型"],
                     raw_text=r["题干"], annotation=r.get("标注", ""))
        t0 = time.perf_counter()
        res = eng.classify(q)
        times.append((time.perf_counter() - t0) * 1000.0)

        dr = res.dims.get(config.DIM_KNOWLEDGE)
        pred = dr.label_name if dr else ""
        gold = r["标准知识点"]
        hit = int(pred == gold)
        ok += hit
        by_subject[r["学科"]].append(hit)
        by_qtype[r["题型"]].append(hit)
        if dr and any(c.label_name == gold for c in dr.candidates):
            recall_hit += 1
        if not hit:
            wrong.append((r["学科"], r["题型"], gold, pred,
                          round(dr.score, 3) if dr else 0, r["题干"][:42]))

    n = len(times)
    print("=" * 78)
    print("样本总数 {}   知识点分类准确率 {:.1%}   粗筛召回率 {:.1%}".format(
        n, ok / n if n else 0, recall_hit / n if n else 0))
    print("单条耗时  平均 {:.1f}ms  中位 {:.1f}ms  P95 {:.1f}ms  最大 {:.1f}ms  (预算 {}ms)".format(
        statistics.mean(times), statistics.median(times),
        sorted(times)[int(n * 0.95) - 1] if n else 0, max(times) if times else 0,
        config.SINGLE_QUESTION_TIME_BUDGET_MS))
    print("-" * 78)
    print("按学科：")
    for k, v in by_subject.items():
        print("   {:<16} {:>3}/{:<3}  {:.1%}".format(k, sum(v), len(v), sum(v) / len(v)))
    print("按题型：")
    for k, v in by_qtype.items():
        print("   {:<16} {:>3}/{:<3}  {:.1%}".format(
            config.QTYPE_LABELS.get(k, k), sum(v), len(v), sum(v) / len(v)))
    if verbose and wrong:
        print("-" * 78)
        print("误判明细（{} 条）：".format(len(wrong)))
        for w in wrong:
            print("   [{}/{}] 应为 {} -> 判为 {} ({}) | {}".format(*w))
    print("=" * 78)
    return ok / n if n else 0.0, times, wrong


class TextOnlyRules(W.WeightRules):
    """对照组：退化为传统单通道纯文本 TF-IDF（关闭公式/题型/学科通道）。"""

    def get(self, subject, qtype, override=None):
        return {config.CHANNEL_TEXT: 1.0, config.CHANNEL_FORMULA: 0.0,
                config.CHANNEL_QTYPE: 0.0, config.CHANNEL_SUBJECT: 0.0}


def ablation_channels(rows, repo):
    """消融实验：多特征融合 vs 传统单通道纯文本 TF-IDF。"""
    print("\n【消融实验 1｜多特征融合 vs 传统单通道 TF-IDF】")
    base = {s.subject: ClassificationEngine(s, rules=TextOnlyRules())
            for s in repo.schemes if s.subject != config.SUBJECT_GENERAL}
    print("--- 对照组：单通道纯文本 TF-IDF + 余弦（传统方案） ---")
    acc_base, _, _ = run(rows, base)
    print("--- 实验组：多特征融合（本项目） ---")
    acc_ours, _, _ = run(rows, build_engines(repo))
    print(">>> 准确率 {:.1%} -> {:.1%}（{:+.1f} 个百分点）".format(
        acc_base, acc_ours, (acc_ours - acc_base) * 100))


def ablation_stages(rows, repo):
    """架构对照：双层分级（粗筛+精匹配） vs 单阶段全量匹配。"""
    print("\n【消融实验 2｜双层分级 vs 单阶段全量匹配（效率）】")
    engines = build_engines(repo)
    stats = {}
    for tag, k in (("单阶段全量匹配", 10 ** 6), ("双层分级（本项目）", config.COARSE_TOP_K)):
        ok, t, scanned = 0, [], 0
        for r in rows:
            eng = engines.get(r["学科"])
            if eng is None:
                continue
            q = Question(subject=r["学科"], qtype=r["题型"],
                         raw_text=r["题干"], annotation=r.get("标注", ""))
            fb = pipeline.preprocess(q)          # 预处理耗时对两者相同，剔除以突出架构差异
            t0 = time.perf_counter()
            res = eng.classify(q, fb=fb, top_k=k)
            t.append((time.perf_counter() - t0) * 1000.0)
            dr = res.dims.get(config.DIM_KNOWLEDGE)
            ok += int(bool(dr) and dr.label_name == r["标准知识点"])
            scanned += sum(d.scanned for d in res.dims.values())
        stats[tag] = (ok / len(rows), statistics.mean(t), scanned)
        print("   {:<20} 准确率 {:.1%}   匹配耗时 {:.2f}ms/条   精匹配标签数 {}".format(
            tag, *stats[tag]))
    a, b = stats["单阶段全量匹配"], stats["双层分级（本项目）"]
    print(">>> 精匹配计算量降至 {:.1%}，匹配环节提速 {:.2f}x，准确率 {:+.1f} 个百分点".format(
        b[2] / a[2] if a[2] else 0, a[1] / b[1] if b[1] else 0, (b[0] - a[0]) * 100))


def scalability_stages(rows, repo):
    """粗筛收益随标签规模的变化。

    内置体系每个维度只有 10-14 个知识点标签，两阶段架构的绝对收益自然有限。
    真实场景里用户会不断细化词库（一门课拆到几十上百个知识点），
    本实验用合成标签把规模放大，验证粗筛的收益随标签数增长而扩大。
    """
    print("\n【消融实验 3｜粗筛收益随知识点标签规模的变化】")
    import copy
    from app.core.models import CategoryLabel

    sc0 = next(s for s in repo.schemes if s.subject == "高等数学")
    sub = [r for r in rows if r["学科"] == "高等数学"]
    print("   {:<10} {:>10} {:>14} {:>14} {:>10}".format(
        "标签数", "准确率", "单阶段", "双层分级", "提速"))
    for extra in (0, 40, 90, 190):
        sc = copy.deepcopy(sc0)
        dim = sc.dim(config.DIM_KNOWLEDGE)
        # 合成干扰标签：词表来自其它学科，语义上与高数无关，只用于放大候选集规模
        pool = [lb for s in repo.schemes if s.subject != "高等数学"
                for d in s.dims if d.key == config.DIM_KNOWLEDGE for lb in d.labels]
        for i in range(extra):
            src = pool[i % len(pool)] if pool else None
            kws = list(src.keywords[:6]) if src else ["占位词{}".format(i)]
            dim.labels.append(CategoryLabel(
                id="syn_{}".format(i), name="合成知识点{}".format(i),
                dim=config.DIM_KNOWLEDGE, keywords=kws,
                samples=list(src.samples[:1]) if src else []))
        sc.touch()
        eng = ClassificationEngine(sc)
        out = {}
        for tag, k in (("full", 10 ** 6), ("two", config.COARSE_TOP_K)):
            ok, t = 0, []
            for r in sub:
                q = Question(subject=r["学科"], qtype=r["题型"],
                             raw_text=r["题干"], annotation=r.get("标注", ""))
                fb = pipeline.preprocess(q)
                t0 = time.perf_counter()
                res = eng.classify(q, fb=fb, top_k=k)
                t.append((time.perf_counter() - t0) * 1000.0)
                dr = res.dims.get(config.DIM_KNOWLEDGE)
                ok += int(bool(dr) and dr.label_name == r["标准知识点"])
            out[tag] = (ok / len(sub), statistics.mean(t))
        print("   {:<10} {:>10.1%} {:>12.2f}ms {:>12.2f}ms {:>9.2f}x".format(
            len(dim.labels), out["two"][0], out["full"][1], out["two"][1],
            out["full"][1] / out["two"][1] if out["two"][1] else 0))


def feedback_experiment(rows, repo):
    """增量优化验证（需求 4.4.1「修正后同类型错题准确率提升≥5%」）。

    实验设计（模拟真实冷启动场景，避免"考过的题再考一遍"式的虚高）：
      1. 剥离全部标准错题样本，让每个标签只剩关键词 —— 相当于一个刚建好词库的新用户；
      2. 数据集按 5:5 划分 修正集 / 复测集；
      3. 只在修正集上执行人工修正（触发增量优化）；
      4. 复测集全程未被系统见过，其准确率变化即为真实泛化收益。
    """
    print("\n【增量反馈优化验证（冷启动 -> 修正 -> 泛化复测）】")
    # 1. 制造冷启动：剥离全部标准样本，每个标签只保留"标签名 + 2 个关键词"。
    #    这模拟一个刚在界面上手敲了几个关键词就开始用的新用户——
    #    也是"零代码自定义分类体系"最真实的初始状态。
    for sc in repo.schemes:
        for d in sc.dims:
            for lb in d.labels:
                lb.samples = []
                lb.keywords = lb.keywords[:2]
        sc.touch()
    engines = build_engines(repo)

    random.seed(42)
    shuffled = list(rows)
    random.shuffle(shuffled)
    cut = len(shuffled) // 2
    fix_set, retest = shuffled[:cut], shuffled[cut:]
    print("冷启动（标签仅剩关键词）  修正集 {} 条 / 复测集 {} 条".format(len(fix_set), len(retest)))

    # 3. 在修正集上分类，只对**判错的**题执行人工修正——这才是用户真实会做的事。
    #    对判对的题重复"确认"不产生新信息，系统也不会拿它去更新画像。
    pending = []
    for r in fix_set:
        eng = engines.get(r["学科"])
        if eng is None:
            continue
        dim = eng.scheme.dim(config.DIM_KNOWLEDGE)
        gold = dim.label_by_name(r["标准知识点"]) if dim else None
        if gold is None:
            continue
        q = Question(subject=r["学科"], qtype=r["题型"],
                     raw_text=r["题干"], annotation=r.get("标注", ""))
        res = eng.classify(q)
        ClassificationEngine.apply_result(q, res, eng.scheme, q.feature_bundle())
        if q.labels.get(config.DIM_KNOWLEDGE) != gold.id:
            pending.append((eng, q, gold, r["学科"], r["标准知识点"]))

    # 需求原文是"修正后**同类型**错题准确率提升"，因此主口径只统计复测集中
    # 知识点收到过修正的题；没被修正过的知识点本来就无从学起，混进来只会稀释信号。
    touched = {(p[3], p[4]) for p in pending}
    same = [r for r in retest if (r["学科"], r["标准知识点"]) in touched]
    print("修正集判错 {} 条，覆盖 {} 个知识点；复测集中同类型题 {} / {} 条".format(
        len(pending), len(touched), len(same), len(retest)))

    print("--- 优化前 ---")
    before_same, _, _ = run(same, engines)
    before_all, _, _ = run(retest, engines)

    n_fb = 0
    for eng, q, gold, _s, _k in pending:
        if FeedbackOptimizer(eng).apply_correction(q, config.DIM_KNOWLEDGE, gold.id):
            n_fb += 1
    print("已应用 {} 次人工修正（每次均为 O(本题词数) 增量更新，无一次全量重训）".format(n_fb))

    print("--- 优化后 ---")
    after_same, _, _ = run(same, engines)
    after_all, _, _ = run(retest, engines)

    n = len(same)
    print(">>> 【需求口径】同类型错题：{:.1%} -> {:.1%}（{:+.1f} 个百分点）".format(
        before_same, after_same, (after_same - before_same) * 100))
    print(">>> 【全量口径】全部复测题：{:.1%} -> {:.1%}（{:+.1f} 个百分点）".format(
        before_all, after_all, (after_all - before_all) * 100))
    print("    注意：同类型子集仅 {} 条，1 条题就相当于 {:.1f} 个百分点，".format(
        n, 100.0 / n if n else 0))
    print("    该量级的差异尚在统计噪声内，不足以支撑\"提升≥5%\"的结论。详见 README「已知局限」。")


EXPERIMENTS = {
    "main": lambda rows, repo, v: (print("\n【主评测｜知识点维度】"),
                                   run(rows, build_engines(repo), verbose=v))[1],
    "channels": lambda rows, repo, v: ablation_channels(rows, repo),
    "stages": lambda rows, repo, v: ablation_stages(rows, repo),
    "scalability": lambda rows, repo, v: scalability_stages(rows, repo),
    "feedback": lambda rows, repo, v: feedback_experiment(rows, repo),
}


def _run_one(name: str, subject: str, verbose: bool) -> None:
    """在当前进程里跑单个实验（由 --only 直接调用，或由 main 派生子进程调用）。"""
    pipeline.warmup()
    EXPERIMENTS[name](load_rows(subject), Repository(), verbose)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--all", action="store_true", help="附加全部消融与增量优化实验")
    ap.add_argument("--only", choices=sorted(EXPERIMENTS), help="只跑指定的单个实验")
    args = ap.parse_args()

    if args.only:
        _run_one(args.only, args.subject, args.verbose)
        return

    names = ["main"] + (["channels", "stages", "scalability", "feedback"]
                        if args.all else [])
    if len(names) == 1:
        _run_one("main", args.subject, args.verbose)
        return

    # 每个实验派生一个独立子进程。
    #
    # 这一步不是洁癖：这些实验会互相污染进程内的全局状态（jieba 词典、
    # 各引擎缓存的特征库与词表 df、WeightRules 的落盘文件），导致后跑的实验
    # 拿到被前面实验改过的起点。曾经就因此把增量优化实验的冷启动基线从
    # 74.4% 压到 67.9%，凭空多算出 6 个百分点的"提升"。同进程内跑出来的数字
    # 依赖执行顺序，不可复现，也就没有资格进结题报告。
    import subprocess
    for n in names:
        env = dict(os.environ)
        cmd = [sys.executable, "-m", "tools.evaluate", "--only", n]
        if args.subject:
            cmd += ["--subject", args.subject]
        if args.verbose:
            cmd += ["-v"]
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)


if __name__ == "__main__":
    main()
