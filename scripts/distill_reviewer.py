#!/usr/bin/env python
# scripts/distill_reviewer.py
"""
审核飞轮 · 离线蒸馏器（P1-6）
==============================

读取 HITL 偏好对（workspace/_training_data/multimedia_review_pairs.jsonl），
为每个审核闸口蒸馏一个**轻量自动审核策略**，落盘 review_policy.json。

策略形态：规则 + 词面对比（lexical contrast）
  - 统计 chosen / rejected 两侧的词元频次差（log-odds 比），得到带符号的判别词表
  - 阈值取两侧得分分布的中点，使得在训练对上分类错误最小
  - 输出可被 review_flywheel.suggest() 直接消费（影子模式，不拦主流程）

为什么不用 embedding：
  当前 DashScope embedding key 不可用；且词面策略在 prompt 类闸口上已足够——
  prompt 的「好/坏」大量体现为具体词的出现（解剖细节、多余修饰、风格冲突词）。
  后续 embedding 通道恢复后可在此处平滑替换为向量质心，接口不变。

用法：
  python scripts/distill_reviewer.py --stats
  python scripts/distill_reviewer.py --min-pairs 8
  python scripts/distill_reviewer.py --out workspace/_training_data/review_policy.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 允许直接以脚本方式运行
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

try:
    from agent.multimedia import review_flywheel as fw
except Exception:
    fw = None  # 允许在无项目环境下降级为独立模式

# 词元切分：英文按词 + 中文按 2-gram（无需分词器依赖）
_EN_WORD = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    toks = [t for t in _EN_WORD.findall(text) if len(t) >= 3]
    cjk = "".join(ch for ch in text if _CJK.match(ch))
    toks += [cjk[i:i + 2] for i in range(len(cjk) - 1)]
    return toks


def _score(text: str, weights: Dict[str, float]) -> float:
    return sum(w for t, w in weights.items() if t in (text or "").lower())


def build_gate_model(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """单闸口：由偏好对学出带符号判别词表 + 阈值。"""
    pos = Counter()  # chosen 侧
    neg = Counter()  # rejected 侧
    n_pos = n_neg = 0
    for p in pairs:
        rej = (p.get("rejected") or {}).get("text", "")
        cho = (p.get("chosen") or {}).get("text", "")
        if not rej or not cho:
            continue
        for t in set(tokenize(cho)):
            pos[t] += 1
        for t in set(tokenize(rej)):
            neg[t] += 1
        n_pos += 1
        n_neg += 1

    if n_pos == 0:
        return {}

    # log-odds 比（带平滑）：chosen 侧倾向 → 正分；rejected 侧倾向 → 负分
    weights: Dict[str, float] = {}
    total_pos, total_neg = sum(pos.values()) or 1, sum(neg.values()) or 1
    for tok in set(pos) | set(neg):
        pp = (pos[tok] + 0.5) / (total_pos + 0.5)
        pn = (neg[tok] + 0.5) / (total_neg + 0.5)
        lo = math.log(pp / pn)
        # 只在词元确实偏向某一侧时保留
        if abs(lo) >= 0.35:
            weights[tok] = round(lo, 4)  # chosen 侧为正、rejected 侧为负

    if not weights:
        return {}

    # 阈值：取「使分类准确率最高的切点」；并列时取最靠近两簇均值的切点。
    # 候选切点 = 相邻得分的中点（含两端外推），避免直接取某个样本得分。
    pos_scores = [_score((p.get("chosen") or {}).get("text", ""), weights) for p in pairs]
    neg_scores = [_score((p.get("rejected") or {}).get("text", ""), weights) for p in pairs]
    all_scores = sorted(set(pos_scores + neg_scores))
    if not all_scores:
        thr = 0.0
        train_acc = 0.0
    elif len(all_scores) == 1:
        thr = all_scores[0]
        train_acc = 1.0
    else:
        cands = [all_scores[0] - 0.5]
        cands += [(all_scores[i] + all_scores[i + 1]) / 2.0 for i in range(len(all_scores) - 1)]
        cands.append(all_scores[-1] + 0.5)
        mean_pos = sum(pos_scores) / len(pos_scores)
        mean_neg = sum(neg_scores) / len(neg_scores)
        mid = (mean_pos + mean_neg) / 2.0
        best_thr, best_acc, best_gap = cands[0], -1.0, 1e18
        for c in cands:
            acc = (sum(1 for s in pos_scores if s >= c) + sum(1 for s in neg_scores if s < c)) / len(pos_scores + neg_scores)
            gap = abs(c - mid)
            if acc > best_acc + 1e-9 or (abs(acc - best_acc) <= 1e-9 and gap < best_gap):
                best_acc, best_thr, best_gap = acc, c, gap
        thr = best_thr
        train_acc = best_acc

    # 取权重绝对值最大的词，作为「教师可读」的解释
    top = sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:14]
    reason = "chosen 侧倾向词：" + ", ".join(t for t, w in top if w > 0) + \
             "；rejected 侧倾向词：" + ", ".join(t for t, w in top if w < 0)

    return {
        "chosen_tokens": {t: w for t, w in weights.items() if w > 0},
        "rejected_tokens": {t: -w for t, w in weights.items() if w < 0},
        "threshold": round(thr, 4),
        "n_pairs": len(pairs),
        "train_acc": round(train_acc, 3),
        "reason": reason[:300],
    }


def distill(pairs: List[Dict[str, Any]], min_pairs: int = 5) -> Dict[str, Any]:
    by_gate: Dict[str, List[Dict[str, Any]]] = {}
    for p in pairs:
        by_gate.setdefault(p.get("gate", "?"), []).append(p)

    gates: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}
    for gate, ps in by_gate.items():
        # 仅文本可比闸口进入蒸馏
        textual = [p for p in ps
                   if (p.get("rejected") or {}).get("text") and (p.get("chosen") or {}).get("text")]
        if len(textual) < min_pairs:
            skipped[gate] = f"文本可比偏好对 {len(textual)} < {min_pairs}"
            continue
        model = build_gate_model(textual)
        if not model:
            skipped[gate] = "未学出判别词（区分度不足）"
            continue
        gates[gate] = model

    return {
        "schema": "multimedia.review_policy.v1",
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "min_pairs": min_pairs,
        "total_pairs": len(pairs),
        "gates": gates,
        "skipped": skipped,
        "note": "影子模式策略：仅建议不拦截。启用见 MULTIMEDIA_AUTO_REVIEW=shadow|<gate1,gate2>",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="审核飞轮离线蒸馏器")
    ap.add_argument("--pairs", default=None, help="偏好对 JSONL 路径")
    ap.add_argument("--out", default=None, help="策略输出路径 review_policy.json")
    ap.add_argument("--min-pairs", type=int, default=5, help="单闸口进入蒸馏的最少偏好对数")
    ap.add_argument("--stats", action="store_true", help="只打印统计，不蒸馏")
    ap.add_argument("--export-dpo", action="store_true", help="同时导出 DPO 格式")
    ap.add_argument("--export-sft", action="store_true", help="同时导出 SFT 格式")
    args = ap.parse_args()

    if fw is None:
        print("✗ 无法导入 agent.multimedia.review_flywheel（请从项目根运行）")
        return 2

    pairs_path = args.pairs or fw.paths()["pairs"]
    print(f"偏好对文件：{pairs_path}")
    pairs = fw._read_pairs(pairs_path)
    stats = fw.summarize(pairs_path)

    print(f"偏好对总数：{stats['total_pairs']}  涉及 thread：{stats['threads']}")
    print(f"来源分布：{stats['by_source']}")
    print("按闸口：")
    for g, d in sorted(stats["by_gate"].items()):
        print(f"  {g:<26} 合计 {d['total']:>4}  (编辑 {d['edit']} / 重写 {d['rewrite']})")

    if args.export_dpo:
        p = fw.export_dpo(pairs_path)
        print(f"✓ 已导出 DPO：{p}")
    if args.export_sft:
        p = fw.export_sft(pairs_path)
        print(f"✓ 已导出 SFT：{p}")

    if args.stats:
        return 0

    if not pairs:
        print("\n⚠ 暂无偏好对——先跑几轮带人审的生成任务（或 dry-run）积累数据。")
        print("  提示：dry-run 会自动注入若干人审决策，可用于打通飞轮链路验证。")
        return 0

    policy = distill(pairs, min_pairs=args.min_pairs)
    out_path = args.out or fw.paths()["policy"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(policy, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 策略已落盘：{out_path}")
    print(f"  可接管闸口：{list(policy['gates'].keys()) or '（暂无）'}")
    for g, m in policy["gates"].items():
        print(f"  · {g}: {m['n_pairs']} 对 · 阈值 {m['threshold']} · 判别词 {len(m['chosen_tokens']) + len(m['rejected_tokens'])}")
    for g, why in policy["skipped"].items():
        print(f"  – 跳过 {g}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
