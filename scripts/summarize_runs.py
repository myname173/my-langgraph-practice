"""
离线聚合 run_metrics.jsonl，输出真实运行指标。

用法（在项目根目录）：
    python scripts/summarize_runs.py
    python scripts/summarize_runs.py --path data/run_metrics.jsonl --top

指标含义（均来自真实埋点，非编造）：
  - 任务总数 / 出片成功率：finalized 且 outcome=='ok' 的占比
  - 关键帧自评一次通过率：cinematic_critic 事件中 pass_gates 且 rewrite_count==0 的任务占比
  - 平均重写次数 / 强制放行占比：因重写达上限被强制放行的任务占比
  - 人工审核弹出总数 / 打回率：human_gate 事件总数，action∈{rewrite,edit_prompt} 的占比
  - 角色一致性重生次数：portrait_regen 事件总数
  - 平均耗时：duration_s 均值

依赖：仅标准库。
"""
import argparse
import json
import os
import statistics
from collections import Counter

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_PATH = os.path.join(_PROJECT_ROOT, "data", "run_metrics.jsonl")


def load(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows):
    total = len(rows)
    s = {"task_total": total}
    if total == 0:
        return s

    ok = [r for r in rows if r.get("outcome") == "ok"]
    s["finalized_ok"] = len(ok)
    s["success_rate"] = round(len(ok) / total, 3)

    # 关键帧自评一次通过率
    first_pass = 0
    critic_tasks = 0
    forced = 0
    rewrite_counts = []
    for r in rows:
        critic = [e for e in r.get("events", []) if e["event"] == "cinematic_critic"]
        if not critic:
            continue
        critic_tasks += 1
        c0 = critic[0]
        rewrite_counts.append(c0.get("rewrite_count", 0))
        if c0.get("pass_gates") and c0.get("rewrite_count", 0) == 0:
            first_pass += 1
        if c0.get("forced"):
            forced += 1
    s["critic_tasks"] = critic_tasks
    s["critic_first_pass_rate"] = round(first_pass / critic_tasks, 3) if critic_tasks else None
    s["avg_rewrite"] = round(statistics.mean(rewrite_counts), 2) if rewrite_counts else None
    s["forced_pass_rate"] = round(forced / critic_tasks, 3) if critic_tasks else None

    # 人工审核 gate
    gates = [e for r in rows for e in r.get("events", []) if e["event"] == "human_gate"]
    s["human_gate_total"] = len(gates)
    s["human_gate_by_stage"] = dict(Counter(e["gate"] for e in gates))
    reject = [e for e in gates if e["action"] in ("rewrite", "edit_prompt")]
    s["human_gate_reject"] = len(reject)
    s["human_gate_reject_rate"] = round(len(reject) / len(gates), 3) if gates else None

    # 角色一致性重生
    regen = [e for r in rows for e in r.get("events", []) if e["event"] == "portrait_regen"]
    s["portrait_regen_total"] = len(regen)

    # 耗时
    durs = [r.get("duration_s") for r in rows if r.get("duration_s")]
    s["avg_duration_s"] = round(statistics.mean(durs), 1) if durs else None

    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=_DEFAULT_PATH)
    ap.add_argument("--top", action="store_true", help="仅打印汇总指标")
    args = ap.parse_args()

    rows = load(args.path)
    s = summarize(rows)

    if not args.top:
        print(f"已加载 {len(rows)} 条 run record（来源：{args.path}）")
    print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
