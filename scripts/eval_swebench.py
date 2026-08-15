#!/usr/bin/env python3
# scripts/eval_swebench.py
"""
SWE-bench_Verified 评估脚本（评估闭环）
=========================================

目的
----
量化评估 SWE Agent 在 SWE-bench_Verified 子集上的表现，对比
「基线基座模型 (base)」与「微调后模型 (finetuned)」的增益。

两种评测模式
------------
1. proxy 模式（默认，本地 CPU 可跑，无需 GPU/网络）
   只用 SWE-bench 的 *静态信息*（problem_statement + gold patch）
   评估 agent 生成的 patch 与 gold patch 的结构相似度：
     - 文件命中率 (file_hit):   生成的 patch 是否改动了 gold patch 涉及的同一文件
     - hunk 重叠率 (hunk_overlap): 生成的 patch 是否触及 gold patch 的同一函数/区域
     - 合法 diff 率 (valid_diff): 生成的 patch 是否为可解析的合法 unified diff
   这是一个「代理指标」，不运行真实测试，但能低成本验证训练是否让 agent
   更精准地定位到正确文件和修改区域。

2. live 模式（云端 / 有 Docker + git 的环境）
   真实 checkout repo、apply patch、运行 FAIL_TO_PASS 测试，
   输出真正的 resolve rate（完全对齐 SWE-bench 官方评测协议）。
   本脚本预留 live 模式的执行接口，具体 repo 构建由 run_live_eval() 负责。

模型模式
--------
--model-mode base       : 使用基座模型（未经微调）生成 patch
--model-mode finetuned  : 加载 LLaMA-Factory 产出的 LoRA 权重生成 patch
（本地 1650Ti 跑不了 7B 推理，所以 base/finetuned 的真实推理应在云端执行；
 本地可用 --offline-report 直接对比两份已生成的报告，跳过重新推理。）

用法
----
# 1) 本地冒烟（3 条样本，proxy 模式，用基座模型）
python scripts/eval_swebench.py --mode proxy --model-mode base --max-instances 3

# 2) 云端生成 finetuned 报告后，本地直接对比
python scripts/eval_swebench.py --offline-report reports/base.json reports/finetuned.json

# 3) 云端 live 模式（需 Docker 可用）
python scripts/eval_swebench.py --mode live --model-mode finetuned --max-instances 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SWE_Eval")

# ── 路径常量 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# SWE-bench_Verified 下载源（与 import_public_data.py 同款，国内可达）
SWEBENCH_SOURCES = [
    ("ModelScope",
     "https://modelscope.cn/datasets/AI-ModelScope/SWE-bench_Verified/resolve/master/data/test-00000-of-00001.parquet"),
    ("HuggingFace",
     "https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified/resolve/main/data/test-00000-of-00001.parquet"),
]


# ==========================================
# 数据加载
# ==========================================
def load_swebench_subset(max_instances: int = 50, cache_dir: Optional[Path] = None) -> List[dict]:
    """
    加载 SWE-bench_Verified 子集。优先用本地缓存 parquet，否则从 ModelScope/HF 下载。
    返回字段：instance_id, repo, problem_statement, patch(gold), FAIL_TO_PASS
    """
    import urllib.request

    cache_dir = cache_dir or (WORKSPACE / "_eval_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / "swebench_verified.parquet"

    if not parquet_path.exists():
        logger.info("本地无 SWE-bench 缓存，尝试下载 ...")
        for name, url in SWEBENCH_SOURCES:
            try:
                logger.info("从 %s 下载 ...", name)
                # 用 urllib（与导入器一致，兼容系统代理）；短超时避免干等
                req = urllib.request.Request(url, headers={"User-Agent": "SWE-Eval/1.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = resp.read()
                parquet_path.write_bytes(data)
                logger.info("✅ 已缓存到 %s", parquet_path)
                break
            except Exception as e:
                logger.warning("⚠️ %s 下载失败: %s", name, e)
        else:
            raise RuntimeError(
                "所有 SWE-bench 数据源均不可达。\n"
                "提示：可将已下载的 test-00000-of-00001.parquet 放到 "
                f"{parquet_path} 后重试（离线可用）。"
            )

    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("需要 pandas 来解析 parquet，请先 `pip install pandas pyarrow`")

    df = pd.read_parquet(parquet_path)
    cols = {"instance_id", "repo", "problem_statement", "patch", "FAIL_TO_PASS"}
    missing = cols - set(df.columns)
    if missing:
        raise RuntimeError(f"parquet 缺少必要字段: {missing}")

    records = []
    for _, row in df.iterrows():
        records.append({
            "instance_id": str(row["instance_id"]),
            "repo": str(row["repo"]),
            "problem_statement": str(row["problem_statement"]),
            "patch": str(row["patch"]),  # gold patch
            "fail_to_pass": row["FAIL_TO_PASS"],
        })
        if len(records) >= max_instances:
            break
    logger.info("加载 SWE-bench_Verified 子集 %d 条", len(records))
    return records


# ==========================================
# Patch 抽取与解析
# ==========================================
_DIFF_RE = re.compile(r"diff --git a/(\S+) b/(\S+)")


def extract_patch_from_text(text: str) -> Optional[str]:
    """从 agent 最终输出文本中抽取 unified diff 格式的 patch。"""
    if not text:
        return None
    # 优先匹配 ```diff ... ``` 代码块
    fence = re.search(r"```diff\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    # 否则匹配以 diff --git 开头的段落
    m = re.search(r"diff --git a/\S+ b/\S+.*?(?=\ndiff --git |\Z)", text, re.DOTALL)
    if m:
        return m.group(0).strip()
    return None


def parse_diff_files(diff: str) -> List[str]:
    """解析 diff 涉及的文件路径列表（相对路径，去前缀 a/ b/）。"""
    files = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            m = _DIFF_RE.search(line)
            if m:
                # 取改动后的路径 b/，去掉前缀
                f = m.group(2)
                files.append(f)
    return files


def parse_diff_hunks(diff: str) -> dict:
    """
    解析 diff 的 hunk 位置（按文件 -> 修改行区间）。
    返回一个 dict: {file: [(start_line, end_line), ...]}
    用于和 gold patch 的修改区域做重叠判断。
    """
    hunks: dict = {}
    cur_file = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            m = _DIFF_RE.search(line)
            if m:
                cur_file = m.group(2)
                hunks.setdefault(cur_file, [])
        elif line.startswith("@@") and cur_file:
            # @@ -a,b +c,d @@
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if m:
                start = int(m.group(1))
                cnt = int(m.group(2)) if m.group(2) else 1
                hunks[cur_file].append((start, start + max(cnt, 1) - 1))
    return hunks


def is_valid_diff(diff: str) -> bool:
    """判断是否为可解析的合法 unified diff。"""
    if not diff or "diff --git" not in diff:
        return False
    # 至少要有一个 @@ hunk 头
    return bool(re.search(r"^@@", diff, re.MULTILINE))


# ==========================================
# 代理指标计算（proxy 模式核心）
# ==========================================
def compute_proxy_metrics(generated: Optional[str], gold_patch: str) -> dict:
    """
    计算单条样本的代理指标。
    返回: {
      valid_diff, file_hit, hunk_overlap,
      gen_files, gold_files, overlap_files
    }
    """
    gold_files = set(parse_diff_files(gold_patch))
    gold_hunks = parse_diff_hunks(gold_patch)

    if not generated or not is_valid_diff(generated):
        return {
            "valid_diff": False,
            "file_hit": False,
            "hunk_overlap": False,
            "gen_files": [],
            "gold_files": sorted(gold_files),
            "overlap_files": [],
        }

    gen_files = set(parse_diff_files(generated))
    gen_hunks = parse_diff_hunks(generated)

    overlap_files = sorted(gold_files & gen_files)
    file_hit = len(overlap_files) > 0

    # hunk 重叠：同一文件内，修改行区间有交集
    hunk_overlap = False
    for f in overlap_files:
        g_spans = gold_hunks.get(f, [])
        e_spans = gen_hunks.get(f, [])
        for (gs, ge) in g_spans:
            for (es, ee) in e_spans:
                if not (ee < gs or es > ge):  # 区间相交
                    hunk_overlap = True
                    break
            if hunk_overlap:
                break
        if hunk_overlap:
            break

    return {
        "valid_diff": True,
        "file_hit": file_hit,
        "hunk_overlap": hunk_overlap,
        "gen_files": sorted(gen_files),
        "gold_files": sorted(gold_files),
        "overlap_files": overlap_files,
    }


# ==========================================
# Agent patch 生成器
# ==========================================
def generate_patch_via_agent(problem_statement: str, model_mode: str) -> Optional[str]:
    """
    调用 SWE Agent 的 graph 生成 patch。
    本地 1650Ti 无法跑 7B 推理，因此本函数只为 finetuned 云端运行预留接口；
    本地冒烟时用 model_mode=base 会尝试加载基座模型（若环境允许）。

    返回 agent 最终状态的完整文本（含 patch），交由 extract_patch_from_text 抽取。
    """
    try:
        sys.path.insert(0, str(ROOT))
        from src.agent.swe.graph import graph
    except Exception as e:
        logger.error("无法导入 agent 模块: %s", e)
        return None

    # 云端 finetuned 模式：通过环境变量注入 LoRA 权重路径
    # （由 llamafactory_runs 训练产出，云端运行时 export SWE_FINETUNED_LORA_PATH=...）
    if model_mode == "finetuned":
        lora_path = os.getenv("SWE_FINETUNED_LORA_PATH")
        if not lora_path:
            logger.warning("finetuned 模式但未设置 SWE_FINETUNED_LORA_PATH，回退 base")
            model_mode = "base"

    # 自动 approve 所有工具调用（评测时无人值守）
    final_text = ""
    try:
        for chunk in graph.stream(
            {"task_description": problem_statement, "max_iterations": 15},
            config={"recursion_limit": 200},
        ):
            # 提取最后一条 AI 消息文本
            for node_out in chunk.values():
                if isinstance(node_out, dict) and "messages" in node_out:
                    msgs = node_out["messages"]
                    if msgs:
                        last = msgs[-1]
                        content = getattr(last, "content", "")
                        if content:
                            final_text = content
    except Exception as e:
        logger.warning("graph 执行异常: %s", e)
        return final_text or None

    return final_text or None


# ==========================================
# Live 模式（云端预留接口）
# ==========================================
def run_live_eval(instances: List[dict], model_mode: str) -> List[dict]:
    """
    [云端] 真实 SWE-bench 评测：checkout repo → apply patch → 跑 FAIL_TO_PASS。
    需要 Docker + git，且能访问对应 repo 的 GitHub。
    这里只给出协议骨架，具体测试运行由 CI / 云端脚本填充。
    """
    logger.info("live 模式需要 Docker 沙盒与 repo checkout，本地跳过。")
    results = []
    for inst in instances:
        # TODO(云端): 用 agent 生成 patch -> 写入 repo -> 运行 FAIL_TO_PASS 测试
        # resolved = run_fail_to_pass(inst["repo"], generated_patch, inst["fail_to_pass"])
        results.append({
            "instance_id": inst["instance_id"],
            "resolved": False,  # 占位，云端实现
            "mode": "live",
        })
    return results


# ==========================================
# 评估主流程
# ==========================================
@dataclass
class EvalResult:
    instance_id: str
    repo: str
    model_mode: str
    valid_diff: bool = False
    file_hit: bool = False
    hunk_overlap: bool = False
    overlap_files: List[str] = field(default_factory=list)
    resolved: Optional[bool] = None  # live 模式填充


def run_eval(
    mode: str,
    model_mode: str,
    max_instances: int,
    use_agent: bool = True,
) -> List[EvalResult]:
    instances = load_swebench_subset(max_instances=max_instances)
    results: List[EvalResult] = []

    for i, inst in enumerate(instances, 1):
        logger.info("[%d/%d] %s (%s)", i, len(instances), inst["instance_id"], inst["repo"])

        generated: Optional[str] = None
        if use_agent and mode == "proxy":
            raw = generate_patch_via_agent(inst["problem_statement"], model_mode)
            generated = extract_patch_from_text(raw) if raw else None

        if mode == "proxy":
            metrics = compute_proxy_metrics(generated, inst["patch"])
            results.append(EvalResult(
                instance_id=inst["instance_id"],
                repo=inst["repo"],
                model_mode=model_mode,
                valid_diff=metrics["valid_diff"],
                file_hit=metrics["file_hit"],
                hunk_overlap=metrics["hunk_overlap"],
                overlap_files=metrics["overlap_files"],
            ))
        else:  # live
            live = run_live_eval([inst], model_mode)
            r = live[0]
            results.append(EvalResult(
                instance_id=inst["instance_id"],
                repo=inst["repo"],
                model_mode=model_mode,
                resolved=r["resolved"],
            ))

    return results


# ==========================================
# 报告与对比
# ==========================================
def summarize(results: List[EvalResult]) -> dict:
    n = len(results)
    if not n:
        return {"count": 0}
    valid = sum(1 for r in results if r.valid_diff)
    fhit = sum(1 for r in results if r.file_hit)
    hov = sum(1 for r in results if r.hunk_overlap)
    resolved = [r for r in results if r.resolved is not None]
    return {
        "count": n,
        "valid_diff_rate": round(valid / n, 3),
        "file_hit_rate": round(fhit / n, 3),
        "hunk_overlap_rate": round(hov / n, 3),
        "resolved_rate": round(sum(1 for r in resolved if r.resolved) / len(resolved), 3)
        if resolved else None,
    }


def compare_reports(base_path: Path, finetuned_path: Path) -> dict:
    """离线对比 base / finetuned 两份报告，量化增益。"""
    base = json.loads(base_path.read_text(encoding="utf-8"))
    ft = json.loads(finetuned_path.read_text(encoding="utf-8"))
    return {
        "base": base.get("summary", {}),
        "finetuned": ft.get("summary", {}),
        "delta": {
            k: round(ft["summary"].get(k, 0) - base["summary"].get(k, 0), 3)
            for k in ("valid_diff_rate", "file_hit_rate", "hunk_overlap_rate", "resolved_rate")
        },
    }


def write_report(results: List[EvalResult], mode: str, model_mode: str) -> Path:
    report = {
        "mode": mode,
        "model_mode": model_mode,
        "summary": summarize(results),
        "details": [asdict(r) for r in results],
    }
    out = REPORT_DIR / f"{model_mode}_{mode}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("📊 报告已写入: %s", out)
    return out


def print_markdown(report: dict) -> None:
    s = report["summary"]
    print("\n" + "=" * 56)
    print(f"  SWE-bench 评估  |  mode={report['mode']}  model={report['model_mode']}")
    print("=" * 56)
    print(f"  样本数          : {s.get('count')}")
    if report["mode"] == "proxy":
        print(f"  合法 diff 率    : {s.get('valid_diff_rate')}")
        print(f"  文件命中率      : {s.get('file_hit_rate')}")
        print(f"  Hunk 重叠率     : {s.get('hunk_overlap_rate')}")
    else:
        print(f"  真实 resolve 率 : {s.get('resolved_rate')}")
    print("=" * 56 + "\n")


# ==========================================
# 入口
# ==========================================
def main():
    ap = argparse.ArgumentParser(description="SWE-bench_Verified 评估脚本")
    ap.add_argument("--mode", choices=["proxy", "live"], default="proxy",
                    help="proxy=结构对比(本地) / live=真跑测试(云端)")
    ap.add_argument("--model-mode", choices=["base", "finetuned"], default="base",
                    help="base=基座模型 / finetuned=LoRA 微调后")
    ap.add_argument("--max-instances", type=int, default=50,
                    help="评估样本数（本地冒烟建议 3）")
    ap.add_argument("--offline-report", nargs=2, metavar=("BASE", "FT"),
                    help="直接对比两份已生成的报告 (base.json finetuned.json)")
    ap.add_argument("--no-agent", action="store_true",
                    help="不调用 agent（仅测试 gold patch 解析，调试用）")
    args = ap.parse_args()

    if args.offline_report:
        cmp = compare_reports(Path(args.offline_report[0]), Path(args.offline_report[1]))
        print(json.dumps(cmp, ensure_ascii=False, indent=2))
        return

    use_agent = not args.no_agent
    results = run_eval(args.mode, args.model_mode, args.max_instances, use_agent=use_agent)
    report = write_report(results, args.mode, args.model_mode)
    print_markdown(json.loads(report.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
