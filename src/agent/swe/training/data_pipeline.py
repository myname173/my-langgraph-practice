# src/agent/swe/training/data_pipeline.py
"""
Training Data Pipeline
======================
将 Agent 执行轨迹 + Skills 库转换为 LlamaFactory 可直接消费的训练数据格式。

流程：
  1. 加载历史轨迹（trajectories.jsonl）
  2. 批量计算奖励（reward_computer）
  3. Three-Index 增强：为每条轨迹注入 Repo Map / 符号上下文
  4. 导出：
     a. sft_success.jsonl    — 成功轨迹 → ShareGPT SFT 格式
     b. dpo_pairs.jsonl      — 成功/失败对比 → LlamaFactory DPO 格式
     c. grpo_all.jsonl       — 全量轨迹（带奖励分） → GRPO/KTO 格式
     d. skills_sft.jsonl     — skills/*.py → 代码生成 SFT 样本
  5. 生成 dataset_info.json  — LlamaFactory 数据集注册表

使用示例（命令行）：
  python -m src.agent.swe.training.data_pipeline --workspace ./workspace
  python -m src.agent.swe.training.data_pipeline --workspace ./workspace --format sft dpo

内部可直接调用：
  from src.agent.swe.training.data_pipeline import run_pipeline
  report = run_pipeline(workspace_dir=Path("./workspace"))
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.agent.swe.training.trajectory_logger import (
    TrajectoryRecord,
    get_training_dir,
    load_trajectories,
    trajectory_to_dpo_pair,
    trajectory_to_sharegpt,
)
from src.agent.swe.training.reward_computer import (
    annotate_trajectories,
    build_dpo_pairs,
    split_by_reward,
)

logger = logging.getLogger("SWE_DataPipeline")


# ==========================================
# Three-Index 上下文增强（离线调用）
# ==========================================

def _try_get_repo_map(workspace_dir: Path, query: str = "") -> str:
    """
    尝试从已构建的 Three-Index 获取 Repo Map。
    失败时静默降级，不阻塞 pipeline。
    """
    try:
        # 仅在已安装 code_index 时调用，避免 import 错误
        from src.agent.swe.code_index import get_repo_map_str, get_index_stats
        stats = get_index_stats()
        if stats.get("status") == "not_built":
            return ""
        return get_repo_map_str(query=query, max_tokens=1500)
    except Exception as e:
        logger.debug(f"Repo Map 获取失败（忽略）: {e}")
        return ""


def _try_get_symbol_context(symbol_name: str) -> str:
    """尝试获取符号上下文，失败时静默降级。"""
    try:
        from src.agent.swe.code_index import get_symbol_context_str
        return get_symbol_context_str(symbol_name)
    except Exception:
        return ""


def _enrich_with_index(record: TrajectoryRecord) -> str:
    """
    为单条轨迹生成 Three-Index 增强的上下文字符串。
    注入：Repo Map（按任务关键词相关性排序）。
    """
    # 提取任务关键词（前 50 字符作为查询）
    query = record.task_description[:50]
    repo_map = _try_get_repo_map(query=query)
    return repo_map


# ==========================================
# SFT 数据生成
# ==========================================

def build_sft_data(
    records: List[TrajectoryRecord],
    min_reward: float = 0.4,
    enrich_with_index: bool = True,
) -> List[Dict]:
    """
    将成功轨迹转换为 ShareGPT SFT 格式。
    仅包含 reward >= min_reward 的轨迹。
    """
    results = []
    skipped = 0
    for rec in records:
        if rec.reward < min_reward:
            skipped += 1
            continue
        repo_map = _enrich_with_index(rec) if enrich_with_index else ""
        sample = trajectory_to_sharegpt(rec, repo_map=repo_map, min_reward=min_reward)
        if sample:
            results.append(sample)
    logger.info(f"SFT 数据: {len(results)} 条有效样本，{skipped} 条因低奖励跳过")
    return results


# ==========================================
# DPO 数据生成
# ==========================================

def build_dpo_data(
    records: List[TrajectoryRecord],
    max_pairs: int = 500,
) -> List[Dict]:
    """
    构造 LlamaFactory DPO 格式训练对。
    LlamaFactory DPO 格式:
      {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    pairs = build_dpo_pairs(records, max_pairs=max_pairs)
    results = []
    for chosen, rejected in pairs:
        sample = trajectory_to_dpo_pair(chosen, rejected)
        if sample:
            # 去掉 _meta（LlamaFactory 不需要）
            clean = {k: v for k, v in sample.items() if not k.startswith("_")}
            results.append(clean)
    logger.info(f"DPO 数据: {len(results)} 对训练对")
    return results


# ==========================================
# GRPO 数据生成（带标量奖励）
# ==========================================

def build_grpo_data(records: List[TrajectoryRecord]) -> List[Dict]:
    """
    GRPO/KTO 格式：全量轨迹 + 标量奖励分。
    LlamaFactory GRPO 使用 DPO 格式，但这里我们输出扩展格式，
    供自定义 Trainer 使用。

    格式:
      {"prompt": "...", "response": "...", "reward": 0.8}
    """
    results = []
    for rec in records:
        if not rec.task_description:
            continue
        # 构建 response：工具调用摘要
        tool_summary_lines = [
            f"[{s.tool_name}] {'成功' if s.success else '失败'}"
            for s in rec.tool_call_steps[:10]
        ]
        response = "\n".join(tool_summary_lines)
        if rec.status == "success":
            response += "\n\nTASK_COMPLETED"

        results.append({
            "prompt": rec.task_description,
            "response": response,
            "reward": rec.reward,
            # 以下字段供自定义 Trainer 使用
            "trajectory_id": rec.trajectory_id,
            "status": rec.status,
            "test_passed": rec.test_passed,
        })
    logger.info(f"GRPO 数据: {len(results)} 条轨迹")
    return results


# ==========================================
# Skills SFT 数据生成
# ==========================================

def build_skills_sft_data(skills_dir: Path) -> List[Dict]:
    """
    将 skills/*.py 格式化为代码生成 SFT 样本。
    Skills 是高质量的人工验证代码，作为正样本注入 SFT 效果极好。

    格式（ShareGPT 单轮）:
    {
      "conversations": [
        {"from": "system", "value": "你是一个代码生成专家..."},
        {"from": "human",  "value": "生成一个 {描述} 的可复用脚本"},
        {"from": "gpt",    "value": "完整 Python 脚本内容"}
      ]
    }
    """
    results = []
    skill_files = list(skills_dir.glob("*.py"))

    for f in skill_files:
        try:
            content = f.read_text(encoding="utf-8")
            # 解析 YAML frontmatter 获取描述
            description = "一个可复用的 Python 技能脚本"
            category = "misc"
            in_fm = False
            for line in content.splitlines()[:15]:
                stripped = line.strip()
                if stripped == "# ---":
                    in_fm = not in_fm
                    continue
                if in_fm and "description: " in stripped:
                    description = stripped.split("description: ", 1)[1]
                if in_fm and "category: " in stripped:
                    category = stripped.split("category: ", 1)[1]

            results.append({
                "conversations": [
                    {
                        "from": "system",
                        "value": (
                            "你是一个代码生成专家。生成规范的、带 YAML frontmatter 和 "
                            "argparse 的可复用 Python 脚本，包含 __test__() 自验证函数。"
                        ),
                    },
                    {
                        "from": "human",
                        "value": (
                            f"请生成一个 [{category}] 类别的可复用 Python 技能脚本，"
                            f"功能描述：{description}"
                        ),
                    },
                    {"from": "gpt", "value": content},
                ],
                "_meta": {"source": "skills_library", "filename": f.name, "category": category},
            })
        except Exception as e:
            logger.debug(f"技能文件解析失败（跳过）: {f.name} — {e}")

    logger.info(f"Skills SFT 数据: {len(results)} 条（来自 {len(skill_files)} 个技能文件）")
    return results


# ==========================================
# dataset_info.json 生成（LlamaFactory 数据集注册）
# ==========================================

def generate_dataset_info(training_dir: Path, formats: List[str]) -> Dict:
    """
    生成 LlamaFactory 的 dataset_info.json。
    注册所有导出的数据集，供 llamafactory-cli 直接使用。
    """
    info = {}

    if "sft" in formats:
        info["swe_agent_sft"] = {
            "file_name": "sft_success.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system",
                "function_tag": "function_call",
                "observation_tag": "observation",
            },
        }

    if "dpo" in formats:
        info["swe_agent_dpo"] = {
            "file_name": "dpo_pairs.jsonl",
            "ranking": True,
            "columns": {
                "prompt": "prompt",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        }

    if "grpo" in formats:
        info["swe_agent_grpo"] = {
            "file_name": "grpo_all.jsonl",
            "columns": {
                "prompt": "prompt",
                "response": "response",
                "reward": "reward",
            },
        }

    if "skills" in formats:
        info["swe_skills_sft"] = {
            "file_name": "skills_sft.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system",
            },
        }

    return info


# ==========================================
# 主 Pipeline
# ==========================================

def run_pipeline(
    workspace_dir: Path,
    formats: Optional[List[str]] = None,
    min_reward_sft: float = 0.4,
    max_dpo_pairs: int = 500,
    enrich_with_index: bool = True,
) -> Dict:
    """
    执行完整数据 Pipeline。

    参数:
      workspace_dir     工作区根目录（包含 _training_data/ 和 skills/）
      formats           要导出的格式列表，默认全部 ["sft", "dpo", "grpo", "skills"]
      min_reward_sft    SFT 数据的最低奖励阈值
      max_dpo_pairs     DPO 数据的最大对数
      enrich_with_index 是否用 Three-Index 增强训练指令

    返回: pipeline 执行报告字典
    """
    if formats is None:
        formats = ["sft", "dpo", "grpo", "skills"]

    training_dir = get_training_dir(workspace_dir)
    skills_dir = workspace_dir / "skills"
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "training_dir": str(training_dir),
        "formats": formats,
        "counts": {},
        "files": [],
    }

    # ── 1. 加载并标注奖励 ────────────────────────────────
    raw_records = load_trajectories(workspace_dir)
    logger.info(f"加载轨迹: {len(raw_records)} 条")

    if not raw_records and "skills" not in formats:
        logger.warning("轨迹数据为空，仅可导出 skills_sft 数据")
        report["warning"] = "无轨迹数据"

    records = annotate_trajectories(raw_records)
    report["counts"]["total_trajectories"] = len(records)

    # ── 2. 导出 SFT ──────────────────────────────────────
    if "sft" in formats and records:
        sft_data = build_sft_data(records, min_reward=min_reward_sft, enrich_with_index=enrich_with_index)
        sft_path = training_dir / "sft_success.jsonl"
        _write_jsonl(sft_data, sft_path)
        report["counts"]["sft_samples"] = len(sft_data)
        report["files"].append(str(sft_path))

    # ── 3. 导出 DPO ──────────────────────────────────────
    if "dpo" in formats and records:
        dpo_data = build_dpo_data(records, max_pairs=max_dpo_pairs)
        dpo_path = training_dir / "dpo_pairs.jsonl"
        _write_jsonl(dpo_data, dpo_path)
        report["counts"]["dpo_pairs"] = len(dpo_data)
        report["files"].append(str(dpo_path))

    # ── 4. 导出 GRPO ─────────────────────────────────────
    if "grpo" in formats and records:
        grpo_data = build_grpo_data(records)
        grpo_path = training_dir / "grpo_all.jsonl"
        _write_jsonl(grpo_data, grpo_path)
        report["counts"]["grpo_samples"] = len(grpo_data)
        report["files"].append(str(grpo_path))

    # ── 5. 导出 Skills SFT ───────────────────────────────
    if "skills" in formats and skills_dir.exists():
        skills_data = build_skills_sft_data(skills_dir)
        skills_path = training_dir / "skills_sft.jsonl"
        _write_jsonl(skills_data, skills_path)
        report["counts"]["skills_samples"] = len(skills_data)
        report["files"].append(str(skills_path))

    # ── 6. 生成 dataset_info.json ────────────────────────
    dataset_info = generate_dataset_info(training_dir, formats)
    info_path = training_dir / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    report["files"].append(str(info_path))

    logger.info(f"✅ Pipeline 完成: {report['counts']}")
    return report


def _write_jsonl(data: List[Dict], path: Path) -> None:
    """写入 JSONL 文件，过滤掉内部 _meta 字段。"""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            # 去除 _meta（LlamaFactory 不识别该字段）
            clean = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    logger.debug(f"写入 {len(data)} 条 → {path}")


# ==========================================
# CLI 入口
# ==========================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="SWE Agent 训练数据 Pipeline")
    parser.add_argument("--workspace", default="./workspace", help="工作区目录")
    parser.add_argument(
        "--format", nargs="+", choices=["sft", "dpo", "grpo", "skills"],
        default=["sft", "dpo", "grpo", "skills"], dest="formats",
    )
    parser.add_argument("--min-reward", type=float, default=0.4)
    parser.add_argument("--max-dpo-pairs", type=int, default=500)
    parser.add_argument("--no-index-enrich", action="store_true")
    args = parser.parse_args()

    report = run_pipeline(
        workspace_dir=Path(args.workspace),
        formats=args.formats,
        min_reward_sft=args.min_reward,
        max_dpo_pairs=args.max_dpo_pairs,
        enrich_with_index=not args.no_index_enrich,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
