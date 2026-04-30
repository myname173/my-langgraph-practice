# src/agent/swe/training/trajectory_logger.py
"""
Trajectory Logger
=================
将每次 Agent 执行过程结构化地捕获为"轨迹记录"，持久化到 JSONL 文件。

设计原则：
  - 从 AgentState.messages 重建完整执行轨迹（tool_calls + tool_results + llm_outputs）
  - 带有基础 outcome 字段（test_passed / status / iteration_count）
  - 不侵入主流程节点，由 graph.py 中专门的 trajectory_export_node 异步调用
  - 保存格式与 LlamaFactory ShareGPT/DPO 格式兼容

目录结构：
  workspace/
  └── _training_data/
      ├── trajectories.jsonl       ← 原始轨迹（带 reward，all-in-one）
      ├── sft_success.jsonl        ← 成功轨迹的 ShareGPT SFT 格式
      ├── dpo_pairs.jsonl          ← 成功/失败 对比对（DPO/GRPO）
      └── skills_sft.jsonl         ← skills/*.py 格式化为 SFT 样本
"""

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger("SWE_TrajectoryLogger")


# ==========================================
# 数据结构
# ==========================================

@dataclass
class ToolCallStep:
    """单次工具调用的完整记录（call + result 配对）"""
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: str
    success: bool           # 结果是否不含 Error 前缀
    result_len: int         # 结果长度，用于计算信息量


@dataclass
class TrajectoryRecord:
    """一次完整 Agent 执行的轨迹快照"""
    trajectory_id: str
    task_description: str
    timestamp: str

    # 执行过程
    tool_call_steps: List[ToolCallStep] = field(default_factory=list)
    # 精简的 llm 推理摘要（仅保留关键思考步骤，不含大段代码）
    llm_reasoning_snippets: List[str] = field(default_factory=list)

    # Outcome 指标（由 graph state 直接填充）
    status: str = "unknown"          # success / failed / unknown
    test_passed: bool = False
    iteration_count: int = 0
    max_iterations: int = 25
    completed_tasks: List[str] = field(default_factory=list)
    summary: str = ""

    # 奖励（由 reward_computer 填充）
    reward: float = 0.0
    reward_breakdown: Dict[str, float] = field(default_factory=dict)

    # 技能库产出（由 evolution 阶段填充）
    evolved_skill_name: str = ""
    evolved_skill_content: str = ""


# ==========================================
# 从 AgentState.messages 重建轨迹
# ==========================================

def _is_error_result(text: str) -> bool:
    """判断工具输出是否为错误结果。"""
    error_prefixes = ("Error:", "Error ", "读取网页失败", "列出技能失败", "代码搜索异常")
    return any(text.startswith(p) for p in error_prefixes)


def _truncate(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"...[截断，原长 {len(text)} 字符]"


def extract_trajectory_from_state(state: dict) -> TrajectoryRecord:
    """
    从 AgentState 字典重建 TrajectoryRecord。
    核心逻辑：遍历 messages，配对 AIMessage.tool_calls 与后续 ToolMessage。
    """
    traj_id = str(uuid.uuid4())[:12]
    messages = list(state.get("messages", []))

    tool_steps: List[ToolCallStep] = []
    llm_snippets: List[str] = []

    i = 0
    while i < len(messages):
        msg = messages[i]
        msg_type = getattr(msg, "type", "")

        # AI 消息：可能含 tool_calls + 推理文本
        if msg_type == "ai":
            ai_content = str(getattr(msg, "content", ""))

            # 提取 LLM 推理片段（排除纯 TASK_COMPLETED 和空内容）
            if ai_content and len(ai_content) > 20 and "TASK_COMPLETED" not in ai_content:
                # 只保留前 200 字符作为推理摘要
                llm_snippets.append(_truncate(ai_content, 200))

            # 提取 tool_calls，与后续 ToolMessage 配对
            tc_list = getattr(msg, "tool_calls", []) or []
            for tc in tc_list:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                tc_id = tc.get("id", "")

                # 向后寻找匹配的 ToolMessage
                tool_result_text = ""
                for j in range(i + 1, min(i + len(tc_list) + 3, len(messages))):
                    candidate = messages[j]
                    if getattr(candidate, "type", "") == "tool":
                        # 按 tool_call_id 匹配（优先）或顺序匹配
                        if (
                            getattr(candidate, "tool_call_id", None) == tc_id
                            or not tc_id
                        ):
                            tool_result_text = str(getattr(candidate, "content", ""))
                            break

                # 清理 args 中的大段代码（节省存储）
                clean_args = {}
                for k, v in tool_args.items():
                    if isinstance(v, str) and len(v) > 500:
                        clean_args[k] = _truncate(v, 200)
                    else:
                        clean_args[k] = v

                success = not _is_error_result(tool_result_text)
                tool_steps.append(ToolCallStep(
                    tool_name=tool_name,
                    tool_args=clean_args,
                    tool_result=_truncate(tool_result_text, 400),
                    success=success,
                    result_len=len(tool_result_text),
                ))

        i += 1

    return TrajectoryRecord(
        trajectory_id=traj_id,
        task_description=state.get("task_description", ""),
        timestamp=datetime.utcnow().isoformat() + "Z",
        tool_call_steps=tool_steps,
        llm_reasoning_snippets=llm_snippets,
        status=state.get("status", "unknown"),
        test_passed=state.get("test_passed", False),
        iteration_count=state.get("iteration_count", 0),
        max_iterations=state.get("max_iterations", 25),
        completed_tasks=list(state.get("completed_tasks", [])),
        summary=state.get("summary", ""),
    )


# ==========================================
# 持久化
# ==========================================

def get_training_dir(workspace_dir: Path) -> Path:
    d = workspace_dir / "_training_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_trajectory(record: TrajectoryRecord, workspace_dir: Path) -> Path:
    """追加写入轨迹到 trajectories.jsonl，返回写入路径。"""
    training_dir = get_training_dir(workspace_dir)
    out_path = training_dir / "trajectories.jsonl"
    try:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        logger.info(f"✅ 轨迹已保存: {record.trajectory_id} → {out_path}")
    except Exception as e:
        logger.warning(f"⚠️ 轨迹保存失败（不影响主流程）: {e}")
    return out_path


def load_trajectories(workspace_dir: Path) -> List[TrajectoryRecord]:
    """从 trajectories.jsonl 读取所有历史轨迹。"""
    path = get_training_dir(workspace_dir) / "trajectories.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                # 重建 ToolCallStep 对象
                steps = [ToolCallStep(**s) for s in d.pop("tool_call_steps", [])]
                rec = TrajectoryRecord(**d)
                rec.tool_call_steps = steps
                records.append(rec)
            except Exception as e:
                logger.debug(f"轨迹行解析失败（跳过）: {e}")
    return records


# ==========================================
# 轨迹转 ShareGPT 格式（供 LlamaFactory SFT）
# ==========================================

def trajectory_to_sharegpt(
    record: TrajectoryRecord,
    system_prompt: str = "",
    repo_map: str = "",
    min_reward: float = 0.3,
) -> Optional[Dict]:
    """
    将单条轨迹转换为 LlamaFactory ShareGPT 格式。
    仅导出 reward >= min_reward 的轨迹（质量过滤）。

    ShareGPT 格式:
    {
      "conversations": [
        {"from": "system", "value": "..."},
        {"from": "human",  "value": "task + repo_map"},
        {"from": "gpt",    "value": "...reasoning..."},
        {"from": "function_call", "value": "{\"name\":..., \"arguments\":...}"},
        {"from": "observation",   "value": "tool_result"},
        ...
        {"from": "gpt", "value": "TASK_COMPLETED"}
      ],
      "tools": "[{\"name\": \"execute_command\", ...}]"
    }
    """
    if record.reward < min_reward:
        return None
    if not record.task_description:
        return None

    convs = []

    # System prompt
    sp = system_prompt or (
        "你是一个顶级的全栈开发工程师 Agent。"
        "根据任务描述，使用可用工具逐步完成编码任务。"
        "完成后回复 TASK_COMPLETED。"
    )
    convs.append({"from": "system", "value": sp})

    # Human turn: task + repo_map 上下文注入（Three-Index 增强）
    human_content = record.task_description
    if repo_map and len(repo_map) > 20:
        human_content += f"\n\n【代码库结构参考（Repo Map）】\n{repo_map[:2000]}"
    if record.completed_tasks:
        human_content += f"\n\n【参考：已完成的分步任务】\n" + "\n".join(
            f"- {t}" for t in record.completed_tasks
        )
    convs.append({"from": "human", "value": human_content})

    # 展开工具调用步骤
    for step in record.tool_call_steps:
        # gpt: 触发工具调用的推理（简化为工具调用本身，实际推理在 llm_reasoning_snippets）
        tc_json = json.dumps(
            {"name": step.tool_name, "arguments": step.tool_args},
            ensure_ascii=False,
        )
        convs.append({"from": "function_call", "value": tc_json})
        convs.append({"from": "observation", "value": step.tool_result})

    # 末尾：gpt 的完成声明（含摘要）
    final_content = "TASK_COMPLETED"
    if record.summary:
        final_content = f"[技术摘要]\n{record.summary[:500]}\n\nTASK_COMPLETED"
    convs.append({"from": "gpt", "value": final_content})

    return {
        "conversations": convs,
        "_meta": {
            "trajectory_id": record.trajectory_id,
            "reward": record.reward,
            "status": record.status,
            "iteration_count": record.iteration_count,
        },
    }


def trajectory_to_dpo_pair(
    chosen: TrajectoryRecord,
    rejected: TrajectoryRecord,
) -> Optional[Dict]:
    """
    构造 DPO 训练对。
    LlamaFactory DPO 格式:
    {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    if not chosen.task_description:
        return None

    prompt = chosen.task_description

    def _summarize(rec: TrajectoryRecord) -> str:
        tool_summary = "\n".join(
            f"[{s.tool_name}] {'✅' if s.success else '❌'} {s.tool_result[:100]}"
            for s in rec.tool_call_steps[:8]
        )
        return (
            f"执行了 {len(rec.tool_call_steps)} 次工具调用。\n"
            f"{tool_summary}\n\n"
            f"最终状态：{'测试通过 ✅' if rec.test_passed else '测试未通过 ❌'}\n"
            f"{'TASK_COMPLETED' if rec.status == 'success' else '任务未完成'}"
        )

    return {
        "prompt": prompt,
        "chosen": _summarize(chosen),
        "rejected": _summarize(rejected),
        "_meta": {
            "chosen_id": chosen.trajectory_id,
            "rejected_id": rejected.trajectory_id,
            "chosen_reward": chosen.reward,
            "rejected_reward": rejected.reward,
        },
    }
