# src/agent/swe/graph.py
import os

# 强制清除代理环境变量，防止 httpx 报错
for key in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(key, None)

import copy
import logging
import re
import uuid
from typing import Any, Dict, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from src.agent.swe.evolution import (
    evolution_generate_node,
    evolution_reflect_node,
    evolution_verify_node,
)
from src.agent.swe.prompts import (
    CODER_PROMPT_TEMPLATE,
    PLANNER_PROMPT,
    REVIEWER_PROMPT,
    SUMMARIZER_PROMPT,
    TASK_MANAGER_PROMPT,
)
from src.agent.swe.state import AgentState
from src.agent.swe.tools import TOOLS, WORKSPACE_DIR

load_dotenv()

logger = logging.getLogger("SWE_Graph")

# ==========================================
# LLM 初始化
# ==========================================
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen3.5-plus"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    temperature=0.1,
)

llm_with_tools = llm.bind_tools(TOOLS)


# ==========================================
# Pydantic 结构化输出 Schema
# ==========================================
class PlanOutput(BaseModel):
    steps: list[str] = Field(
        description="完成任务所需的具体步骤列表。每个元素必须是纯文本字符串，绝对不能是字典！"
    )


class TaskUpdateOutput(BaseModel):
    todo_list: list[str]
    completed_tasks: list[str]


class ReviewOutput(BaseModel):
    decision: Literal["approve", "reject"] = Field(
        description="审查决定：approve 表示通过，reject 表示打回修改"
    )
    feedback: str = Field(description="给 Coder 的反馈意见。")


# ==========================================
# 辅助函数
# ==========================================

def count_tokens(messages: list) -> int:
    try:
        from src.agent.swe.tools import _count_str_tokens
        return sum(_count_str_tokens(str(getattr(m, "content", ""))) for m in messages)
    except ImportError:
        return sum(len(str(getattr(m, "content", ""))) for m in messages) // 4


def get_uncompressed_messages(messages: list) -> list:
    uncompressed = []
    for msg in reversed(messages):
        uncompressed.append(msg)
        if (
            isinstance(msg, AIMessage)
            and isinstance(msg.content, str)
            and "[系统通知：历史记录已压缩" in msg.content
        ):
            break
    return list(reversed(uncompressed))


def get_safe_recent_messages(messages: list, max_history: int = 8) -> list:
    if len(messages) <= max_history:
        return messages
    kept = []
    safety_limit = max_history * 2
    collected = 0
    for msg in reversed(messages):
        if collected >= safety_limit:
            break
        kept.append(msg)
        collected += 1
        if len(kept) >= max_history:
            if getattr(msg, "type", "") != "tool":
                break
    return list(reversed(kept))


def compact_message_history(messages: list) -> list:
    compacted = []
    for i, msg in enumerate(messages):
        new_msg = copy.deepcopy(msg)
        if isinstance(new_msg, AIMessage) and new_msg.tool_calls:
            for tc in new_msg.tool_calls:
                if tc["name"] in ["write_file", "edit_file"]:
                    if "content" in tc["args"] and len(tc["args"]["content"]) > 500:
                        tc["args"]["content"] = f"[代码已写入，已折叠，长度: {len(tc['args']['content'])}]"
                    if "replace_text" in tc["args"] and len(tc["args"]["replace_text"]) > 500:
                        tc["args"]["replace_text"] = f"[代码已修改，已折叠，长度: {len(tc['args']['replace_text'])}]"
        if getattr(new_msg, "type", "") == "tool" and i < len(messages) - 1:
            if isinstance(new_msg.content, str) and len(new_msg.content) > 1000:
                new_msg.content = (
                    new_msg.content[:500]
                    + f"\n\n...[历史工具输出已折叠 {len(new_msg.content) - 1000} 字符]...\n\n"
                    + new_msg.content[-500:]
                )
        compacted.append(new_msg)
    return compacted


def _filter_text_only_messages(messages: list, max_count: int = 6) -> list:
    """
    过滤掉 ToolMessage 和纯工具调用 AIMessage，只保留纯文本消息。
    用于 reviewer、task_manager 等节点，防止 with_structured_output 把
    工具调用 blob 误解析为结构化输出（pydantic "Input should be an object" 根因）。
    """
    def _is_text_only(msg) -> bool:
        msg_type = getattr(msg, "type", "")
        if msg_type == "tool":
            return False
        if msg_type == "ai" and getattr(msg, "tool_calls", None):
            return False
        content = getattr(msg, "content", "")
        return bool(content and len(str(content)) > 5)

    return [m for m in messages[-(max_count * 2):] if _is_text_only(m)][-max_count:]


_TEST_SUCCESS_PATTERNS = [
    r"Return Code: 0\b",
    r"\d+\s+passed",
    r"^OK\s*$",
    r"\bAll tests passed\b",
    r"\bTests passed\b",
    r"\bPASSED\b",
]


def _has_successful_test(messages: list) -> bool:
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "tool":
            content = str(getattr(msg, "content", ""))
            if any(re.search(p, content, re.MULTILINE | re.IGNORECASE) for p in _TEST_SUCCESS_PATTERNS):
                return True
    return False


# ==========================================
# 主流程节点
# ==========================================

def planner_node(state: AgentState) -> Dict[str, Any]:
    if state.get("todo_list"):
        return {}

    task_desc = state.get("task_description", "")
    if not task_desc:
        return {
            "todo_list": ["分析需求并编码"],
            "reviewer_reject_count": 0,
            "evolution_skill_draft": "",
            "evolution_report": "",
            "code_index_ready": False,
            "repo_map": "",
            "trajectory_id": str(uuid.uuid4())[:12],
            "training_export_path": "",
            "messages": [AIMessage(content="[系统] 未提供任务描述，请输入具体需求。")],
        }

    logger.info(">>> [Node] Planner: 执行任务拆解")
    try:
        planner_llm = llm.with_structured_output(PlanOutput)
        result = planner_llm.invoke([
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=task_desc),
        ])
        steps = result.steps if result.steps else ["分析需求并编码"]

        _env_keywords = ["npm", "pip", "install", "node", "python", "apt", "setup", "环境", "配置"]
        needs_env_setup = any(kw in task_desc.lower() for kw in _env_keywords)
        base_iter = state.get("max_iterations", 25)
        dynamic_max = max(base_iter, 35 if needs_env_setup else 25)

        return {
            "todo_list": steps,
            "completed_tasks": [],
            "reviewer_reject_count": 0,
            "evolution_skill_draft": "",
            "evolution_report": "",
            "code_index_ready": False,
            "repo_map": "",
            "status": "coding",
            "max_iterations": dynamic_max,
            "trajectory_id": str(uuid.uuid4())[:12],
            "training_export_path": "",
            "messages": [AIMessage(content=(
                f"📋 计划已生成，共 {len(steps)} 步。"
                + (f"（含环境安装，迭代预算扩展至 {dynamic_max} 轮）" if needs_env_setup else "")
                + f"\n第一步：{steps[0]}"
            ))],
        }
    except Exception as e:
        logger.error(f"Planner 失败: {e}")
        return {
            "todo_list": ["开始任务"],
            "reviewer_reject_count": 0,
            "evolution_skill_draft": "",
            "evolution_report": "",
            "code_index_ready": False,
            "repo_map": "",
            "trajectory_id": str(uuid.uuid4())[:12],
            "training_export_path": "",
            "messages": [AIMessage(content="[系统] 规划失败，直接开始执行。")],
        }


def index_builder_node(state: AgentState) -> Dict[str, Any]:
    if state.get("code_index_ready"):
        return {}

    logger.info(">>> [Node] IndexBuilder: 构建代码智能索引...")
    try:
        from src.agent.swe.code_index import build_workspace_index
        import pathlib
        workspace_path = state.get("workspace")
        ws = WORKSPACE_DIR if not workspace_path else pathlib.Path(workspace_path)

        py_files = list(ws.rglob("*.py")) if ws.exists() else []
        _ignore = {".git", "__pycache__", "node_modules", ".venv", "_evolution_drafts", "_training_data"}
        py_files = [f for f in py_files if not any(d in f.parts for d in _ignore)]

        if not py_files:
            logger.info("IndexBuilder: 工作区暂无 Python 文件，跳过索引构建。")
            return {"code_index_ready": True, "repo_map": "（工作区暂无代码文件）"}

        build_workspace_index()
        logger.info(f"IndexBuilder: 索引构建完成，覆盖 {len(py_files)} 个 Python 文件。")

        try:
            from src.agent.swe.code_index import get_repo_map_str
            repo_map = get_repo_map_str(query=state.get("task_description", ""), max_tokens=1500)
        except Exception:
            repo_map = ""

        return {"code_index_ready": True, "repo_map": repo_map}

    except Exception as e:
        logger.warning(f"IndexBuilder 失败（跳过）: {e}")
        return {"code_index_ready": True, "repo_map": ""}


def coder_node(state: AgentState) -> Dict[str, Any]:
    logger.info(">>> [Node] Coder: 执行编码任务")

    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 25)

    if iteration >= max_iter:
        logger.warning("达到最大迭代次数，强制结束。")
        return {
            "status": "failed",
            "messages": [AIMessage(content="任务失败：达到最大迭代次数。")],
        }

    todo_list = state.get("todo_list", [])
    current_step = todo_list[0] if todo_list else "继续执行任务"

    summary_info = ""
    if state.get("summary"):
        summary_info = f"\n\n【历史摘要】\n{state['summary']}"

    repo_map_info = ""
    repo_map = state.get("repo_map", "")
    if repo_map and len(repo_map) > 10:
        repo_map_snippet = repo_map[:1000] + ("..." if len(repo_map) > 1000 else "")
        repo_map_info = f"\n\n【当前代码库结构（Repo Map）】:\n{repo_map_snippet}"

    sys_prompt_content = CODER_PROMPT_TEMPLATE.format(
        workspace=state.get("workspace", "未知目录"),
        plan="\n".join(state.get("todo_list", [])),
        task_description=state.get("task_description", ""),
        completed_tasks=", ".join(state.get("completed_tasks", [])),
        todo_list=", ".join(state.get("todo_list", [])),
        current_step=current_step,
    )

    sys_prompt = SystemMessage(content=sys_prompt_content + summary_info + repo_map_info)

    if state.get("summary"):
        recent_messages = list(state["messages"][-4:])
    else:
        recent_messages = get_safe_recent_messages(list(state["messages"]), max_history=6)

    compacted = compact_message_history(recent_messages)
    response = llm_with_tools.invoke([sys_prompt] + compacted)

    return {
        "messages": [response],
        "iteration_count": iteration + 1,
    }


def task_manager_node(state: AgentState) -> Dict[str, Any]:
    logger.info(">>> [Node] Task Manager: 更新进度地图")
    manager_llm = llm.with_structured_output(TaskUpdateOutput)

    recent_text_msgs = _filter_text_only_messages(list(state["messages"]), max_count=3)

    context = (
        f"原始任务: {state.get('task_description', '')}\n"
        f"当前待办: {state.get('todo_list', [])}\n"
        f"已完成: {state.get('completed_tasks', [])}"
    )
    try:
        result = manager_llm.invoke(
            [
                SystemMessage(content=TASK_MANAGER_PROMPT),
                HumanMessage(content=f"{context}\n\n请根据最近的执行情况更新清单。"),
            ]
            + recent_text_msgs
        )
        return {"todo_list": result.todo_list, "completed_tasks": result.completed_tasks}
    except Exception as e:
        logger.warning(f"TaskManager 结构化输出失败，保留现有清单: {e}")
        return {}


def reviewer_node(state: AgentState) -> Dict[str, Any]:
    logger.info(">>> [Node] Reviewer: 深度代码审查")

    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 25)

    if iteration >= max_iter:
        logger.warning("达到最大迭代次数，强制结束任务。")
        return {
            "status": "failed",
            "messages": [AIMessage(content="任务失败：达到最大迭代次数。")],
        }

    messages = list(state.get("messages", []))
    if not messages:
        return {"status": "coding"}

    last_message = messages[-1]
    if not (
        isinstance(last_message, AIMessage)
        and last_message.content
        and "TASK_COMPLETED" in last_message.content
    ):
        return {"status": "coding"}

    reject_count = state.get("reviewer_reject_count", 0)
    if reject_count >= 3:
        logger.warning(f"Reviewer 已连续驳回 {reject_count} 次，强制批准。")
        return {
            "status": "success",
            "test_passed": _has_successful_test(messages),
            "reviewer_reject_count": 0,
            "messages": [AIMessage(content="[Reviewer 强制批准]: 已达到最大审查次数，任务视为完成。")],
        }

    logger.info(">>> [Node] Reviewer: 开始 LLM 审查...")
    reviewer_llm = llm.with_structured_output(ReviewOutput)
    sys_prompt = SystemMessage(
        content=REVIEWER_PROMPT.format(task_description=state.get("task_description", ""))
    )

    # ★ Bug 修复：过滤工具消息，只传纯文本给 reviewer LLM
    text_msgs = _filter_text_only_messages(messages, max_count=6)

    try:
        review_result = reviewer_llm.invoke([sys_prompt] + text_msgs)
    except Exception as e:
        logger.warning(f"Reviewer LLM 解析失败，降级为批准: {e}")
        return {
            "status": "success",
            "test_passed": _has_successful_test(messages),
            "reviewer_reject_count": 0,
            "messages": [AIMessage(content="[Reviewer 降级批准]: LLM 解析异常，任务视为完成。")],
        }

    if review_result.decision == "approve":
        logger.info("✅ Reviewer 批准了代码。")
        return {
            "status": "success",
            "test_passed": _has_successful_test(messages),
            "reviewer_reject_count": 0,
            "messages": [AIMessage(content=f"🎉 [Reviewer 审查通过]: {review_result.feedback}")],
        }
    else:
        logger.info(f"❌ Reviewer 驳回了代码: {review_result.feedback}")
        return {
            "status": "coding",
            "reviewer_reject_count": reject_count + 1,
            "messages": [
                AIMessage(content=(
                    f"⚠️ [Reviewer 驳回]: 请根据以下反馈继续修改：\n"
                    f"{review_result.feedback}\n\n"
                    f"请修复后再次提交 (回复 TASK_COMPLETED)。"
                ))
            ],
        }


def summarizer_node(state: AgentState) -> Dict[str, Any]:
    logger.info(">>> [Node] Summarizer: 触发历史记录自动压缩")
    uncompressed = get_uncompressed_messages(list(state["messages"]))
    summary_context = f"现有摘要: {state.get('summary', '无')}\n\n待压缩的近期记录："
    response = llm.invoke([
        SystemMessage(content=SUMMARIZER_PROMPT),
        HumanMessage(content=f"{summary_context}\n\n{str(uncompressed)}"),
    ])
    return {
        "summary": response.content,
        "messages": [AIMessage(content="[系统通知：历史记录已压缩，已开启精简记忆模式]")],
    }


def trajectory_export_node(state: AgentState) -> Dict[str, Any]:
    status = state.get("status", "unknown")
    logger.info(f">>> [Node] TrajectoryExport: 捕获轨迹 (status={status})")

    try:
        from src.agent.swe.training.trajectory_logger import (
            extract_trajectory_from_state,
            save_trajectory,
        )
        from src.agent.swe.training.reward_computer import compute_reward

        state_dict = dict(state)
        record = extract_trajectory_from_state(state_dict)
        bd = compute_reward(record)
        record.reward = bd.total
        record.reward_breakdown = bd.to_dict()

        export_path = save_trajectory(record, WORKSPACE_DIR)
        logger.info(f"轨迹已导出: {record.trajectory_id} reward={record.reward:.3f} -> {export_path}")
        return {"training_export_path": str(export_path)}

    except Exception as e:
        logger.warning(f"轨迹导出失败（不影响主流程）: {e}")
        return {}


# ==========================================
# 路由逻辑
# ==========================================

def route_after_coder(
    state: AgentState,
) -> Literal["tools", "reviewer", "coder", "trajectory_export"]:
    """
    Bug 1 fix: failed -> trajectory_export (not coder loop)
    Bug 3 fix: TASK_COMPLETED always goes to reviewer (no test requirement)
    """
    if state.get("status") == "failed":
        return "trajectory_export"

    messages = list(state.get("messages", []))
    if not messages:
        return "coder"

    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    content = last_message.content if isinstance(last_message.content, str) else ""
    if "TASK_COMPLETED" in content:
        # ★ 修复：移除 _has_successful_test 门槛，不强制要求测试通过
        # 让 reviewer 作为唯一的完成判断器
        return "reviewer"

    return "coder"


def route_after_reviewer(
    state: AgentState,
) -> Literal["evolution_reflect", "coder", "trajectory_export"]:
    """
    Bug 1 fix: failed -> trajectory_export (保存轨迹后结束，不跳过)
    """
    if state.get("status") == "success":
        return "evolution_reflect"
    if state.get("status") == "failed":
        return "trajectory_export"
    return "coder"


def route_after_task_manager(state: AgentState) -> Literal["summarizer", "coder"]:
    uncompressed = get_uncompressed_messages(list(state["messages"]))
    if count_tokens(uncompressed) > 15000:
        return "summarizer"
    return "coder"


def route_after_tools(state: AgentState) -> Literal["task_manager", "summarizer", "coder"]:
    messages = list(state.get("messages", []))
    if not messages:
        return "coder"

    last_msg = messages[-1]
    content = str(getattr(last_msg, "content", ""))
    if any(re.search(p, content, re.MULTILINE | re.IGNORECASE) for p in _TEST_SUCCESS_PATTERNS):
        return "task_manager"
    return route_after_task_manager(state)


def route_after_evolution_verify(
    state: AgentState,
) -> Literal["trajectory_export"]:
    return "trajectory_export"


# ==========================================
# 构建与编译图
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("index_builder", index_builder_node)
workflow.add_node("coder", coder_node)
workflow.add_node("tools", ToolNode(TOOLS))
workflow.add_node("task_manager", task_manager_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("evolution_reflect", evolution_reflect_node)
workflow.add_node("evolution_generate", evolution_generate_node)
workflow.add_node("evolution_verify", evolution_verify_node)
workflow.add_node("trajectory_export", trajectory_export_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "index_builder")
workflow.add_edge("index_builder", "coder")

workflow.add_conditional_edges(
    "coder",
    route_after_coder,
    {
        "tools": "tools",
        "reviewer": "reviewer",
        "coder": "coder",
        "trajectory_export": "trajectory_export",
    },
)
workflow.add_conditional_edges(
    "tools",
    route_after_tools,
    {"task_manager": "task_manager", "summarizer": "summarizer", "coder": "coder"},
)
workflow.add_conditional_edges(
    "task_manager",
    route_after_task_manager,
    {"summarizer": "summarizer", "coder": "coder"},
)
workflow.add_edge("summarizer", "coder")
workflow.add_conditional_edges(
    "reviewer",
    route_after_reviewer,
    {
        "evolution_reflect": "evolution_reflect",
        "coder": "coder",
        "trajectory_export": "trajectory_export",
    },
)
workflow.add_edge("evolution_reflect", "evolution_generate")
workflow.add_edge("evolution_generate", "evolution_verify")
workflow.add_conditional_edges(
    "evolution_verify",
    route_after_evolution_verify,
    {"trajectory_export": "trajectory_export"},
)
workflow.add_edge("trajectory_export", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

print("SWE Agent Graph compiled successfully.")
