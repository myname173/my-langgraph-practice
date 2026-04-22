# src/agent/swe/graph.py
import os

# 强制清除代理环境变量，防止 httpx 报错
for key in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(key, None)

import copy
import logging
import re
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
# 辅助函数（前置，避免前向引用）
# ==========================================

def count_tokens(messages: list) -> int:
    """估算消息列表的 Token 消耗。"""
    try:
        from src.agent.swe.tools import _count_str_tokens
        return sum(_count_str_tokens(str(getattr(m, "content", ""))) for m in messages)
    except ImportError:
        return sum(len(str(getattr(m, "content", ""))) for m in messages) // 4


def get_uncompressed_messages(messages: list) -> list:
    """只提取上一次摘要之后的新消息，防止 Summarizer 无限循环。"""
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
    """
    安全截取最近 N 条消息，保证 tool_calls 与 ToolMessage 成对出现。
    """
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
    """
    双向压缩：折叠大段代码参数 + 折叠历史 ToolMessage 的长输出。
    """
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


# 多测试框架成功模式（词边界精确匹配）
_TEST_SUCCESS_PATTERNS = [
    r"Return Code: 0\b",
    r"\d+\s+passed",
    r"^OK\s*$",
    r"\bAll tests passed\b",
    r"\bTests passed\b",
    r"\bPASSED\b",
]


def _has_successful_test(messages: list) -> bool:
    """检查消息历史中是否存在测试通过的证据。"""
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
    """拆解任务为原子步骤。todo_list 已存在则跳过，避免重复规划。"""
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
        return {
            "todo_list": steps,
            "completed_tasks": [],
            "reviewer_reject_count": 0,
            "evolution_skill_draft": "",
            "evolution_report": "",
            "code_index_ready": False,
            "repo_map": "",
            "status": "coding",
            "messages": [AIMessage(content=f"📋 计划已生成，共 {len(steps)} 步。第一步：{steps[0]}")],
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
            "messages": [AIMessage(content="[系统] 规划失败，直接开始执行。")],
        }


# ★ 新增：代码索引构建节点
def index_builder_node(state: AgentState) -> Dict[str, Any]:
    """
    【三层代码智能索引构建节点】
    在 Planner 之后、Coder 之前执行，一次性构建：
      1. AST 结构索引（tree-sitter 解析函数/类）
      2. 语义向量索引（sentence-transformers 嵌入）
      3. BM25 关键字索引
      4. 依赖图 + PageRank（Repo Map）

    如果索引已构建（code_index_ready=True）则跳过，避免重复构建。
    工作区为空时快速返回，不阻塞 Coder。
    """
    # 如果已构建，跳过
    if state.get("code_index_ready"):
        return {}

    logger.info(">>> [Node] IndexBuilder: 构建代码智能索引...")
    try:
        from src.agent.swe.code_index import build_workspace_index
        workspace_path = state.get("workspace")
        ws = WORKSPACE_DIR if not workspace_path else __import__("pathlib").Path(workspace_path)

        # 扫描工作区是否有 Python 文件
        py_files = list(ws.rglob("*.py")) if ws.exists() else []
        # 过滤掉内部目录
        _ignore = {".git", "__pycache__", "node_modules", ".venv", "_evolution_drafts"}
        py_files = [f for f in py_files if not any(d in f.parts for d in _ignore)]

        if not py_files:
            logger.info("IndexBuilder: 工作区暂无 Python 文件，跳过索引构建。")
            return {
                "code_index_ready": True,
                "repo_map": "（工作区暂无 Python 文件，索引为空）",
            }

        repo_map = build_workspace_index(ws)

        logger.info(f"IndexBuilder: 索引构建完成，共 {len(py_files)} 个文件。")
        return {
            "code_index_ready": True,
            "repo_map": repo_map,
            "messages": [AIMessage(content=(
                f"🗺️ [代码索引] 已分析 {len(py_files)} 个 Python 文件，"
                f"三层索引构建完成。可使用 search_code / get_repo_map / get_symbol_context 工具。"
            ))],
        }

    except Exception as e:
        logger.warning(f"IndexBuilder: 索引构建失败（不影响主流程）: {e}")
        return {
            "code_index_ready": True,   # 标记为完成，避免反复重试
            "repo_map": f"（索引构建失败: {e}）",
        }


def coder_node(state: AgentState) -> Dict[str, Any]:
    """执行当前最优先的任务步骤。"""
    iteration = state.get("iteration_count", 0)
    logger.info(f">>> [Node] Coder: 迭代 {iteration + 1}")

    current_step = state.get("todo_list", ["完成任务"])[0] if state.get("todo_list") else "完成任务"
    summary_info = (
        f"\n\n【前期工作总结】:\n{state['summary']}" if state.get("summary") else ""
    )

    # 如果有 Repo Map，注入到系统提示（最多 1000 字符，节省 token）
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
    """【分层规划核心】动态更新任务清单。"""
    logger.info(">>> [Node] Task Manager: 更新进度地图")
    manager_llm = llm.with_structured_output(TaskUpdateOutput)

    context = (
        f"原始任务: {state.get('task_description', '')}\n"
        f"当前待办: {state.get('todo_list', [])}\n"
        f"已完成: {state.get('completed_tasks', [])}"
    )
    result = manager_llm.invoke(
        [
            SystemMessage(content=TASK_MANAGER_PROMPT),
            HumanMessage(content=f"{context}\n\n请根据最近的执行情况更新清单。"),
        ]
        + list(state["messages"][-3:])
    )
    return {"todo_list": result.todo_list, "completed_tasks": result.completed_tasks}


def reviewer_node(state: AgentState) -> Dict[str, Any]:
    """深度代码审查，连续驳回 ≥3 次后强制批准，防止死循环。"""
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
        logger.warning(f"⚠️ Reviewer 已连续驳回 {reject_count} 次，强制批准以防死锁。")
        return {
            "status": "success",
            "reviewer_reject_count": 0,
            "messages": [AIMessage(content="🎉 [Reviewer 强制批准]: 已达到最大审查次数，任务视为完成。")],
        }

    logger.info(">>> [Node] Reviewer: Coder 提交了完成申请，开始 LLM 审查...")
    reviewer_llm = llm.with_structured_output(ReviewOutput)
    sys_prompt = SystemMessage(
        content=REVIEWER_PROMPT.format(task_description=state.get("task_description", ""))
    )
    review_result = reviewer_llm.invoke([sys_prompt] + messages[-6:])

    if review_result.decision == "approve":
        logger.info("✅ Reviewer 批准了代码。")
        return {
            "status": "success",
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
                    f"⚠️ [Reviewer 驳回]: 任务尚未真正完成。请根据以下反馈继续修改：\n"
                    f"{review_result.feedback}\n\n"
                    f"请修复后再次提交 (回复 TASK_COMPLETED)。"
                ))
            ],
        }


def summarizer_node(state: AgentState) -> Dict[str, Any]:
    """触发历史记录自动压缩，只压缩上次摘要之后的新消息。"""
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


# ==========================================
# 路由逻辑
# ==========================================

def route_after_coder(state: AgentState) -> Literal["tools", "reviewer", "coder"]:
    messages = list(state.get("messages", []))
    if not messages:
        return "coder"

    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    content = last_message.content if isinstance(last_message.content, str) else ""
    if "TASK_COMPLETED" in content:
        if _has_successful_test(messages):
            return "reviewer"
        else:
            return "coder"

    return "coder"


def route_after_reviewer(
    state: AgentState,
) -> Literal["evolution_reflect", "coder", "__end__"]:
    if state.get("status") == "success":
        return "evolution_reflect"
    if state.get("status") == "failed":
        return "__end__"
    return "coder"


def route_after_task_manager(state: AgentState) -> Literal["summarizer", "coder"]:
    uncompressed = get_uncompressed_messages(list(state["messages"]))
    if count_tokens(uncompressed) > 15000:
        return "summarizer"
    return "coder"


def route_after_tools(state: AgentState) -> Literal["task_manager", "summarizer", "coder"]:
    """
    工具执行后路由：
    成功执行 → Task Manager 更新进度
    失败/无进展 → 跳过 Task Manager，直接检查是否需要摘要
    """
    messages = list(state.get("messages", []))
    if not messages:
        return "coder"

    last_msg = messages[-1]
    content = str(getattr(last_msg, "content", ""))
    if any(re.search(p, content, re.MULTILINE | re.IGNORECASE) for p in _TEST_SUCCESS_PATTERNS):
        return "task_manager"
    return route_after_task_manager(state)


# ==========================================
# 构建与编译图
# ==========================================
workflow = StateGraph(AgentState)

# 主流程节点
workflow.add_node("planner", planner_node)
workflow.add_node("index_builder", index_builder_node)   # ★ 新增
workflow.add_node("coder", coder_node)
workflow.add_node("tools", ToolNode(TOOLS))
workflow.add_node("task_manager", task_manager_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("reviewer", reviewer_node)

# Capability Evolution Loop 节点
workflow.add_node("evolution_reflect", evolution_reflect_node)
workflow.add_node("evolution_generate", evolution_generate_node)
workflow.add_node("evolution_verify", evolution_verify_node)

# 主流程连线
# ★ planner → index_builder → coder（原来是 planner → coder）
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "index_builder")
workflow.add_edge("index_builder", "coder")

workflow.add_conditional_edges(
    "coder",
    route_after_coder,
    {"tools": "tools", "reviewer": "reviewer", "coder": "coder"},
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
        "__end__": END,
    },
)

# Evolution Loop
workflow.add_edge("evolution_reflect", "evolution_generate")
workflow.add_edge("evolution_generate", "evolution_verify")
workflow.add_edge("evolution_verify", END)

# 编译（生产环境请换用 PostgresSaver 或 RedisSaver 实现持久化）
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

print("🚀 SWE Agent Graph 编译成功！(含 Three-Index + Evolution Loop)")
