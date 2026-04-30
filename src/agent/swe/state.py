# src/agent/swe/state.py
from typing import Annotated, List, Literal, Sequence
from typing_extensions import TypedDict
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """SWE Agent 的全局状态定义"""

    # 消息历史（operator.add 保证追加而非覆盖）
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # 任务描述与计划
    task_description: str
    plan: List[str]

    # 分层规划
    todo_list: List[str]
    completed_tasks: List[str]

    # 记忆管理
    summary: str

    # 环境与控制
    workspace: str
    iteration_count: int
    max_iterations: int

    # 执行阶段
    status: Literal["planning", "coding", "testing", "success", "failed"]
    test_passed: bool

    # 防死循环
    reviewer_reject_count: int

    # Capability Evolution Loop
    evolution_skill_draft: str
    evolution_report: str

    # ── Three-Index Code Intelligence ──────────────────────────────────────
    # index_builder_node 构建完成后置 True，后续节点不再重复构建
    code_index_ready: bool
    # 最新的 Repo Map 快照（由 index_builder_node 和工具更新时刷新）
    repo_map: str

    # ── LlamaFactory Training Integration ─────────────────────────────────
    # 当前轨迹的唯一标识符（由 planner_node 初始化）
    trajectory_id: str
    # 轨迹导出路径（trajectory_export_node 执行后填充）
    training_export_path: str
