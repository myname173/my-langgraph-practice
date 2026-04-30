# Savant SWE Agent — 项目全解

> 本文档面向希望深度理解本项目的开发者、研究者和学习者。  
> 全文按"概念 → 架构 → 模块 → 数据流 → 安全 → 部署"的顺序展开，  
> 每个专业术语在首次出现时均会给出定义。

---

## 目录

1. [项目定位与核心价值](#1-项目定位与核心价值)
2. [专业术语速查表](#2-专业术语速查表)
3. [整体架构概览](#3-整体架构概览)
4. [核心技术：LangGraph 与状态机](#4-核心技术langgraph-与状态机)
5. [AgentState：系统的血液](#5-agentstate系统的血液)
6. [主流程节点详解](#6-主流程节点详解)
7. [路由逻辑：图的神经系统](#7-路由逻辑图的神经系统)
8. [工具层（Tools）详解](#8-工具层tools详解)
9. [Three-Index 代码智能引擎](#9-three-index-代码智能引擎)
10. [Capability Evolution Loop](#10-capability-evolution-loop)
11. [记忆管理系统](#11-记忆管理系统)
12. [提示词工程](#12-提示词工程)
13. [安全体系](#13-安全体系)
14. [FastAPI 后端](#14-fastapi-后端)
15. [Streamlit 前端](#15-streamlit-前端)
16. [完整数据流演示](#16-完整数据流演示)
17. [依赖关系与安装](#17-依赖关系与安装)
18. [配置参考](#18-配置参考)
19. [项目演进历程](#19-项目演进历程)

---

## 1. 项目定位与核心价值

### 1.1 它是什么

Savant SWE Agent（Software Engineering Agent）是一个**自主软件工程智能体**。给它一个自然语言任务描述，它能够独立完成：需求分析、代码设计、文件读写、命令执行、测试验证，直到任务通过审查。

它不是一个简单的"代码补全工具"，而是一个能够**在真实文件系统上持续工作、自我纠错、积累经验**的自主系统。

### 1.2 核心价值

**vs 传统 IDE 辅助（GitHub Copilot 类）**：不需要人每次手动触发，能独立规划并执行多步任务，能自动运行测试并根据失败信息修正。

**vs 简单 LLM 调用**：有持久化状态，每轮对话不会"失忆"；有工具调用能力，能真实操作文件系统；有闭环验证机制，不允许"口头完成"。

**vs 其他 Coding Agent（Devin、SWE-Agent）**：完全开源、可本地部署、架构清晰可扩展，且具备独特的**能力自演化机制**——每次成功完成任务后，系统会自动分析是否有可复用的逻辑，将其固化为可调用的技能脚本。

---

## 2. 专业术语速查表

| 术语 | 全称/含义 |
|---|---|
| **LLM** | Large Language Model，大语言模型。本项目通过 OpenAI 兼容接口调用，默认使用 Qwen 系列模型 |
| **Agent** | 智能体。一个能够感知环境、做出决策、执行动作、并根据反馈调整行为的自主程序 |
| **LangGraph** | LangChain 生态中用于构建有状态、多角色 Agent 工作流的图执行框架 |
| **StateGraph** | LangGraph 的核心类，将 Agent 逻辑建模为一个有向图（节点 + 边） |
| **Node（节点）** | 图中的一个执行单元，接收当前 State，返回对 State 的局部更新 |
| **Edge（边）** | 节点之间的连接，分为固定边（无条件跳转）和条件边（根据 State 动态选择目标） |
| **State** | 图的全局共享状态，用 TypedDict 定义，所有节点读写同一个 State 对象 |
| **Checkpointer** | LangGraph 的持久化模块，将每次节点执行后的 State 快照存储，支持断点续传 |
| **Tool Call** | LLM 请求调用外部工具的机制。LLM 输出结构化的调用请求，框架负责实际执行 |
| **ToolNode** | LangGraph 预置节点，自动执行 LLM 请求的工具调用并将结果写回消息历史 |
| **InMemorySaver** | Checkpointer 的内存实现，进程重启后状态丢失（适合开发环境） |
| **Structured Output** | LLM 约束输出格式为特定 Pydantic Schema，保证返回 JSON 可解析 |
| **AST** | Abstract Syntax Tree，抽象语法树。将源代码解析为树形结构，用于提取函数、类等语法元素 |
| **Tree-sitter** | 跨语言的增量 AST 解析库，速度极快，支持 Python / JS / Go 等数十种语言 |
| **BM25** | Best Match 25，信息检索领域的经典关键字匹配算法，基于词频和逆文档频率 |
| **Embedding** | 将文本转换为高维向量，语义相近的文本在向量空间中距离近 |
| **Cosine Similarity** | 余弦相似度，衡量两个向量方向的一致程度，用于语义搜索排名 |
| **RRF** | Reciprocal Rank Fusion，倒数排名融合。将多个搜索结果列表合并为单一排名的算法 |
| **PageRank** | Google 原创的网页重要性评分算法，本项目用于评估代码符号的重要性 |
| **HiL** | Human-in-the-Loop，人在回路。在关键决策点暂停并请求人类确认 |
| **SSE** | Server-Sent Events，服务器推送事件。一种 HTTP 长连接协议，用于实时流式推送 |
| **SSRF** | Server-Side Request Forgery，服务端请求伪造。攻击者诱使服务器请求内部资源 |
| **Token** | LLM 的基本处理单元，大约 1 个英文单词或 0.5-1 个中文字符 |
| **Context Window** | LLM 单次能处理的最大 Token 数量上限 |
| **RAG** | Retrieval-Augmented Generation，检索增强生成。先检索相关信息再交给 LLM 生成 |
| **Jaccard Similarity** | Jaccard 相似度，两个集合交集大小与并集大小的比值，用于文本去重 |

---

## 3. 整体架构概览

### 3.1 模块文件地图

```
src/agent/swe/
├── state.py          # AgentState 全局状态定义（项目的数据契约）
├── prompts.py        # 所有提示词模板（LLM 的行为规范）
├── tools.py          # 工具实现 + TOOLS 列表（Agent 的手和眼）
├── code_index.py     # Three-Index 代码智能引擎（全新核心模块）
├── graph.py          # LangGraph 图定义、节点、路由（项目大脑）
├── evolution.py      # Capability Evolution Loop（自我成长机制）
│
src/
├── api.py            # FastAPI 后端（生产部署入口）
└── web_app.py        # Streamlit 前端（开发演示入口）
```

### 3.2 三条主干流水线

本项目在逻辑上由三条独立的流水线组成：

```
┌─────────────────────────────────────────────────────────────────┐
│ 流水线 A：主编码流程（每个任务都会走）                            │
│                                                                  │
│ START → planner → index_builder → coder ⟷ tools ⟷ reviewer   │
│                                      ↑_______↓                  │
│                               task_manager/summarizer            │
└──────────────────────────────────────┬──────────────────────────┘
                                       │ reviewer approve
┌──────────────────────────────────────▼──────────────────────────┐
│ 流水线 B：能力演化（仅在任务成功后触发）                          │
│                                                                  │
│ evolution_reflect → evolution_generate → evolution_verify → END  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 流水线 C：后台持续服务（无状态机节点，按需调用）                  │
│                                                                  │
│ Three-Index Engine（code_index.py）— 独立线程，增量维护         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 层次依赖关系

```
          ┌──────────┐   ┌────────────┐
          │ api.py   │   │ web_app.py │
          │(FastAPI) │   │(Streamlit) │
          └────┬─────┘   └─────┬──────┘
               │               │
          ┌────▼───────────────▼────┐
          │        graph.py         │
          │  (LangGraph StateGraph) │
          └─┬──────┬──────┬────────┘
            │      │      │
      ┌─────▼──┐ ┌─▼────┐ ┌▼──────────┐
      │tools.py│ │evolu-│ │code_index │
      │        │ │tion  │ │  .py      │
      └────────┘ └──────┘ └───────────┘
            │
      ┌─────▼──────────────────────┐
      │   state.py / prompts.py   │
      │   (数据契约 / 行为规范)    │
      └────────────────────────────┘
```

---

## 4. 核心技术：LangGraph 与状态机

### 4.1 为什么用 LangGraph

传统的 LLM 应用是**无状态的**：每次调用都从零开始。对于软件工程任务，这根本行不通——Agent 需要记住"我已经创建了哪些文件"、"第 N 步执行的命令返回了什么错误"。

LangGraph 将 Agent 的行为建模为一个**有限状态机（Finite State Machine）**，核心思想是：

- 整个对话过程的所有信息存储在一个中心化的 **State** 字典中
- 每个 **Node（节点）** 是一个纯函数：`(State) → PartialState`，只负责更新 State 的某些字段
- **Edge（边）** 决定执行完一个节点后去哪里，可以是固定的，也可以是根据当前 State 动态计算的

### 4.2 StateGraph 的生命周期

```python
# 1. 定义状态（TypedDict）
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    todo_list: List[str]
    # ...

# 2. 创建图
workflow = StateGraph(AgentState)

# 3. 添加节点（每个节点是一个函数）
workflow.add_node("planner", planner_node)
workflow.add_node("coder", coder_node)

# 4. 连接节点（固定边）
workflow.add_edge(START, "planner")

# 5. 连接节点（条件边）
workflow.add_conditional_edges(
    "coder",
    route_after_coder,              # 路由函数
    {"tools": "tools", "coder": "coder"}  # 返回值 → 目标节点的映射
)

# 6. 编译（加载 Checkpointer 后支持断点续传）
graph = workflow.compile(checkpointer=InMemorySaver())
```

### 4.3 Checkpointer 与 Thread ID

LangGraph 用 **Thread ID** 来区分不同的对话会话，就像数据库里的主键。

```python
# 运行时通过 config 指定 Thread ID
config = {"configurable": {"thread_id": "task_001"}}
graph.stream(input_data, config=config)

# 下次用同一个 Thread ID 恢复执行
graph.stream(None, config=config)  # input=None 表示从断点继续
```

每次节点执行后，Checkpointer 会将整个 State 序列化存储。这是实现**人在回路（HiL）**的基础：当 Agent 需要人类批准某个工具调用时，图在 `tools` 节点前暂停，等待人类发出指令后再继续。

### 4.4 operator.add 的含义

State 中的 `messages` 字段有一个特殊注解：

```python
messages: Annotated[Sequence[BaseMessage], operator.add]
```

这里 `operator.add` 是 LangGraph 的**归约函数（Reducer）**。它意味着：当某个节点返回 `{"messages": [new_msg]}` 时，LangGraph 不会用 `new_msg` 覆盖原来的消息列表，而是将它**追加**到末尾。这是保证消息历史不丢失的关键设计。

其他字段（如 `todo_list`、`status`）没有指定归约函数，所以节点返回的值会**直接覆盖**原值——这正是所需的行为，因为我们希望 TaskManager 能完全替换旧的待办清单。

---

## 5. AgentState：系统的血液

`state.py` 定义了整个系统的数据契约，所有节点共享同一个 State 实例。

### 5.1 字段分组解析

```python
class AgentState(TypedDict):
    # ── 组 1：对话历史 ──────────────────────────────────────
    # Annotated + operator.add = 只追加，不覆盖
    # BaseMessage 是 HumanMessage / AIMessage / ToolMessage 的基类
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # ── 组 2：任务描述与计划 ──────────────────────────────────
    task_description: str   # 用户的原始自然语言任务
    plan: List[str]         # 保留字段（planner 生成的初始步骤列表）

    # ── 组 3：分层规划（动态维护）────────────────────────────
    todo_list: List[str]        # 当前待执行的原子步骤队列
    completed_tasks: List[str]  # 已完成步骤的记录

    # ── 组 4：记忆管理 ──────────────────────────────────────
    # 当消息历史超过 Token 上限时，Summarizer 将旧消息压缩为此字段
    summary: str

    # ── 组 5：运行环境 ──────────────────────────────────────
    workspace: str          # 工作区绝对路径（Docker 挂载卷的宿主机路径）
    iteration_count: int    # Coder 已执行的迭代次数
    max_iterations: int     # 最大允许迭代次数（防止无限循环）

    # ── 组 6：执行阶段标记 ─────────────────────────────────
    # Literal 约束枚举值，TypedDict 的类型安全保障
    status: Literal["planning", "coding", "testing", "success", "failed"]
    test_passed: bool       # 明确记录测试是否通过（避免 LLM 谎报）

    # ── 组 7：防死循环保护 ─────────────────────────────────
    # 记录 Reviewer 连续驳回次数；≥3 次时强制批准，打破死锁
    reviewer_reject_count: int

    # ── 组 8：能力演化（Evolution Loop 专用）────────────────
    # 在三个演化节点之间传递 JSON 格式的中间状态
    evolution_skill_draft: str  # JSON 字符串，携带决策 + 草稿路径
    evolution_report: str       # 最终演化结果的人类可读摘要

    # ── 组 9：代码智能索引（Three-Index 专用）───────────────
    code_index_ready: bool  # 索引是否已构建（避免重复构建）
    repo_map: str           # 最新的代码库符号地图快照（用于注入 Coder 上下文）
```

### 5.2 State 的更新规则

节点返回的字典是**局部更新**，不需要返回整个 State：

```python
def coder_node(state: AgentState) -> Dict[str, Any]:
    # 只返回需要更新的字段
    return {
        "messages": [response],       # 追加（因为有 operator.add）
        "iteration_count": iteration + 1,  # 覆盖
    }
    # 其他字段（todo_list, summary 等）保持不变
```

节点也可以返回空字典 `{}` 表示"什么都不更新"：

```python
def planner_node(state: AgentState) -> Dict[str, Any]:
    if state.get("todo_list"):  # 计划已存在，跳过
        return {}
```

---

## 6. 主流程节点详解

### 6.1 节点执行顺序图

```
START
  │
  ▼
[planner_node]          ← 将任务拆解为原子步骤列表
  │ (固定边)
  ▼
[index_builder_node]    ← 构建 Three-Index 代码智能索引
  │ (固定边)
  ▼
┌─[coder_node] ──────────────────────────────────────────┐
│   │                                                     │
│   ├── 有 tool_calls → [tools/ToolNode] ─→ 成功 → [task_manager_node]
│   │                          │                    │
│   │                          └→ 失败 → route_after_task_manager
│   │                                         │
│   │              [summarizer_node] ◀─────── ┘ (Token > 15000)
│   │                   │
│   │                   └→ [coder_node] (带压缩后的 summary)
│   │
│   ├── 说 "TASK_COMPLETED" + 有测试证据 → [reviewer_node]
│   │
│   └── 其他 → 继续 [coder_node]
└────────────────────────────────────────────────────────┘
         │
         │ reviewer approve
         ▼
[evolution_reflect_node]
  │ (固定边)
  ▼
[evolution_generate_node]
  │ (固定边)
  ▼
[evolution_verify_node]
  │ (固定边)
  ▼
END
```

### 6.2 planner_node

**职责**：将用户的自然语言任务描述拆解为可执行的原子步骤列表。

**核心技术**：使用 `llm.with_structured_output(PlanOutput)` 强制 LLM 返回结构化的 JSON，避免自由文本输出导致的解析失败。

```python
class PlanOutput(BaseModel):
    steps: list[str] = Field(description="...")

planner_llm = llm.with_structured_output(PlanOutput)
result = planner_llm.invoke([
    SystemMessage(content=PLANNER_PROMPT),
    HumanMessage(content=state["task_description"]),
])
# result.steps = ["创建项目目录", "编写 main.py", "添加单元测试", ...]
```

**幂等性保护**：如果 `todo_list` 已有内容，直接返回 `{}`，防止因图的循环导致重复规划。

**初始化职责**：Planner 负责将所有 State 字段清零（`reviewer_reject_count=0`、`evolution_skill_draft=""`等），这相当于每次新任务的"重置按钮"。

### 6.3 index_builder_node

**职责**：在 Coder 开始工作之前，扫描工作区所有 Python 文件，构建三层代码智能索引。

**设计原则**：
- 工作区为空时快速跳过，不阻塞主流程
- 如果 `code_index_ready=True`（已构建），直接返回 `{}`，防止重复构建
- 任何异常都被捕获并记录，主流程不受影响

**对 Coder 的贡献**：构建完成后，Repo Map 的前 1000 字符会被注入到 Coder 的系统提示中，让 Agent 在开始工作前就对代码库结构有全局认知。

### 6.4 coder_node

**职责**：这是系统的核心执行节点，调用带工具绑定的 LLM，执行当前最优先的待办步骤。

**工具绑定**：
```python
# llm_with_tools = llm + 所有工具的 schema 描述
# LLM 可以在回复中"请求"调用工具，但自身不执行
llm_with_tools = llm.bind_tools(TOOLS)
```

**上下文构建策略**（消息压缩的完整逻辑）：

```
情况 A：有 summary（历史已压缩）
  → 只取最近 4 条消息 + summary 注入系统提示
  → 优点：历史信息以摘要形式保留，近期操作保留完整

情况 B：无 summary（历史未压缩）
  → get_safe_recent_messages(max_history=6)
  → 注意：不能简单截取最后 6 条！必须保证 tool_calls 和 ToolMessage 成对出现

→ compact_message_history()（双向压缩）
  → 折叠历史中 write_file/edit_file 的大段代码参数
  → 折叠历史 ToolMessage 的长输出（保留最后一条完整）
```

**Repo Map 注入**：
```python
if repo_map and len(repo_map) > 10:
    repo_map_snippet = repo_map[:1000]
    repo_map_info = f"\n\n【当前代码库结构（Repo Map）】:\n{repo_map_snippet}"
sys_prompt = SystemMessage(content=prompt + summary_info + repo_map_info)
```

**迭代计数**：每次 `coder_node` 执行，`iteration_count += 1`。由节点内部完成，绝不在外部重复递增（否则计数翻倍）。

### 6.5 task_manager_node

**职责**：动态更新 Todo List，这是"分层规划"的核心——计划不是一成不变的，而是根据实际执行情况动态调整。

**触发时机**：工具执行成功（Return Code: 0 或测试通过模式匹配）后触发。执行失败时跳过，直接回到 Coder 继续修复。

**实现**：同样使用 Structured Output 保证返回格式：
```python
class TaskUpdateOutput(BaseModel):
    todo_list: list[str]
    completed_tasks: list[str]
```

**关键设计**：只传入最近 3 条消息给 Task Manager，而不是全部历史，节省 Token 并聚焦于最新执行结果。

### 6.6 reviewer_node

**职责**：当 Coder 声明任务完成（输出 `TASK_COMPLETED`）时，用独立的 LLM 调用进行质量审查。

**双重防护机制**：

**防护 1 — 前置测试检查**（在路由函数中）：
```python
# 路由函数层面的第一道防线
if "TASK_COMPLETED" in content:
    if _has_successful_test(messages):  # 有测试通过证据才放行
        return "reviewer"
    else:
        return "coder"  # 没有测试证据，打回补测试
```

**防护 2 — LLM 审查**（在 reviewer_node 内）：
```python
# 专用 reviewer_llm，独立于主 llm
reviewer_llm = llm.with_structured_output(ReviewOutput)
# 只看最近 6 条消息（聚焦于最终实现和测试结果）
review_result = reviewer_llm.invoke([sys_prompt] + messages[-6:])
```

**熔断机制**：
```python
reject_count = state.get("reviewer_reject_count", 0)
if reject_count >= 3:
    # 强制批准，防止 Reviewer ↔ Coder 无限循环
    return {"status": "success", "reviewer_reject_count": 0, ...}
```

这解决了一个经典问题：Reviewer 可能有不切实际的标准，导致永远驳回。3 次驳回后强制通过，是在质量保证与避免死锁之间的工程权衡。

### 6.7 summarizer_node

**职责**：当消息历史的 Token 消耗超过阈值（15000 Token）时，将旧消息压缩为一段结构化摘要。

**触发条件**：
```python
def route_after_task_manager(state):
    uncompressed = get_uncompressed_messages(state["messages"])
    if count_tokens(uncompressed) > 15000:
        return "summarizer"
    return "coder"
```

**关键函数 `get_uncompressed_messages`**：
```python
# 从消息列表末尾往前扫描，遇到"已压缩"标记就停止
# 这样每次只压缩"上次摘要之后的新增消息"
# 防止把旧摘要再次压缩（否则信息会无限浓缩导致失真）
for msg in reversed(messages):
    uncompressed.append(msg)
    if "[系统通知：历史记录已压缩" in msg.content:
        break
```

**压缩后的注入方式**：Summarizer 的结果存入 `state.summary`，Coder 下次运行时将其附加到系统提示末尾，而不是作为消息传入——这样既保留了信息，又不占用消息槽位。

---

## 7. 路由逻辑：图的神经系统

路由函数是 LangGraph 的条件边，它们纯粹基于当前 State 做决策，不调用 LLM，因此必须快速、确定。

### 7.1 route_after_coder

这是最复杂的路由，决定 Coder 输出后走哪条路：

```
Coder 输出
    │
    ├── 有 tool_calls ──────────────────────────────→ "tools"
    │   （LLM 请求执行工具）
    │
    ├── 包含 "TASK_COMPLETED"
    │       │
    │       ├── _has_successful_test() = True ──────→ "reviewer"
    │       │   （历史中找到测试通过证据）
    │       │
    │       └── _has_successful_test() = False ─────→ "coder"
    │           （声明完成但没有测试证据，打回）
    │
    └── 其他 ───────────────────────────────────────→ "coder"
        （继续工作）
```

**`_has_successful_test` 的多框架支持**：

```python
_TEST_SUCCESS_PATTERNS = [
    r"Return Code: 0\b",      # Shell 退出码
    r"\d+\s+passed",           # pytest: "3 passed in 0.5s"
    r"^OK\s*$",                # unittest: "OK"
    r"\bAll tests passed\b",   # 自定义
    r"\bTests passed\b",
    r"\bPASSED\b",             # \b 词边界，防止误匹配 "SURPASSED"
]
```

### 7.2 route_after_tools

```
工具执行结果
    │
    ├── 匹配成功模式 ─────────────────────────────→ "task_manager"
    │   （Return Code: 0 / X passed / OK 等）
    │
    └── 不匹配 ──────────────────────────→ route_after_task_manager()
        （工具失败，跳过更新清单，直接检查是否需要摘要）
```

**设计意图**：失败时不更新 Todo List，因为任务还没完成；只有真正成功才让 Task Manager 将步骤移至"已完成"。

### 7.3 route_after_reviewer

```
Reviewer 判断
    │
    ├── status == "success" ─────────────────→ "evolution_reflect"
    │   （触发能力演化流程）
    │
    ├── status == "failed" ──────────────────→ "__end__"
    │   （达到最大迭代次数等不可恢复错误）
    │
    └── status == "coding" ──────────────────→ "coder"
        （Reviewer 驳回，继续修改）
```

---

## 8. 工具层（Tools）详解

### 8.1 工具的定义方式

所有工具用 `@tool` 装饰器定义，LangGraph 会自动从函数签名和 docstring 中提取参数 schema，供 LLM 理解如何调用。

```python
@tool
def write_file(file_path: str, content: str) -> str:
    """创建新文件或完全覆盖写入文件。写入后自动更新代码智能索引。"""
    # 实现...
```

对于参数结构复杂的工具，使用 Pydantic Schema 明确约束：
```python
class EditFileArgs(BaseModel):
    file_path: str = Field(..., description="要修改的文件相对路径")
    start_line: int = Field(..., description="...")
    end_line: int = Field(..., description="...")
    replace_text: str = Field(..., description="...")

@tool(args_schema=EditFileArgs)
def edit_file(file_path: str, start_line: int, ...) -> str:
    ...
```

### 8.2 完整工具清单

| 工具名 | 分类 | 核心功能 |
|---|---|---|
| `read_file` | 文件 I/O | 读取文件内容，超过 250 行时自动截断中间（保留头尾） |
| `write_file` | 文件 I/O | 覆盖写入文件，触发 Linter 检查 + 代码索引增量更新 |
| `edit_file` | 文件 I/O | 精确行号范围替换，触发 Linter 检查 + 索引更新 |
| `execute_command` | 执行 | 在 Docker 沙盒中运行 bash 命令 |
| `search_code` | ★ 智能搜索 | Three-Index 融合搜索，支持语义/关键字/自动模式 |
| `get_repo_map` | ★ 代码导航 | 生成代码库符号地图（PageRank 排序） |
| `get_symbol_context` | ★ 代码导航 | 获取函数/类的完整定义 + 调用关系 |
| `search_codebase` | 文本搜索 | grep 精确字符串搜索（fallback） |
| `search_web` | 网络 | Tavily API 搜索，有缓存和熔断，降级到 DuckDuckGo |
| `read_url` | 网络 | 读取网页全文并解析为纯文本 |
| `list_skills` | 技能库 | 展示已积累的可复用 Skill 脚本（含 YAML 元数据） |
| `run_skill` | 技能库 | 在沙盒中执行指定 Skill 脚本 |

### 8.3 DockerSandbox：安全执行环境

**为什么需要 Docker 沙盒**：Agent 会执行任意 bash 命令，必须与宿主机隔离，防止破坏宿主机文件系统或网络。

```python
class DockerSandbox:
    def __init__(self, workspace_dir: Path, image: str = "python:3.10"):
        # 初始化 Docker 客户端，连接失败则优雅降级（self.client = None）

    def start(self):
        # 懒启动：第一次 execute() 时才真正创建容器
        # 容器挂载 workspace_dir 到 /workspace
        # auto_remove=True：进程退出时自动清理容器

    def execute(self, command: str, timeout: int = 60) -> str:
        # 使用列表模式调用 exec_run，避免 shell 注入
        cmd_list = ["bash", "-c", f"cd /workspace && timeout {timeout}s bash -c {repr(command)}"]
        exit_code, output = self.container.exec_run(cmd_list)
        return f"Return Code: {exit_code}\nOutput:\n{output}"
```

**关键设计**：`repr(command)` 对命令进行 Python 的字符串表示转义，而不是简单的字符串拼接，这能防止命令注入。

### 8.4 隐式 Linter 钩子（_auto_lint）

这是一个**不暴露给 LLM 的后台钩子**，在每次 `write_file` 和 `edit_file` 之后自动触发：

```
_auto_lint(file_path)
    │
    ├── Step 1: Python 内置 ast.parse()
    │   → 极速本地检查，捕获 SyntaxError / IndentationError
    │   → 发现错误 → 返回错误信息（追加到工具返回值末尾）
    │
    └── Step 2: 沙盒内 flake8（如果 Docker 可用）
        → 使用 exec_run(["flake8", workspace_file]) 列表模式
        → 只报告 F 系列（未定义变量等逻辑错误）和 E9 系列
        → 忽略 E501（行长度）等格式性警告
```

Agent 不需要主动调用 Linter，它的反馈自动附加在工具调用结果中，Agent 会看到并主动修复。

### 8.5 搜索缓存与熔断机制

```python
_SEARCH_CACHE: dict = {}    # 搜索词 → 结果
_URL_CACHE: dict = {}       # URL → 解析内容
_CACHE_MAXSIZE = 200        # LRU 最大缓存条目

GLOBAL_STATS = {
    "tavily_count": 0,      # 当前已调用次数
    "max_tavily": 15,       # 单任务最大允许次数
}

# 超过上限 → 自动降级到 DuckDuckGo（免费）
# DuckDuckGo 不可用 → 明确返回错误信息
```

**LRU 缓存实现**：利用 Python 3.7+ 字典保持插入顺序的特性：
```python
def _cache_set(cache, key, value):
    if key in cache:
        del cache[key]          # 移到末尾（最近使用）
    elif len(cache) >= _CACHE_MAXSIZE:
        oldest = next(iter(cache))  # 删除最旧条目
        del cache[oldest]
    cache[key] = value
```

---

## 9. Three-Index 代码智能引擎

### 9.1 为什么要替换 grep

传统的 `grep` 方式（原 `search_codebase`）只能做字面字符串匹配：
- 查询 "处理用户认证的函数" → grep 完全不懂，搜不到
- 查询 `validate_user` → grep 能找到，但不知道这个函数调用了什么、被谁调用

Three-Index 引擎解决了这三个层次的问题：

| 层次 | 问题 | 解决方案 |
|---|---|---|
| 结构层 | "这个函数在哪里定义，签名是什么" | AST 解析 + 结构索引 |
| 语义层 | "实现 X 功能的代码在哪里" | 向量嵌入 + 余弦相似度 |
| 符号层 | "精确搜索标识符名" | BM25 关键字索引 |
| 关系层 | "修改这个函数会影响哪些地方" | 依赖图 + PageRank |

### 9.2 CodeChunk：代码的基本单元

```python
@dataclass
class CodeChunk:
    chunk_id: str      # 唯一 ID："src/auth.py::validate_user::42"
    file_path: str     # 相对路径
    chunk_type: str    # "function" | "method" | "class"
    name: str          # 函数名/类名
    signature: str     # 完整签名："def validate_user(token: str) -> bool"
    docstring: str     # 文档字符串（最多 300 字符）
    body: str          # 完整源码（最多 3000 字符）
    start_line: int    # 起始行（1-indexed）
    end_line: int      # 结束行
    calls: list[str]   # 调用的其他函数名列表（最多 30 个）
    language: str      # "python"
    parent_class: str  # 方法所属类名（普通函数为空字符串）
```

`chunk_id` 的格式 `file_path::name::start_line` 保证了全局唯一性，即使不同文件中有同名函数也不冲突。

### 9.3 第一层：AST 结构索引

**主解析器：tree-sitter**

tree-sitter 是一个**增量解析库**，核心优势：
- 比 Python 内置 `ast` 模块快 10-100 倍
- 返回带有精确行列位置的语法树
- 支持容错解析（有语法错误也能部分解析）

```python
# tree-sitter 解析流程
source_bytes = source.encode("utf-8")
tree = _TS_PY_PARSER.parse(source_bytes)

# 遍历语法树，提取函数节点
for node in tree.root_node.children:
    if node.type == "function_definition":
        name = _node_text(get_child(node, "identifier"))
        params = _node_text(get_child(node, "parameters"))
        # 提取调用关系
        calls = _extract_calls_from_node(node)
```

**Fallback：正则表达式解析**

当 `tree-sitter-languages` 未安装时，自动降级到正则解析：
```python
func_re = re.compile(r'^(\s*)(?:async\s+)?def\s+(\w+)\s*(\([^)]*\))\s*(?:->\s*([^:]+))?\s*:')
```
精度比 tree-sitter 低，但无外部依赖。

### 9.4 第二层：语义向量索引（_SemanticIndex）

**原理**：将每个 CodeChunk 的文本表示（名称 + 签名 + 文档 + 少量代码）用 Sentence Transformer 模型编码为固定维度的向量，存储在 numpy 矩阵中。

```python
# 向量化的文本（embed_text 属性）
"validate_user\ndef validate_user(token: str) -> bool\n验证用户 token 的有效性\nif not token:\n    return False..."

# 编码为向量，L2 归一化（方便余弦相似度计算）
vecs = model.encode(texts, batch_size=32)
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
matrix = (vecs / norms).astype(np.float32)  # shape: (n_chunks, 384)
```

**搜索时**：
```python
q_vec = model.encode([query])  # 查询也归一化
scores = matrix @ q_vec[0]     # 矩阵乘法 = 所有 chunk 的余弦相似度
# scores.shape = (n_chunks,)，取 top_k 即可
```

**线程安全**：`_SemanticIndex` 所有读写操作都通过 `threading.Lock` 保护，支持并发查询。

**增量更新**：
```python
def update_file(self, file_path: str, new_chunks: list):
    # 1. 删除该文件的旧向量（通过 boolean mask 过滤）
    keep_mask = [c.file_path != file_path for c in self._chunks]
    kept_matrix = self._matrix[keep_mask]
    # 2. 编码新 chunks 并拼接
    new_matrix = self._embed_texts([c.embed_text for c in new_chunks])
    self._matrix = np.vstack([kept_matrix, new_matrix])
```

### 9.5 第三层：BM25 关键字索引（_BM25Index）

**BM25（Best Match 25）** 是 TF-IDF 的改进版，是搜索引擎的工业标准算法。它考虑：
- TF（词频）：词出现越多，相关性越高，但有饱和上限
- IDF（逆文档频率）：在很多文档都出现的词（如 `def`、`return`）权重低
- 文档长度归一化：长函数不会因为绝对词频高而虚假排名靠前

**代码专用分词器**：
```python
def _tokenize_code(text: str) -> list[str]:
    # camelCase 拆分：FooBar → foo bar
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # 提取所有标识符（字母、数字、下划线）
    words = re.findall(r"[a-zA-Z_]\w*", text)
    result = []
    for w in words:
        # snake_case 拆分：foo_bar → foo bar
        parts = [p.lower() for p in w.split("_") if len(p) >= 2]
        result.extend(parts)
    return result
```

这个分词器能让 `validateUserToken` 和 `validate_user_token` 都能被 "validate user" 这样的查询匹配到。

### 9.6 搜索融合：RRF（Reciprocal Rank Fusion）

当 `mode="auto"` 时，系统同时执行语义搜索和 BM25 搜索，然后用 RRF 算法融合排名：

```python
# RRF 公式：score = Σ(1 / (rank + k))，k=60 是常数（抑制头部偏差）
for rank, (chunk, _) in enumerate(semantic_results):
    scores[chunk.chunk_id] += 1.0 / (rank + 60)

kw_weight = 1.5 if is_symbol else 1.0  # 包含下划线/驼峰 → 偏关键字
for rank, (chunk, _) in enumerate(bm25_results):
    scores[chunk.chunk_id] += kw_weight * (1.0 / (rank + 60))
```

**自动模式判断**：
```python
# 包含 camelCase 或 snake_case 特征 → 这是一个符号名，加重关键字权重
is_symbol = bool(re.search(r"[_A-Z][a-z]|[a-z][A-Z]|_\w", query))
```

### 9.7 依赖图与 PageRank（_DependencyGraph）

**构建过程**：
```
每个 CodeChunk → NetworkX 图中的一个节点
chunk.calls 中的每个被调用函数名 → 有向边（从调用者指向被调用者）
nx.pagerank(G, alpha=0.85) → 每个节点的重要性分数
```

**PageRank 的含义**：被许多其他函数调用的函数（核心基础函数）分数高；只有几处调用的函数分数低。这与 Google 用链接数量衡量网页重要性的原理完全相同。

**Repo Map 生成**：
```
按文件分组 CodeChunks
→ 用 PageRank 文件分数（文件内所有函数分数之和）排序文件
→ 若有查询，语义相关文件分数额外加权
→ 在每个文件下，按函数 PageRank 降序列出
→ 每个函数显示：签名 + 重要性条形图 + 行号 + 文档摘要
```

典型输出：
```
📦 src/agent/swe/graph.py
  ├── def coder_node(state: AgentState) -> Dict  [■■■□] L280
  │     执行当前最优先的任务步骤。
  ├── def planner_node(state: AgentState) -> Dict  [■■□□] L175
  │     拆解任务为原子步骤。
  ├── def route_after_coder(state) -> Literal  [■□□□] L425
```

### 9.8 增量更新钩子

`write_file` 和 `edit_file` 工具内嵌了增量更新触发器：

```python
@tool
def write_file(file_path: str, content: str) -> str:
    # ... 写入文件 ...
    if file_path.endswith(".py"):
        _trigger_index_update(file_path)  # 非阻塞！
    return f"Success..."

def _trigger_index_update(file_path: str) -> None:
    # 在后台线程中执行，不阻塞工具调用返回
    def _do_update():
        update_file_index(file_path)
    t = threading.Thread(target=_do_update, daemon=True)
    t.start()
```

这保证了：Agent 每次写入新代码后，代码索引会自动更新，下次搜索就能找到新写的函数。

---

## 10. Capability Evolution Loop

### 10.1 设计动机

每次任务成功后，系统会问：**"这次解决的问题有没有可能在其他项目中再次遇到？如果有，我能不能把解决方案固化下来，下次直接复用？"**

这模仿了人类工程师积累"经验工具箱"的方式——解决了一个复杂的 CI/CD 配置问题？写成一个可参数化的脚本存起来，下次一行命令搞定。

### 10.2 三阶段流程

**阶段 1：evolution_reflect_node（观察 + 反思 + 决策）**

输入：`task_description`、`completed_tasks`、`summary`、当前技能库清单

决策框架（用于 LLM 提示）：
```
✅ 高价值，固化为 Skill：
  - 解决了需要查阅文档的通用技术配置问题
  - 包含可参数化的模板代码
  - 未来其他项目很可能再次用到

❌ 低价值，跳过：
  - 高度业务耦合，无法跨项目复用
  - 极其简单（< 15 行代码）
  - 技能库中已有功能高度重叠的版本
```

**双重去重**：
1. LLM 自身判断（语义层面）
2. Jaccard 相似度检查（字面层面）

```python
# Jaccard 相似度：交集大小 / 并集大小
words_a = set(re.findall(r"\w+", text_a.lower()))
words_b = set(re.findall(r"\w+", text_b.lower()))
similarity = len(words_a & words_b) / len(words_a | words_b)
if similarity > 0.65:
    return existing_skill_name  # 跳过，已有类似技能
```

**阶段 2：evolution_generate_node（技能脚本生成）**

调用 LLM 生成完整 Python 脚本（temperature=0.25，比主 LLM 稍高，增加生成创造性）。

生成的脚本格式规范：
```python
# ---
# name: setup_fastapi.py
# category: scaffold
# description: 快速搭建 FastAPI 项目骨架
# version: 1.0
# created_at: 2025-01-15
# usage: python setup_fastapi.py [--help]
# ---
import argparse

def setup_fastapi(project_name: str, port: int = 8000) -> None:
    """核心功能，方便其他脚本 import 复用"""
    ...

def __test__() -> bool:
    """无副作用的自验证函数"""
    # 验证参数解析是否正常工作
    return True  # 验证通过

if __name__ == "__main__":
    if __test__():
        # 执行主逻辑
        ...
```

草稿写入 `workspace/_evolution_drafts/`，而不是直接写入 `skills/`，等待验证通过后再移动。

**阶段 3：evolution_verify_node（验证 + 持久化）**

```
Step 1: AST 语法检查（Python 内置 ast.parse）
    → 失败 → 删除草稿，报告错误

Step 2: 沙盒运行 __test__()（如果 Docker 可用）
    → 以安全的列表模式调用（防止路径注入）
    → exec_run(["python", "-c", test_script, f"/workspace/{rel_draft}"])
    → 输出 "SKILL_TEST_FAIL" → 删除草稿，报告失败

Step 3: 版本化持久化
    → _get_versioned_skill_path()：如果同名已存在，用 _v2、_v3 后缀
    → 超过 99 个版本用时间戳后缀（极端情况兜底）
    → shutil.copy2() 复制到 skills/，删除草稿
```

**任何阶段失败的保护**：整个 Evolution Loop 都被 `try/except` 包裹，任何异常只会更新 `evolution_report` 字段（写入错误信息），绝不影响主任务的 `status: "success"` 状态。

### 10.3 技能库的使用

积累的技能通过 `list_skills` 工具暴露给 Coder：

```
💪 已掌握的肌肉记忆 (技能库):

🏗️ [SCAFFOLD]
   • setup_fastapi.py  —  快速搭建 FastAPI 项目骨架  (v1.0)

🧪 [TEST]
   • setup_pytest.py  —  配置 pytest 和覆盖率报告  (v1.0)

共 2 个技能。使用 run_skill('文件名') 直接执行。
```

---

## 11. 记忆管理系统

### 11.1 问题背景：Context Window 的限制

LLM 有处理上限（Context Window），例如 128K Token。一个复杂任务可能产生数百条工具调用记录，很快就会溢出。简单截断会导致 Agent "忘记"之前的工作，陷入重复犯错的循环。

### 11.2 三层记忆架构

```
┌─────────────────────────────────────────────────────┐
│ 第一层：压缩摘要（Long-Term Memory）                │
│   state.summary                                     │
│   由 Summarizer 节点生成                           │
│   存储：已完成工作 / 测试状态 / 踩坑记录 / 遗留问题 │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│ 第二层：近期消息窗口（Working Memory）              │
│   state.messages[-4:] 或 [-6:]                     │
│   完整保留最近的工具调用和 LLM 回复                 │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│ 第三层：消息双向压缩（Compression）                 │
│   compact_message_history()                         │
│   - 折叠历史中 write_file 的大段代码参数            │
│   - 折叠历史 ToolMessage 的长输出                   │
│   - 保留最后一条工具输出的完整内容                  │
└─────────────────────────────────────────────────────┘
```

### 11.3 安全截取消息的难点

LangGraph 中消息有一个强约束：**工具调用消息（AIMessage 含 tool_calls）和工具结果消息（ToolMessage）必须成对出现**。如果截断点恰好在 ToolMessage，而对应的 AIMessage 被截掉了，LLM API 会报 400 错误。

```python
def get_safe_recent_messages(messages, max_history=8):
    kept = []
    safety_limit = max_history * 2  # 防止极端情况死循环
    for msg in reversed(messages):
        kept.append(msg)
        if len(kept) >= max_history:
            # 如果当前消息是 ToolMessage，继续往前找，
            # 直到找到发起这个工具调用的 AIMessage
            if getattr(msg, "type", "") != "tool":
                break  # 安全截断点
    return list(reversed(kept))
```

---

## 12. 提示词工程

### 12.1 提示词设计原则

本项目中所有提示词都遵循以下原则：

**角色清晰**：每个节点的 LLM 调用都有明确的角色定义（"你是一个资深软件架构师"、"你是一个严苛的高级代码审查员"）。

**约束前置**：最重要的约束（如"严禁使用多个 write_file"）放在显眼位置，用特殊标记（【】）和 emoji 突出。

**输出格式强制**：对于需要结构化输出的节点，提示词中明确说明"请以 JSON 格式输出"，并配合 `with_structured_output` 双重保障。

**正反例提示**：对于容易误解的约束，同时给出"什么要做"和"什么不能做"。

### 12.2 CODER_PROMPT_TEMPLATE 的结构

```
[角色定位]
你是一个顶级的全栈开发工程师 Agent。

[环境信息]（运行时填充）
当前工作区目录: {workspace}
当前计划: {plan}
任务描述: {task_description}

[进度状态]（运行时填充）
✅ 已完成：{completed_tasks}
⏳ 待处理：{todo_list}

[核心指令]
当前目标：完成 "{current_step}"

[行为约束模块]（按优先级排列）
1. 完工协议（任务范围限制）
2. 代码智能导航（优先使用三索引工具）
3. 肌肉记忆（技能库复用）
4. 严禁贪多（原子操作约束）
5. 知识增强（查文档流程）
6. 强制测试闭环（不得口头完成）
7. 操作规范（调用工具前说明意图）

[工具清单]（让 LLM 知道有什么可用）
```

### 12.3 结构化输出的双保险

纯粹依赖 LLM 的 Structured Output 不够可靠（模型可能在 JSON 中嵌套字典），因此配合提示词强调：

```
【重要警告】：JSON 中的 steps 数组的每个元素必须是纯文本字符串，
绝对不要输出嵌套的字典或对象！
```

加上 Pydantic 的类型校验：

```python
class PlanOutput(BaseModel):
    steps: list[str] = Field(
        description="每个元素必须是纯文本字符串，绝对不能是字典！"
    )
```

两道防线确保解析不会崩溃。

---

## 13. 安全体系

### 13.1 路径安全：防止路径穿越攻击

**攻击场景**：Agent 可能被提示输出 `file_path="../../../etc/passwd"`，如果不校验，`write_file` 就会覆盖系统文件。

**防御**：`normalize_path()` 函数是所有文件操作的必经之路：

```python
def normalize_path(file_path: str) -> Path:
    # 1. 绝对路径白名单：只允许 /workspace/ 前缀
    if os.path.isabs(file_path):
        if file_path.startswith("/workspace/"):
            file_path = file_path.replace("/workspace/", "", 1)
        else:
            raise ValueError(f"禁止使用绝对路径")

    # 2. resolve() 解析所有 .. 跳转和符号链接
    resolved = (WORKSPACE_DIR / file_path).resolve()

    # 3. 前缀检查：确认最终路径在工作区内
    workspace_resolved = WORKSPACE_DIR.resolve()
    if not str(resolved).startswith(str(workspace_resolved) + os.sep):
        raise ValueError(f"路径逃逸攻击被拦截: {file_path}")

    return resolved
```

**符号链接绕过防御**：`resolve()` 不只解析 `..`，还会跟随符号链接到真实路径。假设攻击者在工作区内创建了一个指向 `/etc` 的符号链接 `workspace/evil_link`，`resolve()` 会将其展开为 `/etc`，然后前缀检查会发现它不在工作区内并拒绝。

### 13.2 技能名安全：防止技能目录逃逸

`run_skill` 工具接受用户指定的技能文件名，必须严格校验：

```python
# 禁止路径分隔符（防止 ../tricks.py）
if "/" in skill_name or "\\" in skill_name or ".." in skill_name:
    return "Error: 非法的 skill_name"

# 必须以 .py 结尾（防止执行任意可执行文件）
if not skill_name.endswith(".py"):
    return "Error: 只允许 .py 文件"

# resolve() 二次确认（防止边缘 symlink 绕过）
resolved = (SKILLS_DIR / skill_name).resolve()
if not str(resolved).startswith(str(SKILLS_DIR.resolve())):
    return "Error: 路径逃逸攻击被拦截"
```

### 13.3 SSRF 防御：read_url 的 URL 校验

**攻击场景**：Agent 可能被欺骗访问 `file:///etc/passwd`（读取本地文件）或 `http://169.254.169.254`（AWS 元数据服务）。

```python
parsed = urlparse(url)
if parsed.scheme not in ("http", "https"):
    return "Error: 不支持的 URL scheme"
if not parsed.netloc:
    return "Error: 无效的 URL，缺少域名"
```

### 13.4 Shell 注入防御：Docker exec_run 列表模式

**危险写法**：
```python
# 攻击者控制 command = "ls; rm -rf /workspace"
container.exec_run(f"bash -c '{command}'")  # 危险！shell 解释特殊字符
```

**安全写法**：
```python
# 使用列表模式，每个参数是独立的字符串，shell 不会解释特殊字符
cmd_list = ["bash", "-c", f"cd /workspace && timeout {timeout}s bash -c {repr(command)}"]
container.exec_run(cmd_list, workdir="/workspace")
```

`repr(command)` 将命令包裹在 Python 的字符串表示中，内部的特殊字符会被转义。

### 13.5 API 认证

```python
def _verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    if not _API_KEY:
        return  # 未配置 SWE_API_KEY → 开发模式，跳过认证
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
```

通过 `Depends(_verify_api_key)` 注入到所有需要保护的路由，Header 名 `X-API-Key` 是 REST API 的行业惯例。

### 13.6 Thread ID 注入防御

```python
# webapp.py 和 api.py 共同的校验逻辑
_THREAD_ID_RE = re.compile(r"^[\w\-]{1,64}$")
if not _THREAD_ID_RE.match(raw_thread_id):
    raise ValueError("thread_id 只允许字母、数字、下划线、连字符")
```

防止用户输入 `../../secrets` 这类的 thread_id，被 LangGraph checkpointer 用于文件路径。

---

## 14. FastAPI 后端

### 14.1 架构定位

FastAPI (`api.py`) 和 Streamlit (`web_app.py`) 共享同一个 `graph` 对象。这意味着：
- 两者可以同时运行（不同端口）
- 使用相同的 `InMemorySaver`（共享状态，仅限开发）
- 生产环境应换用 `PostgresSaver` 或 `RedisSaver`，两者通过数据库共享状态

### 14.2 SSE 流式推送的实现

```python
@app.get("/api/tasks/{task_id}/stream")
async def stream_task(task_id: str):
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)  # 有界队列，防 OOM

    def _run_stream():
        # 在线程池中运行同步的 graph.stream()
        for event in graph.stream(input_data, config=config, stream_mode="updates"):
            loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    loop.run_in_executor(_executor, _run_stream)  # 线程池执行

    async def event_generator():
        while True:
            try:
                msg_type, data = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"  # 保活
                continue
            # ... 处理 event / done / error ...

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},  # 禁用 Nginx 缓冲，确保实时推送
    )
```

**关键技术点**：
- `graph.stream()` 是同步 API，必须在线程池（executor）中运行
- `asyncio.Queue` 作为线程和协程之间的安全通信管道
- `loop.call_soon_threadsafe` 是线程安全地向异步队列写入的正确方式
- 30 秒超时 + 心跳包，防止代理/负载均衡器断开空闲连接

### 14.3 完整 API 端点清单

| 方法 | 路径 | 功能 |
|---|---|---|
| GET | `/api/health` | 健康检查（无需鉴权） |
| POST | `/api/tasks` | 创建任务，返回 task_id |
| GET | `/api/tasks` | 列出所有任务 |
| GET | `/api/tasks/{id}` | 获取任务详情和当前 State |
| GET | `/api/tasks/{id}/stream` | SSE 实时流（执行 + 推送） |
| POST | `/api/tasks/{id}/action` | 人在回路审批（approve/reject） |
| DELETE | `/api/tasks/{id}` | 删除任务记录 |
| GET | `/api/skills` | 列出技能库 |
| GET | `/api/skills/{name}/source` | 获取技能脚本源码 |
| GET | `/api/index/stats` | 代码索引统计 |
| GET | `/api/index/repo-map` | 获取 Repo Map |
| GET | `/api/index/search` | 三索引代码搜索 |
| GET | `/api/index/symbol/{name}` | 符号上下文查询 |
| POST | `/api/index/rebuild` | 手动触发索引全量重建 |

---

## 15. Streamlit 前端

### 15.1 界面布局

```
侧边栏（持久显示）
├── 会话 ID 输入（thread_id 格式校验）
├── 任务完成度进度条
├── 关键指标（Tavily 消耗 / 迭代轮次 / 技能数量）
├── 代码智能索引状态
└── 任务蓝图（已完成 / 当前 / 后续任务）

主区域
├── 实时位置流水线（主节点行 + Evolution Loop 行）
│
└── Tabs
    ├── 💬 交互中心（聊天界面 + HiL 审批）
    ├── 🧠 记忆矩阵（summary 展示）
    ├── 🗺️ Repo Map（代码地图 + 实时搜索）
    ├── 🧬 技能进化（Evolution 报告 + 技能库）
    └── 📁 工作区看板（文件树 + 代码预览）
```

### 15.2 实时节点位置可视化

```python
state_raw = graph.get_state(config)
next_node = state_raw.next[0] if (state_raw and state_raw.next) else "idle"

for node_id in main_nodes:
    if next_node == node_id:
        st.markdown(f"<div class='step-active'>{name}</div>")  # 蓝色高亮
        st.caption("● 正在此处...")
    else:
        st.markdown(f"<div class='step-inactive'>{name}</div>")  # 灰色
```

`graph.get_state(config)` 返回的 `.next` 字段是 LangGraph 记录的"下一个将执行的节点"列表，利用这个字段就能实时显示 Agent 当前所处的阶段。

### 15.3 自动流转逻辑

```python
if current_state and current_state.next:
    node_now = current_state.next[0]

    if node_now == "tools":
        # 需要人类审批 → 展示审批 UI，停止自动流转
        show_approval_ui()

    elif node_now in evolution_nodes:
        # Evolution 节点 → 自动流转，无需人类干预
        with st.spinner(f"🧬 {NODES_MAP[node_now]} 自动执行中..."):
            run_agent_ui(None)
            st.rerun()

    else:
        # 其他节点（planner, coder 等）→ 自动流转
        with st.spinner(...):
            run_agent_ui(None)
            st.rerun()
```

**熔断机制**：`run_agent_ui` 内部维护 `no_tool_count` 计数器。如果 Coder 连续 3 次都没有发出工具调用（可能陷入循环思考），自动停止并报警。

---

## 16. 完整数据流演示

以任务 **"编写一个 Python HTTP 服务器，返回系统时间"** 为例，逐步追踪数据流。

### 阶段 1：任务启动

```
用户输入: "编写一个 Python HTTP 服务器，返回系统时间"
    ↓
webapp.py 构建初始 State：
{
  "messages": [HumanMessage("编写一个 Python HTTP 服务器，返回系统时间")],
  "task_description": "编写一个 Python HTTP 服务器，返回系统时间",
  "todo_list": [],
  "iteration_count": 0,
  "code_index_ready": False,
  ...
}
    ↓
graph.stream(init_data, config) 开始执行
```

### 阶段 2：Planner

```
planner_node 接收 State
    ↓
LLM（with_structured_output(PlanOutput)）生成：
{
  "steps": [
    "创建 server.py 文件",
    "使用 http.server 模块实现 GET 接口",
    "接口返回当前系统时间（JSON 格式）",
    "创建测试脚本 test_server.py",
    "运行测试并验证返回结果"
  ]
}
    ↓
State 更新：
{
  "todo_list": ["创建 server.py 文件", "使用 http.server...", ...],
  "completed_tasks": [],
  "code_index_ready": False,
  ...
}
```

### 阶段 3：IndexBuilder

```
index_builder_node 接收 State
    ↓
扫描 workspace/ → 无 Python 文件（新任务）
    ↓
State 更新：
{
  "code_index_ready": True,
  "repo_map": "（工作区暂无 Python 文件，索引为空）"
}
```

### 阶段 4：Coder 第一轮

```
coder_node 接收 State
    ↓
构建系统提示（包含 todo_list、current_step="创建 server.py 文件"）
    ↓
LLM 回复（含 tool_call）：
AIMessage(
  content="我将创建 server.py 文件，实现基于 http.server 的 HTTP 服务器。",
  tool_calls=[{
    "name": "write_file",
    "args": {
      "file_path": "server.py",
      "content": "from http.server import HTTPServer, BaseHTTPRequestHandler\n..."
    }
  }]
)
    ↓
route_after_coder → "tools"（因为有 tool_calls）
```

### 阶段 5：Tools 执行

```
ToolNode 执行 write_file("server.py", "...")
    ↓
后台线程：_trigger_index_update("server.py")
           ↓ 解析 server.py → 生成 CodeChunks
           ↓ 更新三层索引
    ↓
_auto_lint("server.py")
  → ast.parse() 通过
  → flake8 检查通过
    ↓
返回 ToolMessage：
"Success: 文件 'server.py' 已成功写入。\n\n✅ [Linter] 语法检查通过。"
    ↓
route_after_tools：
  内容不含 "Return Code: 0" → route_after_task_manager
  Token 未超阈值 → "coder"
```

### 阶段 6：Coder 多轮迭代

```
[重复多次 coder → tools → coder 循环，直到所有步骤完成]

Coder 第 N 轮：运行测试
  tool_call: execute_command("python test_server.py")
    ↓
ToolMessage: "Return Code: 0\nOutput:\n测试通过：HTTP 200，返回时间格式正确"
    ↓
route_after_tools → "task_manager"（匹配 "Return Code: 0"）
    ↓
task_manager_node：
  todo_list = ["运行测试并验证"]
  completed_tasks = ["创建 server.py", "实现接口", ...]
  → 更新 State 中的清单
    ↓
route_after_task_manager → "coder"（Token 未超阈值）
    ↓
Coder 发现所有任务已完成 → 输出：
"所有步骤已完成，测试通过。TASK_COMPLETED"
```

### 阶段 7：Reviewer 审查

```
route_after_coder：
  包含 "TASK_COMPLETED" + _has_successful_test() = True
  → "reviewer"
    ↓
reviewer_node：
  reviewer_llm 审查最近 6 条消息
  → decision: "approve"
  → feedback: "代码实现完整，测试通过，功能符合要求。"
    ↓
State: {status: "success", reviewer_reject_count: 0}
```

### 阶段 8：Evolution Loop

```
route_after_reviewer → "evolution_reflect"
    ↓
evolution_reflect_node：
  任务：HTTP 服务器
  决策：should_evolve = true
  skill_name: "setup_http_server.py"
    ↓
evolution_generate_node：
  LLM 生成可参数化的 HTTP 服务器脚本（带端口参数）
  写入 workspace/_evolution_drafts/setup_http_server.py
    ↓
evolution_verify_node：
  AST 语法检查通过
  沙盒运行 __test__() → SKILL_TEST_PASS
  copy2 到 workspace/skills/setup_http_server.py
    ↓
State: {
  evolution_report: "✅ 新技能已固化入库：setup_http_server.py",
  evolution_skill_draft: ""
}
    ↓
END
```

---

## 17. 依赖关系与安装

### 17.1 核心依赖（必需）

```bash
pip install langchain langchain-openai langgraph
pip install python-dotenv pydantic requests
```

### 17.2 推荐依赖

```bash
# 代码智能索引
pip install tree-sitter-languages    # AST 精确解析
pip install rank_bm25                # BM25 关键字搜索
pip install sentence-transformers    # 语义向量索引（会下载约 90MB 模型）
pip install networkx                 # 依赖图 + PageRank

# 工具支持
pip install docker         # Docker 沙盒
pip install beautifulsoup4 # HTML 解析（read_url 工具）
pip install tiktoken       # 精确 Token 计数
pip install duckduckgo-search  # 搜索降级
```

### 17.3 生产环境依赖

```bash
# FastAPI
pip install fastapi uvicorn

# Streamlit
pip install streamlit

# 持久化 Checkpointer（选其一）
pip install langgraph-checkpoint-postgres  # PostgreSQL
pip install langgraph-checkpoint-redis     # Redis
```

### 17.4 降级策略总结

| 功能 | 最优 | 降级 | 无任何依赖 |
|---|---|---|---|
| AST 解析 | tree-sitter | Python `ast` + 正则 | 无代码智能 |
| 语义搜索 | sentence-transformers | — | 纯 BM25 |
| 关键字搜索 | rank_bm25 | — | grep |
| 依赖图 | networkx | — | 无 PageRank |
| 命令执行 | Docker 沙盒 | — | 不可执行 |
| Token 计数 | tiktoken | 字符数/4 估算 | 字符数/4 |
| 网络搜索 | Tavily API | DuckDuckGo | 不可搜索 |

---

## 18. 配置参考

### 18.1 .env 文件

```bash
# LLM 配置（必需）
OPENAI_API_KEY=sk-xxxx
OPENAI_BASE_URL=https://api.openai.com/v1  # 或阿里云/Azure等兼容端点
MODEL_NAME=gpt-4o                          # 或 qwen3.5-plus 等

# 代码智能索引
SWE_EMBED_MODEL=all-MiniLM-L6-v2          # Sentence Transformer 模型名

# 工作区
SWE_WORKSPACE_DIR=/absolute/path/to/workspace  # 默认：项目根目录下的 workspace/

# 网络搜索
TAVILY_API_KEY=tvly-xxxx                   # 可选，不配置则降级到 DuckDuckGo

# FastAPI
SWE_API_KEY=your-secret-key               # 可选，不配置则无鉴权（开发模式）
CORS_ORIGINS=http://localhost:3000,https://your-frontend.com
SWE_MAX_WORKERS=4                          # SSE 线程池大小
```

### 18.2 启动命令

```bash
# Streamlit（开发模式）
streamlit run web_app.py

# FastAPI（生产模式）
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 2

# 同时运行两者（不同端口）
streamlit run web_app.py --server.port 8501 &
uvicorn src.api:app --port 8000 &
```

---

## 19. 项目演进历程

本项目经历了多轮迭代，每轮都解决了特定的工程问题：

### 第一版：基础 SWE Agent

最初版本包含基本的 Planner → Coder → Reviewer 闭环，以及 Docker 沙盒和文件操作工具。

**发现的核心问题**：
- `get_safe_state` 存在无限递归（调用自身）
- `iteration_count` 被外部和节点内部双重递增
- `graph.resume()` 方法不存在
- `state.values["summary"]` 在错误作用域使用

### 第二版：安全加固

针对第一版的 Bug 进行全面修复，同时发现并修复了多个安全漏洞：

- `_auto_lint` 中的路径穿越漏洞（使用 `WORKSPACE_DIR / file_path` 而非 `normalize_path`）
- `run_skill` 对 skill_name 无校验（可传入 `../evil.py`）
- `read_url` 无 URL scheme 校验（可访问 `file://` 协议）
- Docker `exec_run` 使用字符串拼接（存在 shell 注入风险）

### 第三版：Capability Evolution Loop

新增了自我成长机制，使 Agent 能够将成功的解决方案固化为可复用技能。

**关键设计决策**：
- 原方案的 5 阶段（Observe/Reflect/Decide/Mutate/Verify）合并为 3 节点（节省 LLM 调用）
- "Holdout 验证"替换为更实际的 AST 检查 + `__test__()` 沙盒运行
- "multi-model debate" 替换为更轻量的 Jaccard 去重

### 第四版：FastAPI + 生产化

新增 FastAPI 后端，支持：
- REST API 接口化调用
- SSE 实时流式推送（解决 Streamlit 轮询的延迟问题）
- API Key 认证
- 人在回路 HTTP 接口

### 第五版：Three-Index 代码智能（当前版本）

引入代码智能引擎，彻底替换 grep 搜索：

- **方案选择**：放弃 GraphRAG（需要图数据库）和 CGM（需要修改模型 attention），选择 Agent Brain Three-Index + Aider Repo Map 的混合方案——零新服务依赖，所有依赖均可选且可降级
- `index_builder_node` 插入 planner 和 coder 之间，Repo Map 注入 Coder 上下文
- `write_file`/`edit_file` 工具后台触发增量索引更新
- 新增 3 个智能工具：`search_code`、`get_repo_map`、`get_symbol_context`
- 新增 5 个 FastAPI 端点暴露代码智能能力

---

*本文档基于项目当前版本（含 Three-Index + Evolution Loop + FastAPI）撰写。随着项目迭代，建议定期对照代码更新相关章节。*
