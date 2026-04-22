# src/api.py
"""
FastAPI Backend for Savant SWE Agent
=====================================
提供 REST API + SSE 流式接口，供生产部署、CI/CD 集成和外部调用使用。

与 webapp.py (Streamlit) 的关系：
  - 两者均直接导入 graph.py，可独立运行
  - FastAPI 适合生产部署 / 多用户 / 程序化调用
  - Streamlit 适合本地开发 / 快速演示
  - 注意：两者共享同一个进程内的 InMemorySaver 时会共享状态；
    生产环境建议改用 PostgresSaver/RedisSaver。

启动命令：
  uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator

load_dotenv()

logger = logging.getLogger("SWE_API")

# 延迟导入 graph，避免在 import 时就触发 LLM 初始化（加快测试速度）
from src.agent.swe.graph import graph
from src.agent.swe.tools import WORKSPACE_DIR, SKILLS_DIR, GLOBAL_STATS
from src.agent.swe.evolution import parse_skill_metadata

# ==========================================
# 应用初始化
# ==========================================
app = FastAPI(
    title="Savant SWE Agent API",
    version="1.0.0",
    description="REST + SSE API for the Savant SWE Agent",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 线程池：LangGraph stream() 是同步 API，需在线程中运行
_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("SWE_MAX_WORKERS", "4")),
    thread_name_prefix="swe_agent",
)

# 任务注册表（内存）：task_id → TaskMeta
_tasks: dict[str, dict] = {}

# API Key（不配置则为开放模式，仅用于开发环境）
_API_KEY = os.getenv("SWE_API_KEY", "")

# Thread ID 安全校验正则（只允许字母、数字、下划线、连字符）
_THREAD_ID_RE = re.compile(r"^[\w\-]{1,64}$")


# ==========================================
# 认证
# ==========================================
def _verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    """
    简单的 API Key 认证。
    不配置 SWE_API_KEY 时跳过验证（开发模式）。
    """
    if not _API_KEY:
        return  # 开发模式：无需鉴权
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ==========================================
# 请求 / 响应 Models
# ==========================================
class CreateTaskRequest(BaseModel):
    task_description: str = Field(..., min_length=1, max_length=4000, description="任务描述")
    max_iterations: int = Field(default=25, ge=1, le=50)
    thread_id: Optional[str] = Field(default=None, description="会话ID（留空自动生成）")

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _THREAD_ID_RE.match(v):
            raise ValueError("thread_id 只允许字母、数字、下划线、连字符，长度 1~64")
        return v


class ActionRequest(BaseModel):
    action: str = Field(..., description="approve 或 reject")
    feedback: Optional[str] = Field(default=None, max_length=2000, description="驳回时的反馈")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        if v not in ("approve", "reject"):
            raise ValueError("action 只能是 'approve' 或 'reject'")
        return v


class TaskResponse(BaseModel):
    task_id: str
    thread_id: str
    status: str
    created_at: str
    task_description: str


# ==========================================
# 序列化辅助
# ==========================================
def _serialize_message(msg) -> dict:
    return {
        "type": getattr(msg, "type", "unknown"),
        "content": str(getattr(msg, "content", "")),
        "tool_calls": getattr(msg, "tool_calls", None),
    }


def _serialize_event(event: dict) -> dict:
    """将 LangGraph stream event 序列化为 JSON-safe dict。"""
    result = {}
    for node_name, node_output in event.items():
        serialized = {}
        for k, v in node_output.items():
            if k == "messages":
                serialized[k] = [_serialize_message(m) for m in v]
            else:
                try:
                    json.dumps(v)  # 测试是否可序列化
                    serialized[k] = v
                except TypeError:
                    serialized[k] = str(v)
        result[node_name] = serialized
    return result


def _make_initial_state(task: dict) -> dict:
    return {
        "messages": [HumanMessage(content=task["task_description"])],
        "task_description": task["task_description"],
        "plan": [],
        "todo_list": [],
        "completed_tasks": [],
        "summary": "",
        "workspace": str(WORKSPACE_DIR.absolute()),
        "iteration_count": 0,
        "max_iterations": task.get("max_iterations", 25),
        "status": "coding",
        "test_passed": False,
        "reviewer_reject_count": 0,
        "evolution_skill_draft": "",
        "evolution_report": "",
        # Three-Index Code Intelligence
        "code_index_ready": False,
        "repo_map": "",
    }


# ==========================================
# 路由
# ==========================================

@app.get("/api/health")
async def health():
    """健康检查，无需鉴权。"""
    return {
        "status": "ok",
        "workspace": str(WORKSPACE_DIR),
        "skill_count": len(list(SKILLS_DIR.glob("*.py"))),
        "tavily_usage": f"{GLOBAL_STATS['tavily_count']}/{GLOBAL_STATS['max_tavily']}",
        "active_tasks": len(_tasks),
    }


@app.post(
    "/api/tasks",
    response_model=TaskResponse,
    dependencies=[Depends(_verify_api_key)],
    summary="创建新任务",
)
async def create_task(request: CreateTaskRequest):
    """创建并注册一个新的 Agent 任务，返回 task_id 供后续操作使用。"""
    task_id = str(uuid.uuid4())
    # 用户指定 thread_id 时直接用，否则派生自 task_id
    thread_id = request.thread_id or f"task_{task_id[:12]}"

    now = datetime.utcnow().isoformat() + "Z"
    _tasks[task_id] = {
        "task_id": task_id,
        "thread_id": thread_id,
        "status": "created",
        "created_at": now,
        "task_description": request.task_description,
        "max_iterations": request.max_iterations,
    }
    logger.info(f"任务已创建: {task_id} (thread: {thread_id})")

    return TaskResponse(
        task_id=task_id,
        thread_id=thread_id,
        status="created",
        created_at=now,
        task_description=request.task_description,
    )


@app.get(
    "/api/tasks",
    dependencies=[Depends(_verify_api_key)],
    summary="列出所有任务",
)
async def list_tasks():
    return {"tasks": list(_tasks.values()), "total": len(_tasks)}


@app.get(
    "/api/tasks/{task_id}",
    dependencies=[Depends(_verify_api_key)],
    summary="获取任务详情",
)
async def get_task(task_id: str):
    """返回任务元信息 + 当前 Graph State 摘要。"""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _tasks[task_id]
    config = {"configurable": {"thread_id": task["thread_id"]}}

    graph_state: dict = {}
    next_node: Optional[str] = None
    try:
        state = graph.get_state(config)
        if state and state.values:
            graph_state = state.values
            next_node = state.next[0] if state.next else None
    except Exception:
        pass

    return {
        **task,
        "next_node": next_node,
        "iteration_count": graph_state.get("iteration_count", 0),
        "todo_list": graph_state.get("todo_list", []),
        "completed_tasks": graph_state.get("completed_tasks", []),
        "evolution_report": graph_state.get("evolution_report", ""),
        "code_index_ready": graph_state.get("code_index_ready", False),
        "pending_approval": next_node == "tools",
    }


@app.get(
    "/api/tasks/{task_id}/stream",
    dependencies=[Depends(_verify_api_key)],
    summary="SSE 流式输出 Agent 执行过程",
    response_class=StreamingResponse,
)
async def stream_task(task_id: str):
    """
    SSE 接口，实时推送 Agent 的节点事件、消息输出和状态变更。

    事件格式（每个事件以 \\n\\n 结尾）：
      data: {"type": "start"|"node_event"|"heartbeat"|"done"|"error", ...}

    当 needs_approval=true 时，客户端应调用 POST /api/tasks/{id}/action 来决策工具执行。
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _tasks[task_id]
    thread_id = task["thread_id"]
    config = {"configurable": {"thread_id": thread_id}}

    # 判断是全新启动还是从断点恢复
    input_data: Optional[dict] = None
    try:
        existing_state = graph.get_state(config)
        is_fresh = not (existing_state and existing_state.values)
    except Exception:
        is_fresh = True

    if is_fresh:
        input_data = _make_initial_state(task)

    loop = asyncio.get_event_loop()
    # 使用有界 Queue（maxsize=200），防止内存无限增长
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    def _run_stream():
        """在线程中运行同步的 graph.stream()，通过 Queue 与 async 主线程通信。"""
        try:
            _tasks[task_id]["status"] = "running"
            for event in graph.stream(input_data, config=config, stream_mode="updates"):
                # call_soon_threadsafe 是线程安全的异步入队方式
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
                except asyncio.QueueFull:
                    logger.warning(f"SSE queue full for task {task_id}, dropping event")
            _tasks[task_id]["status"] = "completed"
        except Exception as e:
            logger.error(f"Stream error for task {task_id}: {e}")
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
            _tasks[task_id]["status"] = "error"
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    loop.run_in_executor(_executor, _run_stream)

    async def event_generator():
        yield f"data: {json.dumps({'type': 'start', 'task_id': task_id})}\n\n"
        try:
            while True:
                try:
                    msg_type, data = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # 心跳包，防止代理/负载均衡器因长时间无数据而断开连接
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    continue

                if msg_type == "done":
                    # 获取最终状态中的 evolution_report
                    final_report = ""
                    try:
                        final_state = graph.get_state(config)
                        if final_state and final_state.values:
                            final_report = final_state.values.get("evolution_report", "")
                    except Exception:
                        pass
                    yield f"data: {json.dumps({'type': 'done', 'task_id': task_id, 'evolution_report': final_report})}\n\n"
                    break

                elif msg_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': data})}\n\n"
                    break

                elif msg_type == "event":
                    try:
                        node_name = list(data.keys())[0]
                        serialized = _serialize_event(data)

                        # 检查是否需要工具审批（人在回路）
                        needs_approval = False
                        try:
                            cur_state = graph.get_state(config)
                            needs_approval = bool(
                                cur_state and cur_state.next and cur_state.next[0] == "tools"
                            )
                        except Exception:
                            pass

                        payload = {
                            "type": "node_event",
                            "node": node_name,
                            "data": serialized,
                            "needs_approval": needs_approval,
                        }
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'serialize_error', 'error': str(e)})}\n\n"

        except asyncio.CancelledError:
            logger.info(f"SSE 客户端断开连接: task {task_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # 关闭 Nginx 缓冲，确保实时推送
        },
    )


@app.post(
    "/api/tasks/{task_id}/action",
    dependencies=[Depends(_verify_api_key)],
    summary="审批或驳回工具执行请求（人在回路）",
)
async def task_action(task_id: str, action: ActionRequest):
    """
    当 Agent 请求执行工具（next_node == "tools"）时，
    客户端可通过此接口批准或驳回，实现人在回路 (Human-in-the-loop)。
    操作后需重新调用 /stream 以继续执行。
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _tasks[task_id]
    config = {"configurable": {"thread_id": task["thread_id"]}}

    try:
        state = graph.get_state(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {e}")

    if not state or not state.next:
        raise HTTPException(status_code=400, detail="当前任务没有待审批的操作")

    next_node = state.next[0]
    if next_node != "tools":
        raise HTTPException(
            status_code=400,
            detail=f"当前节点不是工具执行节点（当前: {next_node}），无需审批",
        )

    if action.action == "approve":
        # 批准：直接继续，客户端调用 /stream 即可
        return {
            "status": "approved",
            "message": "工具执行已批准。请重新调用 GET /api/tasks/{task_id}/stream 以继续。",
        }

    else:  # reject
        feedback = action.feedback or "请重新检查逻辑。"
        graph.update_state(
            config,
            {"messages": [HumanMessage(content=f"用户驳回操作。建议：{feedback}")]},
        )
        return {
            "status": "rejected",
            "message": "反馈已注入。请重新调用 GET /api/tasks/{task_id}/stream 以继续。",
        }


@app.get(
    "/api/skills",
    dependencies=[Depends(_verify_api_key)],
    summary="列出技能库",
)
async def list_skills_api():
    """返回 skills/ 目录下所有技能的元数据列表。"""
    skills = []
    for f in sorted(SKILLS_DIR.glob("*.py")):
        try:
            meta = parse_skill_metadata(f)
            skills.append({
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                **meta,
            })
        except Exception:
            skills.append({"filename": f.name, "error": "无法解析元数据"})

    return {"skills": skills, "total": len(skills)}


@app.get(
    "/api/skills/{skill_name}/source",
    dependencies=[Depends(_verify_api_key)],
    summary="获取技能脚本源码",
)
async def get_skill_source(skill_name: str):
    """返回指定技能脚本的完整源代码。"""
    # 路径安全校验：只允许纯文件名，禁止路径分隔符
    if (
        "/" in skill_name
        or "\\" in skill_name
        or ".." in skill_name
        or not skill_name.endswith(".py")
    ):
        raise HTTPException(status_code=400, detail="非法的 skill_name，只允许纯文件名")

    skill_path = SKILLS_DIR / skill_name
    # resolve 确认未逃逸
    try:
        if not str(skill_path.resolve()).startswith(str(SKILLS_DIR.resolve())):
            raise HTTPException(status_code=400, detail="路径安全检查失败")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not skill_path.exists():
        raise HTTPException(status_code=404, detail=f"技能 '{skill_name}' 不存在")

    return {
        "filename": skill_name,
        "source": skill_path.read_text(encoding="utf-8"),
    }


@app.delete(
    "/api/tasks/{task_id}",
    dependencies=[Depends(_verify_api_key)],
    summary="删除任务记录（不影响 Graph 状态）",
)
async def delete_task(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    del _tasks[task_id]
    return {"status": "deleted", "task_id": task_id}


# ==========================================
# 代码智能索引 API（Three-Index）
# ==========================================

@app.get(
    "/api/index/stats",
    dependencies=[Depends(_verify_api_key)],
    summary="获取代码智能索引统计信息",
)
async def get_index_stats_api():
    """
    返回 Three-Index 引擎的运行状态：
    - 已解析的代码块数量
    - 各层索引是否就绪（AST / 语义向量 / BM25 / 依赖图）
    - 可用的解析后端（tree-sitter 或 regex fallback）
    """
    try:
        from src.agent.swe.code_index import get_index_stats
        stats = get_index_stats()
        return {"status": "ok", **stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get(
    "/api/index/repo-map",
    dependencies=[Depends(_verify_api_key)],
    summary="获取代码库符号地图（Repo Map）",
)
async def get_repo_map_api(query: str = "", max_tokens: int = 2000):
    """
    返回工作区所有函数/类的符号地图，按 PageRank 重要性排序。
    可选 query 参数，相关文件会排在前面。
    """
    try:
        from src.agent.swe.code_index import get_repo_map_str, get_index_stats
        stats = get_index_stats()
        if stats.get("status") == "not_built":
            return {"status": "not_built", "repo_map": ""}
        repo_map = get_repo_map_str(query=query, max_tokens=max_tokens)
        return {"status": "ok", "repo_map": repo_map}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/index/search",
    dependencies=[Depends(_verify_api_key)],
    summary="三索引代码搜索",
)
async def search_code_api(
    query: str,
    mode: str = "auto",
    top_k: int = 8,
):
    """
    对工作区代码库执行三索引融合搜索。

    mode:
      "auto"     — 自动选择（推荐）
      "semantic" — 纯语义向量搜索
      "keyword"  — 纯 BM25 关键字搜索
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")
    if mode not in ("auto", "semantic", "keyword"):
        raise HTTPException(status_code=400, detail="mode 必须是 auto / semantic / keyword")
    top_k = max(1, min(top_k, 20))

    try:
        from src.agent.swe.code_index import search_code_index, get_index_stats
        stats = get_index_stats()
        if stats.get("status") == "not_built":
            return {"status": "not_built", "results": []}

        chunks = search_code_index(query, mode=mode, top_k=top_k)
        results = [
            {
                "chunk_id": c.chunk_id,
                "file_path": c.file_path,
                "name": c.display_name,
                "chunk_type": c.chunk_type,
                "signature": c.signature,
                "docstring": c.docstring[:200] if c.docstring else "",
                "start_line": c.start_line,
                "end_line": c.end_line,
                "calls": c.calls[:10],
            }
            for c in chunks
        ]
        return {"status": "ok", "query": query, "mode": mode, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/index/symbol/{symbol_name}",
    dependencies=[Depends(_verify_api_key)],
    summary="获取符号的完整上下文（定义 + 调用关系）",
)
async def get_symbol_api(symbol_name: str):
    """
    返回指定函数/类的：
    - 完整源码
    - 函数签名与文档
    - 调用了哪些函数（callees）
    - 被哪些函数调用（callers）

    相当于「跳转到定义」+「查找所有引用」。
    """
    if not symbol_name.strip():
        raise HTTPException(status_code=400, detail="symbol_name 不能为空")

    try:
        from src.agent.swe.code_index import get_symbol_context_str, get_index_stats
        stats = get_index_stats()
        if stats.get("status") == "not_built":
            return {"status": "not_built", "context": ""}
        context = get_symbol_context_str(symbol_name.strip())
        return {"status": "ok", "symbol": symbol_name, "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/index/rebuild",
    dependencies=[Depends(_verify_api_key)],
    summary="手动触发代码索引全量重建",
)
async def rebuild_index():
    """
    手动触发 Three-Index 全量重建（异步后台执行）。
    一般情况下索引由 index_builder_node 在任务启动时自动构建，
    此接口用于手动刷新（如工作区文件被批量修改后）。
    """
    loop = asyncio.get_event_loop()

    def _do_rebuild():
        try:
            from src.agent.swe.code_index import build_workspace_index
            build_workspace_index()
            logger.info("代码索引手动重建完成")
        except Exception as e:
            logger.error(f"代码索引重建失败: {e}")

    loop.run_in_executor(_executor, _do_rebuild)
    return {"status": "rebuilding", "message": "索引重建已在后台启动，请稍后查询 /api/index/stats"}
