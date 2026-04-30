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
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Header
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
    version="2.0.0",
    description="REST + SSE API for the Savant SWE Agent (含 LlamaFactory 训练集成)",
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


class TrainingExportRequest(BaseModel):
    formats: list[str] = Field(
        default=["sft", "dpo", "grpo", "skills"],
        description="要导出的格式列表：sft / dpo / grpo / skills",
    )
    min_reward_sft: float = Field(default=0.4, ge=0.0, le=1.0, description="SFT 数据的最低奖励阈值")
    max_dpo_pairs: int = Field(default=500, ge=1, le=5000, description="DPO 训练对最大数量")
    enrich_with_index: bool = Field(default=True, description="是否用 Three-Index 增强训练指令")


class LlamaFactoryConfigRequest(BaseModel):
    mode: str = Field(..., description="训练范式：sft / dpo / grpo")
    model_name_or_path: str = Field(..., description="基础模型路径或 HuggingFace 名称")
    output_dir: str = Field(default="./llamafactory_runs", description="配置文件输出目录")
    template: str = Field(default="qwen", description="模型对话模板：qwen / llama3 / deepseek 等")
    adapter_name_or_path: Optional[str] = Field(default=None, description="LoRA checkpoint 路径（DPO 继续训练用）")
    overrides: Optional[dict] = Field(default=None, description="额外超参数覆盖")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("sft", "dpo", "grpo"):
            raise ValueError("mode 只能是 sft / dpo / grpo")
        return v


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
        # [v2] 训练集成
        "trajectory_id": "",
        "training_export_path": "",
    }


# ==========================================
# 路由
# ==========================================

@app.get("/api/health")
async def health():
    """健康检查，无需鉴权。"""
    training_dir = WORKSPACE_DIR / "_training_data"
    traj_count = 0
    if (training_dir / "trajectories.jsonl").exists():
        with open(training_dir / "trajectories.jsonl", "r") as f:
            traj_count = sum(1 for line in f if line.strip())

    return {
        "status": "ok",
        "workspace": str(WORKSPACE_DIR),
        "skill_count": len(list(SKILLS_DIR.glob("*.py"))),
        "tavily_usage": f"{GLOBAL_STATS['tavily_count']}/{GLOBAL_STATS['max_tavily']}",
        "active_tasks": len(_tasks),
        "trajectory_count": traj_count,
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
        "training_export_path": graph_state.get("training_export_path", ""),
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
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    continue

                if msg_type == "done":
                    final_report = ""
                    training_path = ""
                    try:
                        final_state = graph.get_state(config)
                        if final_state and final_state.values:
                            final_report = final_state.values.get("evolution_report", "")
                            training_path = final_state.values.get("training_export_path", "")
                    except Exception:
                        pass
                    yield f"data: {json.dumps({'type': 'done', 'task_id': task_id, 'evolution_report': final_report, 'training_export_path': training_path})}\n\n"
                    break

                elif msg_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': data})}\n\n"
                    break

                elif msg_type == "event":
                    try:
                        node_name = list(data.keys())[0]
                        serialized = _serialize_event(data)

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
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/api/tasks/{task_id}/action",
    dependencies=[Depends(_verify_api_key)],
    summary="审批或驳回工具执行请求（人在回路）",
)
async def task_action(task_id: str, action: ActionRequest):
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
        return {
            "status": "approved",
            "message": "工具执行已批准。请重新调用 GET /api/tasks/{task_id}/stream 以继续。",
        }
    else:
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
    if (
        "/" in skill_name
        or "\\" in skill_name
        or ".." in skill_name
        or not skill_name.endswith(".py")
    ):
        raise HTTPException(status_code=400, detail="非法的 skill_name，只允许纯文件名")

    skill_path = SKILLS_DIR / skill_name
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
async def search_code_api(query: str, mode: str = "auto", top_k: int = 8):
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


# ==========================================
# [v2 新增] LlamaFactory 训练集成 API
# ==========================================

@app.get(
    "/api/training/stats",
    dependencies=[Depends(_verify_api_key)],
    summary="获取训练数据统计",
)
async def get_training_stats():
    """
    返回当前已积累的训练数据概况：
      - 总轨迹数量
      - 成功/失败分布
      - 平均奖励分
      - 技能库规模
      - 已导出的训练文件
    """
    training_dir = WORKSPACE_DIR / "_training_data"
    stats: dict = {
        "trajectory_count": 0,
        "success_count": 0,
        "failed_count": 0,
        "avg_reward": 0.0,
        "skill_count": len(list(SKILLS_DIR.glob("*.py"))),
        "exported_files": [],
    }

    try:
        from src.agent.swe.training.trajectory_logger import load_trajectories
        records = load_trajectories(WORKSPACE_DIR)
        stats["trajectory_count"] = len(records)
        stats["success_count"] = sum(1 for r in records if r.status == "success")
        stats["failed_count"] = sum(1 for r in records if r.status == "failed")
        if records and any(r.reward != 0.0 for r in records):
            stats["avg_reward"] = round(
                sum(r.reward for r in records) / len(records), 4
            )

        # 列出已导出文件
        for fname in ["sft_success.jsonl", "dpo_pairs.jsonl", "grpo_all.jsonl", "skills_sft.jsonl"]:
            fpath = training_dir / fname
            if fpath.exists():
                with open(fpath) as f:
                    line_count = sum(1 for line in f if line.strip())
                stats["exported_files"].append({
                    "filename": fname,
                    "size_bytes": fpath.stat().st_size,
                    "sample_count": line_count,
                })
    except Exception as e:
        stats["error"] = str(e)

    return stats


@app.post(
    "/api/training/export",
    dependencies=[Depends(_verify_api_key)],
    summary="触发训练数据 Pipeline（轨迹 → SFT/DPO/GRPO JSONL）",
)
async def export_training_data(
    request: TrainingExportRequest,
    background_tasks: BackgroundTasks,
):
    """
    异步触发完整的训练数据 Pipeline：
      1. 加载所有历史轨迹
      2. 计算奖励信号
      3. 用 Three-Index 增强指令
      4. 导出 SFT / DPO / GRPO / Skills 格式的 JSONL
      5. 生成 dataset_info.json（供 LlamaFactory 直接读取）

    注意：Pipeline 在后台线程中运行，接口立即返回 202，
    通过 GET /api/training/stats 轮询查看进度。
    """
    _validate_formats(request.formats)

    def _run_pipeline():
        try:
            from src.agent.swe.training.data_pipeline import run_pipeline
            report = run_pipeline(
                workspace_dir=WORKSPACE_DIR,
                formats=request.formats,
                min_reward_sft=request.min_reward_sft,
                max_dpo_pairs=request.max_dpo_pairs,
                enrich_with_index=request.enrich_with_index,
            )
            logger.info(f"训练数据 Pipeline 完成: {report['counts']}")
        except Exception as e:
            logger.error(f"训练数据 Pipeline 失败: {e}")

    background_tasks.add_task(_run_pipeline)
    return {
        "status": "started",
        "message": "训练数据 Pipeline 已在后台启动。",
        "training_dir": str(WORKSPACE_DIR / "_training_data"),
        "formats": request.formats,
    }


@app.post(
    "/api/training/generate-config",
    dependencies=[Depends(_verify_api_key)],
    summary="生成 LlamaFactory YAML 训练配置",
)
async def generate_llamafactory_config(request: LlamaFactoryConfigRequest):
    """
    生成 LlamaFactory 可直接使用的 YAML 训练配置文件。

    返回：
      - config_path：配置文件路径
      - launch_cmd：直接启动训练的命令
      - config_preview：YAML 内容预览（前 50 行）
    """
    try:
        from src.agent.swe.training.llamafactory_config import generate_and_save_config
        config_path = generate_and_save_config(
            mode=request.mode,
            model_name_or_path=request.model_name_or_path,
            data_dir=WORKSPACE_DIR / "_training_data",
            output_dir=Path(request.output_dir) / request.mode,
            template=request.template,
            adapter_name_or_path=request.adapter_name_or_path,
            overrides=request.overrides,
        )
        config_preview = config_path.read_text(encoding="utf-8")
        preview_lines = config_preview.splitlines()[:50]
        return {
            "status": "ok",
            "config_path": str(config_path),
            "launch_cmd": f"llamafactory-cli train {config_path}",
            "config_preview": "\n".join(preview_lines),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/training/trajectories",
    dependencies=[Depends(_verify_api_key)],
    summary="列出历史轨迹记录",
)
async def list_trajectories(limit: int = 20, min_reward: float = -1.0):
    """
    列出已记录的轨迹，支持按奖励过滤。
    用于观察 Agent 的成功/失败分布，选择高质量样本。
    """
    try:
        from src.agent.swe.training.trajectory_logger import load_trajectories
        records = load_trajectories(WORKSPACE_DIR)

        # 奖励过滤
        if min_reward > -1.0:
            records = [r for r in records if r.reward >= min_reward]

        # 倒序（最新的在前）
        records = list(reversed(records))[:limit]

        return {
            "total": len(records),
            "trajectories": [
                {
                    "trajectory_id": r.trajectory_id,
                    "timestamp": r.timestamp,
                    "task_description": r.task_description[:100],
                    "status": r.status,
                    "test_passed": r.test_passed,
                    "iteration_count": r.iteration_count,
                    "tool_calls": len(r.tool_call_steps),
                    "reward": r.reward,
                    "reward_breakdown": r.reward_breakdown,
                    "evolved_skill": r.evolved_skill_name,
                }
                for r in records
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 内部辅助
# ==========================================

def _validate_formats(formats: list) -> None:
    valid = {"sft", "dpo", "grpo", "skills"}
    invalid = set(formats) - valid
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的格式: {invalid}。有效选项: {valid}",
        )
