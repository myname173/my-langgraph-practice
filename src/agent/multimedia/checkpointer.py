# src/agent/multimedia/checkpointer.py
import asyncio
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

import aiosqlite
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


def _resolve_db_path() -> Path:
    """
    SQLite checkpoint 文件路径。
    默认：./data/multimedia_checkpoints.sqlite
    可通过环境变量 MULTIMEDIA_CHECKPOINT_DB 覆盖。
    """
    raw = os.getenv("MULTIMEDIA_CHECKPOINT_DB", "./data/multimedia_checkpoints.sqlite")
    path = Path(raw)

    if not path.is_absolute():
        # 不依赖运行时的 cwd：以本文件位置为锚点解析项目根，
        # 确保 LangGraph Platform server（其 cwd 可能不是项目根）与本地脚本
        # 始终连同一份 ./data/multimedia_checkpoints.sqlite。
        project_root = Path(__file__).resolve().parents[3]
        path = (project_root / path).resolve()

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_checkpointer() -> SqliteSaver:
    """
    构造一个真正的 BaseCheckpointSaver 实例。
    SqliteSaver.from_conn_string() 是 context manager，不适合直接传给 compile。

    健壮性加固（方案A 定位到的持久化层隐患）：
    - timeout=30：Windows 下多连接/线程并发访问 SQLite 时，写锁等待上限从默认 5s
      提高到 30s，避免瞬间锁竞争直接抛 "database is locked"。
    - PRAGMA journal_mode=WAL：显式声明 WAL 模式（db 级属性，幂等）。WAL 下读写
      可并发、写不阻塞读，显著降低大 state 序列化入库（如 film_studio 的
      studio_output）时的文件锁竞争与潜在持久化阻塞。
    """
    db_path = _resolve_db_path()
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error:
        # WAL 设置失败不应阻断启动（极少数只读/权限场景），记录即可
        pass
    return SqliteSaver(conn)


checkpointer = build_checkpointer()


@asynccontextmanager
async def create_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """
    供 LangGraph Platform（langgraph.json 的 checkpointer.path 字段）使用的
    async context manager 工厂。

    平台模式不允许在 graph.compile() 中传入自定义 checkpointer（否则 GraphLoadError），
    但允许通过 langgraph.json 的 checkpointer.path 声明一个工厂函数，由平台注入。

    注意：LangGraph Platform 的 getState / threads 接口走 async 调用路径，
    必须使用支持 async 方法的 AsyncSqliteSaver（同步 SqliteSaver 的 aget/aget_tuple
    会抛 NotImplementedError，导致 getState 对历史 thread 返回 404、前端"点击没反应"）。

    这里复用与本地脚本完全相同的 sqlite 存储（同一份 ./data/multimedia_checkpoints.sqlite，
    保留 WAL / timeout=30 加固），从而让平台前端能列出并恢复本地已跑一半的历史会话
    （如 9e4a9dae）。
    """
    db_path = _resolve_db_path()
    # aiosqlite.connect 是协程，必须在 async 上下文内 await，否则拿到的是未执行的
    # coroutine 对象而非真实连接，会导致查询静默返回空 / 失败。
    conn = await aiosqlite.connect(str(db_path), timeout=30)
    try:
        await conn.execute("PRAGMA journal_mode=WAL")
    except aiosqlite.Error:
        # WAL 设置失败不应阻断启动（极少数只读/权限场景），记录即可
        pass
    yield AsyncSqliteSaver(conn)


def make_thread_config(thread_id: str, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
    """
    生成 LangGraph 运行配置。
    """
    configurable: Dict[str, Any] = {"thread_id": thread_id}
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}
