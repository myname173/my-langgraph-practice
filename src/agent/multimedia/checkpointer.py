# src/agent/multimedia/checkpointer.py
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from langgraph.checkpoint.sqlite import SqliteSaver


def _resolve_db_path() -> Path:
    """
    SQLite checkpoint 文件路径。
    默认：./data/multimedia_checkpoints.sqlite
    可通过环境变量 MULTIMEDIA_CHECKPOINT_DB 覆盖。
    """
    raw = os.getenv("MULTIMEDIA_CHECKPOINT_DB", "./data/multimedia_checkpoints.sqlite")
    path = Path(raw)

    if not path.is_absolute():
        path = Path.cwd() / path

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


def make_thread_config(thread_id: str, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
    """
    生成 LangGraph 运行配置。
    """
    configurable: Dict[str, Any] = {"thread_id": thread_id}
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}
