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
    """
    db_path = _resolve_db_path()
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
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
