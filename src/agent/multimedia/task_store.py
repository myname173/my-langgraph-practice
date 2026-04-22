# src/agent/multimedia/task_store.py
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DB_PATH = Path(os.getenv("MULTIMEDIA_TASK_DB", "./data/multimedia_tasks.sqlite"))


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(raw: Optional[str]) -> Any:
    if raw is None or raw == "":
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _normalize_tags(tags: Optional[Any]) -> List[str]:
    if tags is None:
        return []

    if isinstance(tags, list):
        out = []
        for item in tags:
            text = str(item).strip()
            if text:
                out.append(text)
        return out

    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]

    text = str(tags).strip()
    return [text] if text else []


def _table_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(task_registry)").fetchall()
    return {row["name"] for row in rows}


def init_task_store() -> None:
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_registry (
                thread_id TEXT PRIMARY KEY,
                task_text TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                template_name TEXT NOT NULL DEFAULT '',
                project_type TEXT NOT NULL DEFAULT '',
                mode TEXT NOT NULL DEFAULT 'unknown',
                duration INTEGER,
                shot_count INTEGER,
                parent_thread_id TEXT,
                fork_from_checkpoint_id TEXT,
                status TEXT NOT NULL DEFAULT 'RUNNING',
                stage TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                scene_index INTEGER,
                scene_count INTEGER,
                final_movie_path TEXT,
                abort_reason TEXT,
                last_error TEXT,
                summary_json TEXT NOT NULL DEFAULT '{}',
                pending_payload_json TEXT
            )
            """
        )

        existing = _table_columns(conn)
        migrations = [
            ("title", "TEXT NOT NULL DEFAULT ''"),
            ("tags_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("template_name", "TEXT NOT NULL DEFAULT ''"),
            ("project_type", "TEXT NOT NULL DEFAULT ''"),
            ("mode", "TEXT NOT NULL DEFAULT 'unknown'"),
            ("duration", "INTEGER"),
            ("shot_count", "INTEGER"),
            ("parent_thread_id", "TEXT"),
            ("fork_from_checkpoint_id", "TEXT"),
            ("status", "TEXT NOT NULL DEFAULT 'RUNNING'"),
            ("stage", "TEXT"),
            ("summary_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("pending_payload_json", "TEXT"),
            ("scene_index", "INTEGER"),
            ("scene_count", "INTEGER"),
            ("final_movie_path", "TEXT"),
            ("abort_reason", "TEXT"),
            ("last_error", "TEXT"),
            ("task_text", "TEXT NOT NULL DEFAULT ''"),
            ("created_at", "TEXT NOT NULL DEFAULT ''"),
            ("updated_at", "TEXT NOT NULL DEFAULT ''"),
        ]

        for column_name, ddl in migrations:
            if column_name not in existing:
                conn.execute(f"ALTER TABLE task_registry ADD COLUMN {column_name} {ddl}")

        conn.commit()


def save_task_state(
    *,
    thread_id: str,
    task_text: str,
    status: str,
    stage: Optional[str],
    summary: Dict[str, Any],
    pending_payload: Optional[Dict[str, Any]] = None,
    title: str = "",
    tags: Optional[Sequence[str]] = None,
    template_name: str = "",
    project_type: str = "",
    mode: str = "unknown",
    duration: Optional[int] = None,
    shot_count: Optional[int] = None,
    parent_thread_id: Optional[str] = None,
    fork_from_checkpoint_id: Optional[str] = None,
    final_movie_path: Optional[str] = None,
    abort_reason: Optional[str] = None,
    last_error: Optional[str] = None,
) -> None:
    """
    保存任务注册表状态。
    这是应用层长期记忆：用于刷新后恢复 thread_id、状态、摘要、暂停点，以及任务元数据。
    """
    init_task_store()
    now = _now_iso()
    scene_index = summary.get("current_scene_index")
    scene_count = summary.get("scene_count")

    payload_json = _dumps(pending_payload) if pending_payload is not None else None
    summary_json = _dumps(summary)
    tags_json = _dumps(_normalize_tags(tags))

    with _connect() as conn:
        row = conn.execute(
            "SELECT created_at FROM task_registry WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        created_at = row["created_at"] if row and row["created_at"] else now

        conn.execute(
            """
            INSERT INTO task_registry (
                thread_id,
                task_text,
                title,
                tags_json,
                template_name,
                project_type,
                mode,
                duration,
                shot_count,
                parent_thread_id,
                fork_from_checkpoint_id,
                status,
                stage,
                created_at,
                updated_at,
                scene_index,
                scene_count,
                final_movie_path,
                abort_reason,
                last_error,
                summary_json,
                pending_payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                task_text = excluded.task_text,
                title = excluded.title,
                tags_json = excluded.tags_json,
                template_name = excluded.template_name,
                project_type = excluded.project_type,
                mode = excluded.mode,
                duration = excluded.duration,
                shot_count = excluded.shot_count,
                parent_thread_id = excluded.parent_thread_id,
                fork_from_checkpoint_id = excluded.fork_from_checkpoint_id,
                status = excluded.status,
                stage = excluded.stage,
                updated_at = excluded.updated_at,
                scene_index = excluded.scene_index,
                scene_count = excluded.scene_count,
                final_movie_path = excluded.final_movie_path,
                abort_reason = excluded.abort_reason,
                last_error = excluded.last_error,
                summary_json = excluded.summary_json,
                pending_payload_json = excluded.pending_payload_json
            """,
            (
                thread_id,
                task_text,
                title,
                tags_json,
                template_name,
                project_type,
                mode,
                duration,
                shot_count,
                parent_thread_id,
                fork_from_checkpoint_id,
                status,
                stage,
                created_at,
                now,
                scene_index,
                scene_count,
                final_movie_path,
                abort_reason,
                last_error,
                summary_json,
                payload_json,
            ),
        )
        conn.commit()


def load_task_state(thread_id: str) -> Optional[Dict[str, Any]]:
    init_task_store()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM task_registry WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "thread_id": row["thread_id"],
        "task_text": row["task_text"],
        "title": row["title"],
        "tags": _loads(row["tags_json"]) or [],
        "template_name": row["template_name"],
        "project_type": row["project_type"],
        "mode": row["mode"],
        "duration": row["duration"],
        "shot_count": row["shot_count"],
        "parent_thread_id": row["parent_thread_id"],
        "fork_from_checkpoint_id": row["fork_from_checkpoint_id"],
        "status": row["status"],
        "stage": row["stage"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "scene_index": row["scene_index"],
        "scene_count": row["scene_count"],
        "final_movie_path": row["final_movie_path"],
        "abort_reason": row["abort_reason"],
        "last_error": row["last_error"],
        "summary": _loads(row["summary_json"]) or {},
        "pending_payload": _loads(row["pending_payload_json"]),
    }


def list_task_states(limit: int = 50) -> List[Dict[str, Any]]:
    init_task_store()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM task_registry
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    tasks: List[Dict[str, Any]] = []
    for row in rows:
        tasks.append(
            {
                "thread_id": row["thread_id"],
                "task_text": row["task_text"],
                "title": row["title"],
                "tags": _loads(row["tags_json"]) or [],
                "template_name": row["template_name"],
                "project_type": row["project_type"],
                "mode": row["mode"],
                "duration": row["duration"],
                "shot_count": row["shot_count"],
                "parent_thread_id": row["parent_thread_id"],
                "fork_from_checkpoint_id": row["fork_from_checkpoint_id"],
                "status": row["status"],
                "stage": row["stage"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "scene_index": row["scene_index"],
                "scene_count": row["scene_count"],
                "final_movie_path": row["final_movie_path"],
                "abort_reason": row["abort_reason"],
                "last_error": row["last_error"],
                "summary": _loads(row["summary_json"]) or {},
                "pending_payload": _loads(row["pending_payload_json"]),
            }
        )
    return tasks


def get_latest_task_state() -> Optional[Dict[str, Any]]:
    tasks = list_task_states(limit=1)
    return tasks[0] if tasks else None


def get_latest_active_task_state() -> Optional[Dict[str, Any]]:
    init_task_store()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM task_registry
            WHERE status IN ('RUNNING', 'PAUSED', 'INTERRUPTED')
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        return None

    return {
        "thread_id": row["thread_id"],
        "task_text": row["task_text"],
        "title": row["title"],
        "tags": _loads(row["tags_json"]) or [],
        "template_name": row["template_name"],
        "project_type": row["project_type"],
        "mode": row["mode"],
        "duration": row["duration"],
        "shot_count": row["shot_count"],
        "parent_thread_id": row["parent_thread_id"],
        "fork_from_checkpoint_id": row["fork_from_checkpoint_id"],
        "status": row["status"],
        "stage": row["stage"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "scene_index": row["scene_index"],
        "scene_count": row["scene_count"],
        "final_movie_path": row["final_movie_path"],
        "abort_reason": row["abort_reason"],
        "last_error": row["last_error"],
        "summary": _loads(row["summary_json"]) or {},
        "pending_payload": _loads(row["pending_payload_json"]),
    }


def delete_task_state(thread_id: str) -> None:
    init_task_store()
    with _connect() as conn:
        conn.execute("DELETE FROM task_registry WHERE thread_id = ?", (thread_id,))
        conn.commit()
