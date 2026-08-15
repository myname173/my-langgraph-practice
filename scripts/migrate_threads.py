"""One-off migration: register historical thread_ids from the SqliteSaver
checkpoint DB into the langgraph_runtime_inmem threads store (persisted in
.langgraph_api/.langgraph_ops.pckl), so the frontend ThreadSidebar can list
and resume previously interrupted sessions.

Run this BEFORE starting langgraph dev (server must be stopped so the pckl
is not overwritten on shutdown).
"""
import os
import sqlite3
from datetime import datetime, timezone

from langgraph_runtime_inmem.database import GLOBAL_STORE, OPS_FILENAME

DB = "data/multimedia_checkpoints.sqlite"
OPS = OPS_FILENAME  # .langgraph_api/.langgraph_ops.pckl

# Ensure the ops store file exists / loads
os.makedirs(os.path.dirname(OPS), exist_ok=True)
if os.path.exists(OPS):
    try:
        GLOBAL_STORE.load()
    except Exception as e:  # noqa: BLE001
        print("WARN: failed to load existing ops pckl:", e)

if "threads" not in GLOBAL_STORE or GLOBAL_STORE["threads"] is None:
    GLOBAL_STORE["threads"] = []

# 始终基于 sqlite 的 checkpoint 重建历史 thread 列表，并补上 graph_id。
# 仅保留非迁移来源的用户/平台自建 thread（避免覆盖前端实时创建的会话）。
existing_user = [
    t for t in GLOBAL_STORE["threads"]
    if isinstance(t.get("metadata"), dict) and not t["metadata"].get("migrated_from")
]
GLOBAL_STORE["threads"] = []
existing = set()
print("preserving", len(existing_user), "non-migrated threads")

# collect historical thread_ids + their latest checkpoint metadata from sqlite
conn = sqlite3.connect(DB, timeout=10)
conn.execute("PRAGMA busy_timeout=10000")
conn.row_factory = sqlite3.Row
rows = conn.execute(
    """
    SELECT thread_id, metadata
    FROM checkpoints
    WHERE checkpoint_ns = ''
    """
).fetchall()

seen = {}
for r in rows:
    tid = r["thread_id"]
    if tid in seen:
        continue
    seen[tid] = r["metadata"]

# Normalize any existing thread records so all date fields are datetime objects
# (prevents '<' not supported between str and datetime' at search-sort time).
def _as_dt(v):
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except Exception:  # noqa: BLE001
            return now
    return now


now = datetime.now(timezone.utc)
for t in GLOBAL_STORE["threads"]:
    for fld in ("created_at", "updated_at", "state_updated_at"):
        t[fld] = _as_dt(t.get(fld))

added = 0
now2 = datetime.now(timezone.utc)
for tid, meta_blob in seen.items():
    if tid in existing:
        # 已存在的 thread：确保其 metadata 含 graph_id（历史迁移时可能缺失，
        # 缺失会导致 getState 跳过 checkpoint 读取，前端点击"没反应"）。
        for t in GLOBAL_STORE["threads"]:
            if str(t.get("thread_id")) == tid:
                md = t.setdefault("metadata", {})
                if isinstance(md, dict) and not md.get("graph_id"):
                    md["graph_id"] = "agent"
                break
        continue
    # try to extract a created/updated time from checkpoint metadata
    created = now2
    try:
        import json
        m = json.loads(meta_blob) if meta_blob else {}
        if m.get("created_at"):
            raw = m["created_at"]
            # accept ISO strings or datetime objects
            created = datetime.fromisoformat(raw) if isinstance(raw, str) else raw
    except Exception:  # noqa: BLE001
        created = now
    GLOBAL_STORE["threads"].append(
        {
            "thread_id": tid,
            "created_at": created,
            "updated_at": created,
            "state_updated_at": created,
            # graph_id 必须存在，否则 platform 的 get_thread_state 在读取历史
            # checkpoint 前会跳过 checkpointer.aget（见 langgraph_runtime_inmem/ops.py
            # `if graph_id:` 分支），导致 getState 返回空 / 前端"点击没反应"。
            # 该值须与 langgraph.json 注册的 graph key 一致（本项目为 "agent"）。
            "metadata": {
                "created_by": "migration",
                "migrated_from": "sqlite_checkpoints",
                "graph_id": "agent",
            },
            "status": "idle",
        }
    )
    added += 1

# 将非迁移来源（用户/平台自建）的 thread 加回，避免覆盖前端实时会话
for t in existing_user:
    if str(t.get("thread_id")) not in {str(x.get("thread_id")) for x in GLOBAL_STORE["threads"]}:
        GLOBAL_STORE["threads"].append(t)

GLOBAL_STORE["threads"].sort(key=lambda t: str(t.get("updated_at") or ""), reverse=True)
# PersistentDict.sync() 以 pickle 格式写 dict(self)（与 langgraph 读取格式一致）。
GLOBAL_STORE.sync()
print(f"added {added} historical threads; total now {len(GLOBAL_STORE['threads'])}")
print("sample metadata:", GLOBAL_STORE["threads"][0].get("metadata"))
print("saved to", OPS)
