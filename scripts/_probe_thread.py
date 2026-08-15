# -*- coding: utf-8 -*-
import sqlite3, os
DB = os.path.join("c:/Users/13682/Desktop/my-langgraph-practice-main/data", "multimedia_checkpoints.sqlite")
TARGET = "01a0020f-e136-7c62-848a-c667a7cf8962"
conn = sqlite3.connect(DB, timeout=30)
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id=?", (TARGET,))
print("checkpoint 总数:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM writes WHERE thread_id=?", (TARGET,))
print("writes 总数:", cur.fetchone()[0])
# 最近的 metadata（rowid 最大）
cur.execute("SELECT metadata FROM checkpoints WHERE thread_id=? ORDER BY rowid DESC LIMIT 1", (TARGET,))
print("最近 metadata 前80字:", (cur.fetchone()[0] or "")[:80])
conn.close()
