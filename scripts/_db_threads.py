# -*- coding: utf-8 -*-
import sqlite3, os

DB = os.path.join("c:/Users/13682/Desktop/my-langgraph-practice-main/data", "multimedia_checkpoints.sqlite")

conn = sqlite3.connect(DB, timeout=30)
cur = conn.cursor()

# 查看有哪些表
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("TABLES:", tables)

TARGET = "01a0020f-e136-7c62-848a-c667a7cf8962"

# 在常见线程表结构中查找
for tbl in tables:
    try:
        cur.execute(f"PRAGMA table_info('{tbl}')")
        cols = [c[1] for c in cur.fetchall()]
        if "thread_id" in cols:
            cur.execute(f"SELECT COUNT(*) FROM '{tbl}' WHERE thread_id LIKE ?", (TARGET[:8] + "%",))
            cnt = cur.fetchone()[0]
            print(f"表 {tbl}: 匹配 {TARGET[:8]}* 的记录数 = {cnt}")
    except Exception as e:
        pass

conn.close()
