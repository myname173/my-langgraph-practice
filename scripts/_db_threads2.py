# -*- coding: utf-8 -*-
import sqlite3, os

DB = os.path.join("c:/Users/13682/Desktop/my-langgraph-practice-main/data", "multimedia_checkpoints.sqlite")
conn = sqlite3.connect(DB, timeout=30)
cur = conn.cursor()

# 列出所有表
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("ALL TABLES:", [r[0] for r in cur.fetchall()])

TARGET = "01a0020f-e136-7c62-848a-c667a7cf8962"

# 是否有 threads 表
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%thread%'")
print("thread 相关表:", [r[0] for r in cur.fetchall()])

# 检查 checkpoints 表里 distinct 的 thread_id 数量
cur.execute("SELECT DISTINCT thread_id FROM checkpoints")
tids = [r[0] for r in cur.fetchall()]
print(f"\ncheckpoints 中 distinct thread_id 数量: {len(tids)}")
for tid in tids:
    print(" -", tid)

conn.close()
