import sqlite3, json, sys, os
sys.stdout.reconfigure(encoding="utf-8")

DB = "c:/Users/13682/Desktop/my-langgraph-practice-main/data/multimedia_checkpoints.sqlite"
conn = sqlite3.connect(DB)
cur = conn.cursor()

# 列出所有 thread_id 和最新 checkpoint 时间
rows = cur.execute("SELECT thread_id, MAX(timestamp) FROM checkpoints GROUP BY thread_id ORDER BY MAX(timestamp) DESC LIMIT 10").fetchall()
print("=== checkpoints by thread ===")
for tid, ts in rows:
    print(f"  {tid}  {ts}")

# 找到有 scenes 数据的 checkpoint blob
print("\n=== searching for non-empty scenes ===")
for tid, _ in rows:
    blobs = cur.execute(
        "SELECT checkpoint_id, timestamp, thread_id FROM checkpoints WHERE thread_id=? ORDER BY timestamp DESC LIMIT 5",
        (tid,)
    ).fetchall()
    for cid, ts2, t in blobs:
        row = cur.execute("SELECT checkpoint FROM checkpoint_blobs WHERE checkpoint_id=?", (cid,)).fetchone()
        if not row: continue
        try:
            d = json.loads(row[0])
            # channel_values 里找 scenes
            cv = d.get("channel_values", {})
            scenes = cv.get("scenes", [])
            idx = cv.get("index")
            if scenes and len(scenes) > 0:
                print(f"  {t[:8]}...  ts={ts2}  scenes={len(scenes)}  index={idx}")
                # 打印每个 scene 的 description + image_url 前缀
                for i,s in enumerate(scenes[:12]):
                    desc = s.get("description","")[:40]
                    iu = s.get("image_url","")[-40:]
                    print(f"    [{i}] desc={desc}  img={iu}")
                break
        except:
            pass

conn.close()
