import sqlite3, json

db = "c:/Users/13682/Desktop/my-langgraph-practice-main/data/multimedia_checkpoints.sqlite"
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("TABLES:", [r[0] for r in cur.fetchall()])

target = "030b4b82-ea0d-48e4-a1b9-38e9d7af239b"

# list columns of checkpoints
cur.execute("PRAGMA table_info(checkpoints)")
print("CHECKPOINTS cols:", [r[1] for r in cur.fetchall()])

# get the earliest checkpoint (smallest checkpoint_ns / first) for the thread -> its state
cur.execute("SELECT checkpoint_id, parent_checkpoint_id, checkpoint_ns, type, checkpoint FROM checkpoints WHERE thread_id=? ORDER BY rowid ASC LIMIT 3", (target,))
for row in cur.fetchall():
    cid, pid, ns, typ, blob = row
    print("--- checkpoint_id:", cid, "ns:", ns, "type:", typ)
    # extract top-level keys of the state msgpack without full parse
    try:
        import msgpack
        data = msgpack.unpackb(blob, raw=False, strict_map_key=False)
        if isinstance(data, dict) and "channel_values" in data:
            cv = data["channel_values"]
            print("   channel keys:", list(cv.keys())[:40])
            # print a few interesting channels
            for k in ("input", "user_input", "topic", "story", "theme", "prompt", "messages"):
                if k in cv:
                    v = cv[k]
                    s = str(v)
                    print(f"   [{k}] =", s[:400])
    except Exception as e:
        print("   parse err:", e)
con.close()
