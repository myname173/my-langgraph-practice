import sqlite3, json

db = "data/multimedia_checkpoints.sqlite"
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("TABLES:", [r[0] for r in cur.fetchall()])
cur.execute("PRAGMA table_info(checkpoints)")
print("CHECKPOINTS COLS:", [r[1] for r in cur.fetchall()])
cur.execute("SELECT thread_id, checkpoint FROM checkpoints ORDER BY rowid DESC LIMIT 3")
for tid, cp in cur.fetchall():
    print("TID", tid, "cp_len", len(cp))
    try:
        c = json.loads(cp)
        print("  keys:", list(c.keys())[:10])
        if "channel_values" in c:
            print("  cv keys:", list(c["channel_values"].keys())[:20])
    except Exception as e:
        print("  err", e)
con.close()
