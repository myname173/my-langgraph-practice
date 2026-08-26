import os, glob, sqlite3, json

root = "c:/Users/13682/Desktop/my-langgraph-practice-main"
dbs = []
for f in glob.glob(os.path.join(root, "**", "*.sqlite*"), recursive=True):
    if "chroma" in f or "node_modules" in f:
        continue
    dbs.append(f)
print("=== candidate db files ===")
for f in dbs:
    print(f, os.path.getsize(f))

# try each db for a checkpoints table and the 030b thread
target = "030b4b82-ea0d-48e4-a1b9-38e9d7af239b"
for f in dbs:
    try:
        con = sqlite3.connect(f)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tabs = [r[0] for r in cur.fetchall()]
        if "checkpoints" in tabs:
            cur.execute("SELECT thread_id FROM checkpoints WHERE thread_id LIKE ? LIMIT 5", (target[:8] + "%",))
            rows = cur.fetchall()
            if rows:
                print("=== FOUND in", f, "===")
                for (tid,) in rows:
                    print("thread:", tid)
        con.close()
    except Exception as e:
        pass
