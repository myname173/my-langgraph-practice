import sqlite3, msgpack

db = "c:/Users/13682/Desktop/my-langgraph-practice-main/data/multimedia_checkpoints.sqlite"
con = sqlite3.connect(db)
cur = con.cursor()
target = "030b4b82-ea0d-48e4-a1b9-38e9d7af239b"
cur.execute("SELECT checkpoint FROM checkpoints WHERE thread_id=? AND checkpoint_id=?",
            (target, "1f19a6e8-3f34-6a5a-8000-c2a39bbd04a3"))
blob = cur.fetchone()[0]
data = msgpack.unpackb(blob, raw=False, strict_map_key=False)
cv = data["channel_values"]
task = cv.get("task")
with open("scripts/_task_full.txt", "w", encoding="utf-8") as f:
    f.write(task)
print("LEN:", len(task))
print("FULL TASK:")
print(task)
con.close()
