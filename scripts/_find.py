import urllib.request, json
def post(path, body):
    req = urllib.request.Request("http://localhost:2024"+path, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)
# list threads via POST search
try:
    resp = post("/threads/search", {"limit": 20, "offset": 0})
    print("THREADS FOUND:", len(resp))
    for t in resp:
        tid = t.get("thread_id")
        upd = t.get("updated_at")
        print(" ", tid, upd)
        # try state
        try:
            st = post(f"/threads/{tid}/state", {})
            v = st.get("values", {})
            sc = len(v.get("scenes", []) or [])
            idx = v.get("index")
            if sc or idx is not None:
                print(f"      -> scenes={sc} index={idx}")
        except Exception as e:
            print("      state err:", e)
except Exception as e:
    print("ERR", e)
