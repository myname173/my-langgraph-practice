import urllib.request, json

BASE = "http://localhost:2024"
def get(path, timeout=10):
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")

for ep in ["/threads", "/threads/01a01604-4141-7383-90e9-b9c0d2cae1a6", "/threads/01a01604-4141-7383-90e9-b9c0d2cae1a6/state", "/assistants", "/ok"]:
    try:
        st, body = get(ep)
        print(f"{ep} -> {st} ({len(body)}B)")
        if ep == "/threads" and st == 200:
            try:
                arr = json.loads(body)
                print("  thread count:", len(arr))
                for t in arr[-6:]:
                    print("   ", t.get("thread_id"), t.get("metadata"))
            except Exception:
                pass
        if ep.endswith("/state") and st == 200:
            try:
                v = json.loads(body).get("values", {})
                print("  state scenes:", len(v.get("scenes", [])), "idx:", v.get("current_scene_index"))
            except Exception:
                pass
    except Exception as e:
        print(f"{ep} -> ERR {type(e).__name__}: {e}")
