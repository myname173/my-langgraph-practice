import urllib.request, json
def get(path):
    with urllib.request.urlopen("http://localhost:2024"+path, timeout=10) as r:
        return json.load(r)
# thread 0ced7442 (yesterday's fresh run)
tid = "0ced7442-ffd8-4caa-846a-5925aebbed9f"
try:
    st = get(f"/threads/{tid}/state")
    vals = st.get("values", {})
    print("THREAD 0ced7442:")
    print("  scenes:", len(vals.get("scenes", []) or []))
    print("  index:", vals.get("index"))
    print("  video_gen_failures:", vals.get("video_gen_failures"))
    print("  has raw_video_url:", bool(vals.get("raw_video_url")))
except Exception as e:
    print("0ced7442 ERR", e)
# also list recent threads to see if there's an older partial one
try:
    threads = get("/threads?limit=8")
    print("\nRECENT THREADS:")
    for t in threads:
        print(" ", t.get("thread_id"), t.get("updated_at"))
except Exception as e:
    print("threads ERR", e)
