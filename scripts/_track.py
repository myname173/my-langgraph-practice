import urllib.request, json, os

BASE = "http://localhost:2024"
tid = "0ced7442-ffd8-4caa-846a-5925aebbed9f"

def get(path, timeout=12):
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

# run status
try:
    runs = get(f"/threads/{tid}/runs")
    last = runs[-1]
    print(f"run={last['run_id']} status={last['status']}")
    if last.get("status") == "error":
        print("error:", json.dumps(last.get("error"), ensure_ascii=False)[:600])
except Exception as e:
    print("runs err:", e)

# state
try:
    st = get(f"/threads/{tid}/state")
    cv = st.get("values", {})
    scenes = cv.get("scenes", [])
    print(f"scenes={len(scenes)} idx={cv.get('current_scene_index')}")
    for i, s in enumerate(scenes):
        print(f"  shot{i+1}: img={'Y' if s.get('image_url') else '-'} vid={'Y' if s.get('video_url') else '-'} iters={s.get('iterations')}")
except Exception as e:
    print("state err:", e)

# clips
d = os.path.join("output", "clips", tid)
if os.path.isdir(d):
    print("clips:")
    for root, _, files in os.walk(d):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            print(f"  {fn} {os.path.getsize(p)//1024}KB")
else:
    print("no clips yet")
