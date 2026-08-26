import urllib.request, json, os, sys

BASE = "http://localhost:2024"
tid = "01a01604-4141-7383-90e9-b9c0d2cae1a6"

def get(path):
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))

runs = get(f"/threads/{tid}/runs")
last = runs[-1]
rid = last["run_id"]
status = last["status"]
print(f"latest run={rid} status={status}")
if status == "error":
    print("run.error:", json.dumps(last.get("error"), ensure_ascii=False)[:1500])

# state
try:
    st = get(f"/threads/{tid}/state")
    cv = st.get("values", {})
    scenes = cv.get("scenes", [])
    print(f"scenes={len(scenes)} current_scene_index={cv.get('current_scene_index')}")
    for i, s in enumerate(scenes):
        iu = s.get("image_url"); vu = s.get("video_url")
        print(f"  shot{i+1}: img={'Y' if iu else '-'} vid={'Y' if vu else '-'} iters={s.get('iterations')} last_img={'Y' if s.get('last_image_url') else '-'}")
    ab = cv.get("aborted")
    if ab: print("ABORTED:", cv.get("abort_reason"))
except Exception as e:
    print("state err:", e)

# clips
d = os.path.join("output", "clips", tid)
if os.path.isdir(d):
    print("=== clips ===")
    for root, _, files in os.walk(d):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            print(f"  {fn} {os.path.getsize(p)//1024}KB")
