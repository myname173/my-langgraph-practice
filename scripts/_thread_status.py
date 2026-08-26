import urllib.request, json

BASE = "http://localhost:2024"
tid = "01a01604-4141-7383-90e9-b9c0d2cae1a6"

def get(path):
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))

# runs for this thread
try:
    runs = get(f"/threads/{tid}/runs")
    print("=== RUNS ===")
    for run in runs[-5:]:
        print(f"run_id={run.get('run_id')} status={run.get('status')} started={run.get('start_time')} ended={run.get('end_time')}")
except Exception as e:
    print("runs err:", e)

# latest state (scenes progress)
try:
    st = get(f"/threads/{tid}/state")
    cv = st.get("values", {})
    scenes = cv.get("scenes", [])
    print(f"\n=== STATE === scenes count={len(scenes)}")
    for i, s in enumerate(scenes):
        iu = s.get("image_url")
        vu = s.get("video_url")
        print(f"  shot{i+1}: image_url={'YES' if iu else 'NO'} video_url={'YES' if vu else 'NO'} iters={s.get('iterations')} use_ff={s.get('use_first_last_frame')}")
    print("  current_scene_index:", cv.get("current_scene_index"))
    print("  aborted:", cv.get("aborted"), "abort_reason:", cv.get("abort_reason"))
except Exception as e:
    print("state err:", e)
