import urllib.request, json, os

BASE = "http://localhost:2024"
tid = "01a01604-4141-7383-90e9-b9c0d2cae1a6"

def get(path):
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))

runs = get(f"/threads/{tid}/runs")
print("=== ALL RUNS (last 8) ===")
for run in runs[-8:]:
    print(f"  {run['run_id']}  status={run['status']}  started={run.get('start_time')}")

# specifically the new run
new_id = "01a01622-6dad-7de1-b9ba-1b0bef361a45"
try:
    nr = get(f"/threads/{tid}/runs/{new_id}")
    print(f"\n=== NEW RUN {new_id} ===")
    print("status:", nr.get("status"))
    print("error:", json.dumps(nr.get("error"), ensure_ascii=False)[:800])
except Exception as e:
    print("new run fetch err:", e)

# current state
st = get(f"/threads/{tid}/state")
cv = st.get("values", {})
scenes = cv.get("scenes", [])
print(f"\n=== STATE === scenes={len(scenes)} current_scene_index={cv.get('current_scene_index')}")
for i, s in enumerate(scenes):
    print(f"  shot{i+1}: img={'Y' if s.get('image_url') else '-'} vid={'Y' if s.get('video_url') else '-'}")
