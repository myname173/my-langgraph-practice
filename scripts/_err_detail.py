import urllib.request, json

BASE = "http://localhost:2024"
tid = "01a01604-4141-7383-90e9-b9c0d2cae1a6"
run_id = "01a01604-4965-7323-afae-b2bd57fdf127"

# get run error detail
def get(path):
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))

try:
    run = get(f"/threads/{tid}/runs/{run_id}")
    print("=== RUN ERROR ===")
    print("status:", run.get("status"))
    print("error:", json.dumps(run.get("error"), ensure_ascii=False, indent=2)[:3000])
except Exception as e:
    print("err:", e)
