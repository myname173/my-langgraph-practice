import urllib.request, json
def get(path):
    with urllib.request.urlopen("http://localhost:2024"+path, timeout=10) as r:
        return json.load(r)
tid = "0ced7442-ffd8-4caa-846a-5925aebbed9f"
try:
    runs = get(f"/threads/{tid}/runs")
    print("RUNS:", len(runs))
    for run in runs:
        print(run.get("run_id"), run.get("status"), run.get("started_at"), run.get("updated_at"))
except Exception as e:
    print("ERR", e)
