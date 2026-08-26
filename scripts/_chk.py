import urllib.request, json
try:
    url = "http://localhost:2024/threads/0ced7442-ffd8-4caa-846a-5925aebbed9f/runs"
    with urllib.request.urlopen(url, timeout=10) as r:
        runs = json.load(r)
    for run in runs[-5:]:
        print(run.get("run_id"), run.get("status"), run.get("updated_at"))
except Exception as e:
    print("ERR", e)
