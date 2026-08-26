import urllib.request, json, sys
sys.stdout.reconfigure(encoding="utf-8")

tid = "0ced7442-ffd8-4caa-846a-5925aebbed9f"
run_id = "01a01879-945a-7001-ba0b-eeb08f1a05be"

# 1) 尝试 cancel 当前 run
try:
    req = urllib.request.Request(
        f"http://localhost:2024/threads/{tid}/runs/{run_id}/cancel",
        data=b"{}", headers={"Content-Type":"application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        print("cancel status:", r.status)
except Exception as e:
    print("cancel err:", e)

# 2) 确认 server 活着
try:
    with urllib.request.urlopen("http://localhost:2024/ok", timeout=5) as r:
        print("server /ok:", r.status)
except Exception as e:
    print("server /ok err:", e)
