import urllib.request, json, sys
sys.stdout.reconfigure(encoding="utf-8")
out = open("c:/Users/13682/Desktop/my-langgraph-practice-main/scripts/_stop_result.txt", "w", encoding="utf-8")

tid = "0ced7442-ffd8-4caa-846a-5925aebbed9f"
run_id = "01a01879-945a-7001-ba0b-eeb08f1a05be"
try:
    req = urllib.request.Request(
        f"http://localhost:2024/threads/{tid}/runs/{run_id}/cancel",
        data=b"{}", headers={"Content-Type":"application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=8) as r:
        out.write(f"cancel status: {r.status}\n")
except Exception as e:
    out.write(f"cancel err: {e}\n")

try:
    with urllib.request.urlopen("http://localhost:2024/ok", timeout=5) as r:
        out.write(f"server /ok: {r.status}\n")
except Exception as e:
    out.write(f"server /ok err: {e}\n")
out.close()
