import urllib.request, json, time

# 1) check agnes backend health / quota
AGNES = "https://apihub.agnes-ai.com/v1/videos"
print("=== AGNES endpoint probe ===")
try:
    req = urllib.request.Request(AGNES, method="GET")
    with urllib.request.urlopen(req, timeout=12) as r:
        print("GET status:", r.status, r.read(200).decode("utf-8", "replace"))
except Exception as e:
    print("agnes GET err:", repr(e))

# 2) check recent media-server / langgraph logs for agnes 403 / quota
import subprocess
print("\n=== recent agnes mentions in server logs ===")
# try common log locations
for cand in ["scripts/_rerun_stdout.log", "scripts/_rerun_stderr.log", "scripts/_rerun_01a01604.log"]:
    try:
        with open(cand, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        hits = [l.rstrip() for l in lines if "agnes" in l.lower() or "403" in l or "quota" in l.lower() or "Free quota" in l]
        if hits:
            print(f"-- {cand} --")
            for h in hits[-15:]:
                print("  ", h)
    except Exception:
        pass
