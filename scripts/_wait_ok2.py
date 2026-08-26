import urllib.request, time, json, subprocess

statuses = []
# also capture the langgraph server PID listening on 2024
def get_listener():
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if ":2024" in line and "LISTENING" in line:
                parts = line.split()
                return parts[-1]
    except Exception:
        pass
    return None

for i in range(15):  # up to ~300s
    pid = get_listener()
    t0 = time.time()
    ok = False
    err = None
    try:
        req = urllib.request.Request("http://localhost:2024/ok")
        with urllib.request.urlopen(req, timeout=8) as r:
            code = r.status
        ok = True
        err = str(code)
    except Exception as e:
        err = type(e).__name__
    elapsed = round(time.time() - t0, 2)
    statuses.append({"i": i, "ok": ok, "err": err, "pid": pid, "elapsed": elapsed})
    if ok:
        break
    time.sleep(20)

with open("scripts/_wait_ok2.json", "w", encoding="utf-8") as f:
    json.dump(statuses, f, ensure_ascii=False)
print(json.dumps(statuses, ensure_ascii=False))
