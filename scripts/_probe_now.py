import urllib.request, json, subprocess

# listener pid
pid = None
try:
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if ":2024" in line and "LISTENING" in line:
            pid = line.split()[-1]
except Exception:
    pass

res = {"port_listener_pid": pid}
try:
    req = urllib.request.Request("http://localhost:2024/ok")
    with urllib.request.urlopen(req, timeout=8) as r:
        res["ok"] = True
        res["code"] = r.status
except Exception as e:
    res["ok"] = False
    res["err"] = type(e).__name__

with open("scripts/_probe_now.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False)
print(json.dumps(res, ensure_ascii=False))
