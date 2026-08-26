import urllib.request, time, json

statuses = []
for i in range(12):  # up to ~180s
    t0 = time.time()
    try:
        req = urllib.request.Request("http://localhost:2024/ok")
        with urllib.request.urlopen(req, timeout=8) as r:
            code = r.status
        elapsed = round(time.time() - t0, 2)
        statuses.append({"i": i, "ok": True, "code": code, "elapsed": elapsed})
        break
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        statuses.append({"i": i, "ok": False, "err": type(e).__name__, "elapsed": elapsed})
    time.sleep(15)

with open("scripts/_wait_ok.json", "w", encoding="utf-8") as f:
    json.dump(statuses, f, ensure_ascii=False)
print(json.dumps(statuses, ensure_ascii=False))
