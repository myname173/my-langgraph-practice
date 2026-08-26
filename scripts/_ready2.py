import urllib.request

def check(path, timeout=6):
    try:
        req = urllib.request.Request("http://localhost:2024" + path)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return f"{path} -> {r.status}"
    except Exception as e:
        return f"{path} -> ERR {type(e).__name__}: {e}"

print(check("/ok"))
print(check("/threads/01a01604-4141-7383-90e9-b9c0d2cae1a6/runs"))

# also read resume result file if present
import os
rf = "scripts/_resume_result.json"
if os.path.exists(rf):
    with open(rf, encoding="utf-8", errors="replace") as f:
        print("RESUME_RESULT:", f.read()[:300])
else:
    print("RESUME_RESULT: file not present")
