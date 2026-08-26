import urllib.request, json

def get(path, timeout=8):
    req = urllib.request.Request("http://localhost:2024" + path)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")

for ep in ["/ok", "/threads", "/threads/01a01604-4141-7383-90e9-b9c0d2cae1a6/state"]:
    try:
        st, body = get(ep)
        print(f"{ep} -> {st} ({len(body)} bytes)")
    except Exception as e:
        print(f"{ep} -> ERR: {type(e).__name__}: {e}")
