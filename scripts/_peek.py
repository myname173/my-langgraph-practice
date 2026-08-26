import urllib.request, json

BASE = "http://localhost:2024"
tid = "01a01604-4141-7383-90e9-b9c0d2cae1a6"
rid = "01a01622-6dad-7de1-b9ba-1b0bef361a45"

# stream a few events quickly to see current node (short timeout)
payload = {"stream_mode": ["events"]}
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(BASE + f"/threads/{tid}/runs/{rid}/stream", data=data,
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=8) as r:
        for i, line in enumerate(r):
            line = line.decode("utf-8", "replace")
            if "on_chain_start" in line or "on_chain_end" in line:
                try:
                    obj = json.loads(line[6:] if line.startswith("data: ") else line)
                    nm = obj.get("data", {}).get("name") or obj.get("data", {}).get("metadata", {}).get("langgraph_node")
                    print("EVENT node:", nm)
                except Exception:
                    print("RAW:", line[:120])
            if i > 40:
                break
except Exception as e:
    print("stream peek done/timeout:", e)
