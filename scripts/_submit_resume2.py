import urllib.request, json

BASE = "http://localhost:2024"
tid = "01a01604-4141-7383-90e9-b9c0d2cae1a6"
headers = {"Content-Type": "application/json"}

def post(path, payload, timeout=40):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

resume_input = {
    "task": "Write and produce a short xianxia live-action style movie trailer, fully photorealistic CGI. Cast anchored to reference images char_main.jpg / char_villain1.jpg / char_villain2.jpg. 3 acts, 6-8 shots, Chinese narration + soundtrack.",
    "reference_images": [
        "http://localhost:8900/media/assets_uploaded/char_main.jpg",
        "http://localhost:8900/media/assets_uploaded/char_villain1.jpg",
        "http://localhost:8900/media/assets_uploaded/char_villain2.jpg",
    ],
    "reference_sheets": {"characters": {
        "主角": {"name": "主角", "urls": ["/media/assets_uploaded/char_main.jpg"], "description": "素材库主角人物立绘，作为角色一致性参考锚点"},
        "反派1": {"name": "反派1", "urls": ["/media/assets_uploaded/char_villain1.jpg"], "description": "素材库反派人物立绘，作为角色一致性参考锚点"},
        "反派2": {"name": "反派2", "urls": ["/media/assets_uploaded/char_villain2.jpg"], "description": "素材库反派人物立绘，作为角色一致性参考锚点"}},
        "props": {}, "environments": {}},
}

result = {"status": "unknown"}
try:
    # 1) recreate/attach thread (reuses existing checkpoint by thread_id)
    try:
        t = post(f"/threads", {"thread_id": tid, "metadata": {"resume": "after-restart"}})
        result["thread"] = "reused"
    except Exception as e:
        result["thread_err"] = f"{type(e).__name__}: {e}"
    # 2) submit run
    run = post(f"/threads/{tid}/runs", {
        "assistant_id": "agent",
        "input": resume_input,
        "config": {"configurable": {"thread_id": tid}},
    })
    result["status"] = "ok"
    result["run_id"] = run["run_id"]
except Exception as e:
    result["status"] = "err"
    result["error"] = f"{type(e).__name__}: {e}"

with open("scripts/_resume_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
print(json.dumps(result, ensure_ascii=False))
