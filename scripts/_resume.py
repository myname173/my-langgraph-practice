import urllib.request, json

BASE = "http://localhost:2024"
ASSISTANT = "agent"
tid = "01a01604-4141-7383-90e9-b9c0d2cae1a6"

# RESUME: send only task, do NOT reset scenes (graph uses checkpoint scenes=5, index=3)
resume_input = {
    "task": (
        "Write and produce a short xianxia live-action style movie trailer, fully photorealistic CGI. "
        "Cast anchored to reference images char_main.jpg / char_villain1.jpg / char_villain2.jpg. "
        "3 acts, 6-8 shots, Chinese narration + soundtrack."
    ),
    "reference_images": [
        "http://localhost:8900/media/assets_uploaded/char_main.jpg",
        "http://localhost:8900/media/assets_uploaded/char_villain1.jpg",
        "http://localhost:8900/media/assets_uploaded/char_villain2.jpg",
    ],
    "reference_sheets": {
        "characters": {
            "主角": {"name": "主角", "urls": ["/media/assets_uploaded/char_main.jpg"], "description": "素材库主角人物立绘，作为角色一致性参考锚点"},
            "反派1": {"name": "反派1", "urls": ["/media/assets_uploaded/char_villain1.jpg"], "description": "素材库反派人物立绘，作为角色一致性参考锚点"},
            "反派2": {"name": "反派2", "urls": ["/media/assets_uploaded/char_villain2.jpg"], "description": "素材库反派人物立绘，作为角色一致性参考锚点"},
        },
        "props": {}, "environments": {},
    },
}

def post(path, payload, timeout=30):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

run = post(f"/threads/{tid}/runs", {
    "assistant_id": ASSISTANT,
    "input": resume_input,
    "config": {"configurable": {"thread_id": tid}},
})
print("RESUME_RUN_ID=" + run["run_id"])
print("Server resuming from checkpoint (scenes=5, index=3). Poll with: python scripts/_poll.py")
