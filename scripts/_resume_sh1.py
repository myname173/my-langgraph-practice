import urllib.request, json

BASE = "http://localhost:2024"
tid = "01a01604-4141-7383-90e9-b9c0d2cae1a6"
payload = {
    "assistant_id": "agent",
    "input": {
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
    },
    "config": {"configurable": {"thread_id": tid}},
}
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(BASE + f"/threads/{tid}/runs", data=data, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        run = json.loads(r.read().decode("utf-8"))
    print("RESUME_RUN_ID=" + run["run_id"])
except Exception as e:
    print("POST_ERR:", type(e).__name__, e)
