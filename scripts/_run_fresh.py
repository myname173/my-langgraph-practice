import urllib.request, json

BASE = "http://localhost:2024"
headers = {"Content-Type": "application/json"}

def call(method, path, payload=None, timeout=40):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

full_input = {
    "task": (
        "Write and produce a short xianxia (Chinese cultivation fantasy) live-action style movie trailer, "
        "fully photorealistic CGI. Visual style: AAA game CG trailer, Unreal Engine 5 render, cinematic lighting, "
        "ray tracing, physically based rendering, real skin and fabric textures, volumetric fog, film grain, 8k, "
        "epic and grounded, fully photorealistic, not stylized or animated.\n\n"
        "Cast (anchored to provided reference images):\n"
        "- 主角 (Protagonist): appearance MUST match char_main.jpg.\n"
        "- 反派1 (Villain 1): appearance MUST match char_villain1.jpg.\n"
        "- 反派2 (Villain 2): appearance MUST match char_villain2.jpg.\n\n"
        "Story (3 acts, about 6-8 shots):\n"
        "Act 1 - At dawn on a misty immortal mountain, 主角 trains alone; an old mentor warns a sealed relic must never fall into darkness.\n"
        "Act 2 - 反派1 descends with 反派2 and a dark army, shattering the mountain altar; 主角 stands against them.\n"
        "Act 3 - A decisive duel at the collapsing crimson altar; 主角 strikes 反派1 down, the relic dissolving into spirit light.\n"
        "Requirements: photorealistic CGI, cinematic camera, 6-8 shots, Chinese narration + soundtrack.\n"
    ),
    "global_setting": "",
    "scenes": [],
    "current_scene_index": 0,
    "voice_role": "yunxi",
    "enable_audio": True,
    "bgm_mood": "tense",
    "prefer_native_audio": False,
    "use_first_last_frame": True,
    "reference_sheets": {"characters": {
        "主角": {"name": "主角", "urls": ["/media/assets_uploaded/char_main.jpg"], "description": "角色一致性参考锚点"},
        "反派1": {"name": "反派1", "urls": ["/media/assets_uploaded/char_villain1.jpg"], "description": "角色一致性参考锚点"},
        "反派2": {"name": "反派2", "urls": ["/media/assets_uploaded/char_villain2.jpg"], "description": "角色一致性参考锚点"}},
        "props": {}, "environments": {}},
    "reference_images": [
        "http://localhost:8900/media/assets_uploaded/char_main.jpg",
        "http://localhost:8900/media/assets_uploaded/char_villain1.jpg",
        "http://localhost:8900/media/assets_uploaded/char_villain2.jpg",
    ],
    "assets_imported": True,
    "auto_mode": True,
}

result = {}
try:
    t = call("POST", "/threads", {"metadata": {"purpose": "verify-fix-A-B-C-D-fresh"}})
    tid = t["thread_id"]
    result["thread_id"] = tid
    run = call("POST", f"/threads/{tid}/runs", {
        "assistant_id": "agent",
        "input": full_input,
        "config": {"configurable": {"thread_id": tid}},
    })
    result["status"] = "ok"
    result["run_id"] = run["run_id"]
except Exception as e:
    result["status"] = "err"
    result["error"] = f"{type(e).__name__}: {e}"

with open("scripts/_run_fresh_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
print(json.dumps(result, ensure_ascii=False))
