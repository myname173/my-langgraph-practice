import urllib.request, json, time, sys

BASE = "http://localhost:2024"
ASSISTANT = "agent"

# Reconstructed initial state from 030b checkpoint (input-only fields)
initial_state = {
    "task": (
        "Write and produce a short xianxia (Chinese cultivation fantasy) live-action style movie trailer, "
        "fully photorealistic CGI. Visual style: AAA game CG trailer, Unreal Engine 5 render, cinematic lighting, "
        "ray tracing, physically based rendering, real skin and fabric textures, volumetric fog, film grain, 8k, "
        "epic and grounded, fully photorealistic, not stylized or animated.\n\n"
        "Cast (use these exact names and appearance, anchored to the provided reference images):\n"
        "- 主角 (Protagonist): the hero of the story, appearance MUST match the provided reference image char_main.jpg "
        "(use it as the visual anchor for face/body/clothing consistency).\n"
        "- 反派1 (Villain 1): a primary antagonist, appearance MUST match char_villain1.jpg.\n"
        "- 反派2 (Villain 2): a secondary antagonist, appearance MUST match char_villain2.jpg.\n\n"
        "Story (3 acts, about 6-8 shots):\n"
        "Act 1 - At dawn on a misty immortal mountain, 主角 trains alone, the wind stirring their robe; "
        "an old mentor warns that a sealed relic must never fall into darkness.\n"
        "Act 2 - 反派1 descends with 反派2 and a dark army, shattering the mountain altar in search of the relic; "
        "主角 stands against them.\n"
        "Act 3 - A decisive duel at the collapsing crimson altar; 主角 channels sword light and strikes 反派1 down, "
        "the relic dissolving into spirit light, peace restored.\n"
        "Requirements:\n"
        "- Every shot must be photorealistic CGI, cinematic camera work, realistic materials, no stylized or cartoon look.\n"
        "- Include about 6-8 shots.\n"
        "- Add a narration voice-over in Chinese and a matching soundtrack.\n"
    ),
    "global_setting": "",
    "scenes": [],
    "current_scene_index": 0,
    "voice_role": "yunxi",
    "enable_audio": True,
    "bgm_mood": "tense",
    "prefer_native_audio": False,
    "use_first_last_frame": True,
    "reference_sheets": {
        "characters": {
            "主角": {"name": "主角", "urls": ["/media/assets_uploaded/char_main.jpg"], "description": "素材库主角人物立绘，作为角色一致性参考锚点"},
            "反派1": {"name": "反派1", "urls": ["/media/assets_uploaded/char_villain1.jpg"], "description": "素材库反派人物立绘，作为角色一致性参考锚点"},
            "反派2": {"name": "反派2", "urls": ["/media/assets_uploaded/char_villain2.jpg"], "description": "素材库反派人物立绘，作为角色一致性参考锚点"},
        },
        "props": {},
        "environments": {},
    },
    "reference_images": [
        "http://localhost:8900/media/assets_uploaded/char_main.jpg",
        "http://localhost:8900/media/assets_uploaded/char_villain1.jpg",
        "http://localhost:8900/media/assets_uploaded/char_villain2.jpg",
    ],
    "assets_imported": True,
    "auto_mode": True,
}

def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

# 1) create thread
tid = post("/threads", {"metadata": {"rerun_of": "030b4b82", "purpose": "verify-fix-A-B-C-D"}})
thread_id = tid["thread_id"]
print("THREAD_ID=" + thread_id)

# 2) stream run
payload = {
    "assistant_id": ASSISTANT,
    "input": initial_state,
    "stream_mode": ["events", "messages"],
}
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(BASE + f"/threads/{thread_id}/runs/stream", data=data,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=7200) as r:
    log_path = f"scripts/_rerun_{thread_id[:8]}.log"
    with open(log_path, "w", encoding="utf-8") as lf:
        print("LOG=" + log_path)
        for line in r:
            line = line.decode("utf-8", "replace")
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
print("DONE")
