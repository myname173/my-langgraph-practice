"""Run a full xianxia pipeline test using REAL imported assets (manifest) with photorealistic CG style.

- Uses langgraph_sdk (same pattern as scripts/_drive_stream.py) so the thread is created
  properly and interrupt auto-approval works.
- Reference sheets use the EXACT names present in the asset manifest so the matcher
  attaches them to the script.
- Photorealistic CG only: NO watercolor / ink-wash / anime wording anywhere, so the
  backend auto-selects the `aaa_cg` style (AAA game CG trailer, UE5, ray tracing, 8k).
"""
import os
import re
import json
import asyncio
import sys

from pathlib import Path

import os as _os
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "output" / "reference_sheets" / "library" / "manifest.json"
# 端口可用 LG_PORT 覆盖（默认 2024，与 `langgraph dev` 默认端口一致）。
LG_BASE = "http://localhost:%s" % _os.environ.get("LG_PORT", "2024")
MEDIA_BASE = "http://localhost:8900"
# 与前端 frontend/src/lib/langgraphClient.ts 的 ASSISTANT_ID 完全一致，
# 即“前端有的接口”实际调用的就是这个 agent 图（langgraph.json 仅注册了 agent）。
ASSISTANT_ID = "agent"

# 冒烟模式：SMOKE=1 时只跑 2 个镜头，验证即梦生图链路是否跑通，避免浪费每日免费积分。
SMOKE = _os.environ.get("SMOKE", "") in ("1", "true", "True")

# 角色设定：与素材库里“已有的素材图”登记名一致（仙尊/魔尊/女剑仙，source:user）。
# 这等价于用户在前端「素材库」面板勾选这些真人参考图后填写的 task。
# assets_imported=True 时，graph 复用注入的 reference_images 作一致性锚点，
# CAST 仅作语义引导，角色外观以素材库真实人物图为准。
CAST = (
    "Cast (use these exact names and appearance, anchored to the provided reference images):\n"
    "- 仙尊 (Immortal Venerable): the hero of the story, appearance MUST match the provided reference "
    "image char_elder_master_real.jpg (use it as the visual anchor for face/body/clothing consistency).\n"
    "- 魔尊 (Demon Venerable): a primary antagonist, appearance MUST match char_villain_lord_real.jpg.\n"
    "- 女剑仙 (Sword Fairy): a secondary character / ally, appearance MUST match char_female_companion_real.jpg.\n"
    "- 青锋剑 (Azure Edge Sword) and 神秘玉佩 (Mystic Jade Pendant): key props, MUST match the "
    "provided reference images prop_qingfeng_sword_real.png / prop_mystic_jade_pendant_real.png.\n\n"
)

STYLE = (
    "Visual style: AAA game CG trailer, Unreal Engine 5 render, cinematic lighting, ray "
    "tracing, physically based rendering, real skin and fabric textures, volumetric fog, film "
    "grain, 8k, epic and grounded, fully photorealistic, not stylized or animated.\n\n"
)

if SMOKE:
    # 冒烟：2 个镜头，验证素材库人物参考图注入 + agnes 视频链。
    # 注意：内容需温和，避免写实暴力触发安全审核（photorealistic 风格下流血细节会被拒）。
    STORY = (
        "Story (about 2 shots, gentle & scenic, no violence or blood):\n"
        "Shot 1 - At dawn on a misty immortal mountain, 仙尊 meditates peacefully, "
        "the wind gently stirring their robe as spirit light gathers around them.\n"
        "Shot 2 - 仙尊 walks through a glowing bamboo forest at sunrise, 魔尊 and 女剑仙 "
        "watching from afar with calm curiosity as golden petals drift down.\n"
    )
    SHOT_REQ = "about 2 shots"
else:
    STORY = (
        "Story (3 acts, about 6-8 shots):\n"
        "Act 1 - At dawn on a misty immortal mountain, 仙尊 trains alone with the 青锋剑, the wind "
        "stirring their robe; an old mentor warns that the sealed 神秘玉佩 must never fall into darkness.\n"
        "Act 2 - 魔尊 descends with a dark army, shattering the mountain altar in search of the relic; "
        "仙尊 stands against them, 女剑仙 by their side.\n"
        "Act 3 - A decisive duel at the collapsing crimson altar; 仙尊 channels sword light and strikes "
        "魔尊 down, the relic dissolving into spirit light, peace restored.\n"
    )
    SHOT_REQ = "about 6-8 shots"

TASK = (
    "Write and produce a short xianxia (Chinese cultivation fantasy) live-action style "
    "movie trailer, fully photorealistic CGI. " + STYLE + CAST + STORY +
    "Requirements:\n"
    "- Every shot must be photorealistic CGI, cinematic camera work, realistic materials, "
    "no stylized or cartoon look.\n"
    f"- Include {SHOT_REQ}.\n"
    "- Add a narration voice-over in Chinese and a matching soundtrack."
)


# 素材库里「已有的素材图」：output/reference_sheets/library 下登记的参考图
# （source:user，manifest.json 登记名=仙尊/魔尊/女剑仙/青锋剑/神秘玉佩）。
# 这等价于用户在前端「素材库」面板勾选这些图后提交的素材。
ASSET_RECORDS = [
    {"filename": "char_elder_master_real.jpg", "category": "角色", "role_name": "仙尊",
     "description": "素材库真人参考图：仙尊，作为主角一致性参考锚点"},
    {"filename": "char_villain_lord_real.jpg", "category": "角色", "role_name": "魔尊",
     "description": "素材库真人参考图：魔尊，作为反派一致性参考锚点"},
    {"filename": "char_female_companion_real.jpg", "category": "角色", "role_name": "女剑仙",
     "description": "素材库真人参考图：女剑仙，作为盟友一致性参考锚点"},
    {"filename": "prop_qingfeng_sword_real.png", "category": "道具", "role_name": "青锋剑",
     "description": "素材库真人参考图：青锋剑，关键道具"},
    {"filename": "prop_mystic_jade_pendant_real.png", "category": "道具", "role_name": "神秘玉佩",
     "description": "素材库真人参考图：神秘玉佩，关键道具"},
]

# 素材库真人参考图 URL（经 /media 静态服务访问），供 reference_images 注入视频生成作一致性参考
ASSET_IMAGE_URLS = [
    f"http://localhost:8900/media/reference_sheets/library/{r['category'].replace('角色','characters').replace('道具','props')}/{r['filename']}" for r in ASSET_RECORDS
]

def build_sheets():
    chars = {}
    props = {}
    for r in ASSET_RECORDS:
        name = (r.get("role_name") or r.get("label") or "").strip()
        cat = (r.get("category") or "").strip()
        url = f"/media/reference_sheets/library/{r['category'].replace('角色','characters').replace('道具','props')}/{r['filename']}"
        entry = {"name": name, "urls": [url], "description": r.get("description", "")}
        if cat == "角色":
            chars.setdefault(name, []).append(entry)
        elif cat == "道具":
            props.setdefault(name, []).append(entry)
    # reference_sheets expects a flat name->entry map under each category
    chars_map = {name: (entries[0] if len(entries) == 1 else {"name": name, "urls": [e["urls"][0] for e in entries], "description": "；".join(e["description"] for e in entries)}) for name, entries in chars.items()}
    props_map = {name: (entries[0] if len(entries) == 1 else {"name": name, "urls": [e["urls"][0] for e in entries], "description": "；".join(e["description"] for e in entries)}) for name, entries in props.items()}
    return {"characters": chars_map, "props": props_map, "environments": {}}


async def main():
    from langgraph_sdk import get_client
    from langgraph_sdk.schema import Command

    client = get_client(url=LG_BASE, api_key="")
    sheets = build_sheets()
    run_input = {
        "task": TASK,
        "global_setting": "",
        "scenes": [],  # 场景不指定，交给 LLM 按故事自动生成
        "current_scene_index": 0,
        "use_first_last_frame": True,
        "enable_audio": True,
        # 冒烟模式限制 2 镜，让 showrunner 跳过三幕弧线完整性校验（短故事必然缺幕）
        "max_shots": 2 if SMOKE else None,
        "prefer_native_audio": False,
        "voice_role": "yunxi",
        "bgm_mood": "tense",
        # 注入素材库真实人物参考图：作视频生成一致性锚点（agnes keyframes 多图模式消费）
        "reference_images": ASSET_IMAGE_URLS,
        "reference_embeddings": [],
        "reference_sheets": sheets,
        # 素材已导入：graph 复用 reference_sheets 与 reference_images，角色一致、省额度
        "assets_imported": True,
        "auto_mode": True,
        "aborted": False,
        "final_movie_path": None,
        "error_log": None,
        "rewrite_count": 0,
    }
    thread = await client.threads.create()
    tid = thread["thread_id"]
    print(f"[start] thread_id={tid}")
    print(f"[start] characters={list(sheets['characters'].keys())}")
    print(f"[start] props={list(sheets['props'].keys())}")

    command = None
    interrupt_count = 0
    while True:
        if command is None:
            stream = client.runs.stream(
                tid, ASSISTANT_ID, input=run_input, stream_mode=["values", "updates"]
            )
        else:
            stream = client.runs.stream(
                tid, ASSISTANT_ID, command=command, stream_mode=["values", "updates"]
            )
            command = None
        async for chunk in stream:
            if chunk.event == "values":
                v = chunk.data or {}
                nn = v.get("__node__")
                if nn:
                    print(f"  -> node: {nn}", flush=True)
            elif chunk.event == "updates":
                for node, out in (chunk.data or {}).items():
                    if isinstance(out, dict) and out.get("__interrupt__"):
                        print(f"[interrupt] {node}", flush=True)
            elif chunk.event == "error":
                print(f"[error] {chunk.data}", flush=True)
        state = await client.threads.get_state(tid)
        values = state.get("values", {})
        pending = []
        for t in state.get("tasks", []):
            if t.get("interrupts"):
                pending.extend(t["interrupts"])
        if pending:
            interrupt_count += 1
            print(f"[interrupt #{interrupt_count}] auto-approve", flush=True)
            command = Command(resume={"action": "approve"})
            continue
        runs_now = await client.runs.list(thread_id=tid)
        if any(r.get("status") == "running" for r in runs_now):
            print("[pending] run still running, empty resume to continue", flush=True)
            command = Command(resume={})
            continue
        final = values.get("final_movie_path")
        err = values.get("error_log")
        if final or err:
            print(f"[done] interrupts={interrupt_count}")
            if err:
                print(f"[error_log] {err}")
            if final:
                print(f"[final_movie] {final}")
            break
        print("[done] ended without terminal state")
        break


if __name__ == "__main__":
    asyncio.run(main())
