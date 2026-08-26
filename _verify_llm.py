import asyncio
from langgraph_sdk import get_client

INPUT = {
    "task": (
        "Write and produce a short xianxia (Chinese cultivation fantasy) live-action style "
        "movie trailer with 6 scenes. Inject the provided reference assets and produce a full "
        "multimedia video with keyframes, video clips, voiceover and subtitles."
    ),
    "assets": {
        "imported": True,
        "references": [
            {"category": "characters", "name": "仙尊", "filename": "char_elder_master_real.jpg"},
            {"category": "characters", "name": "魔尊", "filename": "char_villain_lord_real.jpg"},
            {"category": "characters", "name": "女剑仙", "filename": "char_female_companion_real.jpg"},
            {"category": "props", "name": "青锋剑", "filename": "prop_qingfeng_sword_real.png"},
            {"category": "props", "name": "神秘玉佩", "filename": "prop_mystic_jade_pendant_real.png"},
        ],
    },
    "use_first_last_frame": True,
    "multimodal_backend": "jimeng",
    "video_provider": "agnes",
}

async def main():
    client = get_client(url="http://localhost:2024")
    thread = await client.threads.create()
    tid = thread["thread_id"]
    print("THREAD_ID:", tid)
    run = await client.runs.create(
        thread_id=tid,
        assistant_id="agent",
        input=INPUT,
        stream_mode=["values", "updates"],
    )
    print("RUN_ID:", run.get("run_id"))

asyncio.run(main())
