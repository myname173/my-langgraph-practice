import asyncio, json
from langgraph_sdk import get_client

TID = "6c988ec4-b5db-474b-8fe0-a49809f3e1ed"

async def main():
    client = get_client(url="http://localhost:2024")
    try:
        state = await client.threads.get_state(TID)
    except Exception as e:
        print("STATE_ERR", repr(e))
        return
    vals = state.get("values", {})
    print("NEXT:", state.get("next"))
    print("TASKS:", state.get("tasks"))
    for k in ["visual_context", "shot_strategy", "director", "scenes", "images", "keyframes", "video_tasks", "audio", "subtitles", "final_video"]:
        v = vals.get(k)
        if isinstance(v, dict):
            print(f"{k}: dict keys={list(v.keys())[:8]}")
        elif isinstance(v, list):
            print(f"{k}: list len={len(v)}")
        elif v is None:
            print(f"{k}: None")
        else:
            print(f"{k}: {type(v).__name__}")

asyncio.run(main())
