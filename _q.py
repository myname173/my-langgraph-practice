import asyncio
from langgraph_sdk import get_client
TID = "2211a9f5-d951-4996-a433-19e98e8e98ae"
async def main():
    c = get_client(url="http://localhost:2024")
    st = await c.threads.get_state(TID)
    print("NEXT:", st.get("next"))
    v = st.get("values", {})
    for k in ["showrunner","visual_context","shot_strategy","scenes","images","keyframes","video_tasks","final_video"]:
        val = v.get(k)
        if isinstance(val, dict): print(f"{k}: dict keys={list(val.keys())[:6]}")
        elif isinstance(val, list): print(f"{k}: list len={len(val)}")
        else: print(f"{k}: {val}")
asyncio.run(main())
