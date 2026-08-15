# -*- coding: utf-8 -*-
import asyncio, time, os
from langgraph_sdk import get_client

TID = "01a0054b-28de-7572-b075-01b2d2392db5"

async def main():
    client = get_client(url="http://localhost:2024")
    deadline = time.time() + 200
    while time.time() < deadline:
        try:
            runs = await client.runs.list(TID, limit=1)
            r = runs[0] if runs else None
            status = r.get("status") if r else "none"
            if status in ("success", "error", "timeout", "crashed"):
                print(f"DONE run status={status} at {time.strftime('%H:%M:%S')}")
                state = await client.threads.get_state(TID)
                v = state.get("values", {}) or {}
                for k in ("final_movie_path","final_movie_with_audio","aborted","error_log","abort_reason"):
                    if v.get(k):
                        print(f"{k} = {v.get(k)}")
                print("next =", state.get("next"))
                return
            print(f"still running @ {time.strftime('%H:%M:%S')} status={status}")
        except Exception as e:
            print("err", e)
        await asyncio.sleep(10)
    print("TIMEOUT waiting")

asyncio.run(main())
