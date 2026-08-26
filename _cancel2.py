import asyncio
from langgraph_sdk import get_client

TID = "3dc8284e-b10f-4073-be36-e703c1406c95"

async def main():
    c = get_client(url="http://localhost:2024")
    runs = await c.runs.list(thread_id=TID)
    for r in runs:
        if r.get("status") in ("pending", "running"):
            try:
                await c.runs.cancel(TID, r["run_id"])
                print("CANCELLED", r["run_id"], r.get("status"))
            except Exception as e:
                print("ERR", r["run_id"], repr(e))
    print("done")

asyncio.run(main())
