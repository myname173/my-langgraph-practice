import asyncio
from langgraph_sdk import get_client
async def main():
    c = get_client(url="http://localhost:2024")
    runs = await c.runs.list(assistant_id="agent", limit=3)
    for r in runs:
        print(r.get("run_id"), r.get("thread_id"), r.get("status"))
asyncio.run(main())
