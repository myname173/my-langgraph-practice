import asyncio
from langgraph_sdk import get_client

async def main():
    client = get_client(url="http://localhost:2024")
    try:
        runs = await client.runs.list(status="pending")
    except Exception as e:
        print("LIST_ERR", repr(e))
        return
    print("PENDING RUNS:", len(runs))
    for r in runs:
        print("---")
        print("run_id:", r.get("run_id"))
        print("thread_id:", r.get("thread_id"))
        print("status:", r.get("status"))
        print("kwargs:", {k: str(v)[:120] for k, v in (r.get("kwargs") or {}).items()})

asyncio.run(main())
