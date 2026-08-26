import asyncio
from langgraph_sdk import get_client

OLD = "01a02f96-96f6-7743-9a3a-c7c022ba1ddd"

async def main():
    client = get_client(url="http://localhost:2024")
    try:
        runs = await client.runs.list(thread_id=OLD)
    except Exception as e:
        print("LIST_ERR", repr(e))
        return
    print("OLD THREAD RUNS:", len(runs))
    for r in runs[-10:]:
        print("---")
        print("run_id:", r.get("run_id"))
        print("status:", r.get("status"))
        err = r.get("error")
        if err:
            print("ERROR:", str(err)[:400])

asyncio.run(main())
