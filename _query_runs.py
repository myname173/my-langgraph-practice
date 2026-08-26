import asyncio
from langgraph_sdk import get_client

TID = "6c988ec4-b5db-474b-8fe0-a49809f3e1ed"

async def main():
    client = get_client(url="http://localhost:2024")
    try:
        runs = await client.runs.list(thread_id=TID)
    except Exception as e:
        print("LIST_ERR", repr(e))
        return
    print("TOTAL RUNS:", len(runs))
    for r in runs[-6:]:
        print("---")
        print("run_id:", r.get("run_id"))
        print("status:", r.get("status"))
        err = r.get("error")
        if err:
            print("ERROR:", str(err)[:1500])

asyncio.run(main())
