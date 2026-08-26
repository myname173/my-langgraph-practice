import asyncio
from langgraph_sdk import get_client

NEW = "6c988ec4-b5db-474b-8fe0-a49809f3e1ed"

async def main():
    client = get_client(url="http://localhost:2024")
    runs = await client.runs.list(thread_id=NEW)
    print("NEW THREAD RUNS:", len(runs))
    for r in runs:
        print("---")
        print("run_id:", r.get("run_id"))
        print("status:", r.get("status"))
        print("kwargs:", {k: str(v)[:100] for k, v in (r.get("kwargs") or {}).items()})
        err = r.get("error")
        if err:
            print("ERROR:", str(err)[:800])
    # 同时查 assistant 当前是否有 running 的 run
    try:
        all_runs = await client.runs.list(assistant_id="fe096781-5601-53d2-b2f6-0d3403f7e9ca", limit=5)
        print("=== recent runs for assistant ===")
        for r in all_runs:
            print(r.get("run_id"), r.get("thread_id"), r.get("status"))
    except Exception as e:
        print("ASSIST_ERR", repr(e))

asyncio.run(main())
