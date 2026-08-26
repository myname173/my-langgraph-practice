import asyncio
from langgraph_sdk import get_client

# 找出最近 running/pending 的 run 并取消（避免持续刷 jimeng 代理）
async def main():
    c = get_client(url="http://localhost:2024")
    # 列出 assistant 最近 runs
    runs = await c.runs.list(assistant_id="agent", limit=10)
    for r in runs:
        if r.get("status") in ("pending", "running"):
            try:
                await c.runs.cancel(r["thread_id"], r["run_id"])
                print("CANCELLED", r["thread_id"], r["run_id"], r.get("status"))
            except Exception as e:
                print("ERR", r["run_id"], repr(e)[:120])
    print("done")

asyncio.run(main())
