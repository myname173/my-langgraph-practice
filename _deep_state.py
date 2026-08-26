import asyncio, json
from langgraph_sdk import get_client

TID = "3dc8284e-b10f-4073-be36-e703c1406c95"

async def main():
    c = get_client(url="http://localhost:2024")
    st = await c.threads.get_state(TID)
    print("NEXT:", st.get("next"))
    print("TASKS:", json.dumps(st.get("tasks"), ensure_ascii=False)[:1500])
    print("interrupts:", st.get("interrupts"))
    v = st.get("values", {})
    scenes = v.get("scenes")
    if isinstance(scenes, list):
        for i, s in enumerate(scenes[:6]):
            dur = s.get("duration_seconds") or s.get("duration")
            print(f"scene[{i}] dur={dur} dialogue={(s.get('dialogue') or '')[:30]!r}")
    # 查这个 thread 的 runs
    runs = await c.runs.list(thread_id=TID)
    print("THREAD RUNS:", len(runs))
    for r in runs:
        print("  ", r.get("run_id"), r.get("status"), "interrupts:", r.get("interrupts"))

asyncio.run(main())
