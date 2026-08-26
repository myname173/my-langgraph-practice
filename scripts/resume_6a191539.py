"""针对线程 6a191539 自动续跑：空 resume 继续中断点 + 自动 approve 所有 HITL，直到产出成片。"""
import asyncio
from langgraph_sdk import get_client
from langgraph_sdk.schema import Command

TID = "6a191539-bb54-405e-8c3b-634f961525da"
AID = "fe096781-5601-53d2-b2f6-0d3403f7e9ca"
LG_BASE = "http://127.0.0.1:2024"

async def main():
    client = get_client(url=LG_BASE, api_key="")
    command = Command(resume={})  # 空 resume：从 interrupted 点继续
    interrupt_count = 0
    while True:
        stream = client.runs.stream(TID, AID, command=command, stream_mode=["values", "updates"])
        command = None
        async for chunk in stream:
            if chunk.event == "values":
                v = chunk.data or {}
                nn = v.get("__node__")
                if nn:
                    print(f"  -> node: {nn}", flush=True)
            elif chunk.event == "updates":
                for node, out in (chunk.data or {}).items():
                    if isinstance(out, dict) and out.get("__interrupt__"):
                        print(f"[interrupt] {node}", flush=True)
            elif chunk.event == "error":
                print(f"[error] {chunk.data}", flush=True)
        # 检查是否需要继续
        state = await client.threads.get_state(TID)
        values = state.get("values", {})
        pending = []
        for t in state.get("tasks", []):
            if t.get("interrupts"):
                pending.extend(t["interrupts"])
        if pending:
            interrupt_count += 1
            print(f"[interrupt #{interrupt_count}] auto-approve", flush=True)
            command = Command(resume={"action": "approve"})
            continue
        runs_now = await client.runs.list(thread_id=TID)
        if any(r.get("status") == "running" for r in runs_now):
            print("[pending] run still running, empty resume to continue", flush=True)
            command = Command(resume={})
            continue
        final = values.get("final_movie_path")
        err = values.get("error_log")
        if final or err:
            print(f"[done] interrupts={interrupt_count}")
            if err:
                print(f"[error_log] {err}")
            if final:
                print(f"[final_movie] {final}")
            break
        print("[done] ended without terminal state")
        break

if __name__ == "__main__":
    asyncio.run(main())
