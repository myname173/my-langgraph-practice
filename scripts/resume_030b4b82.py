"""Resume the existing xianxia run (thread 030b4b82) from its last checkpoint.

- Reuses the persisted checkpoint (data/multimedia_checkpoints.sqlite); the run was
  stopped at node `director_refine` (step 9). Completed nodes (showrunner/writer/
  keyframe anchors) are NOT re-run.
- First loop iteration issues an empty resume command (Command(resume={})) to
  continue from the existing checkpoint instead of submitting a fresh input (which
  would re-execute the entry nodes).
- Auto-approves any interrupt so the pipeline can proceed unattended.
- Reference: scripts/run_xianxia_real.py (same auto-approve loop).
"""
import asyncio
from pathlib import Path
from langgraph_sdk import get_client
from langgraph_sdk.schema import Command

ROOT = Path(__file__).resolve().parents[1]
LG_BASE = "http://localhost:2024"
ASSISTANT_ID = "fe096781-5601-53d2-b6f0-0d3403f7e9ca"
# correct assistant id from langgraph.json metadata
ASSISTANT_ID = "fe096781-5601-53d2-b2f6-0d3403f7e9ca"

TID = "030b4b82-ea0d-48e4-a1b9-38e9d7af239b"


async def main():
    client = get_client(url=LG_BASE, api_key="")

    # sanity: thread must exist
    try:
        thread = await client.threads.get(TID)
    except Exception as e:
        print(f"[fatal] thread {TID} not found: {e}")
        return
    print(f"[resume] thread_id={TID} status={thread.get('status')}")

    command = Command(resume={})  # first: empty resume from existing checkpoint
    interrupt_count = 0
    while True:
        if command is not None:
            stream = client.runs.stream(
                TID, ASSISTANT_ID, command=command, stream_mode=["values", "updates"]
            )
            command = None
        else:
            # poll-style: keep ticking a running run forward
            stream = client.runs.stream(
                TID, ASSISTANT_ID, command=Command(resume={}), stream_mode=["values", "updates"]
            )
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
