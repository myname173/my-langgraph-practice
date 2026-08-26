import asyncio, json
from langgraph_sdk import get_client

TID = "3dc8284e-b10f-4073-be36-e703c1406c95"
RUN = "01a034e5-e88e-7332-a447-c9f7874a7afb"

async def main():
    c = get_client(url="http://localhost:2024")
    st = await c.threads.get_state(TID)
    interrupts = st.get("interrupts") or []
    if not interrupts:
        print("NO INTERRUPTS - current next:", st.get("next"))
        return
    for itr in interrupts:
        val = itr.get("value", {})
        stage = val.get("stage")
        print(f"approving interrupt stage={stage} id={itr.get('id')}")
        # resume the run with approve action
        await c.runs.create(
            thread_id=TID,
            assistant_id="agent",
            input={"action": "approve", "stage": stage},
            config={"configurable": {"thread_id": TID}},
            stream_mode=["values", "updates"],
        )
        print(f"resumed run for stage={stage}")

asyncio.run(main())
