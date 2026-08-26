import asyncio
from langgraph_sdk import get_client
tid = "c7152ce2-ffb8-4a31-9bab-23a203483172"
async def main():
    c = get_client(url="http://localhost:2024", api_key="")
    st = await c.threads.get_state(tid)
    vals = st.get("values", {})
    print("NEXT_NODE:", vals.get("__node__"))
    print("ERROR_LOG:", vals.get("error_log"))
    print("FINAL:", vals.get("final_movie_path"))
    for t in st.get("tasks", []):
        print("TASK_STATUS:", t.get("status"))
    runs = await c.runs.list(thread_id=tid)
    for r in runs[:5]:
        print("RUN:", r.get("run_id"), r.get("status"), r.get("name"))
asyncio.run(main())
