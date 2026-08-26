import asyncio
from langgraph_sdk import get_client
async def main():
    c = get_client(url='http://localhost:2024')
    st = await c.threads.get_state('3dc8284e-b10f-4073-be36-e703c1406c95')
    v = st.get('values', {})
    print('NEXT:', st.get('next'))
    for k in ['showrunner','visual_context','shot_strategy','scenes','images','keyframes','video_tasks','final_video']:
        val = v.get(k)
        if isinstance(val, dict):
            print(f'{k}: dict keys={list(val.keys())[:8]}')
        elif isinstance(val, list):
            print(f'{k}: list len={len(val)}')
        else:
            print(f'{k}: {val}')
asyncio.run(main())
