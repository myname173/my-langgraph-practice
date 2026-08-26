import asyncio
from langgraph_sdk import get_client
async def main():
    c = get_client(url='http://localhost:2024')
    st = await c.threads.get_state('0674a20f-438e-4028-bd9d-f61980a106c8')
    nxt = st.get('next')
    v = st.get('values', {})
    lines = [f'NEXT: {nxt}']
    for k in ['showrunner','visual_context','shot_strategy','scenes','images','keyframes','video_tasks','final_video']:
        val = v.get(k)
        if isinstance(val, dict): lines.append(f'{k}: dict {list(val.keys())[:6]}')
        elif isinstance(val, list): lines.append(f'{k}: list len={len(val)}')
        else: lines.append(f'{k}: {val}')
    open(r'c:\Users\13682\Desktop\my-langgraph-practice-main\poll_result.txt','w',encoding='utf-8').write(chr(10).join(lines))
asyncio.run(main())
