$project = 'c:\Users\13682\Desktop\my-langgraph-practice-main'
$py = 'C:\Users\13682\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe'
$site = Join-Path $project '.venv\Lib\site-packages'
$env:PYTHONPATH = $site
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location $project

$out = Join-Path $project 'poll_result.txt'
$tid = "0674a20f-438e-4028-bd9d-f61980a106c8"
$q = @"
import asyncio
from langgraph_sdk import get_client
async def main():
    c = get_client(url='http://localhost:2024')
    st = await c.threads.get_state('$tid')
    nxt = st.get('next')
    v = st.get('values', {})
    lines = [f'NEXT: {nxt}']
    for k in ['showrunner','visual_context','shot_strategy','scenes','images','keyframes','video_tasks','final_video']:
        val = v.get(k)
        if isinstance(val, dict): lines.append(f'{k}: dict {list(val.keys())[:6]}')
        elif isinstance(val, list): lines.append(f'{k}: list len={len(val)}')
        else: lines.append(f'{k}: {val}')
    open(r'$out','w',encoding='utf-8').write(chr(10).join(lines))
asyncio.run(main())
"@
$q | Out-File -Encoding utf8 _q_state.py

for ($i = 0; $i -lt 50; $i++) {
    & $py _q_state.py 2>&1 | Out-Null
    Start-Sleep -Seconds 30
}
