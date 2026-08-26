$project = 'c:\Users\13682\Desktop\my-langgraph-practice-main'
$py = 'C:\Users\13682\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe'
$site = Join-Path $project '.venv\Lib\site-packages'
$env:PYTHONPATH = $site
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location $project

$tid = "3dc8284e-b10f-4073-be36-e703c1406c95"
$code = @"
import asyncio
from langgraph_sdk import get_client
async def main():
    c = get_client(url='http://localhost:2024')
    st = await c.threads.get_state('$tid')
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
"@
$env:PYTHONCODE = $code
# 写成临时 py 执行
$code | Out-File -Encoding utf8 _monitor_state.py
for ($i = 0; $i -lt 9; $i++) {
    Start-Sleep -Seconds 40
    Write-Host "===== iter $i ====="
    & $py _monitor_state.py 2>&1
    Write-Host "--- lg tail ---"
    Get-Content -Encoding utf8 (Join-Path $project 'logs_langgraph.txt') -Tail 4
}
