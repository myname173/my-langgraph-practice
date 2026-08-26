$project = 'c:\Users\13682\Desktop\my-langgraph-practice-main'
$py = 'C:\Users\13682\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe'
$site = Join-Path $project '.venv\Lib\site-packages'
$env:PYTHONPATH = $site
$env:PYTHONDONTWRITEBYTECODE = '1'
$code = @'
import asyncio, json
from langgraph_sdk import get_client
async def main():
    client = get_client(url="http://localhost:2024")
    tid = "6c988ec4-b5db-474b-8fe0-a49809f3e1ed"
    try:
        state = await client.threads.get_state(tid)
    except Exception as e:
        print("STATE_ERR", repr(e)); return
    vals = state.get("values", {})
    print("NEXT:", state.get("next"))
    print("TASKS:", state.get("tasks"))
    # 打印主要字段存在性
    for k in ["visual_context","shot_strategy","director","scenes","images","keyframes","video_tasks","audio","subtitles","final_video"]:
        v = vals.get(k)
        if isinstance(v, dict):
            print(f"{k}: dict keys={list(v.keys())[:8]}")
        elif isinstance(v, list):
            print(f"{k}: list len={len(v)}")
        elif v is None:
            print(f"{k}: None")
        else:
            print(f"{k}: {type(v).__name__}")
asyncio.run(main())
'@
Set-Location $project
& $py -c $code 2>&1
