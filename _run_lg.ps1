$ErrorActionPreference = 'Continue'
$project = 'c:\Users\13682\Desktop\my-langgraph-practice-main'
$py = 'C:\Users\13682\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe'
$site = Join-Path $project '.venv\Lib\site-packages'

# 用可运行的 uv python，把 .venv 的 site-packages 挂上去，直接跑 langgraph_cli 模块
$env:PYTHONPATH = $site
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location $project
& $py -m langgraph_cli --help 2>&1 | Select-Object -First 12
