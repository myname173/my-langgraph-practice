$project = 'c:\Users\13682\Desktop\my-langgraph-practice-main'
$py = 'C:\Users\13682\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe'
$env:PYTHONPATH = Join-Path $project '.venv\Lib\site-packages'
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location $project
& $py -m langgraph_cli dev --no-reload --port 2024 *> 'logs_lg_job.txt'
