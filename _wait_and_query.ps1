$project = 'c:\Users\13682\Desktop\my-langgraph-practice-main'
$py = 'C:\Users\13682\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe'
$site = Join-Path $project '.venv\Lib\site-packages'
$env:PYTHONPATH = $site
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location $project

Write-Host "waiting 75s for old render to finish..."
Start-Sleep -Seconds 75

& $py _query_state.py 2>&1
Write-Host "=== langgraph log tail ==="
Get-Content -Encoding utf8 (Join-Path $project 'logs_langgraph.txt') -Tail 4
