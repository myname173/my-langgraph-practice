$project = 'c:\Users\13682\Desktop\my-langgraph-practice-main'
$py = 'C:\Users\13682\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe'
$site = Join-Path $project '.venv\Lib\site-packages'
$env:PYTHONPATH = $site
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location $project

for ($i = 0; $i -lt 12; $i++) {
    Start-Sleep -Seconds 30
    & $py _query_state.py 2>&1
    Write-Host "=== iter $i - langgraph tail ==="
    Get-Content -Encoding utf8 (Join-Path $project 'logs_langgraph.txt') -Tail 3
    Write-Host "==================================="
}
