$ErrorActionPreference = "Continue"
$root = "c:\Users\13682\Desktop/my-langgraph-practice-main"
$uv = "$env:USERPROFILE\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe"

Write-Host "=== [A] kill stale procs ==="
# kill any leftover langgraph / media / run scripts
Get-Process -Name python -ErrorAction SilentlyContinue | ForEach-Object {
    $cmd = ($_.CommandLine)
    if ($cmd -match 'langgraph|static_server|run_xianxia_real') {
        Write-Host "kill pid $($_.Id): $cmd"
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "=== [A] start jimeng (docker) ==="
docker start jimeng-free-api-all
docker update --restart=always jimeng-free-api-all 2>$null
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8090/v1/models" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
    Start-Sleep -Seconds 2
}
if (-not $ready) { Write-Host "WARN 8090 not responding" } else { Write-Host "OK 8090 jimeng up" }

Write-Host "=== [A] start media server 8900 ==="
Start-Process -FilePath "cmd.exe" -ArgumentList "/c","cd /d $root && python -m src.agent.multimedia.static_server" -WindowStyle Normal
Write-Host "media server launching..."

Write-Host "=== [A] start langgraph 2024 ==="
Start-Process -FilePath "cmd.exe" -ArgumentList "/c","cd /d $root && uv run langgraph dev --no-reload --port 2024" -WindowStyle Normal
Write-Host "langgraph launching..."

Write-Host "=== [A] wait for 2024 ==="
$up = $false
for ($i = 0; $i -lt 40; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:2024/ok" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $up = $true; break }
    } catch { }
    Start-Sleep -Seconds 3
}
if (-not $up) { Write-Host "ERR 2024 NOT UP"; exit 1 }
Write-Host "OK 2024 langgraph up"

Write-Host "=== [A] wait for 8900 ==="
$up2 = $false
for ($i = 0; $i -lt 20; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8900/ok" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $up2 = $true; break }
    } catch { }
    Start-Sleep -Seconds 2
}
Write-Host ($(if ($up2) {"OK 8900 media up"} else {"WARN 8900 not up (will fall back to local file)"}))

Write-Host "=== [A] launch run_xianxia_real.py ==="
Start-Process -FilePath "cmd.exe" -ArgumentList "/c","cd /d $root && python scripts/run_xianxia_real.py > logs_run_a.txt 2>&1" -WindowStyle Normal
Write-Host "run_xianxia_real.py launched -> logs_run_a.txt"
