$ErrorActionPreference = "Continue"
$JIMENG_CONTAINER = "jimeng-free-api-all"
$JimengPort = 8090

Write-Host "start existing jimeng container"
docker start $JIMENG_CONTAINER
docker update --restart=always $JIMENG_CONTAINER 2>$null

Write-Host "wait for 8090 (max 60s)"
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:$JimengPort/v1/models" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
    Start-Sleep -Seconds 2
}
if (-not $ready) {
    Write-Host "WARN 8090 not responding, check: docker logs $JIMENG_CONTAINER"
    exit 0
}
Write-Host "OK 8090 responded 200, jimeng is up. Run: python scripts/run_xianxia_full.py"
