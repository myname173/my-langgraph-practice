$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $r = Invoke-WebRequest -Uri http://localhost:2024/ok -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) {
            Write-Host "READY after $($i)s"
            $ready = $true
            break
        }
    } catch {}
    Start-Sleep -Seconds 1
}
if (-not $ready) { Write-Host "NOT READY" }
Write-Host "--- log tail ---"
Get-Content logs_langgraph.txt -Tail 30 -ErrorAction SilentlyContinue
