$ErrorActionPreference = "Continue"
$allup = $false
for ($i = 0; $i -lt 50; $i++) {
    $lg = $false; $md = $false; $jm = $false
    try { $r = Invoke-WebRequest -Uri "http://localhost:2024/ok" -TimeoutSec 2 -UseBasicParsing; if ($r.StatusCode -eq 200) { $lg = $true } } catch {}
    try { $r = Invoke-WebRequest -Uri "http://localhost:8900/ok" -TimeoutSec 2 -UseBasicParsing; if ($r.StatusCode -eq 200) { $md = $true } } catch {}
    try { $r = Invoke-WebRequest -Uri "http://localhost:8090/v1/models" -TimeoutSec 2 -UseBasicParsing; if ($r.StatusCode -eq 200) { $jm = $true } } catch {}
    Write-Host ("tick $i : 2024=$lg 8900=$md 8090=$jm")
    if ($lg -and $md -and $jm) { $allup = $true; break }
    Start-Sleep -Seconds 3
}
if ($allup) { Write-Host "ALL_UP" } else { Write-Host "NOT_ALL_UP" }
