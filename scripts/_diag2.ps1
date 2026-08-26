$ErrorActionPreference = "Continue"
$lines = @()

# wsl docker-desktop running state
$job = Start-Job -ScriptBlock { wsl -d docker-desktop echo ok 2>&1 }
$ok = Wait-Job -Job $job -Timeout 6
if ($ok) { $lines += "wsl_docker_desktop: $(Receive-Job -Job $job)" }
else { Stop-Job -Job $job; $lines += "wsl_docker_desktop: TIMEOUT" }

# docker info with timeout
$job2 = Start-Job -ScriptBlock { docker info --format "{{.ServerVersion}}" 2>&1 }
$ok2 = Wait-Job -Job $job2 -Timeout 10
if ($ok2) { $lines += "docker_info: $(Receive-Job -Job $job2)" }
else { Stop-Job -Job $job2; $lines += "docker_info: TIMEOUT" }

# container status
$job3 = Start-Job -ScriptBlock { docker ps -a --filter name=jimeng-free-api-all --format "{{.Names}} {{.State}}" 2>&1 }
$ok3 = Wait-Job -Job $job3 -Timeout 10
if ($ok3) { $lines += "jimeng_container: $(Receive-Job -Job $job3)" }
else { Stop-Job -Job $job3; $lines += "jimeng_container: TIMEOUT" }

# 8090 probe
try {
    $r = Invoke-WebRequest -Uri "http://localhost:8090/v1/models" -TimeoutSec 4 -UseBasicParsing
    $lines += "8090: OK $($r.StatusCode)"
} catch { $lines += "8090: FAIL" }

$lines -join "`n" | Set-Content -Path "workspace\_diag2.txt" -Encoding ascii
$lines -join "`n"
