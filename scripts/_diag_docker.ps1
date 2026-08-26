# probe docker/wsl responsiveness with hard timeout, detached
$ErrorActionPreference = "Continue"
$out = @()

# docker version with 8s timeout
$job = Start-Job -ScriptBlock { docker version --format "{{.Server.Version}}" 2>&1 }
$ok = Wait-Job -Job $job -Timeout 8
if ($ok) {
    $res = Receive-Job -Job $job
    $out += "docker_version: $res"
} else {
    Stop-Job -Job $job
    $out += "docker_version: TIMEOUT (daemon not responding)"
}

# wsl list with 8s timeout
$job2 = Start-Job -ScriptBlock { wsl --list --quiet 2>&1 }
$ok2 = Wait-Job -Job $job2 -Timeout 8
if ($ok2) {
    $res2 = Receive-Job -Job $job2
    $out += "wsl_list: $res2"
} else {
    Stop-Job -Job $job2
    $out += "wsl_list: TIMEOUT"
}

$out -join "`n" | Set-Content -Path "workspace\_diag_docker.txt" -Encoding ascii
Write-Host ($out -join "`n")
