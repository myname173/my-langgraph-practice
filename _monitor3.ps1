$project = 'c:\Users/13682\Desktop\my-langgraph-practice-main'
$py = 'C:\Users\13682\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe'
$site = Join-Path $project '.venv\Lib\site-packages'
$env:PYTHONPATH = $site
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location $project

$job = Start-Job -ScriptBlock {
    param($project, $py, $site)
    $env:PYTHONPATH = $site
    $env:PYTHONDWRITEBYTECODE = '1'
    Set-Location $project
    for ($i = 0; $i -lt 8; $i++) {
        Start-Sleep -Seconds 40
        Write-Host "===== iter $i ====="
        Write-Host "--- run_real tail ---"
        Get-Content -Encoding utf8 (Join-Path $project 'logs_run_real.txt') -Tail 6
        Write-Host "--- lg tail ---"
        Get-Content -Encoding utf8 (Join-Path $project 'logs_langgraph.txt') -Tail 3
        $kfBase = Join-Path $project 'output\keyframes'
        if (Test-Path $kfBase) {
            $latest = Get-ChildItem $kfBase -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($latest) {
                $kfs = Get-ChildItem $latest.FullName -ErrorAction SilentlyContinue
                Write-Host "latest keyframe dir: $($latest.Name) count=$($kfs.Count)"
            }
        }
    }
} -ArgumentList $project, $py, $site
$wait = Wait-Job -Job $job -Timeout 360
if ($wait.State -eq 'Completed') { Receive-Job -Job $job } else { "MONITOR TIMEOUT 360s" }
