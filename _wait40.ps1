Start-Sleep -Seconds 40
$srv = 'c:\Users\13682\Desktop\my-langgraph-practice-main\logs_lg_job.txt'
Write-Host '=== SERVER LOG (last 18) ==='
if (Test-Path $srv) { Get-Content $srv -Tail 18 } else { Write-Host 'NO_SRV_LOG' }
$py = Get-Process -Name python -ErrorAction SilentlyContinue
Write-Host ('PYTHON_PROCS: ' + $py.Count)
