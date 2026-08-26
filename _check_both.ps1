$run = 'c:\Users\13682\Desktop\my-langgraph-practice-main\logs_run_real.txt'
$srv = 'c:\Users\13682\Desktop\my-langgraph-practice-main\logs_lg_job.txt'
Write-Host '=== RUN LOG (full) ==='
if (Test-Path $run) { Get-Content $run } else { Write-Host 'NO_RUN_LOG' }
Write-Host ''
Write-Host '=== SERVER LOG (last 20) ==='
if (Test-Path $srv) { Get-Content $srv -Tail 20 } else { Write-Host 'NO_SRV_LOG' }
$py = Get-Process -Name python -ErrorAction SilentlyContinue
Write-Host ('PYTHON_PROCS: ' + $py.Count)
