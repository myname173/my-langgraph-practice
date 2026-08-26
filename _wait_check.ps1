Start-Sleep -Seconds 25
$log = 'c:\Users\13682\Desktop\my-langgraph-practice-main\logs_run_real.txt'
if (Test-Path $log) {
    Write-Host '=== RUN LOG (tail) ==='
    Get-Content $log -Tail 35
} else {
    Write-Host 'NO_LOG_YET'
}
$py = Get-Process -Name python -ErrorAction SilentlyContinue
Write-Host ('PYTHON_PROCS: ' + $py.Count)
