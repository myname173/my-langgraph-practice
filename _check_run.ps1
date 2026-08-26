$log = 'c:\Users\13682\Desktop\my-langgraph-practice-main\logs_run_real.txt'
if (Test-Path $log) {
    Write-Host 'LOG_EXISTS, lines:'
    Get-Content $log -Tail 30
} else {
    Write-Host 'NO_LOG_YET'
}
