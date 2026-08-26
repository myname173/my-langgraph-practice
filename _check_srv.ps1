$log = 'c:\Users\13682\Desktop\my-langgraph-practice-main\logs_lg_job.txt'
if (Test-Path $log) {
    Write-Host '=== SERVER LOG (tail 12) ==='
    Get-Content $log -Tail 12
} else {
    Write-Host 'NO_SRV_LOG'
}
