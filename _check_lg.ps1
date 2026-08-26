$log = 'c:\Users\13682\Desktop\my-langgraph-practice-main\logs_lg_job.txt'
if (Test-Path $log) {
    Write-Host 'LOG_EXISTS'
    Get-Content $log -Tail 6
} else {
    Write-Host 'NO_LOG'
}
try {
    $r = Invoke-WebRequest -Uri http://127.0.0.1:2024/threads -TimeoutSec 4 -UseBasicParsing
    Write-Host ('UP:' + $r.StatusCode)
} catch {
    Write-Host 'DOWN'
}
