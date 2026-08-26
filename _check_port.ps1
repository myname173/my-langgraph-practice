try {
    $a = Invoke-WebRequest -Uri 'http://127.0.0.1:2024/ok' -TimeoutSec 5 -UseBasicParsing
    Write-Host ('OK:' + $a.StatusCode)
} catch {
    Write-Host ('OKFAIL:' + $_.Exception.Message)
}
try {
    $b = Invoke-WebRequest -Uri 'http://127.0.0.1:2024/threads' -TimeoutSec 5 -UseBasicParsing
    Write-Host ('THREADS:' + $b.StatusCode)
} catch {
    Write-Host ('THREADSFAIL:' + $_.Exception.Message)
}
