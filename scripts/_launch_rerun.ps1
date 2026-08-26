$proc = Start-Process -FilePath python -ArgumentList 'scripts/_rerun_030b.py' `
    -RedirectStandardOutput 'scripts/_rerun_stdout.log' `
    -RedirectStandardError 'scripts/_rerun_stderr.log' `
    -PassThru
Write-Host ("STARTED PID=" + $proc.Id)
Start-Sleep -Seconds 12
if (Test-Path 'scripts/_rerun_stdout.log') {
    Get-Content 'scripts/_rerun_stdout.log' | Select-Object -First 20
}
if (Test-Path 'scripts/_rerun_stderr.log') {
    $err = Get-Content 'scripts/_rerun_stderr.log' | Select-Object -First 10
    if ($err) { Write-Host "=== STDERR ==="; $err }
}
