Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3
Write-Host ('REMAINING_PY:' + (Get-Process -Name python -ErrorAction SilentlyContinue | Measure-Object).Count)
