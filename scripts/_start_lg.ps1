$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$ROOT = "C:\Users\13682\Desktop\my-langgraph-practice-main"
$p = Start-Process -FilePath "python" `
    -ArgumentList "-m","langgraph_cli","dev","--port","2024","--no-reload" `
    -WorkingDirectory $ROOT -NoNewWindow -PassThru
$p.Id | Out-File -FilePath "$ROOT\workspace\.lg_pid" -Encoding ascii
Write-Host "LG_PID=$($p.Id)"
