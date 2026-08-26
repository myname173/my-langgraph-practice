$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$Root = "C:\Users\13682\Desktop\my-langgraph-practice-main"
Start-Process -FilePath python.exe `
    -ArgumentList "scripts/run_xianxia_full.py" `
    -WorkingDirectory $Root `
    -NoNewWindow `
    -RedirectStandardOutput (Join-Path $Root "workspace\_xianxia_run.log") `
    -RedirectStandardError (Join-Path $Root "workspace\_xianxia_err.log")
Write-Output "STARTED"
