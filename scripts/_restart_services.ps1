$root = "c:\Users\13682\Desktop\my-langgraph-practice-main"

# restart langgraph server in its own window (persists after this script ends)
Start-Process -FilePath "cmd.exe" -ArgumentList "/c","cd /d $root && uv run langgraph dev" `
    -WindowStyle Normal
Write-Host "langgraph dev launching..."

# restart media server in its own window
Start-Process -FilePath "cmd.exe" -ArgumentList "/c","cd /d $root && python -m src.agent.multimedia.static_server" `
    -WindowStyle Normal
Write-Host "media server launching..."

Start-Sleep -Seconds 18
Write-Host "=== ports after start ==="
netstat -ano | Select-String ':2024|:8900' | Select-Object -First 10
if (-not (netstat -ano | Select-String ':2024')) { Write-Host "2024 NOT UP YET" }
if (-not (netstat -ano | Select-String ':8900')) { Write-Host "8900 NOT UP YET" }
