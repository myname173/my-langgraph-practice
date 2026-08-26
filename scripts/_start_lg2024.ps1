$Root = "C:\Users\13682\Desktop\my-langgraph-practice-main"
$log = Join-Path $Root "workspace\_lg2024_new.log"
$cmd = "cd /d $Root && set `"PYTHONUTF8=1`" && python -m langgraph_cli dev --port 2024 --no-reload >> `"$log`" 2>&1"
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "cmd.exe"
$psi.Arguments = "/c $cmd"
$psi.WorkingDirectory = $Root
$psi.UseShellExecute = $false
$p = [System.Diagnostics.Process]::Start($psi)
$p.Id | Out-File (Join-Path $Root "workspace\.lg_pid") -Encoding ascii
Write-Output ("PID_" + $p.Id)
