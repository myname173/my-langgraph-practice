$Root = "C:\Users\13682\Desktop\my-langgraph-practice-main"
# 杀掉所有 static_server 进程（清除历史多实例残留），仅保留一个干净实例
$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*static_server*" }
foreach ($p in $procs) {
    Write-Output ("Killing media server PID " + $p.ProcessId)
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 3
$log = Join-Path $Root "workspace\_media8900.log"
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "python"
$psi.Arguments = "-m src.agent.multimedia.static_server"
$psi.WorkingDirectory = $Root
$psi.UseShellExecute = $false
$psi.Environment["MEDIA_SERVER_PORT"] = "8900"
$p = [System.Diagnostics.Process]::Start($psi)
Write-Output ("Restarted media server PID " + $p.Id)
