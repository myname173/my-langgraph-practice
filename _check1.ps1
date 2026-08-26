$project = 'c:\Users\13682\Desktop\my-langgraph-practice-main'
$job = Start-Job -ScriptBlock {
    Start-Sleep -Seconds 45
    Write-Host "--- run_real tail ---"
    Get-Content -Encoding utf8 "c:\Users\13682\Desktop\my-langgraph-practice-main\logs_run_real.txt" -Tail 8
    Write-Host "--- lg img2img/keyframe lines ---"
    Get-Content -Encoding utf8 "c:\Users\13682\Desktop\my-langgraph-practice-main\logs_langgraph.txt" -Tail 6
    $b = "c:\Users\13682\Desktop\my-langgraph-practice-main\output\keyframes"
    Get-ChildItem $b -Directory | ForEach-Object {
        $c = (Get-ChildItem $_.FullName -ErrorAction SilentlyContinue).Count
        Write-Host "KF $($_.Name) count=$c"
    }
}
$wait = Wait-Job -Job $job -Timeout 55
if ($wait.State -eq 'Completed') { Receive-Job -Job $job } else { "TIMEOUT" }
