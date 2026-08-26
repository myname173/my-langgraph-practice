$project = 'c:\Users\13682\Desktop\my-langgraph-practice-main'
$b = Join-Path $project 'output\keyframes'
if (Test-Path $b) {
    Get-ChildItem $b -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 3 | ForEach-Object {
        $c = (Get-ChildItem $_.FullName -ErrorAction SilentlyContinue).Count
        Write-Host "KF $($_.Name) count=$c"
    }
} else {
    Write-Host "no keyframes dir"
}
Write-Host "--- lg img2img lines ---"
Get-Content -Encoding utf8 (Join-Path $project 'logs_langgraph.txt') -Tail 5
