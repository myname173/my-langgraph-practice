$tid = '6c988ec4-b5db-474b-8fe0-a49809f3e1ed'
$root = 'c:\Users\13682\Desktop\my-langgraph-practice-main'
$kfDir = Join-Path $root "output\keyframes\$tid"
Write-Host "=== keyframes for new thread ==="
if (Test-Path $kfDir) {
    Get-ChildItem $kfDir | Sort-Object LastWriteTime | ForEach-Object { Write-Host "$($_.Name)  $($_.Length)  $($_.LastWriteTime)" }
} else {
    Write-Host "no keyframes dir yet"
}
Write-Host "=== run_test tail ==="
Get-Content -Encoding utf8 (Join-Path $root 'logs_run_test.txt') -Tail 25
