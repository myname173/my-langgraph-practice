$project = 'c:\Users\13682\Desktop\my-langgraph-practice-main'
$kfBase = Join-Path $project 'output\keyframes'
if (Test-Path $kfBase) {
    Get-ChildItem $kfBase -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 4 | ForEach-Object {
        $c = (Get-ChildItem $_.FullName -ErrorAction SilentlyContinue).Count
        Write-Host "KF_DIR $($_.Name) count=$c"
    }
}
# 也看 _generated 里的文件数
$gen = Join-Path $kfBase '_generated'
if (Test-Path $gen) { Write-Host "_generated files: $((Get-ChildItem $gen -ErrorAction SilentlyContinue).Count)" }
Write-Host "--- lg img2img/kf lines (last 6) ---"
Get-Content -Encoding utf8 (Join-Path $project 'logs_langgraph.txt') -Tail 6
