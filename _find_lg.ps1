$lang = Get-Command langgraph -ErrorAction SilentlyContinue
if ($lang) { Write-Host "PATH langgraph: $($lang.Source)" }
$roots = @("$env:USERPROFILE\miniconda3\envs", "$env:USERPROFILE\anaconda3\envs", "c:\Users\13682\Desktop\my-langgraph-practice-main")
foreach ($root in $roots) {
    if (-not (Test-Path $root)) { continue }
    Get-ChildItem -Path $root -Filter langgraph.exe -Recurse -ErrorAction SilentlyContinue -Depth 5 | ForEach-Object { Write-Host "FOUND: $($_.FullName)" }
}
Write-Host "--- pip show ---"
pip show langgraph-cli 2>$null | Select-String -Pattern '^(Name|Location|Version)' | ForEach-Object { Write-Host $_ }
