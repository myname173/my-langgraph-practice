$envs = @(
    "$env:USERPROFILE\miniconda3\envs",
    "$env:USERPROFILE\anaconda3\envs",
    "$env:USERPROFILE\.conda\envs"
)
foreach ($e in $envs) {
    if (Test-Path $e) {
        Get-ChildItem -Path $e -Depth 0 | Where-Object { $_.PSIsContainer } | ForEach-Object { Write-Host "ENV: $($_.FullName)" }
    }
}

$dirs = @(
    "c:\Users\13682\Desktop\my-langgraph-practice-main\.venv\Lib\site-packages",
    "c:\Users\13682\Desktop\my-langgraph-practice-main\.venv_blocked\Lib\site-packages"
)
foreach ($d in $dirs) {
    if (Test-Path $d) {
        $f = Get-ChildItem $d -Filter 'langgraph_cli*' -ErrorAction SilentlyContinue
        if ($f) { Write-Host "FOUND langgraph_cli in $d" } else { Write-Host "NO langgraph_cli in $d" }
    } else {
        Write-Host "MISSING $d"
    }
}

Write-Host "--- venv_blocked python ---"
& "c:\Users\13682\Desktop\my-langgraph-practice-main\.venv_blocked\Scripts\python.exe" --version 2>&1
