$project = 'c:\Users\13682\Desktop\my-langgraph-practice-main'
$py = 'C:\Users\13682\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe'
$site = Join-Path $project '.venv\Lib\site-packages'
$env:PYTHONPATH = $site
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location $project
$job = Start-Job -ScriptBlock {
    param($py, $site, $project)
    $env:PYTHONPATH = $site
    $env:PYTHONDONTWRITEBYTECODE = '1'
    Set-Location $project
    & $py _probe_img2img.py 2>&1
} -ArgumentList $py, $site, $project
$wait = Wait-Job -Job $job -Timeout 55
if ($wait.State -eq 'Completed') { Receive-Job -Job $job } else { "TIMEOUT 55s" }
