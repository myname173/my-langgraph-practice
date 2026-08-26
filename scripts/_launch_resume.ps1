$root = "c:\Users\13682\Desktop\my-langgraph-practice-main"
$payload = Join-Path $root "scripts\_resume_payload.json"
$out = Join-Path $root "scripts\_resume_result.json"
Start-Process -FilePath "curl.exe" -ArgumentList @(
    "-s","-m","60","-X","POST",
    "http://localhost:2024/threads/01a01604-4141-7383-90e9-b9c0d2cae1a6/runs",
    "-H","Content-Type: application/json",
    "-d",("@"+$payload),
    "-o",$out
)
Write-Host "POST_LAUNCHED_BACKGROUND"
