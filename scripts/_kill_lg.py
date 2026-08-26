import subprocess, time, json
out = subprocess.run(["powershell", "-NoProfile", "-Command",
    "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'langgraph|_start_lg_utf8' } | Select-Object ProcessId | ConvertTo-Json -Compress"],
    capture_output=True, text=True, timeout=30)
try:
    arr = json.loads(out.stdout)
    if not isinstance(arr, list):
        arr = [arr]
    for p in arr:
        pid = str(p.get("ProcessId"))
        r = subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True, text=True)
        print(f"kill {pid}: {r.stdout.strip()[:60]}")
except Exception as e:
    print("parse err:", e)
time.sleep(3)
out2 = subprocess.run(["powershell", "-NoProfile", "-Command",
    "(Get-NetTCPConnection -LocalPort 2024 -ErrorAction SilentlyContinue).Count"],
    capture_output=True, text=True, timeout=30)
print("port 2024 count:", out2.stdout.strip())
