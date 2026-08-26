import subprocess, time

up = False
for i in range(12):  # up to 120s
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    if ":2024" in out:
        up = True
        print("2024 UP after", (i+1)*10, "s")
        break
    time.sleep(10)
if not up:
    print("2024 STILL DOWN after 120s")
