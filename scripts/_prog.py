import os

tid = "01a01604-4141-7383-90e9-b9c0d2cae1a6"
log = f"scripts/_rerun_{tid[:8]}.log"
print("=== LOG TAIL ===")
if os.path.exists(log):
    with open(log, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    pats = ("node ->", "step=", "on_chain_start", "on_chain_end", "image_url",
            "jimeng", "agnes", "保底", "不入帧", "ERROR", "Traceback", "DONE", "FINAL")
    shown = [l.rstrip("\n") for l in lines if any(p in l for p in pats)]
    for l in shown[-30:]:
        print(l)
    print(f"[total log lines: {len(lines)}]")
else:
    print("no log")

print("\n=== CLIPS ===")
d = os.path.join("output", "clips", tid)
if os.path.isdir(d):
    for root, _, files in os.walk(d):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            print(f"{fn}\t{os.path.getsize(p)//1024}KB\t{os.path.getmtime(p):.0f}")
else:
    print("no clips dir yet")

# process alive?
import subprocess
try:
    out = subprocess.run(["tasklist", "/FI", "PID eq 20592"], capture_output=True, text=True).stdout
    print("\n=== PROC 20592 ===")
    print(out.strip().splitlines()[-1] if "python" in out else "NOT RUNNING")
except Exception as e:
    print("proc check err", e)
