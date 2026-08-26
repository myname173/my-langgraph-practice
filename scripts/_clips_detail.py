import os, time

tid = "01a01604-4141-7383-90e9-b9c0d2cae1a6"
d = os.path.join("output", "clips", tid)
print("=== clips dir ===", os.path.abspath(d))
if os.path.isdir(d):
    items = []
    for root, _, files in os.walk(d):
        for fn in files:
            p = os.path.join(root, fn)
            items.append((os.path.getmtime(p), fn, os.path.getsize(p)))
    items.sort()
    now = time.time()
    for m, fn, sz in items:
        age = int(now - m)
        print(f"  {fn}  {sz//1024}KB  mtime={time.strftime('%H:%M:%S', time.localtime(m))}  ({age}s ago)")
else:
    print("no dir")

# also check the FINAL output dir
final_dir = os.path.join("output", "clips", tid, "final")
if os.path.isdir(final_dir):
    print("=== final ===")
    for fn in os.listdir(final_dir):
        p = os.path.join(final_dir, fn)
        print(f"  {fn} {os.path.getsize(p)//1024}KB")
