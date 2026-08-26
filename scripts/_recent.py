import os, glob, time, sys
sys.stdout.reconfigure(encoding="utf-8")

ROOT = "c:/Users/13682/Desktop/my-langgraph-practice-main/output"
cut = time.time() - 4*86400  # 最近4天

def walk(label, d):
    print(f"\n=== {label}: {d} ===")
    if not os.path.isdir(d):
        print("  (目录不存在)")
        return
    items = []
    for f in glob.glob(os.path.join(d, "**", "*"), recursive=True):
        if os.path.isfile(f):
            items.append((f, os.path.getmtime(f)))
    recent = sorted([(f,mt) for f,mt in items if mt>cut], key=lambda x:-x[1])
    print(f"  最近4天文件: {len(recent)} 个 (该目录总文件 {len(items)} 个)")
    for f,mt in recent[:50]:
        rel = os.path.relpath(f, ROOT)
        sz = os.path.getsize(f)
        print(f"   {time.strftime('%m-%d %H:%M', time.localtime(mt))}  {sz:>8}  {rel}")

walk("keyframes", os.path.join(ROOT, "keyframes"))
walk("clips", os.path.join(ROOT, "clips"))
walk("reference_sheets", os.path.join(ROOT, "reference_sheets"))
walk("review", os.path.join(ROOT, "review"))
