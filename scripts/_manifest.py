import os, json, sys
sys.stdout.reconfigure(encoding="utf-8")
ROOT = "c:/Users/13682/Desktop/my-langgraph-practice-main/output/keyframes/_legacy"
m = os.path.join(ROOT, "manifest.json")
if os.path.isfile(m):
    d = json.load(open(m, encoding="utf-8"))
    print("manifest entries:", len(d) if isinstance(d, list) else list(d.keys()))
    # 打印前几个和最后几个，看 thread/task 关联
    items = d if isinstance(d, list) else d.get("items", [])
    for it in items[:3]:
        print("  ", json.dumps(it, ensure_ascii=False)[:200])
    print("  ...")
    for it in items[-5:]:
        print("  ", json.dumps(it, ensure_ascii=False)[:200])
else:
    print("no manifest")

# 列 keyframes 下非 _legacy 的子目录（按线程隔离的）
print("\n=== keyframes 子目录（按线程的）===")
for name in sorted(os.listdir("c:/Users/13682/Desktop/my-langgraph-practice-main/output/keyframes")):
    p = os.path.join("c:/Users/13682/Desktop/my-langgraph-practice-main/output/keyframes", name)
    if os.path.isdir(p):
        cnt = len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p,f))])
        print(f"  {name}: {cnt} files")
