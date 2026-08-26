import os, sys, time, json, urllib.request
sys.stdout.reconfigure(encoding="utf-8")

# 1) 列出所有仙侠风格的关键帧（8/18 之后，排除赛博朋克那4张）
LEG = "c:/Users/13682/Desktop/my-langgraph-practice-main/output/keyframes/_legacy"
xianxia_files = []
for f in sorted(os.listdir(LEG)):
    if not f.startswith("jimeng_kf_"): continue
    p = os.path.join(LEG, f)
    mt = os.path.getmtime(p)
    # 赛博朋克那4张是 00:08~00:38，仙侠从 00:38 开始
    if mt < time.mktime(time.strptime("2026-08-18 00:35","%Y-%m-%d %H:%M")):
        continue
    xianxia_files.append((mt, f))

print(f"=== 仙侠关键帧共 {len(xianxia_files)} 张 ===")
for mt, f in xianxia_files:
    sz = os.path.getsize(os.path.join(LEG, f))
    print(f"  {time.strftime('%m-%d %H:%M', time.localtime(mt))}  {sz:>8}B  {f}")

# 2) 当前线程状态
tid = "0ced7442-ffd8-4caa-846a-5925aebbed9f"
try:
    req = urllib.request.Request(
        f"http://localhost:2024/threads/{tid}/state",
        data=json.dumps({}).encode(),
        headers={"Content-Type":"application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        st = json.load(r)
    v = st.get("values", {})
    print(f"\n=== 线程 {tid[:8]}... state ===")
    print("  scenes:", len(v.get("scenes", []) or []))
    print("  index:", v.get("index"))
    print("  task:", str(v.get("task", ""))[:80])
except Exception as e:
    print("\nstate err:", e)

# 3) run 详情
try:
    with urllib.request.urlopen(f"http://localhost:2024/threads/{tid}/runs", timeout=10) as r:
        runs = json.load(r)
    for run in runs:
        print(f"\n  run {run['run_id'][:8]}... status={run['status']} updated={run.get('updated_at')}")
except Exception as e:
    print("runs err:", e)
