import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
# 把 _legacy 里 8/18 之后生成的关键帧，按时间戳分组，列出可给你看的
LEG = "c:/Users/13682/Desktop/my-langgraph-practice-main/output/keyframes/_legacy"
files = []
for f in os.listdir(LEG):
    p = os.path.join(LEG, f)
    if os.path.isfile(p) and f.startswith("jimeng_kf_"):
        files.append((f, os.path.getmtime(p)))
files.sort(key=lambda x:x[1])
print("8/18 之后的 jimeng 关键帧（按时间）:")
for f,mt in files:
    if mt > time.mktime(time.strptime("2026-08-18","%Y-%m-%d")):
        print(f"  {time.strftime('%m-%d %H:%M', time.localtime(mt))}  {f}")
