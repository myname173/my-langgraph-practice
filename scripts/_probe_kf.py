import os, sys, time, struct
sys.stdout.reconfigure(encoding="utf-8")

LEG = "c:/Users/13682/Desktop/my-langgraph-practice-main/output/keyframes/_legacy"

# 1) 这张 13:27 图的基本信息
f = os.path.join(LEG, "jimeng_kf_1787117267385.png")
print("=== 13:27 图 ===")
print("exists:", os.path.isfile(f), "size:", os.path.getsize(f))
print("mtime:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(f))))

# 2) 看 clips 里有没有对应 8/19 13:27 之后的视频（说明哪个 run 产出了这张图）
print("\n=== 8/19 13:00 之后的 clips ===")
ROOT = "c:/Users/13682/Desktop/my-langgraph-practice-main/output/clips"
cut = time.mktime(time.strptime("2026-08-19 13:00","%Y-%m-%d %H:%M"))
seen = []
for d in os.listdir(ROOT):
    dp = os.path.join(ROOT, d)
    if os.path.isdir(dp):
        for fn in os.listdir(dp):
            fp = os.path.join(dp, fn)
            if os.path.isfile(fp) and os.path.getmtime(fp) > cut:
                seen.append((os.path.getmtime(fp), d, fn))
seen.sort()
for mt,d,fn in seen:
    print(f"  {time.strftime('%H:%M', time.localtime(mt))}  {d}/{fn}")

# 3) 确认这张图是 jimeng 生成的（文件名含 jimeng_kf_ 且时间戳 1787117267385 -> 对应 run）
ts = 1787117267385/1000
print("\n图内嵌时间戳 1787117267385 ->", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)), "(UTC+8 约)")
