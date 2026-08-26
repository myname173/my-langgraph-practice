import os, sys, glob
sys.path.insert(0, r'c:\Users\13682\Desktop\my-langgraph-practice-main\.venv\Lib\site-packages')
envf = r'c:\Users\13682\Desktop\my-langgraph-practice-main\.env'
for line in open(envf, encoding='utf-8'):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if not os.environ.get(k):
            os.environ[k] = v
from src.agent.multimedia.tools import jimeng

kf_dir = r'c:\Users\13682\Desktop\my-langgraph-practice-main\output\keyframes'
pngs = glob.glob(os.path.join(kf_dir, '**', 'jimeng_kf_*.png'), recursive=True)
pngs.sort(key=os.path.getmtime, reverse=True)
print("jimeng_kf pngs:", len(pngs))
if pngs:
    p = pngs[0]
    print("using:", p)
    print("size bytes:", os.path.getsize(p))
    try:
        r = jimeng.generate_image("a heroic cinematic shot, different pose", size="1920x1080",
                                  reference_image_urls=[p], ref_strength=0.1, thread_id="probepath2", timeout=60)
        print("ABS_PATH OK ->", r)
    except Exception as e:
        print("ABS_PATH FAIL ->", repr(e)[:150])
