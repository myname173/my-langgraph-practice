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
pngs = glob.glob(os.path.join(kf_dir, '**', '*.png'), recursive=True)
print("found kf pngs:", len(pngs))
if pngs:
    p = pngs[0]
    print("using local abs path:", p)
    try:
        r = jimeng.generate_image("a heroic cinematic shot, different pose", size="1920x1080",
                                  reference_image_urls=[p], ref_strength=0.1, thread_id="probepath")
        print("ABS_PATH OK ->", r)
    except Exception as e:
        print("ABS_PATH FAIL ->", repr(e)[:150])

    # 同时测 /media/ 形式
    rel = os.path.relpath(p, r'c:\Users\13682\Desktop\my-langgraph-practice-main\output').replace("\\", "/")
    media_url = "/media/" + rel
    print("using /media/ url:", media_url)
    try:
        r2 = jimeng.generate_image("a heroic cinematic shot, different pose", size="1920x1080",
                                   reference_image_urls=[media_url], ref_strength=0.1, thread_id="probepath")
        print("MEDIA_URL OK ->", r2)
    except Exception as e:
        print("MEDIA_URL FAIL ->", repr(e)[:150])
