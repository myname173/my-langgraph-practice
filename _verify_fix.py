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

root = r'c:\Users\13682\Desktop\my-langgraph-practice-main\output'
refs = glob.glob(os.path.join(root, 'reference_sheets', 'library', '**', '*.png'), recursive=True)
print("refs:", len(refs))
if refs:
    rel = os.path.relpath(refs[0], root).replace("\\", "/")
    # 模拟 graph 传入的 localhost:8900 形式（媒体服务器已宕）
    url = f"http://localhost:8900/media/{rel}"
    print("using URL:", url)
    try:
        p = jimeng.generate_image("a heroic cinematic shot", size="1920x1080",
                                  reference_image_urls=[url], ref_strength=0.5, thread_id="verifyfix")
        print("IMG2IMG (localhost-url, 8900 down) OK ->", p)
    except Exception as e:
        print("IMG2IMG FAIL ->", repr(e)[:200])
