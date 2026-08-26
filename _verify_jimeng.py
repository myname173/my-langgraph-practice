import os, sys, glob
sys.path.insert(0, r'c:\Users\13682\Desktop\my-langgraph-practice-main\.venv\Lib\site-packages')

# 加载 .env
envf = r'c:\Users\13682\Desktop\my-langgraph-practice-main\.env'
for line in open(envf, encoding='utf-8'):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if not os.environ.get(k):
            os.environ[k] = v

from src.agent.multimedia.tools import jimeng

print("JIMENG_SESSION_ID set:", bool(os.environ.get("JIMENG_SESSION_ID")))

# 1) 纯文生图
print("=== TXT2IMG ===")
try:
    p = jimeng.generate_image("a majestic immortal cultivator standing on a cloud, cinematic", size="1920x1080", thread_id="verify")
    print("TXT2IMG OK ->", p)
except Exception as e:
    print("TXT2IMG FAIL ->", repr(e)[:200])

# 2) img2img，用本地一张素材图（/media 形式）
root = r'c:\Users\13682\Desktop\my-langgraph-practice-main\output'
refs = glob.glob(os.path.join(root, 'reference_sheets', 'library', '**', '*.png'), recursive=True)
print("refs found:", len(refs))
if refs:
    rp = refs[0]
    rel = os.path.relpath(rp, root).replace("\\", "/")
    url = "/media/" + rel
    print("using ref url:", url)
    try:
        p2 = jimeng.generate_image("a heroic cinematic shot of a sword cultivator", size="1920x1080", reference_image_urls=[url], ref_strength=0.5, thread_id="verify")
        print("IMG2IMG OK ->", p2)
    except Exception as e:
        print("IMG2IMG FAIL ->", repr(e)[:200])
