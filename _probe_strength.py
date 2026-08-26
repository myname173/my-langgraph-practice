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
ref = "/media/" + os.path.relpath(refs[0], root).replace("\\", "/")
print("ref:", ref)

for ss in [0.1, 0.2, 0.3, 0.5]:
    try:
        p = jimeng.generate_image("a heroic cinematic shot", size="1920x1080",
                                  reference_image_urls=[ref], ref_strength=ss, thread_id="probe")
        print(f"SS={ss} OK ->", p)
    except Exception as e:
        print(f"SS={ss} FAIL ->", repr(e)[:120])
