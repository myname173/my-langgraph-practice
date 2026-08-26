import os, base64, requests
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\13682\Desktop\my-langgraph-practice-main")
base = os.getenv("JIMENG_BASE_URL", "http://localhost:8090/v1").strip().rstrip("/")
sid = os.getenv("JIMENG_SESSION_ID", "").strip().split(",")[0]
hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {sid}"}

# 取一张真实素材图作为参考图
ref = PROJECT_ROOT / "output" / "reference_sheets" / "library" / "characters" / "char_elder_master_real.jpg"
b = ref.read_bytes()
b64 = "data:image/jpeg;base64," + base64.b64encode(b).decode()

def call(label, payload):
    try:
        r = requests.post(f"{base}/images/generations", headers=hdr, json=payload, timeout=120)
        print(f"[{label}] HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[{label}] EXC {repr(e)[:150]}")

# 1) 纯文生图（不含参考图）
call("txt2img", {"model":"jimeng-5.0","prompt":"a xianxia hero standing on mountain","ratio":"16:9","resolution":"2k","n":1})
# 2) img2img sample_strength=1.0（当前代码默认值）
call("img2img_s1.0", {"model":"jimeng-5.0","prompt":"a xianxia hero","ratio":"16:9","resolution":"2k","n":1,"images":[b64],"sample_strength":1.0})
# 3) img2img sample_strength=0.7
call("img2img_s0.7", {"model":"jimeng-5.0","prompt":"a xianxia hero","ratio":"16:9","resolution":"2k","n":1,"images":[b64],"sample_strength":0.7})
# 4) img2img resolution=1k
call("img2img_1k", {"model":"jimeng-5.0","prompt":"a xianxia hero","ratio":"16:9","resolution":"1k","n":1,"images":[b64],"sample_strength":0.7})
