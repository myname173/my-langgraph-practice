import os, base64, requests
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\13682\Desktop\my-langgraph-practice-main")
base = "http://localhost:8090/v1"
sid = os.getenv("JIMENG_SESSION_ID", "d4c003afc5a339e01983bbef4f3a64f1").strip().split(",")[0]
hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {sid}"}

ref = PROJECT_ROOT / "output" / "reference_sheets" / "library" / "characters" / "char_elder_master_real.jpg"
b = ref.read_bytes()
b64 = "data:image/jpeg;base64," + base64.b64encode(b).decode()

def call(label, payload):
    try:
        r = requests.post(f"{base}/images/generations", headers=hdr, json=payload, timeout=20)
        txt = r.text
        ok = ('"url"' in txt) or ('"data"' in txt and 'http' in txt)
        print(f"[{label}] HTTP {r.status_code} OK={ok}: {txt[:160]}")
    except Exception as e:
        print(f"[{label}] EXC {repr(e)[:100]}")

P = "a xianxia hero standing on a mountain peak, cinematic"
# A: 当前代码默认（resolution=2k, sample_strength=1.0）
call("A_cur", {"model":"jimeng-5.0","prompt":P,"ratio":"16:9","resolution":"2k","n":1,"images":[b64],"sample_strength":1.0})
# B: 去掉 sample_strength + resolution=1k
call("B_fix", {"model":"jimeng-5.0","prompt":P,"ratio":"16:9","resolution":"1k","n":1,"images":[b64]})
