import os, requests
from pathlib import Path

sid = "d4c003afc5a339e01983bbef4f3a64f1"  # 用户提供的 sessionid（与 .env 一致）
base = "http://localhost:8090/v1"
hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {sid}"}

# 不带参考图的纯文生图，最小 payload，确认是否还是 check login error
payload = {"model": "jimeng-5.0", "prompt": "a blue sky", "ratio": "16:9", "resolution": "2k", "n": 1}
try:
    r = requests.post(f"{base}/images/generations", headers=hdr, json=payload, timeout=120)
    print("HTTP", r.status_code, "BODY:", r.text[:400])
except Exception as e:
    print("EXC", repr(e)[:200])
