import os, urllib.request, json

KEY = os.getenv("DASHSCOPE_API_KEY") or "sk-ws-H.ERPMHXH.VFyD.MEUCIQCN0_qnSR7yoXN74YHXZr8YEJgN4WIKaziWP-Few5iCYwIgTzglRXubm_h1xCtaBOG8Xf1ktz5W11l2FGeOjoYteUg"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

candidates = [
    "qwen3.7-max-preview", "qwen3.7-max", "qwen3.7-max-2026-06-08",
    "qwen3.8-27b", "qwen3.5-flash", "qwen-plus", "qwen-max",
    "qwen3.7-flash-2026-07-15", "qwen3.7-plus",
]

for m in candidates:
    body = json.dumps({
        "model": m,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
    }).encode()
    req = urllib.request.Request(URL, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {KEY}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            code = r.status
            data = json.loads(r.read().decode())
            if code == 200 and "choices" in data:
                print(f"OK    {m}")
            else:
                print(f"CODE {code} {m} {str(data)[:120]}")
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} {m} {e.read().decode()[:160]}")
    except Exception as e:
        print(f"ERR   {m} {repr(e)[:120]}")
