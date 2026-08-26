import os
p = r"c:\Users\13682\Desktop\my-langgraph-practice-main\.env"
for line in open(p, encoding="utf-8"):
    line = line.strip()
    if line.startswith(("JIMENG_SESSION_ID", "JIMENG_BASE_URL", "MULTIMODIA_BACKEND")):
        # 打码只显示前8位
        k, _, v = line.partition("=")
        print(f"{k}={v[:8]}...(len={len(v)})")
