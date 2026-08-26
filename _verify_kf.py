import asyncio, sys, os
from pathlib import Path
sys.path.insert(0, r"c:\Users\13682\Desktop\my-langgraph-practice-main")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
# 手动加载 .env（模拟 langgraph dev 初始化）
try:
    from dotenv import load_dotenv
    load_dotenv(r"c:\Users\13682\Desktop\my-langgraph-practice-main\.env")
except Exception as e:
    print("dotenv err", e)

print("SESSION_ID set:", bool(os.getenv("JIMENG_SESSION_ID")))

from src.agent.multimedia.tools.jimeng import generate_image

ref = "http://localhost:8900/media/reference_sheets/library/characters/char_elder_master_real.jpg"
prompt = "仙侠真人电影风格，白发剑修立于雪山之巅，长袍猎猎，远景云海翻涌，电影感前景虚化"

try:
    url = generate_image(prompt=prompt, size="1920x1080", reference_image_urls=[ref], ref_strength=0.9)
    print("KF_OK:", str(url)[:200])
except Exception as e:
    print("KF_FAIL:", repr(e)[:300])
