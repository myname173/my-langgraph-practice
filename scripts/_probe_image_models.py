import os, sys, json, time
import requests
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY"))

# 候选图像模型：覆盖 image_gen.py 链 + .env 指向的 + 可能的其他免费模型
CANDIDATES = [
    ("qwen-image-3.0-pro", "multimodal-generation/generation"),
    ("qwen-image-3.0", "multimodal-generation/generation"),
    ("qwen-image-2.0-pro-2026-06-22", "multimodal-generation/generation"),
    ("qwen-image-2.0-pro", "multimodal-generation/generation"),
    ("wan2.7-image", "image-generation/generation"),
    ("wan2.6-image", "image-generation/generation"),
    ("wanx-v1", "text2image/image-synthesis"),
    ("wanx2.1-t2i-turbo", "text2image/image-synthesis"),
    ("qwen-image-plus", "multimodal-generation/generation"),
]

SYNC = {"multimodal-generation/generation"}
prompt = "a cyberpunk city, neon rain, cinematic, simple test image"
size = "1024*1024"

def is_quota(err):
    low = err.lower()
    pats = ["quota","free tier","freitier","free_tier","balance","insufficient",
            "arrearage","overdue","余额","欠费","余额不足","throttling","ratelimit",
            "rate limit","model not","not found","not exist","invalidmodel",
            "datainspectionfailed","invalidparameter","url error","no quota",
            "quota exhausted","quota exceeded"]
    return any(p in low for p in pats)

results = {}
for model, ep in CANDIDATES:
    is_sync = ep in SYNC
    url = f"https://dashscope.aliyuncs.com/api/v1/services/aigc/{ep}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if not is_sync:
        headers["X-DashScope-Async"] = "enable"
    payload = {"model": model, "input": {"messages":[{"role":"user","content":[{"text":prompt}]}]},
               "parameters": {"size": size, "n": 1, "watermark": False}}
    if not is_sync:
        payload["input"] = {"prompt": prompt}
        payload.pop("parameters")
        payload["parameters"] = {"size": size, "n": 1}
    t0 = time.time()
    status = "???"
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        dt = round(time.time()-t0, 1)
        if r.status_code == 200:
            d = r.json()
            code = d.get("code", "")
            if code:
                msg = d.get("message","") or str(d)
                status = f"CODE:{code} {msg[:120]}"
                ok = False
            else:
                out = d.get("output", {})
                task_id = out.get("task_id")
                if is_sync:
                    content = out.get("choices",[{}])[0].get("message",{}).get("content",[]) if "choices" in out else []
                    img = None
                    for it in (content if isinstance(content,list) else []):
                        if isinstance(it,dict) and (it.get("image") or it.get("url")):
                            img = it.get("image") or it.get("url")
                    status = "SYNC_OK" if img else f"SYNC_NOIMG {str(d)[:80]}"
                    ok = bool(img)
                elif task_id:
                    status = f"ASYNC_SUBMITTED task={task_id}"
                    ok = True
                else:
                    status = f"NO_TASKID {str(d)[:100]}"
                    ok = False
        else:
            status = f"HTTP{r.status_code} {r.text[:160]}"
            ok = False
    except Exception as e:
        dt = round(time.time()-t0,1)
        status = f"EXC:{type(e).__name__} {str(e)[:100]}"
        ok = False
    results[f"{model} ({ep})"] = {"ok": ok, "status": status, "sec": dt}
    print(f"[{'OK ' if ok else 'FAIL'}] {model} ({ep}) -> {status} ({dt}s)", flush=True)

print("\n=== SUMMARY ===", flush=True)
for k,v in results.items():
    print(f"{'OK ' if v['ok'] else 'FAIL'}  {k}: {v['status']}", flush=True)
