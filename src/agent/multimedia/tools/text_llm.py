# src/agent/multimedia/tools/text_llm.py
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
import httpx

load_dotenv()

# ── 超时与重试配置 ──
_LLM_TIMEOUT = 120.0          # 单次请求超时（秒），总导演提示词很长需要宽松超时
_MAX_LLM_RETRIES = 3
_LLM_BACKOFF = [5, 10, 20]    # 指数退避（秒）

http_client = httpx.Client(trust_env=False, timeout=_LLM_TIMEOUT)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", os.getenv("DASHSCOPE_API_KEY")),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    http_client=http_client,
    timeout=_LLM_TIMEOUT,
)

def _is_retryable_llm_error(exc: Exception) -> bool:
    """判断是否为可重试的 LLM 调用瞬断错误"""
    err_str = str(exc).lower()
    # 超时、限流、服务端错误、免费额度瞬时限额都是可重试/可切换的
    retryable_keywords = [
        "timed out", "timeout", "rate limit", "rate_limit",
        "429", "500", "502", "503", "504",
        "connection", "internal server error",
        "overloaded", "server_error",
        "free quota", "free_tier", "quota", "allocation",
    ]
    return any(kw in err_str for kw in retryable_keywords)

# ── 多模型 fallback 链：百炼免费额度常出现单模型瞬时限额（FreeTierOnly），
#    按优先级尝试可用模型，避免整个 pipeline 因单一模型 403 而崩。 ──
# 模型均走 DashScope 兼容 OpenAI 端点（OPENAI_BASE_URL），免费额度可用清单见 .env。
# 排序原则：能力最强/带快照日期的稳定版优先，其次裸名兜底，最后跨厂商保底。
_LLM_FALLBACK_MODELS = [
    os.getenv("DASHSCOPE_TEXT_MODEL"),          # 允许 .env 覆盖首选
    "qwen3.7-max-2026-06-08",                   # 带快照日期稳定版（免费额度可用）
    "qwen3.7-max",                              # 裸名兜底（免费额度对裸名开放）
    "qwen3.7-max-preview",
    "qwen3.7-plus",
    "qwen3.7-flash",
    "qwen3.7-flash-2026-07-15",
    "deepseek-v4-flash-0731",
    "glm-5.2",
    "qwen3.5-ocr",                              # OCR 能力兜底，最后保底
]
# 去空（覆盖变量未设置时）并保持顺序去重
_LLM_FALLBACK_MODELS = list(dict.fromkeys([m for m in _LLM_FALLBACK_MODELS if m]))

# 进程内「失效模型」缓存：一旦某模型被确认不可达（如免费额度 403 / 鉴权失败），
# 后续所有 call_llm 直接跳过它，不再浪费 3 次重试（约 35s）去打注定失败的请求。
_DISABLED_MODELS = set()

def call_llm(prompt_text: str, role_name: str = "LLM") -> str:
    """通用的 LLM 调用接口（带超时、多模型 fallback 与重试）"""
    # 对总导演增加温度，其它角色保持较低温度
    temperature = 0.9 if "总导演" in role_name else 0.7

    last_err = None
    tried_any = False
    # 外层：遍历 fallback 模型；内层：每个模型做指数退避重试
    for model in _LLM_FALLBACK_MODELS:
        if model in _DISABLED_MODELS:
            continue  # 已知失效模型，直接跳过
        tried_any = True
        disabled = False
        for attempt in range(_MAX_LLM_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=temperature,
                    top_p=0.95,
                    presence_penalty=0.2,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                last_err = e
                if _is_retryable_llm_error(e) and attempt < _MAX_LLM_RETRIES:
                    wait = _LLM_BACKOFF[min(attempt, len(_LLM_BACKOFF) - 1)]
                    print(f"    ⚠️ [{role_name}] 模型 {model} 异常 ({type(e).__name__})，{wait}s 后重试 ({attempt+1}/{_MAX_LLM_RETRIES})...")
                    time.sleep(wait)
                    continue
                # 不可重试错误（如 403 免费额度/鉴权失败）→ 标记该模型进程内失效，直接切下一个
                print(f"    ⚠️ [{role_name}] 模型 {model} 失败并标记为不可用，切换下一个 fallback 模型")
                _DISABLED_MODELS.add(model)
                disabled = True
                break

        if disabled:
            continue  # 该模型已失效，继续尝试下一个

    if not tried_any:
        raise Exception(f"{role_name} 思考失败：所有 fallback 模型均已被标记为不可用（兜底模型={_LLM_FALLBACK_MODELS}）")
    raise Exception(f"{role_name} 思考失败（所有 fallback 模型均失败）: {str(last_err)}")
