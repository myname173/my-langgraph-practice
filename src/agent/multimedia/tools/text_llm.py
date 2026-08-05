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
    # 超时、限流、服务端错误都是可重试的
    retryable_keywords = [
        "timed out", "timeout", "rate limit", "rate_limit",
        "429", "500", "502", "503", "504",
        "connection", "internal server error",
        "overloaded", "server_error",
    ]
    return any(kw in err_str for kw in retryable_keywords)

def call_llm(prompt_text: str, role_name: str = "LLM") -> str:
    """通用的 LLM 调用接口（带超时和重试）"""
    # 对总导演增加温度，其它角色保持较低温度
    temperature = 0.9 if "总导演" in role_name else 0.7

    last_err = None
    for attempt in range(_MAX_LLM_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=os.getenv("DASHSCOPE_TEXT_MODEL", "tongyi-xiaomi-analysis-pro"),
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
                print(f"    ⚠️ [{role_name}] LLM 调用异常 ({type(e).__name__})，{wait}s 后重试 ({attempt+1}/{_MAX_LLM_RETRIES})...")
                time.sleep(wait)
                continue
            # 不可重试的错误 或 重试耗尽 → 抛出
            raise Exception(f"{role_name} 思考失败: {str(e)}")

    raise Exception(f"{role_name} 思考失败（重试 {_MAX_LLM_RETRIES} 次后仍失败）: {str(last_err)}")
