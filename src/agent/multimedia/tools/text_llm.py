# src/agent/multimedia/tools/text_llm.py
import os
import time
from openai import OpenAI
from openai import APITimeoutError
from dotenv import load_dotenv

load_dotenv()

import sys as _sys


def sprint(*args, **kwargs):
    """安全打印：在 GBK 等非 UTF-8 控制台下，emoji/中文不会触发 UnicodeEncodeError 崩溃。"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        msg = " ".join(str(a) for a in args)
        _sys.stdout.write(msg.encode("utf-8", "replace").decode("utf-8", "replace") + "\n")


# ── 超时与重试配置 ──
# 关键：给 OpenAI SDK 传 timeout 作为【总请求时长】硬上限。
# 注意不能同时使用 http_client 自定义 timeout——SDK 会优先 http_client 的
# read 超时（read 是“两次读取间隔”语义，不限制总时长），导致 DashScope 慢速
# 流式响应永远不触发超时而永久挂起。只用 SDK 的 timeout 参数即可正确限制总时长。
_LLM_TIMEOUT = 300.0
_MAX_LLM_RETRIES = 2
_LLM_BACKOFF = [5, 5]         # 退避（秒），缩短以快速 fail-fast

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", os.getenv("DASHSCOPE_API_KEY")),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    timeout=_LLM_TIMEOUT,
)

# ── SiliconFlow 文本兜底 ──
# 背景：DashScope（百炼）免费额度耗尽后，qwen 系模型全部 403（Free quota exhausted），
# 导致 showrunner/导演/风格抽取等文本环节永久卡死。SiliconFlow 提供 OpenAI 兼容端点
# （实测 0.5~12s 响应，无 DashScope 额度限制），作为 DashScope 全挂后的兜底渠道。
# 模型实测可用：deepseek-ai/DeepSeek-V3（语义最准）、Qwen/Qwen2.5-72B-Instruct、
# Qwen/Qwen2.5-7B-Instruct、THUDM/GLM-4-9B-0414。env 可用 SILICONFLOW_TEXT_MODEL 覆盖首选。
_sf_api_key = os.getenv("SILICONFLOW_API_KEY", "")
_sf_base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
_sf_client = None
if _sf_api_key:
    _sf_client = OpenAI(
        api_key=_sf_api_key,
        base_url=_sf_base_url,
        timeout=_LLM_TIMEOUT,
    )
_SILICONFLOW_FALLBACK_MODELS = list(
    dict.fromkeys([
        os.getenv("SILICONFLOW_TEXT_MODEL"),
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "THUDM/GLM-4-9B-0414",
    ])
)

def _is_retryable_llm_error(exc: Exception) -> bool:
    """判断是否为可重试的 LLM 调用瞬断错误。

    注意：DashScope 免费层在长链路连续请求后可能触发 400 Access denied / 限流风控，
    这类错误是瞬时的（不是鉴权永久失效），应退避重试而非永久禁用模型，否则多镜头
    长 pipeline 会在中后段因全部 fallback 模型被禁用而 FATAL。因此 400 / access denied
    / limit 也纳入可重试集合。
    """
    err_str = str(exc).lower()
    # 超时、限流、服务端错误、免费额度瞬时限额都是可重试/可切换的
    retryable_keywords = [
        "timed out", "timeout", "rate limit", "rate_limit",
        "429", "500", "502", "503", "504",
        # 免费层长链路风控：400 Access denied / bad request 限流也按可重试处理
        "400", "access denied", "bad request", "limit", "throttl",
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
    "qwen3.7-max-2026-05-20",                   # 免费额度内快照版（100万，2026/08/20 到期）
    "qwen3.8-max",                              # 最新版（免费额度内）
    "qwen3.7-max",                              # 裸名兜底（免费额度内）
    "qwen3.7-plus",                             # 裸名兜底（免费额度内；注意不能用带快照日期的 2026-05-26，会 403）
]
# 去空（覆盖变量未设置时）并保持顺序去重
_LLM_FALLBACK_MODELS = list(dict.fromkeys([m for m in _LLM_FALLBACK_MODELS if m]))

# 进程内「失效模型」缓存：一旦某模型被确认不可达（如免费额度 403 / 鉴权失败），
# 后续所有 call_llm 直接跳过它，不再浪费 3 次重试（约 35s）去打注定失败的请求。
# 关键：DashScope 免费层在长链路连续请求后常触发 400 Access denied / 限流风控，
# 这类失效是瞬时的——过一段时间（_DISABLED_TTL 秒）后会自动恢复，因此禁用带 TTL，
# 到期自动解禁，避免多镜头长 pipeline 在中后段因全部 fallback 被永久禁用而 FATAL。
_DISABLED_MODELS = {}  # model -> 解禁时间戳（time.monotonic 绝对秒）
_DISABLED_TTL = 120.0  # 失效模型 120s 后自动恢复（免费层限流风控多为短时）


def _is_model_disabled(model: str) -> bool:
    """模型当前是否被禁用；若已过期则自动解禁并返回 False。"""
    ts = _DISABLED_MODELS.get(model)
    if ts is None:
        return False
    if time.monotonic() > ts:
        _DISABLED_MODELS.pop(model, None)
        return False
    return True


def _disable_model(model: str) -> None:
    _DISABLED_MODELS[model] = time.monotonic() + _DISABLED_TTL

def call_llm(prompt_text: str, role_name: str = "LLM", max_tokens: int | None = None) -> str:
    """通用的 LLM 调用接口（带超时、多模型 fallback 与重试）"""
    # 对总导演增加温度，其它角色保持较低温度
    temperature = 0.9 if "总导演" in role_name else 0.7
    # 长输出（总导演分镜 / 安全审核 / 导演精修）若不限制会触发 DashScope 兼容端点超时挂起，
    # 按角色给一个合理上限，避免无限生成长内容导致 APITimeoutError。
    if max_tokens is None:
        if "总导演" in role_name or "安全审核" in role_name or "导演" in role_name:
            max_tokens = 1536
        else:
            max_tokens = 768

    last_err = None
    tried_any = False
    # 模型 fallback 链：先试 DashScope（首选），全挂后切 SiliconFlow 兜底。
    # 每项为 (provider 标签, client, model)；用 provider 区分，避免跨厂商模型名串用。
    _llm_chain = [(m, client, m) for m in _LLM_FALLBACK_MODELS]
    if _sf_client is not None:
        _llm_chain += [("siliconflow", _sf_client, m) for m in _SILICONFLOW_FALLBACK_MODELS if m]

    # 外层：一轮 fallback 遍历；若全部临时禁用（网络抖动典型场景）则等待解禁后重试整轮
    _round = 0
    while _round < 6:
        _round += 1
        tried_any = False
        # 遍历 fallback 模型；内层：每个模型做指数退避重试
        for _provider, _cli, model in _llm_chain:
            _key = f"{_provider}:{model}"
            if _is_model_disabled(_key):
                continue  # 已知失效模型（TTL 内），直接跳过
            tried_any = True
            disabled = False
            for attempt in range(_MAX_LLM_RETRIES + 1):
                try:
                    # 使用流式 + 总时长累计超时：DashScope 兼容端点对长生成会“慢速持续吐数据”，
                    # 导致 httpx 的 read timeout（间隔语义）永远不触发、请求永久挂起。
                    # 流式逐 chunk 读取，用 time.monotonic() 累计总时长，超时就主动断开并抛超时。
                    start = time.monotonic()
                    stream = _cli.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=temperature,
                        top_p=0.95,
                        presence_penalty=0.2,
                        max_tokens=max_tokens,
                        stream=True,
                    )
                    pieces = []
                    for chunk in stream:
                        if time.monotonic() - start > _LLM_TIMEOUT:
                            try:
                                stream.close()
                            except Exception:
                                pass
                            raise APITimeoutError(
                                f"总时长超过 {_LLM_TIMEOUT:.0f}s 未生成完成（模型 {model}）"
                            )
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            pieces.append(delta.content)
                    if _provider == "siliconflow":
                        sprint(f"    [OK] [{role_name}] 已通过 SiliconFlow 兜底模型 {model} 完成输出（DashScope 额度耗尽或不可用）")
                    return "".join(pieces).strip()

                except Exception as e:
                    last_err = e
                    if _is_retryable_llm_error(e) and attempt < _MAX_LLM_RETRIES:
                        wait = _LLM_BACKOFF[min(attempt, len(_LLM_BACKOFF) - 1)]
                        sprint(f"    [WARN] [{role_name}] 模型 {model} 异常 ({type(e).__name__})，{wait}s 后重试 ({attempt+1}/{_MAX_LLM_RETRIES})...")
                        time.sleep(wait)
                        continue
                    # 不可重试错误（如 403 免费额度/鉴权失败）→ 标记该模型进程内临时失效（TTL 后自动恢复），直接切下一个
                    sprint(f"    [WARN] [{role_name}] 模型 {model} 失败并标记为不可用（{_DISABLED_TTL:.0f}s 后恢复），切换下一个 fallback 模型")
                    _disable_model(_key)
                    disabled = True
                    break

            if disabled:
                continue  # 该模型已失效，继续尝试下一个

        if not tried_any:
            # 所有 fallback 模型当前都处于 TTL 禁用态（典型场景：网络抖动期间每个模型都瞬时失败被禁用）。
            # 标准 retry-backoff 自愈：等待最近一个模型解禁后再重试整轮，而不是立即 FATAL。
            _now = time.monotonic()
            _earliest = min(
                (ts for ts in _DISABLED_MODELS.values() if ts > _now),
                default=None,
            )
            if _earliest is not None:
                _wait = min(_earliest - _now, _DISABLED_TTL) + 1.0
                sprint(f"    [WARN] [{role_name}] 所有 fallback 模型暂不可用，等待 {_wait:.0f}s 让模型恢复后重试...")
                time.sleep(_wait)
                # 解禁已过期项后重跑整轮
                for _m in list(_DISABLED_MODELS.keys()):
                    _is_model_disabled(_m)
                continue
            raise Exception(f"{role_name} 思考失败：所有 fallback 模型均已被标记为不可用（兜底模型={_LLM_FALLBACK_MODELS}）")
        raise Exception(f"{role_name} 思考失败（所有 fallback 模型均失败）: {str(last_err)}")
