# src/agent/multimedia/tools/image_gen.py
import time
import os
import base64
import requests
from dotenv import load_dotenv
from .video_gen import _retry_request, _SUBMIT_TIMEOUT, _POLL_TIMEOUT
from ..config_loader import get_visual_backend, get_jimeng_config
from . import jimeng
from pathlib import Path as _Path

# 项目根目录：image_gen.py -> tools -> multimedia -> agent -> src -> <root>
PROJECT_ROOT = _Path(__file__).resolve().parents[4]

load_dotenv()

# ── 图片生成模型链 ──
# 端点调用方式说明：
#   image-generation/generation        → 异步（返回 task_id，需轮询）
#   text2image/image-synthesis         → 异步（返回 task_id，需轮询）
#   multimodal-generation/generation   → 同步（直接返回图片，不可带 async 头）
# qwen-image 系列经实测走同步 multimodal 端点；带 X-DashScope-Async 头会报
# "current user api does not support asynchronous calls"。
_SYNC_ENDPOINTS = {"multimodal-generation/generation"}

# 支持 img2img（参考图）的模型 — 按优先级排列（仅用百炼免费额度 qwen-image 系列）
_IMG2IMG_CHAIN = [
    # qwen-image 3.0：Arena 榜国内第一，4K，超长文本理解与精细渲染（实测可用）
    ("qwen-image-3.0-pro", "multimodal-generation/generation"),  # 旗舰版，10张免费
    ("qwen-image-3.0",     "multimodal-generation/generation"),  # 标准版，10张免费
    # qwen-image 2.0-pro：额度最充足（100张），作为主力承载批量镜头
    ("qwen-image-2.0-pro-2026-06-22", "multimodal-generation/generation"),
]
# 纯文生图模型（不支持参考图，仅作为最终 fallback）
_TEXT2IMAGE_CHAIN = [
    ("qwen-image-3.0",     "multimodal-generation/generation"),  # 纯文本也可走 messages
    ("qwen-image-3.0-pro", "multimodal-generation/generation"),
    ("qwen-image-2.0-pro-2026-06-22", "multimodal-generation/generation"),
]


def _build_payload(model: str, endpoint_type: str, prompt: str, size: str,
                   reference_image_urls: list = None, ref_strength: float = 1.0,
                   negative_prompt: str = "") -> dict:
    """根据模型类型构建不同的 payload。

    reference_image_urls: 资产下多张参考图（已内联为 base64/data URL 或 http URL），
    全部注入 messages 做多参考图融合，对齐 Vidu「最多 7 张智能融合」。
    """
    ref_urls = reference_image_urls or []

    def _content(ref_list):
        content_list = []
        for r in ref_list:
            if r:
                content_list.append({"image": r})
        content_list.append({"text": prompt})
        return content_list

    # ── qwen-image 系列：同步 multimodal messages 格式 ──
    # 与 wan2.x-image 的 messages 结构一致，但不支持 enable_interleave /
    # ref_strength / negative_prompt 等 wan 专有参数，多传会被拒。
    if model.startswith("qwen-image"):
        return {
            "model": model,
            "input": {
                "messages": [
                    {"role": "user", "content": _content(ref_urls)}
                ]
            },
            "parameters": {
                "size": size,
                "n": 1,
                "watermark": False,
            }
        }

    # ── wan2.x-image 系列：多模态 messages 格式 ──
    # 覆盖 wan2.6-image / wan2.7-image / wan2.7-image-pro 等带后缀的变体
    if model.startswith("wan2.") and "-image" in model and not model.endswith("-t2i"):
        payload = {
            "model": model,
            "input": {
                "messages": [
                    {"role": "user", "content": _content(ref_urls)}
                ]
            },
            "parameters": {
                "size": size,
                "n": 1,
                "watermark": False,
            }
        }
        if ref_urls:
            payload["parameters"]["enable_interleave"] = False
        if ref_urls and ref_strength < 1.0:
            payload["parameters"]["ref_strength"] = ref_strength
        # wan2.7 image models do NOT support negative_prompt (official docs)
        if negative_prompt and not model.startswith("wan2.7"):
            payload["parameters"]["negative_prompt"] = negative_prompt
        return payload

    # ── wanx-v1：V1 text2image 格式，支持 ref_image + ref_mode ──
    elif model == "wanx-v1":
        payload = {
            "model": model,
            "input": {
                "prompt": prompt,
            },
            "parameters": {
                "size": size,
                "n": 1,
            }
        }
        if ref_urls:
            payload["input"]["ref_image"] = ref_urls[0]
            payload["parameters"]["ref_mode"] = "repaint"
            payload["parameters"]["ref_strength"] = ref_strength
        if negative_prompt:
            payload["parameters"]["negative_prompt"] = negative_prompt
        return payload

    # ── wanx2.1 系列：纯文生图，不支持参考图 ──
    else:
        payload = {
            "model": model,
            "input": {
                "prompt": prompt,
            },
            "parameters": {
                "size": size,
                "n": 1,
            }
        }
        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt
        return payload


def _is_quota_or_model_error(error_text: str) -> bool:
    """检测是否为额度/余额/限流/模型不可用等可恢复错误。"""
    low = error_text.lower()
    patterns = [
        # 额度相关
        "quota", "free tier", "freitier", "free_tier",
        # 余额/计费相关（之前遗漏导致 fallback 链中断）
        "balance", "insufficient", "arrearage", "overdue",
        "余额", "欠费", "余额不足",
        # 积分/权益相关（jimeng 免费额度耗尽返回「积分不足或没有相关权益」-2001，
        # 措辞与「余额」不同，此前遗漏导致 fallback 链在真实额度耗尽时从未触发）
        "积分", "credit", "没有相关权益", "相关权益", "-2001", "2001",
        # 限流
        "throttling", "ratelimit", "rate limit",
        # 模型不可用
        "model not", "not found", "not exist", "invalidmodel",
        # 内容审核/参数错误（也属于该模型无法处理当前请求）
        "datainspectionfailed", "invalidparameter", "url error",
        # 额度用尽的通用模式
        "quota exhausted", "quota exceeded", "no quota",
    ]
    return any(p in low for p in patterns)


def _extract_image_url(output: dict) -> str:
    """从 API output 中提取图片 URL。"""
    # messages 格式
    if "choices" in output and output["choices"]:
        content = output["choices"][0].get("message", {}).get("content", [])
        if content and isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    url = item.get("image") or item.get("url")
                    if url:
                        return url
    # 直接 url
    if "url" in output:
        return output["url"]
    # results 格式 (wanx)
    results = output.get("results", [])
    if results and isinstance(results, list):
        url = results[0].get("image") or results[0].get("url") or results[0].get("b64_image")
        if url:
            return url
    return None


# 常见尺寸缩写 → 像素尺寸映射（wanx 系列模型要求像素格式）
_SIZE_MAP = {
    "2K":    "1024*1024",
    "2k":    "1024*1024",
    "1K":    "1024*1024",
    "1k":    "1024*1024",
    "4K":    "2048*2048",
    "4k":    "2048*2048",
    "HD":    "1280*720",
    "FHD":   "1920*1080",
}


def _normalize_size(size: str) -> str:
    """将尺寸参数标准化为像素格式（wanx 模型兼容）。"""
    return _SIZE_MAP.get(size, size)


# 限流退避重试配置：429/Throttling 是 API key 级别的速率限制，切换模型无效
# （所有模型共享同一配额），必须在当前模型上退避等待速率窗口恢复。
_RATE_LIMIT_RETRIES = 5
_RATE_LIMIT_BACKOFF = [8, 15, 30, 45, 60]  # 递增退避（秒）


def _is_rate_limit(text: str) -> bool:
    """判断是否为限流类错误（429 / Throttling / RateQuota）。"""
    low = text.lower()
    return any(p in low for p in ("429", "throttling", "ratelimit", "rate limit",
                                   "ratequota", "too many requests"))


# ── 多生图后端自助接入（P0-4 收敛后的再开放）──
# jimeng 仍是首选（免费积分 + 多参考图融合）。当它因积分/额度类错误失败时，
# 按 MULTIMEDIA_IMAGE_FALLBACK（默认 "dashscope,siliconflow"）顺序尝试
# **已配置 key** 的备选后端——现场连通验证，失败继续下一个，全部失败抛原错误。
# 用户无需预先在设置面板点测试：key 填了、能用就用（"连通的话就给他们用"）。
# 设置面板的「连通测试」按钮（/settings/probe）可预先手动验证 key 有效性。
_IMAGE_FALLBACK_ORDER = [
    b.strip().lower()
    for b in os.getenv("MULTIMEDIA_IMAGE_FALLBACK", "dashscope,siliconflow").split(",")
    if b.strip()
]

_DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/api/v1"


def _generate_via_dashscope(
    prompt: str, size: str, ref_urls: list, negative_prompt: str, thread_id: str, timeout: int = 120
) -> str:
    """DashScope qwen-image 系列生图（同步 multimodal messages 端点）。

    复用历史遗留的 _IMG2IMG_CHAIN / _build_payload / _extract_image_url：
    - 有参考图走 _IMG2IMG_CHAIN（多参考图融合，messages content: image+text）；
    - 无参考图走 _TEXT2IMAGE_CHAIN。
    key 无效/额度不足/模型不可用 → 抛异常，由 fallback 链继续下一个。
    """
    key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY 未配置")
    base = (os.getenv("DASHSCOPE_BASE_URL") or _DASHSCOPE_API_BASE).strip().rstrip("/")
    chain = _IMG2IMG_CHAIN if ref_urls else _TEXT2IMAGE_CHAIN
    norm_size = _normalize_size(size)
    last_err = ""
    for model, endpoint in chain:
        payload = _build_payload(model, endpoint, prompt, norm_size,
                                 reference_image_urls=ref_urls,
                                 ref_strength=1.0, negative_prompt=negative_prompt)
        try:
            resp = requests.post(
                f"{base}/services/aigc/{endpoint}",
                json=payload,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                timeout=timeout,
            )
            text = resp.text[:400]
            if resp.status_code != 200:
                last_err = f"{model}: HTTP {resp.status_code} {text}"
                print(f"    [IMG][DashScope] {last_err}")
                continue
            url = _extract_image_url(resp.json())
            if not url:
                last_err = f"{model}: 响应中未找到图片 URL: {text}"
                print(f"    [IMG][DashScope] {last_err}")
                continue
            print(f"    [IMG][DashScope] {model} 生成成功")
            return url
        except Exception as e:  # noqa: BLE001
            last_err = f"{model}: {str(e)[:160]}"
            print(f"    [IMG][DashScope] {last_err}")
    raise RuntimeError(f"DashScope 生图失败: {last_err}")


def _generate_via_siliconflow(
    prompt: str, size: str, ref_urls: list, negative_prompt: str, thread_id: str, timeout: int = 120
) -> str:
    """SiliconFlow Qwen-Image 生图（images/generations，参考图走 image 数组）。"""
    key = (os.getenv("SILICONFLOW_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("SILICONFLOW_API_KEY 未配置")
    base = (os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    model = (os.getenv("SILICONFLOW_IMAGE_MODEL") or "Qwen/Qwen-Image").strip()
    # 尺寸映射：SiliconFlow 接受 1024x1024 等 WxH；jimeng 的 2K 语义近似 2048 长边
    sz = {"2K": "2048x2048", "1K": "1024x1024"}.get(size.upper(), size if "x" in size else "1024x1024")
    body: dict = {
        "model": model,
        "prompt": prompt,
        "image_size": sz,
    }
    if ref_urls:
        body["image"] = ref_urls[:3]  # Qwen-Image 图生图参考（data URL 或公网 URL）
    if negative_prompt and negative_prompt.strip():
        body["negative_prompt"] = negative_prompt.strip()

    resp = requests.post(
        f"{base}/v1/images/generations",
        json=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} {resp.text[:300]}")
    data = resp.json()
    # 兼容两种响应：{"images":[{"url":...}]} / OpenAI 风格 {"data":[{"url":...}]}
    items = data.get("images") or data.get("data") or []
    for it in items:
        u = (it or {}).get("url") or ""
        if u:
            print(f"    [IMG][SiliconFlow] {model} 生成成功")
            return u
    raise RuntimeError(f"响应中未找到图片 URL: {str(data)[:200]}")


def _probe_dashscope_image(api_key: str, model: str, post_fn=None, timeout: float = 15.0):
    """零额度探测 DashScope qwen-image 生图可达性（与视频 probe 同手法）。

    向同步生图端点发送「缺参」请求（input 为空），服务端校验阶段即拒绝，
    不创建任务、不消耗额度：401=Key 无效 / Model.NotFound=模型无效 /
    InvalidParameter=Key 与模型路由均有效 → 通过。
    Returns: (ok: bool, detail: str)
    """
    if post_fn is None:
        post_fn = requests.post
    url = f"{_DASHSCOPE_API_BASE}/services/aigc/multimodal-generation/generation"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": {}}
    try:
        resp = post_fn(url, headers=headers, json=payload, timeout=timeout)
    except Exception as e:
        return False, f"网络错误（无法连接百炼）: {e}"
    try:
        data = resp.json()
    except Exception:
        data = {}
    code = str(data.get("code") or "")
    msg = str(data.get("message") or resp.text[:160])
    low = (code + " " + msg).lower().replace(" ", "")
    if resp.status_code == 401 or "invalidapikey" in low:
        return False, "Key 无效（401 InvalidApiKey）"
    if "model.notfound" in low or "modelnotexist" in low or "模型不存在" in msg:
        return False, f"模型名无效: {model}"
    if "invalidparameter" in low or resp.status_code == 400:
        return True, f"Key 与模型路由有效（{model}）"
    if resp.status_code == 200:
        return True, f"可达（{model}）"
    if "throttling" in low or "quota" in low:
        return True, f"Key/模型有效，但额度或限流受限: {msg[:120]}"
    return False, f"未知响应: HTTP {resp.status_code} {msg[:120]}"


def _probe_siliconflow_image(api_key: str, model: str, timeout: float = 15.0):
    """零额度探测 SiliconFlow：GET /v1/models 验证 Key 有效性。

    models 列表不校验具体生图模型权限（诚实标注）；不发起生图请求，不消耗额度。
    Returns: (ok: bool, detail: str)
    """
    base = (os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    try:
        resp = requests.get(
            f"{base}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
    except Exception as e:
        return False, f"网络错误（无法连接硅基流动）: {e}"
    if resp.status_code == 401:
        return False, "Key 无效（401）"
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:140]}"
    try:
        ids = [m.get("id", "") for m in (resp.json().get("data") or [])]
    except Exception:
        ids = []
    if model and any(model in i for i in ids):
        return True, f"Key 有效，模型 {model} 在可用列表中"
    return True, "Key 有效（未在模型列表中检索到目标模型名，路由以实际生成为准）"


def generate_keyframe(prompt: str, size: str = "2K", reference_image_urls: list = None,
                      ref_strength: float = 1.0, negative_prompt: str = "",
                      thread_id: str = "", timeout: int = 90) -> str:
    """调用即梦(jimeng)图片生成模型生成关键帧（支持多参考图融合）。

    设计约束：关键帧生成只用 jimeng，不再走 DashScope / SiliconFlow / 任何 fallback。

    Args:
        prompt: 图片生成提示词
        size: 图片尺寸
        reference_image_urls: 资产下多张参考图 URL 列表（用于角色/道具一致性融合）
        ref_strength: 参考图影响力，0.0-1.0。1.0=强参考（默认），越低越弱
        negative_prompt: 负面提示词（jimeng 不支持该字段，自动融入 prompt 正文 "Avoid: ..."）
    """
    # 归一化为列表（兼容旧调用方传单字符串）
    ref_urls = reference_image_urls if isinstance(reference_image_urls, list) else (
        [reference_image_urls] if reference_image_urls else []
    )
    # 内联多张参考图为 base64/data URL（即梦需内联）
    inline_refs = [jimeng._as_inline(u) for u in ref_urls if u]

    # ── 关键帧仅走即梦(jimeng)免费后端（白嫖网页版每日免费积分）──
    # jimeng 支持多参考图 img2img 融合，保证角色/道具一致性。
    # jimeng 接口不支持 negative_prompt 字段，若调用方传入则融入 prompt 正文
    # （转成 "Avoid: ..." 文本），既保留抑制语义又避免 invalid parameter 报错导致尾帧降级。
    jimeng_prompt = prompt
    if negative_prompt and negative_prompt.strip():
        jimeng_prompt = f"{prompt}\nAvoid: {negative_prompt.strip()}"
    # S1（P0-4）：生图调用同样接入熔断器 + 成本台账。jimeng 为唯一生图后端，
    # 熔断主要用于 /settings/health 状态可视化与连续失败可见性。
    from . import backend_health as _bh

    _t0 = time.time()
    try:
        out = jimeng.generate_image(
            jimeng_prompt, size=size, reference_image_urls=inline_refs,
            ref_strength=ref_strength, thread_id=thread_id, timeout=timeout,
        )
        _bh.record_result("jimeng", True, time.time() - _t0, kind="image", thread_id=thread_id)
        return out
    except Exception as e:  # noqa: BLE001
        _bh.record_result("jimeng", False, time.time() - _t0, kind="image",
                          thread_id=thread_id, meta=str(e))
        # ── 自助多后端 fallback ──
        # 仅对积分/额度/限流类错误启用（配置类错误直接抛出，避免掩盖真问题）。
        # 备选后端必须已在 .env / 设置面板配置 key——现场连通验证，失败继续下一个；
        # 全部失败时抛出 jimeng 原错误，保持旧行为。
        _err = str(e)
        if not _is_quota_or_model_error(_err):
            raise
        _chain = [b for b in _IMAGE_FALLBACK_ORDER if b in ("dashscope", "siliconflow")]
        if not _chain:
            raise
        print(f"    [IMG] jimeng 额度/积分类失败，尝试备选生图后端: {_chain}")
        for backend in _chain:
            _t1 = time.time()
            try:
                if backend == "dashscope" and (os.getenv("DASHSCOPE_API_KEY") or "").strip():
                    print("    [IMG] fallback → DashScope qwen-image")
                    out = _generate_via_dashscope(prompt, size, inline_refs, negative_prompt, thread_id, timeout=timeout)
                    _bh.record_result("dashscope", True, time.time() - _t1, kind="image", thread_id=thread_id)
                    return out
                if backend == "siliconflow" and (os.getenv("SILICONFLOW_API_KEY") or "").strip():
                    print("    [IMG] fallback → SiliconFlow Qwen-Image")
                    out = _generate_via_siliconflow(prompt, size, inline_refs, negative_prompt, thread_id, timeout=timeout)
                    _bh.record_result("siliconflow", True, time.time() - _t1, kind="image", thread_id=thread_id)
                    return out
                if backend not in ("dashscope", "siliconflow"):
                    print(f"    [WARN] 未知备选生图后端 '{backend}'，跳过")
            except Exception as e2:  # noqa: BLE001
                print(f"    [WARN] 备选后端 {backend} 失败: {str(e2)[:140]}")
                if backend in ("dashscope", "siliconflow"):
                    _bh.record_result(backend, False, time.time() - _t1, kind="image",
                                      thread_id=thread_id, meta=str(e2))
        raise








