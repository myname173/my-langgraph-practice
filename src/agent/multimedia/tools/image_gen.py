# src/agent/multimedia/tools/image_gen.py
import os
import base64
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


def generate_keyframe(prompt: str, size: str = "2K", reference_image_urls: list = None,
                      ref_strength: float = 1.0, negative_prompt: str = "",
                      thread_id: str = "") -> str:
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
    return jimeng.generate_image(
        jimeng_prompt, size=size, reference_image_urls=inline_refs,
        ref_strength=ref_strength, thread_id=thread_id,
    )








