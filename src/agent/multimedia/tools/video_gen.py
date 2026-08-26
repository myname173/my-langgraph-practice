# src/agent/multimedia/tools/video_gen.py
import os
import time
import json
import requests
from requests.exceptions import (
    ConnectionError, Timeout, SSLError,
    ChunkedEncodingError, ReadTimeout, ConnectTimeout,
)
from dotenv import load_dotenv
from ..config_loader import get_visual_backend, get_jimeng_config
from . import jimeng
from . import agnes
from . import zhipu
from pathlib import Path as _Path

# 项目根目录：video_gen.py -> tools -> multimedia -> agent -> src -> <root>
PROJECT_ROOT = _Path(__file__).resolve().parents[4]

load_dotenv()

# ── 网络瞬断重试配置 ──
# 重试次数与退避都相对宽松：观察到 dashscope 网关偶发 SSL EOF 抖动会持续数十秒，
# 若只重试 3 次(最长 12s)极易直接 FATAL 终止整条流水线。调高到 6 次、退避最长 ~30s，
# 可在约 1~2 分钟的抖动窗口内自愈，避免单点网络抖动拖垮全部 8 个镜头的生成。
_MAX_NETWORK_RETRIES = 6
_RETRY_BACKOFF = [3, 6, 12, 18, 24, 30]  # 指数退避（秒）
_SUBMIT_TIMEOUT = 60         # 提交请求超时（秒）
_POLL_TIMEOUT = 30           # 轮询请求超时（秒）
# 单个视频任务轮询的墙钟超时（秒）。防止服务端任务永久 RUNNING 导致进程挂死。
# 可通过环境变量 DASHSCOPE_POLL_MAX_WAIT 覆盖。
_POLL_MAX_WAIT = int(os.getenv("DASHSCOPE_POLL_MAX_WAIT", "600"))


class VideoPollTimeoutError(RuntimeError):
    """轮询超过墙钟上限仍未完成，视为该任务失败。"""
    pass

def _is_transient_error(exc: Exception) -> bool:
    """判断是否为可重试的瞬断网络错误"""
    return isinstance(exc, (
        SSLError, ConnectionError, Timeout,
        ChunkedEncodingError, ReadTimeout, ConnectTimeout,
    ))

def _retry_request(request_fn, max_retries=_MAX_NETWORK_RETRIES, backoff=_RETRY_BACKOFF, label="请求"):
    """
    对 HTTP 请求执行指数退避重试。
    request_fn: 无参可调用对象，返回 requests.Response
    仅对瞬断网络错误重试；其他异常原样抛出。
    """
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return request_fn()
        except Exception as e:
            if not _is_transient_error(e):
                raise  # 非瞬断错误，立即抛出
            last_err = e
            if attempt < max_retries:
                wait = backoff[min(attempt, len(backoff) - 1)]
                print(f"    [WARN] [{label}] 网络瞬断 ({type(e).__name__})，{wait}s 后重试 ({attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                print(f"    [FAIL] [{label}] 网络重试 {max_retries} 次后仍失败: {e}")
    raise last_err  # type: ignore[misc]

class FreeTierQuotaExhaustedError(RuntimeError):
    pass

class VideoSchemaMismatchError(RuntimeError):
    pass

def _is_recoverable_api_error(payload) -> bool:
    """检测所有可恢复的 API 错误（额度/余额/计费/限流），触发 fallback 到下一个模型。"""
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except Exception:
        text = str(payload)
    low = text.lower()
    patterns = [
        # DashScope 额度相关
        "allocationquota", "free quota", "free tier", "free_tier",
        # 余额/计费相关（关键！之前遗漏导致 fallback 链中断）
        "balance", "insufficient", "arrearage", "overdue",
        "余额", "欠费", "余额不足",
        # 限流相关
        "throttling", "ratelimit", "rate limit",
        # 模型不可用
        "invalidmodel", "model not exist", "not found",
        # 额度用尽的通用模式
        "quota exhausted", "quota exceeded", "no quota",
    ]
    return any(p in low for p in patterns)

def _contains_schema_error(payload) -> bool:
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except Exception:
        text = str(payload)

    lowered = text.lower()
    keywords =[
        "input.media.0.type",
        "first_frame",
        "last_frame",
        "driving_audio",
        "first_clip",
        "invalidparameter",
        "field required",
        "input.media",
    ]
    return any(k in lowered for k in keywords)

def _extract_error_message(payload) -> str:
    if isinstance(payload, dict):
        return payload.get("message") or payload.get("msg") or str(payload)
    return str(payload)

def _build_parameters(
    resolution: str = None,
    duration: int = None,
    prompt_extend: bool = True,
    watermark: bool = False,
    shot_type: str = "single",
    model_name: str = "",
    negative_prompt: str = "",
) -> dict:
    # 从环境变量读取默认值，允许参数覆盖
    resolution = resolution or os.getenv("VIDEO_RESOLUTION", "1080P")
    # 单镜头默认时长：由 5s 提升至 8s，给运动展开留出时间。
    # 仍可由环境变量 VIDEO_DURATION 覆盖（如 10），或调用方显式传 duration 覆盖。
    duration = duration or int(os.getenv("VIDEO_DURATION", "8"))

    # wan2.7 系列：不支持 shot_type，使用 ratio 控制画面比例，分辨率上限 720P
    if model_name.startswith("wan2.7"):
        if resolution not in ("720P", "480P"):
            resolution = "720P"
        params = {
            "resolution": resolution,
            "duration": duration,
            "prompt_extend": prompt_extend,
            "watermark": watermark,
            "ratio": "16:9",
        }
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
        return params

    params = {
        "resolution": resolution,
        "duration": duration,
        "prompt_extend": prompt_extend,
        "watermark": watermark,
        "shot_type": shot_type,
    }
    if negative_prompt:
        params["negative_prompt"] = negative_prompt
    return params

def _build_payload_media_frames(model_name: str, first_frame_url: str, last_frame_url: str, prompt: str, **kwargs) -> dict:
    """
    支持单首帧或首尾双帧的 media 结构。可传入 resolution, duration 等参数。
    """
    media =[
        {
            "type": "first_frame",
            "url": first_frame_url,
        }
    ]
    
    if last_frame_url:
        media.append({
            "type": "last_frame",
            "url": last_frame_url,
        })

    return {
        "model": model_name,
        "input": {
            "prompt": prompt,
            "media": media,
        },
        "parameters": _build_parameters(model_name=model_name, **kwargs),
    }

def _build_payload_img_url(model_name: str, image_url: str, prompt: str, **kwargs) -> dict:
    return {
        "model": model_name,
        "input": {
            "prompt": prompt,
            "img_url": image_url,
        },
        "parameters": _build_parameters(model_name=model_name, **kwargs),
    }

def _build_parameters_kf2v(
    resolution: str = None,
    prompt_extend: bool = True,
    watermark: bool = False,
    negative_prompt: str = "",
) -> dict:
    """kf2v models don't support shot_type or custom duration (fixed 5s)."""
    resolution = resolution or os.getenv("VIDEO_RESOLUTION", "720P")
    params = {
        "resolution": resolution,
        "duration": 5,
        "prompt_extend": prompt_extend,
        "watermark": watermark,
    }
    if negative_prompt:
        params["negative_prompt"] = negative_prompt
    return params

def _as_data_url(url_or_path: str) -> str:
    """
    将图像输入统一为服务端可接受的 url 形式：
      - 已是 http(s)/oss:///data: 开头 → 原样返回
      - 本地文件路径 → 读取并内联为 data:{mime};base64,{data}（避免 "url error"）
    """
    if not url_or_path:
        return url_or_path
    if url_or_path.startswith(("http://", "https://", "oss://", "data:")):
        return url_or_path
    # 视为本地路径
    try:
        import mimetypes
        import base64 as _b64
        mime = mimetypes.guess_type(url_or_path)[0] or "image/png"
        with open(url_or_path, "rb") as f:
            b64 = _b64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"    [WARN] 首帧图转 base64 失败，回退原值: {e}")
        return url_or_path


def _build_payload_kf2v(model_name: str, first_frame_url: str, last_frame_url: str, prompt: str, **kwargs) -> dict:
    """
    Build payload for kf2v (keyframe-to-video) models.
    万相2.7-图生视频（wan2.7-r2v / happyhorse-1.1-r2v）使用 media 数组：
    [{type:first_frame,url}, {type:last_frame,url}]，端点 video-generation/video-synthesis。
    url 支持 http(s)/oss:///data:base64 三种；本地路径自动内联 base64（修正 "url error"）。
    """
    first_url = _as_data_url(first_frame_url)
    last_url = _as_data_url(last_frame_url) if last_frame_url else None

    media = [{"type": "first_frame", "url": first_url}]
    if last_url:
        media.append({"type": "last_frame", "url": last_url})

    input_obj = {
        "media": media,
        "prompt": prompt,
    }

    return {
        "model": model_name,
        "input": input_obj,
        "parameters": _build_parameters_kf2v(**kwargs),
    }


def _build_payload_t2v(model_name: str, prompt: str, **kwargs) -> dict:
    """
    文生视频 (text-to-video) payload。仅含 prompt，不传任何图像。
    适用模型：wan2.7-t2v-2026-06-12 / wan3.0-video / happyhorse-1.1-t2v。
    端点与 i2v 相同（video-generation/video-synthesis）。
    """
    return {
        "model": model_name,
        "input": {
            "prompt": prompt,
        },
        "parameters": _build_parameters(**kwargs),
    }


def _submit_task(session: requests.Session, submit_url: str, headers: dict, payload: dict) -> str:
    for attempt in range(_MAX_NETWORK_RETRIES + 1):
        def _do_post():
            return session.post(submit_url, headers=headers, json=payload, timeout=_SUBMIT_TIMEOUT)

        response = _retry_request(_do_post, label="提交任务")

        # 服务端 5xx 临时故障 → 重试
        if response.status_code >= 500 and attempt < _MAX_NETWORK_RETRIES:
            # Agnes 免费后端队列满（video_queue_full）属于可控限流，需更长退避等待排空，
            # 而非短间隔猛刷。专项退避 20/40/60/90/120/150s，总等待约 9.5 分钟自愈窗口。
            _queue_full = False
            try:
                _ep = response.json()
                if isinstance(_ep, dict) and _ep.get("code") == "video_queue_full":
                    _queue_full = True
            except Exception:
                pass
            if _queue_full:
                wait = [20, 40, 60, 90, 120, 150][min(attempt, 5)]
                print(f"    ⚠️ [提交任务] Agnes 视频队列已满 (video_queue_full)，{wait}s 后重试 ({attempt+1}/{_MAX_NETWORK_RETRIES})...")
            else:
                wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                print(f"    ⚠️ [提交任务] 服务端 {response.status_code}，{wait}s 后重试 ({attempt+1}/{_MAX_NETWORK_RETRIES})...")
            time.sleep(wait)
            continue

        try:
            response_payload = response.json()
        except Exception:
            response_payload = response.text

        if response.status_code != 200:
            if _is_recoverable_api_error(response_payload):
                raise FreeTierQuotaExhaustedError(
                    f"视频生成被服务端拒绝（额度/余额/限流）。原始错误：{_extract_error_message(response_payload)}"
                )
            if _contains_schema_error(response_payload):
                raise VideoSchemaMismatchError(
                    f"视频生成请求 schema 不匹配。原始错误：{_extract_error_message(response_payload)}"
                )
            raise Exception(f"视频渲染提交异常: {response.text}")

        task_id = response_payload.get("output", {}).get("task_id")
        if not task_id:
            raise Exception(f"视频任务已提交但未返回 task_id: {response_payload}")

        return task_id

    raise Exception(f"提交任务重试 {_MAX_NETWORK_RETRIES} 次后仍失败")

def _poll_task(session: requests.Session, poll_url: str, headers: dict,
               max_wait: int = _POLL_MAX_WAIT) -> str:
    poll_fail_count = 0  # 连续轮询失败计数
    deadline = time.monotonic() + max_wait  # 墙钟超时截止点

    while True:
        # ── 全局墙钟超时保护：防止任务永久 RUNNING 挂死 ──
        if time.monotonic() > deadline:
            raise VideoPollTimeoutError(
                f"视频任务轮询超过 {max_wait}s 仍未完成，判定为超时失败。"
            )

        # ── 轮询请求加网络重试 ──
        try:
            def _do_get():
                return session.get(poll_url, headers=headers, timeout=_POLL_TIMEOUT)

            poll_resp = _retry_request(_do_get, max_retries=2, backoff=[2, 5], label="轮询状态")
            poll_data = poll_resp.json()
            poll_fail_count = 0  # 成功则重置

        except Exception as poll_err:
            poll_fail_count += 1
            if poll_fail_count >= 5:
                raise Exception(f"轮询连续 {poll_fail_count} 次失败，放弃重试: {poll_err}")
            print(f"    ⚠️ [轮询] 网络异常 ({type(poll_err).__name__})，5s 后继续轮询 ({poll_fail_count}/5)...")
            time.sleep(5)
            continue

        output = poll_data.get("output", {})
        task_status = output.get("task_status", "")

        if task_status == "SUCCEEDED":
            video_url = (
                output.get("video_url")
                or output.get("url")
                or (output.get("results", [{}])[0].get("url") if output.get("results") else None)
            )
            if video_url:
                print(f"    ✅ 视频生成成功！URL: {video_url}")
                return video_url
            raise Exception(f"视频生成成功但未能解析视频 URL: {poll_data}")

        if task_status == "FAILED":
            if _is_recoverable_api_error(poll_data):
                raise FreeTierQuotaExhaustedError(
                    f"视频生成任务失败（额度/余额/限流）。原始错误：{_extract_error_message(poll_data)}"
                )
            if _contains_schema_error(poll_data):
                raise VideoSchemaMismatchError(
                    f"视频生成任务失败：请求 schema 与服务端不匹配。原始错误：{_extract_error_message(poll_data)}"
                )
            raise Exception(f"视频生成失败: {poll_data}")

        print("    [视频渲染进度] 视频逐帧生成中，请耐心等待 (约5秒/次)...")
        time.sleep(5)

# ── 视频生成模型优先级（主 → fallback）──
# 自 2026-08-11 起，用户仅保有【文生视频(t2v)】与【文生图】免费额度，
# r2v / i2v 模型已无额度（happyhorse-1.1-i2v 已耗尽会直接中止流程），
# 因此默认【优先走 t2v 链】，且 t2v 链仅包含有额度的模型。
#
#   t2v (文生视频，有额度)：wan2.7-t2v-2026-06-12, wan3.0-video,
#                            happyhorse-1.1-t2v, happyhorse-t2v
#       → 端点 video-generation/video-synthesis，input 仅 prompt（无图）
#   r2v / i2v（无额度，保留代码但默认不触发）：
#       仅当 env DASHSCOPE_VIDEO_MODEL 显式指定为 r2v/i2v 模型时才启用。
_R2V_MODEL_CHAIN = [
    "wan2.7-r2v-2026-06-12",    # 万相首尾帧双控（当前无额度）
    "happyhorse-1.1-r2v",       # 参考图生视频（当前无额度）
]
_I2V_MODEL_CHAIN = [
    "happyhorse-1.1-i2v",        # 图生视频（额度已耗尽，默认不触发）
]
_T2V_MODEL_CHAIN = [
    "wan2.7-t2v-2026-06-12",     # 文生视频，50秒免费
    "wan3.0-video",              # 文生视频，30秒免费
    "happyhorse-1.1-t2v",        # 文生视频（env 注释名）
    "happyhorse-t2v",            # 文生视频（用户口头清单名，双写做 fallback）
]

# kf2v = r2v 链别名；默认不再优先。仅 env 显式指定时启用。
_KF2V_MODEL_CHAIN = _R2V_MODEL_CHAIN
# 修正（2026-08-12）：wan2.7-r2v / happyhorse-1.1-r2v 属"万相2.7-图生视频"族，
# 官方端点统一为 video-generation/video-synthesis（与 i2v 同端点），
# 旧 image2video 路径会导致 404 / "url error"。
_KF2V_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
_I2V_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"

# 默认负面提示词：抑制静态/僵硬动作
_MOTION_NEGATIVE_PROMPT = (
    "static, frozen, still frame, motionless, stiff, "
    "no movement, slow motion, slideshow, freeze frame, "
    "mannequin, statue, rigid body, no expression change"
)


def _resolve_agnes_num_frames(duration: int = 8, fps: int = 24) -> int:
    """把时长（秒）换算为 Agnes 帧数，满足 8n+1 约束。

    Agnes 要求 num_frames 满足 8n+1 且 <= 441（约 5~18s @24fps）。
    此处仅做粗换算，agnes.generate_video 内部还会再做一次 8n+1 对齐与上下限钳制。
    """
    n = max(1, round(duration * fps / 8))
    return n * 8 + 1


def generate_video_from_image(
    image_url: str,
    prompt: str,
    last_image_url: str = "",
    first_frame_path: str = "",
    thread_id: str = "",
    reference_images: list = None,
    reference_note: str = "",
    negative_prompt: str = "",
    duration: float = None,
) -> str:
    """
    调用视频生成模型（支持多后端 fallback）。

    视频后端优先级（VIDEO_PROVIDER 环境变量，默认 agnes）：
      - agnes:    Agnes AI 免费视频后端（默认）。支持 keyframes 多图模式，
                  把「首尾帧关键帧 + 素材参考图(角色转面图/道具/环境参考图)」
                  完整注入，免费且无需信用卡。每分钟限 1 次（内置退避）。
      - jimeng:   即梦免费后端（图生/文生视频，仅首帧，无多参考图）。
      - dashscope:通义万相（t2v 免费链默认；i2v/r2v 仅在 DASHSCOPE_VIDEO_MODEL 显式指定时启用）。
      - zhipu:    智谱 AI 免费视频后端（CogVideoX-Flash，model 级免费，无需信用卡）。
                  仅首尾帧参与图生视频，返回本地 mp4，与 agnes 同契约。

    reference_images: 素材参考图列表（URL/本地路径），供 agnes keyframes 多图模式注入。
    reference_note:   素材参考图语义锚点文本，拼入 prompt（jimeng/dashscope 分支用）。

    说明：默认策略（2026-08-11 起）对 dashscope 仅走 t2v 免费链以保额度可用；
    r2v/i2v 链仅在 DASHSCOPE_VIDEO_MODEL 显式指定时启用。
    """
    # 规范化 thread_id（空串 -> _generated），供 jimeng 分支按会话隔离产物
    _thread_id = thread_id or ""
    reference_images = reference_images or []

    # ── Agnes AI 免费视频后端（默认 VIDEO_PROVIDER=agnes）──
    # 完整融入首尾帧关键帧 + 素材参考图（keyframes 多图模式）。
    # 注意：agnes 使用自有 API key（AGNES_API_KEY），与 MULTIMODIA_BACKEND 解耦，
    # 因此只要 VIDEO_PROVIDER=agnes 且 key 已配，即走此免费渠道，与生图后端无关。
    _video_provider = os.getenv("VIDEO_PROVIDER", "agnes").strip().lower()
    if _video_provider == "agnes":
        try:
            ff = first_frame_path or image_url
            _final_prompt = (prompt + "\n\n[参考图一致性约束]\n" + reference_note).strip() if reference_note else prompt
            # 【修复-D】Agnes 不支持独立 negative_prompt 参数，把"抑制静态/增强运动"约束
            # 以文本指令并入 prompt，避免生成"人物定格、画面静止"的弱运动视频。
            if negative_prompt:
                _final_prompt = (_final_prompt + "\n\n[Motion Constraint / Negative]\n" + negative_prompt).strip()
            # 透传 VIDEO_DURATION（默认 8s）：此前写死 num_frames=81(≈3.4s)，
            # 导致"延长时长"改造对 agnes 不生效。agnes 内部仍会做 8n+1 对齐与上限钳制。
            # 若调用方显式传入 duration（按台词估算的 duration_seconds），优先采用，
            # 使视频时长与台词长度匹配，避免统一短时长导致台词念不完。
            _env_dur = int(os.getenv("VIDEO_DURATION", "8"))
            if duration and 5 <= float(duration) <= 8:
                _agnes_duration = int(round(float(duration)))
            else:
                _agnes_duration = _env_dur
            _agnes_num_frames = _resolve_agnes_num_frames(_agnes_duration, fps=24)
            print(f"    [Agnes] 时长 {_agnes_duration}s -> num_frames={_agnes_num_frames} (≈{_agnes_num_frames/24:.1f}s)")
            return agnes.generate_video(
                _final_prompt,
                reference_images=reference_images,
                first_frame_url=ff,
                last_frame_url=last_image_url if last_image_url else "",
                width=1280, height=720, num_frames=_agnes_num_frames, frame_rate=24,
                thread_id=_thread_id,
            )
        except Exception as e:
            # 【修复】Agnes 是用户指定的唯一视频后端，不得静默回退到无权限的
            # dashscope 视频链（会掩盖真实错误并必然失败 AccessDenied）。
            # 直接抛出，交由 video_gen_node 的 except 分支按失败/重试/跳过处理。
            raise RuntimeError(f"视频生成失败（Agnes 后端）: {e}") from e

    # ── 智谱 AI 免费视频后端（VIDEO_PROVIDER=zhipu）──
    # 走 CogVideoX-Flash（model 级免费，无需信用卡）。仅首尾帧参与图生视频
    # （CogVideoX 不支持多参考图），返回本地 mp4 路径，与 agnes 同契约。
    if _video_provider == "zhipu":
        try:
            ff = first_frame_path or image_url
            _final_prompt = (prompt + "\n\n[参考图一致性约束]\n" + reference_note).strip() if reference_note else prompt
            return zhipu.generate_video(
                _final_prompt,
                reference_images=reference_images,
                first_frame_url=ff,
                last_frame_url=last_image_url if last_image_url else "",
                width=1280, height=720,
                thread_id=_thread_id,
            )
        except Exception as e:
            raise RuntimeError(f"视频生成失败（智谱后端）: {e}") from e

    # ── 即梦 AI Free 后端分支（白嫖网页版每日免费积分）──
    # 当 MULTIMODIA_BACKEND == "jimeng" 时，走即梦免费生视频（图生/文生视频）。
    if get_visual_backend() == "jimeng":
        ff = first_frame_path or image_url
        _jimeng_prompt = (prompt + "\n\n" + reference_note).strip() if reference_note else prompt
        return jimeng.generate_video(_jimeng_prompt, first_frame_url=ff if ff else None, thread_id=_thread_id)

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未配置视频生成 API Key：请设置环境变量 DASHSCOPE_API_KEY（或 OPENAI_API_KEY）。"
        )
    session = requests.Session()
    session.trust_env = False

    headers = {
        "X-DashScope-Async": "enable",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 从环境变量读取主模型（覆盖默认链）
    env_model = os.getenv("DASHSCOPE_VIDEO_MODEL")
    last_error = None

    # 若 env 显式指定了 r2v / i2v 模型，则优先走对应的图生视频链
    env_is_r2v = env_model in _R2V_MODEL_CHAIN
    env_is_i2v = env_model in _I2V_MODEL_CHAIN

    if env_is_r2v:
        # ── 显式 r2v（image2video 端点）──
        kf2v_chain = [env_model] + [m for m in _R2V_MODEL_CHAIN if m != env_model]
        for model_name in kf2v_chain:
            print(f"    [视频渲染] 尝试 kf2v/r2v 模型: {model_name}")
            resolution = "720P"
            payload = _build_payload_kf2v(
                model_name, image_url, last_image_url, prompt,
                resolution=resolution, negative_prompt=_MOTION_NEGATIVE_PROMPT,
            )
            try:
                task_id = _submit_task(session, _KF2V_ENDPOINT, headers, payload)
                print(f"    [视频渲染进度] kf2v/r2v 生成中 ({model_name})，Task ID: {task_id}")
                poll_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
                return _poll_task(session, poll_url, headers)
            except FreeTierQuotaExhaustedError:
                print(f"    [WARN] kf2v {model_name} 免费额度耗尽，切换...")
                last_error = FreeTierQuotaExhaustedError(f"{model_name} 额度耗尽")
                continue
            except VideoPollTimeoutError as e:
                print(f"    [WARN] kf2v {model_name} 轮询超时，切换: {e}")
                last_error = e
                continue
            except VideoSchemaMismatchError as e:
                print(f"    [WARN] kf2v {model_name} schema 不匹配: {e}")
                last_error = e
                continue
            except Exception as e:
                err_text = str(e)
                if _is_recoverable_api_error(err_text):
                    print(f"    [WARN] kf2v {model_name} 不可用（额度/余额/限流）: {e}")
                    last_error = e
                    continue
                raise

    if env_is_i2v:
        # ── 显式 i2v（video-generation 端点）──
        model_chain = [env_model] + [m for m in _I2V_MODEL_CHAIN if m != env_model]
        for model_name in model_chain:
            print(f"    [视频渲染] 尝试 i2v 模型: {model_name}")
            payload_candidates = [
                ("media.frames", _build_payload_media_frames(
                    model_name, image_url, "", prompt,
                    negative_prompt=_MOTION_NEGATIVE_PROMPT)),
                ("img_url", _build_payload_img_url(
                    model_name, image_url, prompt,
                    negative_prompt=_MOTION_NEGATIVE_PROMPT)),
            ]
            for attempt_name, payload in payload_candidates:
                print(f"    [视频渲染提交] 尝试 schema: {attempt_name} ({model_name})")
                try:
                    task_id = _submit_task(session, _I2V_ENDPOINT, headers, payload)
                    print(f"    [视频渲染进度] i2v 生成中 ({model_name})，Task ID: {task_id}")
                    poll_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
                    return _poll_task(session, poll_url, headers)
                except FreeTierQuotaExhaustedError:
                    print(f"    [WARN] 模型 {model_name} 免费额度耗尽，切换 fallback...")
                    last_error = FreeTierQuotaExhaustedError(f"{model_name} 额度耗尽")
                    break
                except VideoPollTimeoutError as e:
                    print(f"    [WARN] 模型 {model_name} 轮询超时，切换 fallback: {e}")
                    last_error = e
                    break
                except VideoSchemaMismatchError as e:
                    last_error = e
                    print(f"    [WARN] schema 不匹配，准备切换备用入参结构: {e}")
                    continue
                except Exception as e:
                    last_error = e
                    err_text = str(e)
                    if _contains_schema_error(err_text):
                        print(f"    [WARN] 捕获到 schema 类错误，准备切换备用入参结构: {err_text}")
                        continue
                    if _is_recoverable_api_error(err_text):
                        print(f"    [WARN] 模型 {model_name} 不可用（额度/余额/限流），切换 fallback: {e}")
                        break
                    raise

    # ── 默认主链：t2v 文生视频（有额度）──
    # 无论是否提供首帧图，统一走 t2v。env_model 若属于 t2v 链则前置。
    print("    [INFO] 默认走 t2v 文生视频链（用户持有额度）")
    if env_model and env_model not in _T2V_MODEL_CHAIN:
        # env 指定的不是 t2v 模型且上面已处理，这里忽略
        pass
    t2v_chain = ([env_model] + _T2V_MODEL_CHAIN) if (env_model and env_model in _T2V_MODEL_CHAIN) else _T2V_MODEL_CHAIN
    for model_name in t2v_chain:
        print(f"    [视频渲染] 尝试 t2v 模型: {model_name}")
        payload = _build_payload_t2v(
            model_name, prompt, negative_prompt=_MOTION_NEGATIVE_PROMPT)
        try:
            task_id = _submit_task(session, _I2V_ENDPOINT, headers, payload)
            print(f"    [视频渲染进度] 正在文生视频 ({model_name})，Task ID: {task_id}")
            poll_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
            return _poll_task(session, poll_url, headers)
        except FreeTierQuotaExhaustedError:
            print(f"    [WARN] t2v {model_name} 免费额度耗尽，切换...")
            last_error = FreeTierQuotaExhaustedError(f"{model_name} 额度耗尽")
            continue
        except VideoPollTimeoutError as e:
            print(f"    [WARN] t2v {model_name} 轮询超时，切换: {e}")
            last_error = e
            continue
        except Exception as e:
            err_text = str(e)
            if _is_recoverable_api_error(err_text):
                print(f"    [WARN] t2v {model_name} 不可用: {e}")
                last_error = e
                continue
            raise

    # ── 即梦 Free 兜底：百炼全链失败后，若配置了 session 则白嫖即梦免费视频额度 ──
    # 即梦支持图生视频（首帧）→ 真动态 + 尽量维持角色一致，缓解百炼额度紧张。
    if get_jimeng_config().get("has_session"):
        try:
            print(f"    [视频渲染] 百炼全链失败，兜底走即梦 Free 生视频...")
            ff = first_frame_path or image_url
            return jimeng.generate_video(prompt, first_frame_url=ff if ff else None)
        except Exception as je:
            print(f"    [WARN] 即梦兜底生视频也失败: {je}")

    raise last_error or Exception("所有视频生成模型均不可用，请检查 API Key 和模型额度。")
