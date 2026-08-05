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

load_dotenv()

# ── 网络瞬断重试配置 ──
_MAX_NETWORK_RETRIES = 3
_RETRY_BACKOFF = [3, 6, 12]  # 指数退避（秒）
_SUBMIT_TIMEOUT = 60         # 提交请求超时（秒）
_POLL_TIMEOUT = 30           # 轮询请求超时（秒）

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
                print(f"    ⚠️ [{label}] 网络瞬断 ({type(e).__name__})，{wait}s 后重试 ({attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                print(f"    ❌ [{label}] 网络重试 {max_retries} 次后仍失败: {e}")
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
    duration = duration or int(os.getenv("VIDEO_DURATION", "5"))

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

def _build_payload_kf2v(model_name: str, first_frame_url: str, last_frame_url: str, prompt: str, **kwargs) -> dict:
    """
    Build payload for kf2v (keyframe-to-video) models.
    Uses first_frame_url / last_frame_url directly in input (not media array).
    Endpoint: image2video/video-synthesis (different from i2v's video-generation/video-synthesis).
    """
    input_obj = {
        "first_frame_url": first_frame_url,
        "prompt": prompt,
    }
    if last_frame_url:
        input_obj["last_frame_url"] = last_frame_url

    return {
        "model": model_name,
        "input": input_obj,
        "parameters": _build_parameters_kf2v(**kwargs),
    }

def _submit_task(session: requests.Session, submit_url: str, headers: dict, payload: dict) -> str:
    for attempt in range(_MAX_NETWORK_RETRIES + 1):
        def _do_post():
            return session.post(submit_url, headers=headers, json=payload, timeout=_SUBMIT_TIMEOUT)

        response = _retry_request(_do_post, label="提交任务")

        # 服务端 5xx 临时故障 → 重试
        if response.status_code >= 500 and attempt < _MAX_NETWORK_RETRIES:
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

def _poll_task(session: requests.Session, poll_url: str, headers: dict) -> str:
    poll_fail_count = 0  # 连续轮询失败计数

    while True:
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
# wan2.7 系列: media.frames schema (首帧+尾帧), 720P 上限, 用 ratio 代替 shot_type, 2-15秒
# wan2.6 系列: media.frames 优先, 兼容 img_url, 2-15秒
# wan2.5 系列: img_url schema, 预览版
# wan2.2 系列: media.frames 优先, 兼容 img_url, 固定5秒
# wanx2.1 系列: img_url schema, 720P/480P, 固定5秒
_VIDEO_MODEL_CHAIN = [
    "wan2.7-i2v",               # 最新快照，原生首尾帧双控，50秒免费
    "wan2.7-i2v-2026-04-25",    # 固定快照版本
    "wan2.6-i2v",               # 50秒免费
    "wan2.6-i2v-flash",         # 快速版，50秒免费
    "wan2.5-i2v-preview",       # 预览版 fallback
    "wan2.2-i2v-plus",
    "wan2.2-i2v-flash",
    "wanx2.1-i2v-plus",
    "wanx2.1-i2v-turbo",
]

# kf2v (首尾帧双控) 模型链 — 当提供 last_image_url 时优先使用
# 使用独立 endpoint: image2video/video-synthesis
_KF2V_MODEL_CHAIN = ["wan2.2-kf2v-flash", "wanx2.1-kf2v-plus"]
_KF2V_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2video/video-synthesis"
_I2V_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"

# 默认负面提示词：抑制静态/僵硬动作
_MOTION_NEGATIVE_PROMPT = (
    "static, frozen, still frame, motionless, stiff, "
    "no movement, slow motion, slideshow, freeze frame, "
    "mannequin, statue, rigid body, no expression change"
)


def generate_video_from_image(image_url: str, prompt: str, last_image_url: str = "") -> str:
    """
    调用通义万相图生视频模型（支持多模型 fallback）。

    当提供 last_image_url 时，优先使用 kf2v (首尾帧双控) 模型链，
    再降级到 i2v 模型链。kf2v 模型使用独立 endpoint 和 schema。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY"))
    session = requests.Session()
    session.trust_env = False

    headers = {
        "X-DashScope-Async": "enable",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 从环境变量读取主模型（覆盖默认链）
    env_model = os.getenv("DASHSCOPE_VIDEO_MODEL")

    # ── Phase 1: kf2v 首尾帧双控（当提供 last_image_url 时）──
    if last_image_url:
        kf2v_chain = list(_KF2V_MODEL_CHAIN)
        if env_model and env_model not in kf2v_chain:
            pass  # env_model 留给 i2v phase
        for model_name in kf2v_chain:
            print(f"    [视频渲染] 尝试 kf2v 模型: {model_name}")
            resolution = "720P" if model_name.startswith("wanx2.1") else "1080P"
            payload = _build_payload_kf2v(
                model_name, image_url, last_image_url, prompt,
                resolution=resolution,
                negative_prompt=_MOTION_NEGATIVE_PROMPT,
            )

            try:
                task_id = _submit_task(session, _KF2V_ENDPOINT, headers, payload)
                print(f"    [视频渲染进度] kf2v 首尾帧双控生成中 ({model_name})，Task ID: {task_id}")
                poll_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
                return _poll_task(session, poll_url, headers)

            except FreeTierQuotaExhaustedError:
                print(f"    [WARN] kf2v {model_name} 免费额度耗尽，切换...")
                continue
            except VideoSchemaMismatchError as e:
                print(f"    [WARN] kf2v {model_name} schema 不匹配: {e}")
                continue
            except Exception as e:
                err_text = str(e)
                if _is_recoverable_api_error(err_text):
                    print(f"    [WARN] kf2v {model_name} 不可用（额度/余额/限流）: {e}")
                    continue
                raise

        print("    [INFO] kf2v 链全部不可用，降级到 i2v 链")

    # ── Phase 2: i2v 图生视频（传统模式）──
    if env_model:
        model_chain = [env_model] + [m for m in _VIDEO_MODEL_CHAIN if m != env_model]
    else:
        model_chain = _VIDEO_MODEL_CHAIN

    last_error = None

    for model_name in model_chain:
        print(f"    [视频渲染] 尝试模型: {model_name}")

        # wanx2.1 系列模型只支持 img_url 格式和 720P/480P 分辨率
        is_wanx21 = model_name.startswith("wanx2.1")
        if is_wanx21:
            payload_candidates = [
                ("img_url", _build_payload_img_url(
                    model_name, image_url, prompt,
                    resolution="720P", negative_prompt=_MOTION_NEGATIVE_PROMPT)),
                ("media.frames", _build_payload_media_frames(
                    model_name, image_url, last_image_url, prompt,
                    resolution="720P", negative_prompt=_MOTION_NEGATIVE_PROMPT)),
            ]
        else:
            payload_candidates = [
                ("media.frames", _build_payload_media_frames(
                    model_name, image_url, last_image_url, prompt,
                    negative_prompt=_MOTION_NEGATIVE_PROMPT)),
                ("img_url", _build_payload_img_url(
                    model_name, image_url, prompt,
                    negative_prompt=_MOTION_NEGATIVE_PROMPT)),
            ]

        for attempt_name, payload in payload_candidates:
            print(f"    [视频渲染提交] 尝试 schema: {attempt_name} ({model_name})")

            try:
                task_id = _submit_task(session, _I2V_ENDPOINT, headers, payload)
                print(f"    [视频渲染进度] 正在生成动态宣传片 ({model_name})，Task ID: {task_id}")

                poll_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
                return _poll_task(session, poll_url, headers)

            except FreeTierQuotaExhaustedError:
                # 免费额度耗尽：尝试下一个模型
                print(f"    [WARN] 模型 {model_name} 免费额度耗尽，切换 fallback...")
                last_error = FreeTierQuotaExhaustedError(f"{model_name} 额度耗尽")
                break  # 跳出 schema 循环，尝试下一个模型

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
                # 统一的可恢复错误检测：额度/余额/限流/模型不可用 → fallback
                if _is_recoverable_api_error(err_text):
                    print(f"    [WARN] 模型 {model_name} 不可用（额度/余额/限流），切换 fallback: {e}")
                    break
                raise

    raise last_error or Exception("所有视频生成模型均不可用，请检查 API Key 和模型额度。")
