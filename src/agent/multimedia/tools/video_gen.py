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
    duration: float = None,
) -> dict:
    """kf2v models don't support shot_type or custom duration (only 5s / 8s tiers).

    【修复】此前 duration 写死 5，导致按台词估算为 7~8s 的长镜头只出 5s 视频，
    表现为「台词还没念完画面就切走」。这里改为按估算时长分档取 5 或 8。
    """
    resolution = resolution or os.getenv("VIDEO_RESOLUTION", "720P")
    # kf2v 仅支持 5s / 8s 两档，按估算时长就近取档
    _dur = 5
    try:
        if duration is not None:
            _d = float(duration)
            if _d > 5:
                _dur = 8
    except (TypeError, ValueError):
        _dur = 5
    params = {
        "resolution": resolution,
        "duration": _dur,
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
    # 反静止（原有）：抑制弱运动 / 定格
    "static, frozen, still frame, motionless, stiff, "
    "no movement, slow motion, slideshow, freeze frame, "
    "mannequin, statue, rigid body, no expression change, "
    # 防穿模 / 防肢体畸变（新增）：agnes 等免费模型在动态生成中
    # 极易出现角色肢体错位、多余肢体、肢体穿插物体等畸变，必须显式禁止。
    "deformed anatomy, deformed body, extra limbs, extra arms, extra legs, "
    "floating limbs, missing limbs, dismembered, detached limbs, "
    "twisted body proportions, distorted body, mutation, morphed anatomy, "
    "limbs clipping through objects, limbs passing through body, "
    "body intersecting scenery, body passing through terrain, limbs sinking into floor, "
    "feet floating above ground, feet sunk into ground, torso clipping through walls, "
    "broken skeleton, unnatural pose, contorted limbs, "
    # 防过曝 / 光效吞结构（新增）：发光武器与法宝光效过强时，会把背景漂成纯白、
    # 把手部与面部细节整个吞掉（成片中女剑仙分镜背景全白、玉佩白光淹没手掌）。
    "overexposed, blown-out highlights, whiteout background, blinding glow, "
    "washed-out background, excessive bloom, glowing object obscuring the hand, "
    "glow covering the face, lens flare washing out details, "
    # 防武器软化（新增，针对"剑被糊成光/柔化成能量"）：武器必须是实体不透明硬物。
    "weapon dissolving into light, weapon turning to mist, sword blurred into glow, "
    "silky weapon, soft energy blade, rubbery sword, weapon losing solidity, "
    "weapon melting, translucent blade, weapon fading to vapor, "
    # 防身份漂移（新增）：动态过程中角色被重画成另一个人 / 服装中途变化，
    # 是免费档视频模型最常见的质量事故，必须显式禁止（尤其"仙尊变白须老者"）。
    "wrong character identity, swapped character, changed costume mid-shot, "
    "altered appearance between frames, different person in later frames, "
    "inconsistent face, inconsistent outfit colors, old man with white beard, "
    "elderly character, aged face on a young character, "
    # 防手部崩坏（新增）：武器悬空无握持、手指缺失或被光效糊成一团。
    "obscured hands, hidden fingers, melted fingers, weapon floating without grip, "
    "hand not touching the weapon, blurry hand stump, fingers not wrapping the hilt"
)


def _resolve_agnes_num_frames(duration: int = 8, fps: int = 24) -> int:
    """把时长（秒）换算为 Agnes 帧数，满足 8n+1 约束。

    Agnes 要求 num_frames 满足 8n+1 且 <= 441（约 5~18s @24fps）。
    此处仅做粗换算，agnes.generate_video 内部还会再做一次 8n+1 对齐与上下限钳制。
    """
    n = max(1, round(duration * fps / 8))
    return n * 8 + 1


# ─────────────────────────────────────────────────────────────────────────────
# S1 重构（架构评审 P0-1 + P0-4）：质量档位候选编排 + 熔断器 + 成本台账
#
# 原 generate_video_from_image 按 VIDEO_PROVIDER 单后端直派，失败直接抛出。
# 现改为「候选链编排」：
#   1) 显式 VIDEO_PROVIDER 仍最优先（向后兼容旧行为）；
#   2) 其后按 QUALITY_TIER（fast/standard/cinema，config_loader.get_tier_plan）
#      给出的后端优先级依次尝试；
#   3) 每次尝试前后经 backend_health 记录结果（熔断器 + 成本台账），连续失败达
#      阈值的后端进入冷却（down）被自动跳过，冷却期满半开重试；
#   4) dashscope 仅在配置了 DASHSCOPE_API_KEY 时进入候选，保持「不落入无权限
#      链路掩盖真实错误」的既有约束。
# 各 provider 分支体保持原逻辑不变，仅从 if/elif 直派提取为独立 helper。
# ─────────────────────────────────────────────────────────────────────────────

from . import backend_health  # noqa: E402  (S1: 熔断器 + 成本台账)


def _resolve_video_candidates(quality_tier: str = "") -> tuple:
    """解析视频后端候选顺序。返回 (tier, [provider, ...])。

    候选来源：显式 VIDEO_PROVIDER 置顶 + QUALITY_TIER 档位计划；随后按
    「是否有对应凭证」硬过滤，避免把注定失败的候选排进链路。
    """
    from ..config_loader import get_tier_plan

    tier = (quality_tier or os.getenv("QUALITY_TIER", "standard")).strip().lower()
    if tier not in ("fast", "standard", "cinema"):
        tier = "standard"
    plan = get_tier_plan(tier)

    candidates: list = []
    explicit = os.getenv("VIDEO_PROVIDER", "").strip().lower()
    if explicit:
        candidates.append(explicit)
    for p in plan.get("providers", []):
        if p not in candidates:
            candidates.append(p)

    def _configured(p: str) -> bool:
        if p == "agnes":
            # agnes 例外：即使 key 未配置也保留候选，让底层抛出明确的「未配置」报错
            return True
        if p == "zhipu":
            return bool((os.getenv("ZHIPU_API_KEY") or "").strip())
        if p == "jimeng":
            return bool((os.getenv("JIMENG_SESSION_ID") or "").strip())
        if p == "dashscope":
            return bool((os.getenv("DASHSCOPE_API_KEY") or "").strip())
        return False

    candidates = [p for p in candidates if _configured(p)]
    if not candidates:
        candidates = [explicit or "agnes"]
    return tier, candidates


def _via_agnes(image_url, prompt, last_image_url, first_frame_path, thread_id,
               reference_images, reference_note, negative_prompt, duration) -> str:
    """Agnes AI 免费视频后端（keyframes 多图模式，参考图身份锚点注入）。"""
    ff = first_frame_path or image_url
    # 镜头稳定性引导（质量类，不作死定义）：保持单镜头连贯、主体清晰、环境稳定。
    _shot_stability = (
        "\n\n[Shot Stability] Keep this shot as one continuous, coherent action with "
        "a clearly readable subject and a stable, consistent environment from start "
        "to end. Use soft, controlled glow only — no overexposure, no blown highlights; "
        "keep environment, fabric and facial detail clearly visible."
    )
    _final_prompt = (prompt + "\n\n[Reference Consistency]\n" + reference_note).strip() if reference_note else prompt
    _final_prompt = (_final_prompt + _shot_stability).strip()
    # 透传 VIDEO_DURATION（默认 8s）；调用方显式传入 duration（按台词估算）时优先采用。
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
        width=1920, height=1080, num_frames=_agnes_num_frames, frame_rate=24,
        thread_id=thread_id,
        negative_prompt=negative_prompt,
    )


def _via_zhipu(image_url, prompt, last_image_url, first_frame_path, thread_id,
               reference_images, reference_note, negative_prompt, duration) -> str:
    """智谱 CogVideoX-Flash 免费后端（model 级免费；仅首尾帧参与图生视频）。"""
    ff = first_frame_path or image_url
    _shot_stability = (
        "\n\n[镜头稳定性] 本镜头保持单一连贯动作，主体清晰可辨，环境自始至终一致稳定。"
        "光效以柔和受控的辉光呈现，勿过曝、勿高光溢出，环境与人物细节清晰可见。"
    )
    _final_prompt = (prompt + "\n\n[参考图一致性约束]\n" + reference_note).strip() if reference_note else prompt
    _final_prompt = (_final_prompt + _shot_stability).strip()
    return zhipu.generate_video(
        _final_prompt,
        reference_images=reference_images,
        first_frame_url=ff,
        last_frame_url=last_image_url if last_image_url else "",
        width=1280, height=720,
        thread_id=thread_id,
    )


def _via_jimeng(image_url, prompt, first_frame_path, thread_id, reference_note) -> str:
    """即梦 Free 后端（图生/文生视频，首帧锚定）。"""
    ff = first_frame_path or image_url
    _jimeng_prompt = (prompt + "\n\n" + reference_note).strip() if reference_note else prompt
    return jimeng.generate_video(_jimeng_prompt, first_frame_url=ff if ff else None, thread_id=thread_id)


def _probe_dashscope_video(api_key: str, model: str, post_fn=None, timeout: float = 15.0):
    """零额度探测百炼视频模型可达性（设置面板「探测」与生成前校验共用）。

    原理：向视频合成端点发送「缺参」请求（input 为空），服务端在校验阶段即拒绝，
    **不会创建任务、不消耗额度**；用错误类型分类问题来源：
      - 401 / InvalidApiKey            → Key 无效
      - Model.NotFound / 模型不存在     → 模型名无效
      - InvalidParameter（缺 input 等） → Key 与模型路由均有效 → 通过
      - Throttling / Quota             → Key 与模型有效，但额度/限流受限
      - HTTP 200（极端情况下受理）      → 通过（附 task_id 提示）
    Returns: (ok: bool, detail: str)
    """
    if post_fn is None:
        post_fn = requests.post
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
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

    if resp.status_code == 200:
        task_id = str((data.get("output") or {}).get("task_id") or "")
        return True, ("探测通过（服务端已受理" + (f"，task_id={task_id}" if task_id else "") + "）")
    if resp.status_code in (401, 403) or "invalidapikey" in low or "unauthorized" in low:
        return False, f"API Key 无效或无权限（HTTP {resp.status_code}）: {msg}"
    if ("model" in low and ("notexist" in low or "notfound" in low or "invalid" in low or "不存在" in msg)) \
            or "modelnot" in low:
        return False, f"模型名无效: {msg}"
    if "invalidparameter" in low or "missinginput" in low or "input" in low:
        return True, "探测通过（Key 有效，模型路由成功）"
    if "throttl" in low or "quota" in low or "allocation" in low or "欠费" in msg or "余额" in msg:
        return False, f"Key 与模型可达，但额度/限流受限: {msg}"
    return False, f"探测未通过（HTTP {resp.status_code}）: {code} {msg}"


def _via_dashscope(image_url, prompt, last_image_url, first_frame_path,
                   negative_prompt, duration) -> str:
    """通义万相 DashScope 链（百炼）。

    用户语义（探测-钉死）：DASHSCOPE_VIDEO_MODEL 显式指定模型名时——
      先零额度探测（Key + 模型名 + 端点）；探测失败 → 明确报错（不静默回落）；
      探测通过 → 本次仅使用该模型（按名称路由 r2v / i2v / t2v 端点与入参），
      生成失败同样明确报错，不回落其他模型/厂商。
    未指定模型时保持原默认免费链 + 即梦兜底。
    """
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

    # 从环境变量读取主模型（覆盖默认链）。任意模型名均生效：按名称路由到
    # r2v / i2v / t2v 之一，未命中已知链的新模型同样置于链首，失败自动回落。
    env_model = (os.getenv("DASHSCOPE_VIDEO_MODEL") or "").strip() or None
    last_error = None

    # ── 探测-钉死：用户显式指定模型时，先用零额度探测校验 Key/模型名 ──
    if env_model:
        _pok, _pdetail = _probe_dashscope_video(api_key, env_model)
        if not _pok:
            raise RuntimeError(
                f"百炼视频模型探测失败（{env_model}）：{_pdetail} "
                f"—— 请在设置面板修正模型名 / API Key 后重试（本次不回落其他模型）。"
            )
        print(f"    [视频渲染] 百炼探测通过: {env_model}（{_pdetail}）→ 本次仅使用该模型")

    # 若 env 显式指定了模型，则优先走对应的生成链
    env_is_r2v = bool(env_model) and (
        env_model in _R2V_MODEL_CHAIN or "r2v" in env_model or "kf2v" in env_model
    )
    env_is_i2v = bool(env_model) and (not env_is_r2v) and (
        env_model in _I2V_MODEL_CHAIN or "i2v" in env_model
    )

    if env_is_r2v:
        # ── 显式 r2v（image2video 端点）──
        kf2v_chain = [env_model] + [m for m in _R2V_MODEL_CHAIN if m != env_model]
        for model_name in kf2v_chain:
            print(f"    [视频渲染] 尝试 kf2v/r2v 模型: {model_name}")
            resolution = "720P"
            payload = _build_payload_kf2v(
                model_name, image_url, last_image_url, prompt,
                resolution=resolution, negative_prompt=_MOTION_NEGATIVE_PROMPT,
                duration=duration,
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

        # 探测-钉死：用户显式指定 r2v 模型，失败即明确报错（不回落 i2v/t2v/即梦）
        if env_model:
            raise RuntimeError(
                f"百炼 r2v 模型 {env_model} 生成失败: {last_error} "
                f"—— 已按你的配置仅使用该模型；请检查模型名、额度或首帧图后重试。"
            )

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

        # 探测-钉死：用户显式指定 i2v 模型，失败即明确报错（不回落 t2v/即梦）
        if env_model:
            raise RuntimeError(
                f"百炼 i2v 模型 {env_model} 生成失败: {last_error} "
                f"—— 已按你的配置仅使用该模型；请检查模型名、额度或首帧图后重试。"
            )

    # ── 默认主链：t2v 文生视频（有额度）──
    # 无论是否提供首帧图，统一走 t2v。env_model 若属于 t2v 链则前置。
    _pinned_t2v = bool(env_model) and not (env_is_r2v or env_is_i2v)
    print(
        f"    [INFO] t2v 文生视频链"
        + (f"（用户指定模型: {env_model}，探测已通过，仅用该模型）" if _pinned_t2v else "（默认免费链）")
    )
    t2v_chain = [env_model] if _pinned_t2v else list(_T2V_MODEL_CHAIN)
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

    # 探测-钉死：用户显式指定 t2v 模型，失败即明确报错（不回落默认链/即梦）
    if _pinned_t2v:
        raise RuntimeError(
            f"百炼模型 {env_model} 生成失败: {last_error} "
            f"—— 已按你的配置仅使用该模型，未回落默认链/即梦；请检查模型名与额度。"
        )

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
    quality_tier: str = "",
) -> str:
    """
    调用视频生成模型（S1：质量档位候选链 + 熔断降级 + 成本台账）。

    视频后端候选顺序：
      1) 显式 VIDEO_PROVIDER 环境变量（向后兼容，置顶优先）；
      2) QUALITY_TIER 档位计划（fast / standard / cinema，见 config_loader.get_tier_plan）：
         - fast:     zhipu(flash) → jimeng → agnes   （秒级免费档，用于草稿/预览）
         - standard: agnes → jimeng → zhipu          （默认档，保持既有行为）
         - cinema:   agnes → jimeng → dashscope      （高质感档，dashscope 需显式模型）
      3) 每次调用经 backend_health 记录台账；连续失败达阈值的后端被熔断跳过，
         冷却期满自动半开重试。

    Args:
        image_url: 首帧图（URL / data URL / 本地路径）
        prompt: 视频运动 prompt
        last_image_url: 尾帧图（首尾帧双控模式）
        first_frame_path: 本地首帧文件路径（data URL 解码产物）
        thread_id: 会话隔离 ID（台账与产物目录）
        reference_images: 素材参考图列表（agnes keyframes 多图注入）
        reference_note: 参考图语义锚点文本
        negative_prompt: 负面提示词（agnes 独立字段 / dashscope payload）
        duration: 目标时长（秒）；fast 档会钳制到 ≤5s 以省额度
        quality_tier: 质量档位 fast / standard / cinema（空则读 QUALITY_TIER env）
    """
    _thread_id = thread_id or ""
    reference_images = reference_images or []

    tier, candidates = _resolve_video_candidates(quality_tier)

    # fast 档省额度：目标时长钳制 ≤5s（各 helper 内部再做 8n+1 / 上限对齐）
    if tier == "fast" and duration:
        duration = min(float(duration), 5.0)

    last_exc: Exception | None = None
    attempted: list = []

    for provider in candidates:
        status = backend_health.provider_status(provider)
        if status == "down":
            print(f"    [熔断] 视频后端 {provider} 冷却中，跳过（候选链继续）")
            continue
        attempted.append(provider)
        _t0 = time.time()
        try:
            print(f"    [视频后端] tier={tier} → {provider}")
            if provider == "agnes":
                out = _via_agnes(
                    image_url, prompt, last_image_url, first_frame_path,
                    _thread_id, reference_images, reference_note,
                    negative_prompt, duration,
                )
            elif provider == "zhipu":
                out = _via_zhipu(
                    image_url, prompt, last_image_url, first_frame_path,
                    _thread_id, reference_images, reference_note,
                    negative_prompt, duration,
                )
            elif provider == "jimeng":
                out = _via_jimeng(
                    image_url, prompt, first_frame_path, _thread_id, reference_note,
                )
            elif provider == "dashscope":
                out = _via_dashscope(
                    image_url, prompt, last_image_url, first_frame_path,
                    negative_prompt, duration,
                )
            else:
                print(f"    [WARN] 未知视频后端 {provider}，跳过")
                continue
            backend_health.record_result(
                provider, True, time.time() - _t0, kind="video", thread_id=_thread_id,
            )
            return out
        except Exception as e:  # noqa: BLE001 — 候选链语义：单后端失败换下一个
            backend_health.record_result(
                provider, False, time.time() - _t0, kind="video",
                thread_id=_thread_id, meta=str(e),
            )
            last_exc = e
            print(
                f"    [WARN] 视频后端 {provider} 失败（耗时 {time.time() - _t0:.1f}s），"
                f"切换下一候选: {e}"
            )

    if last_exc is None:
        raise RuntimeError(
            "没有可用的视频后端：请在前端「设置」面板或 .env 配置至少一个后端凭证"
            "（AGNES_API_KEY / ZHIPU_API_KEY / JIMENG_SESSION_ID / DASHSCOPE_API_KEY）。"
        )
    raise last_exc


def effective_primary_provider(quality_tier: str = "") -> str:
    """返回当前候选链首个视频后端（graph 节点据此决定是否做 agnes 分钟级退避）。"""
    try:
        _tier, _cands = _resolve_video_candidates(quality_tier)
        return _cands[0] if _cands else "agnes"
    except Exception:  # noqa: BLE001
        return os.getenv("VIDEO_PROVIDER", "agnes").strip().lower() or "agnes"
