# src/agent/multimedia/tools/image_gen.py
import os
import time
import base64
import requests
from dotenv import load_dotenv
from .video_gen import _retry_request, _SUBMIT_TIMEOUT, _POLL_TIMEOUT
from ..config_loader import get_visual_backend, get_jimeng_config, get_siliconflow_config
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
    """调用 DashScope 图片生成模型生成关键帧（支持多参考图融合 + 多模型 fallback）

    Args:
        prompt: 图片生成提示词
        size: 图片尺寸
        reference_image_urls: 资产下多张参考图 URL 列表（用于角色/道具一致性融合）
        ref_strength: 参考图影响力，0.0-1.0。1.0=强参考（默认），越低越弱
        negative_prompt: 负面提示词，抑制不想要的元素
    """
    # 归一化为列表（兼容旧调用方传单字符串）
    ref_urls = reference_image_urls if isinstance(reference_image_urls, list) else (
        [reference_image_urls] if reference_image_urls else []
    )
    # 内联多张参考图为 base64/data URL（即梦与 qwen messages 均需内联）
    inline_refs = [jimeng._as_inline(u) for u in ref_urls if u]

    # ── 即梦 AI Free 后端分支（白嫖网页版每日免费积分）──
    # 当 MULTIMODIA_BACKEND == "jimeng" 时，走即梦免费生图（支持多参考图 img2img 融合）。
    if get_visual_backend() == "jimeng":
        return jimeng.generate_image(
            prompt, size=size, reference_image_urls=inline_refs,
            ref_strength=ref_strength, thread_id=thread_id,
        )

    # ── SiliconFlow 后端分支（硅基流动，图生图 img2img）──
    # 当 MULTIMODIA_BACKEND == "siliconflow" 时，用素材参考图作锚点生成本镜头专属首/尾帧。
    # 与即梦不同：SiliconFlow 直接调用 OpenAI 兼容 images/edits 接口，支持 image[] 参考条件。
    if get_visual_backend() == "siliconflow":
        return generate_keyframe_siliconflow(prompt, reference_image_urls=ref_urls, size=size, ref_strength=ref_strength, thread_id=thread_id)

    api_key = os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY"))
    session = requests.Session()
    session.trust_env = False

    # 标准化尺寸参数（wanx 模型只接受像素格式如 "1024*1024"）
    size = _normalize_size(size)

    # 构建模型链：有参考图时优先用 img2img 链（qwen-image-3.0-pro → qwen-image-3.0 → qwen-image-2.0-pro），最后 fallback 纯文生图
    if inline_refs:
        model_chain = list(_IMG2IMG_CHAIN) + list(_TEXT2IMAGE_CHAIN)
    else:
        model_chain = list(_TEXT2IMAGE_CHAIN)

    # 环境变量覆盖主模型
    env_model = os.getenv("DASHSCOPE_IMAGE_MODEL")
    if env_model:
        # 端点不能一律写死成 text2image，否则 qwen/wan2.x 系列会被打到错误端点。
        # 优先复用链中已知的端点配置，未知模型再按模型名推断。
        known = {m: e for m, e in (_IMG2IMG_CHAIN + _TEXT2IMAGE_CHAIN)}
        if env_model in known:
            env_endpoint = known[env_model]
        elif env_model.startswith("qwen-image"):
            env_endpoint = "multimodal-generation/generation"
        elif env_model.startswith("wan2.") and "-image" in env_model:
            env_endpoint = "image-generation/generation"
        else:
            env_endpoint = "text2image/image-synthesis"
        model_chain = [(env_model, env_endpoint)] + [
            (m, e) for m, e in model_chain if m != env_model
        ]

    last_error = None

    for model, endpoint_type in model_chain:
        submit_url = f"https://dashscope.aliyuncs.com/api/v1/services/aigc/{endpoint_type}"
        is_sync = endpoint_type in _SYNC_ENDPOINTS
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # 同步端点不能带 async 头，否则报 AccessDenied
        if not is_sync:
            headers["X-DashScope-Async"] = "enable"

        payload = _build_payload(model, endpoint_type, prompt, size, inline_refs, ref_strength, negative_prompt)

        print(f"    [画师提交] 尝试模型: {model} ({endpoint_type})")

        # 针对当前模型做限流退避重试：429/Throttling 是 key 级速率限制，切换模型无效，
        # 必须在原模型上等待速率窗口恢复；只有非限流的可恢复错误才切模型。
        rate_limited = False
        for rl_attempt in range(_RATE_LIMIT_RETRIES + 1):
            try:
                # 同步端点需等待出图，超时要放宽（异步端点只是提交，很快返回）
                post_timeout = 180 if is_sync else _SUBMIT_TIMEOUT

                def _do_post():
                    return session.post(submit_url, headers=headers, json=payload, timeout=post_timeout)

                response = _retry_request(_do_post, label="画师提交")

                if response.status_code != 200:
                    err_text = response.text
                    # 限流：退避重试当前模型，不切模型
                    if _is_rate_limit(err_text):
                        rate_limited = True
                        if rl_attempt < _RATE_LIMIT_RETRIES:
                            wait = _RATE_LIMIT_BACKOFF[min(rl_attempt, len(_RATE_LIMIT_BACKOFF) - 1)]
                            print(f"    [WARN] 模型 {model} 触发限流(429)，{wait}s 后重试当前模型 ({rl_attempt+1}/{_RATE_LIMIT_RETRIES})...")
                            time.sleep(wait)
                            continue
                        print(f"    [WARN] 模型 {model} 限流重试 {_RATE_LIMIT_RETRIES} 次仍失败，切换 fallback...")
                        last_error = Exception(f"画师提交限流: {err_text}")
                        break  # 跳出限流重试，切下一个模型
                    # 其他可恢复错误（额度/模型不可用）：直接切模型
                    if _is_quota_or_model_error(err_text):
                        print(f"    [WARN] 模型 {model} 不可用，切换 fallback...")
                        last_error = Exception(f"画师提交异常: {err_text}")
                        break  # 跳出限流重试，切下一个模型
                    raise Exception(f"画师提交异常: {err_text}")
                else:
                    # 成功拿到响应，跳出限流重试循环，进入后续解析
                    rate_limited = False
                    break
            except Exception as e:
                if _is_rate_limit(str(e)):
                    rate_limited = True
                    if rl_attempt < _RATE_LIMIT_RETRIES:
                        wait = _RATE_LIMIT_BACKOFF[min(rl_attempt, len(_RATE_LIMIT_BACKOFF) - 1)]
                        print(f"    [WARN] 模型 {model} 提交异常限流，{wait}s 后重试 ({rl_attempt+1}/{_RATE_LIMIT_RETRIES})...")
                        time.sleep(wait)
                        continue
                    last_error = e
                    break
                raise

        # 若因限流重试耗尽而跳出，继续下一个模型
        if rate_limited:
            continue

        # 若因其他可恢复错误跳出（last_error 已设置）且仍在本模型循环内，继续下一个模型
        if response.status_code != 200:
            continue

        try:
            resp_data = response.json()

            # 检查提交级别的错误（如额度不足）
            if "code" in resp_data and resp_data["code"] != "":
                err_msg = resp_data.get("message", str(resp_data))
                if _is_quota_or_model_error(err_msg) or _is_quota_or_model_error(str(resp_data.get("code", ""))):
                    print(f"    [WARN] 模型 {model} 额度不足或不可用，切换 fallback...")
                    last_error = Exception(f"画师提交异常: {err_msg}")
                    continue
                raise Exception(f"画师提交异常: {err_msg}")

            # ── 同步端点：响应即结果，无需轮询 ──
            if is_sync:
                image_url = _extract_image_url(resp_data.get("output", {}))
                if image_url:
                    print(f"    [画师完成] 关键帧生成成功 ({model})")
                    return image_url
                print(f"    [WARN] 模型 {model} 同步返回无图片，切换 fallback...")
                last_error = Exception(f"同步响应缺少图片: {str(resp_data)[:200]}")
                continue

            task_id = resp_data.get("output", {}).get("task_id")
            if not task_id:
                raise Exception(f"画师提交成功但无 task_id: {resp_data}")

            print(f"    [画师进度] 关键帧渲染中 ({model})，Task ID: {task_id}")

            # ── 轮询 ──
            poll_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
            poll_fail_count = 0

            while True:
                try:
                    def _do_get():
                        return session.get(poll_url, headers=headers, timeout=_POLL_TIMEOUT)

                    poll_resp = _retry_request(_do_get, max_retries=2, backoff=[2, 5], label="画师轮询")
                    poll_data = poll_resp.json()
                    poll_fail_count = 0
                except Exception as poll_err:
                    poll_fail_count += 1
                    if poll_fail_count >= 5:
                        raise Exception(f"画师轮询连续 {poll_fail_count} 次失败: {poll_err}")
                    print(f"    [WARN] [画师轮询] 网络异常 ({type(poll_err).__name__})，5s 后继续 ({poll_fail_count}/5)...")
                    time.sleep(5)
                    continue

                output = poll_data.get("output", {})
                task_status = output.get("task_status", "")

                if task_status == "SUCCEEDED":
                    url = _extract_image_url(output)
                    if url and isinstance(url, str) and url.startswith("http"):
                        print(f"    [OK] 关键帧生成成功 ({model})! URL: {url}")
                        return url
                    else:
                        raise Exception(f"画师出图成功但未能解析 URL: {poll_data}")

                elif task_status == "FAILED":
                    err_msg = str(poll_data)
                    if _is_quota_or_model_error(err_msg):
                        print(f"    [WARN] 模型 {model} 任务失败（额度/模型问题），切换 fallback...")
                        last_error = Exception(f"画师出图失败: {err_msg}")
                        break  # 跳出轮询，尝试下一个模型
                    raise Exception(f"画师出图失败: {poll_data}")

                time.sleep(3)

            # 如果是从轮询 break 出来的（fallback），继续下一个模型
            if last_error and "切换 fallback" in str(last_error):
                continue
            elif last_error:
                raise last_error

        except Exception as e:
            if _is_quota_or_model_error(str(e)):
                print(f"    [WARN] 模型 {model} 异常，切换 fallback: {e}")
                last_error = e
                continue
            raise

    # ── 即梦 Free 兜底：DashScope 全链失败后，若配置了 session 则白嫖即梦额度 ──
    # 缓解百炼免费额度紧张；即梦支持参考图 img2img（尽力维持角色一致）。
    if get_jimeng_config().get("has_session"):
        try:
            print(f"    [画师提交] DashScope 全链失败，兜底走即梦 Free 生图...")
            return jimeng.generate_image(
                prompt, size=size, reference_image_urls=inline_refs,
                ref_strength=ref_strength, thread_id=thread_id,
            )
        except Exception as je:
            print(f"    [WARN] 即梦兜底生图也失败: {je}")

    raise last_error or Exception("所有图片生成模型均不可用，请检查 API Key 和模型额度。")


def generate_keyframe_siliconflow(prompt: str, reference_image_urls: list = None,
                                  size: str = "2K", ref_strength: float = 1.0,
                                  thread_id: str = "") -> str:
    """调用 SiliconFlow（硅基流动）OpenAI 兼容 images/generations 接口，用素材参考图作锚点
    生成本镜头专属首/尾帧（img2img 图生图）。

    与即梦不同：SiliconFlow 直接 post 到 {base_url}/images/generations，模型支持 image 单图
    作为生成条件（Qwen-Image-Edit 等 image-edit 模型必须带 image 字段）。参考图被当作
    「风格/构图锚点」，而非直接当首尾帧，符合大厂视频 Agent 把参考图融入生成的做法。
    返回 base64 data URL，便于下游 Agnes keyframes 与视觉审核直接消费。

    失败抛异常，由上层 fallback/降级处理。同时落盘一份副本供排查。

    注意：SiliconFlow 是云端服务，无法 fetch 本地 URL（如 http://localhost:8900/...），
    因此所有锚点图都先在本地下载为 base64 data URL 再上传。
    """
    cfg = get_siliconflow_config()
    if not cfg["enabled"]:
        raise RuntimeError("SiliconFlow 后端未启用：未在 .env 配置 SILICONFLOW_API_KEY")

    ref_urls = reference_image_urls if isinstance(reference_image_urls, list) else (
        [reference_image_urls] if reference_image_urls else []
    )
    # 把锚点图下载为 base64 data URL（云端无法访问本地/远程 URL，必须内联上传）
    inline_refs = []
    for u in ref_urls:
        if not u:
            continue
        try:
            _bytes = _data_url_to_bytes(u)
            import mimetypes
            _mime = mimetypes.guess_type(u)[0] or "image/png"
            inline_refs.append(f"data:{_mime};base64,{base64.b64encode(_bytes).decode('ascii')}")
        except Exception as e:
            print(f"    [WARN] SiliconFlow 锚点图下载失败，跳过: {e}")

    px = _SIZE_MAP.get(size, "1024*1024")
    w, h = (int(x) for x in px.split("*"))

    edit_url = f"{cfg['base_url']}/images/generations"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }

    # 判断当前模型是否为「必须带 image 字段」的 image-edit 类模型（img2img only）。
    # 这类模型（如 Qwen/Qwen-Image-Edit）不支持纯文生图，无锚点时若仍发请求必 400。
    _edit_only_models = ("qwen-image-edit", "qwen-image-edit-", "image-edit", "kolors")
    _is_edit_only = any(m in cfg["image_model"].lower() for m in _edit_only_models)

    # SiliconFlow 图像接口（Qwen-Image-Edit 与 text2img 共用 /images/generations 端点，
    # 区别仅在于是否带 image 字段）。img2img 时把参考图作为 base64 data URL 传入 image 字段。
    # 多张参考图：以第一张作主参考（img2img 单图），其余通过 prompt 文本描述锚定风格/身份。
    _extra = ""
    if inline_refs:
        _primary = inline_refs[0]
        if len(inline_refs) > 1:
            _extra = "。参考素材还包括：" + "、".join([f"素材图{i+1}" for i in range(1, len(inline_refs))])
        data = {
            "model": cfg["image_model"],
            "prompt": prompt + _extra,
            "image": _primary,
            "size": f"{w}x{h}",
            "n": 1,
            "response_format": "b64_json",
        }
        # strength 控制对原参考图的保留程度（0~1），与即梦 ref_strength 语义一致
        if ref_strength is not None:
            data["strength"] = max(0.0, min(1.0, float(ref_strength)))
        print(f"    [画师提交] SiliconFlow img2img 生图 (模型 {cfg['image_model']}, 参考图 {len(inline_refs)} 张, strength={ref_strength}): {prompt[:60]}...")
    else:
        # 无锚点：编辑类模型（img2img only）必须带 image，纯文生图会 400 —— 抛清晰错误让上层降级，
        # 而不是发一个必败的纯文生图请求。text2img 模型（如 Kolors）仍可正常纯文生。
        if _is_edit_only:
            raise RuntimeError(
                f"SiliconFlow 当前模型 {cfg['image_model']} 为 image-edit 类（需 image 锚点），"
                f"但本次调用无可用参考图，已停止以免触发 400。建议补充 reference_sheets 锚点或改用 text2img 模型。"
            )
        data = {
            "model": cfg["image_model"],
            "prompt": prompt,
            "size": f"{w}x{h}",
            "n": 1,
            "response_format": "b64_json",
        }
        print(f"    [画师提交] SiliconFlow 纯文生图 (模型 {cfg['image_model']}, 无参考图): {prompt[:60]}...")

    # 落盘目录，便于出错排查
    out_dir = PROJECT_ROOT / "output" / "keyframes" / (thread_id or "_legacy")
    out_dir.mkdir(parents=True, exist_ok=True)

    _max_retry = 2
    for _attempt in range(_max_retry + 1):
        try:
            resp = requests.post(edit_url, headers=headers, json=data, timeout=180)
        except Exception as e:
            if _attempt < _max_retry:
                time.sleep(5 * (_attempt + 1))
                continue
            raise RuntimeError(f"SiliconFlow img2img 网络异常: {e}")
        if 200 <= resp.status_code < 300:
            try:
                j = resp.json()
                _item = (j.get("data") or [{}])[0]
                b64 = _item.get("b64_json") or _item.get("url")
                if b64 and b64.startswith("http"):
                    b64 = base64.b64encode(requests.get(b64, timeout=60).content).decode()
            except Exception:
                raise RuntimeError(f"SiliconFlow img2img 返回解析失败: {resp.text[:200]}")
            if not b64:
                raise RuntimeError(f"SiliconFlow img2img 返回未含图像: {resp.text[:200]}")
            out_path = out_dir / f"siliconflow_kf_{int(time.time() * 1000)}.png"
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(b64))
            print(f"    [OK] SiliconFlow img2img 完成: {out_path}")
            return f"data:image/png;base64,{b64}"
        # 429 / 5xx 退避重试
        if (resp.status_code == 429 or 500 <= resp.status_code < 600) and _attempt < _max_retry:
            time.sleep(5 * (_attempt + 1))
            continue
        raise RuntimeError(f"SiliconFlow img2img 返回异常 (HTTP {resp.status_code}): {resp.text[:200]}")


def _data_url_to_bytes(data_url: str) -> bytes:
    """将 data URL 或远程/本地 URL 转成原始 bytes（仅处理 data URL 与内联已下载情形）。"""
    if data_url.startswith("data:"):
        _header, _b64 = data_url.split(",", 1)
        return base64.b64decode(_b64)
    # 远程/本地文件：直接读取
    if data_url.startswith("http"):
        return requests.get(data_url, timeout=60).content
    with open(data_url, "rb") as f:
        return f.read()
