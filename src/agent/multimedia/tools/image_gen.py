# src/agent/multimedia/tools/image_gen.py
import os
import time
import requests
from dotenv import load_dotenv
from .video_gen import _retry_request, _SUBMIT_TIMEOUT, _POLL_TIMEOUT

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
                   reference_image_url: str = None, ref_strength: float = 1.0,
                   negative_prompt: str = "") -> dict:
    """根据模型类型构建不同的 payload。"""

    # ── qwen-image 系列：同步 multimodal messages 格式 ──
    # 与 wan2.x-image 的 messages 结构一致，但不支持 enable_interleave /
    # ref_strength / negative_prompt 等 wan 专有参数，多传会被拒。
    if model.startswith("qwen-image"):
        content_list = []
        if reference_image_url:
            content_list.append({"image": reference_image_url})
        content_list.append({"text": prompt})
        return {
            "model": model,
            "input": {
                "messages": [
                    {"role": "user", "content": content_list}
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
        content_list = []
        if reference_image_url:
            content_list.append({"image": reference_image_url})
        content_list.append({"text": prompt})
        payload = {
            "model": model,
            "input": {
                "messages": [
                    {"role": "user", "content": content_list}
                ]
            },
            "parameters": {
                "size": size,
                "n": 1,
                "watermark": False,
            }
        }
        if reference_image_url:
            payload["parameters"]["enable_interleave"] = False
        if reference_image_url and ref_strength < 1.0:
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
        if reference_image_url:
            payload["input"]["ref_image"] = reference_image_url
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


def generate_keyframe(prompt: str, size: str = "2K", reference_image_url: str = None,
                      ref_strength: float = 1.0, negative_prompt: str = "") -> str:
    """调用 DashScope 图片生成模型生成关键帧（支持参考图 + 多模型 fallback）

    Args:
        prompt: 图片生成提示词
        size: 图片尺寸
        reference_image_url: 参考图 URL（用于风格/角色一致性）
        ref_strength: 参考图影响力，0.0-1.0。1.0=强参考（默认），越低越弱
        negative_prompt: 负面提示词，抑制不想要的元素
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY"))
    session = requests.Session()
    session.trust_env = False

    # 标准化尺寸参数（wanx 模型只接受像素格式如 "1024*1024"）
    size = _normalize_size(size)

    # 构建模型链：有参考图时优先用 img2img 链（qwen-image-3.0-pro → qwen-image-3.0 → qwen-image-2.0-pro），最后 fallback 纯文生图
    if reference_image_url:
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

        payload = _build_payload(model, endpoint_type, prompt, size, reference_image_url, ref_strength, negative_prompt)

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

    raise last_error or Exception("所有图片生成模型均不可用，请检查 API Key 和模型额度。")
