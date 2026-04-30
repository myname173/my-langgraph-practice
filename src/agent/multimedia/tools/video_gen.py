# src/agent/multimedia/tools/video_gen.py
import os
import time
import json
import requests
from dotenv import load_dotenv

load_dotenv()

class FreeTierQuotaExhaustedError(RuntimeError):
    pass

class VideoSchemaMismatchError(RuntimeError):
    pass

def _contains_free_tier_quota_error(payload) -> bool:
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except Exception:
        text = str(payload)
    return "AllocationQuota.FreeTierOnly" in text

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
    resolution: str = "1080P",
    duration: int = 5,
    prompt_extend: bool = True,
    watermark: bool = False,
    shot_type: str = "single",
) -> dict:
    return {
        "resolution": resolution,
        "duration": duration,
        "prompt_extend": prompt_extend,
        "watermark": watermark,
        "shot_type": shot_type,
    }

def _build_payload_media_frames(model_name: str, first_frame_url: str, last_frame_url: str, prompt: str) -> dict:
    """
    支持单首帧或首尾双帧的 media 结构
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
        "parameters": _build_parameters(),
    }

def _build_payload_img_url(model_name: str, image_url: str, prompt: str) -> dict:
    return {
        "model": model_name,
        "input": {
            "prompt": prompt,
            "img_url": image_url,
        },
        "parameters": _build_parameters(),
    }

def _submit_task(session: requests.Session, submit_url: str, headers: dict, payload: dict) -> str:
    response = session.post(submit_url, headers=headers, json=payload)

    try:
        response_payload = response.json()
    except Exception:
        response_payload = response.text

    if response.status_code != 200:
        if _contains_free_tier_quota_error(response_payload):
            raise FreeTierQuotaExhaustedError(
                f"视频生成被服务端拒绝：Free Quota Only 已开启且免费额度已耗尽。原始错误：{_extract_error_message(response_payload)}"
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

def _poll_task(session: requests.Session, poll_url: str, headers: dict) -> str:
    while True:
        poll_resp = session.get(poll_url, headers=headers)
        poll_data = poll_resp.json()
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
            if _contains_free_tier_quota_error(poll_data):
                raise FreeTierQuotaExhaustedError(
                    f"视频生成任务失败：Free Quota Only 已开启且免费额度已耗尽。原始错误：{_extract_error_message(poll_data)}"
                )
            if _contains_schema_error(poll_data):
                raise VideoSchemaMismatchError(
                    f"视频生成任务失败：请求 schema 与服务端不匹配。原始错误：{_extract_error_message(poll_data)}"
                )
            raise Exception(f"视频生成失败: {poll_data}")

        print("    [视频渲染进度] 视频逐帧生成中，请耐心等待 (约5秒/次)...")
        time.sleep(5)

def generate_video_from_image(image_url: str, prompt: str, last_image_url: str = "") -> str:
    """
    调用通义万相图生视频模型。
    支持传入 last_image_url 开启首尾帧双控模式。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY"))
    model_name = os.getenv("DASHSCOPE_VIDEO_MODEL", "wan2.7-i2v")
    session = requests.Session()
    session.trust_env = False

    submit_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"

    headers = {
        "X-DashScope-Async": "enable",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload_candidates =[
        ("media.frames", _build_payload_media_frames(model_name, image_url, last_image_url, prompt)),
        ("img_url", _build_payload_img_url(model_name, image_url, prompt)),
    ]

    last_error = None

    for attempt_name, payload in payload_candidates:
        print(f"    [视频渲染提交] 尝试 schema: {attempt_name}")

        try:
            task_id = _submit_task(session, submit_url, headers, payload)
            print(f"    [视频渲染进度] 正在生成动态宣传片，Task ID: {task_id}")

            poll_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
            return _poll_task(session, poll_url, headers)

        except FreeTierQuotaExhaustedError:
            raise

        except VideoSchemaMismatchError as e:
            last_error = e
            print(f"    ⚠️ schema 不匹配，准备切换备用入参结构: {e}")
            continue

        except Exception as e:
            last_error = e
            err_text = str(e)
            if _contains_schema_error(err_text):
                print(f"    ⚠️ 捕获到 schema 类错误，准备切换备用入参结构: {err_text}")
                continue
            raise

    raise last_error or Exception("视频生成失败：两个兼容 schema 都未成功。")
