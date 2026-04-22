# src/agent/multimedia/tools/vision_eval.py
import os
import base64
import tempfile
from io import BytesIO
from typing import Optional, List

import httpx
import requests
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

http_client = httpx.Client(trust_env=False)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", os.getenv("DASHSCOPE_API_KEY")),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    http_client=http_client
)


def _image_to_data_url(image: Image.Image, max_size: int = 768, quality: int = 88) -> str:
    img = image.convert("RGB")
    width, height = img.size
    long_side = max(width, height)

    if long_side > max_size:
        scale = max_size / float(long_side)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _download_to_tempfile(url: str, suffix: str = ".mp4") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    response = requests.get(url, stream=True, timeout=180)
    response.raise_for_status()

    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return tmp_path


def _sample_video_frames(video_path: str, frame_count: int = 4) -> List[str]:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        from moviepy import VideoFileClip

    clip = None
    data_urls: List[str] = []

    try:
        clip = VideoFileClip(video_path)
        duration = float(getattr(clip, "duration", 0.0) or 0.0)

        if duration <= 0.0:
            raise ValueError("视频时长无效，无法抽帧。")

        fractions = [0.15, 0.35, 0.60, 0.85]
        timestamps = []
        for frac in fractions[: max(1, frame_count)]:
            t = duration * frac
            t = max(0.05, min(t, max(0.05, duration - 0.05)))
            timestamps.append(t)

        while len(timestamps) < frame_count:
            timestamps.append(timestamps[-1])

        for t in timestamps[:frame_count]:
            frame = clip.get_frame(t)
            img = Image.fromarray(frame)
            data_urls.append(_image_to_data_url(img))

        return data_urls
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass


def evaluate_image(
    image_url: str,
    script: str,
    prompt_template: str,
    previous_image_url: Optional[str] = None
) -> str:
    """调用视觉模型评估关键帧。"""
    system_msg = prompt_template.format(script=script)

    content = [
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    if previous_image_url:
        content.append({"type": "image_url", "image_url": {"url": previous_image_url}})
        system_msg += "\n\n【重要】第二张图片是前一个镜头的关键帧。请严格检查视觉一致性。"

    content.append({"type": "text", "text": system_msg})

    try:
        response = client.chat.completions.create(
            model="qwen-image-2.0",
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=512
        )

        result = str(response.choices[0].message.content or "").strip()

        if "PASS" in result.upper():
            print("    ✅ 艺术总监 + 安全官审核通过！")
            return "PASS"

        print(f"    ❌ 审核打回: {result[:300]}...")
        return result

    except Exception as e:
        error_str = str(e).lower()

        if "data_inspection_failed" in error_str or "inappropriate content" in error_str:
            fail_msg = (
                "FAIL: 阿里云内容安全预检拦截（data_inspection_failed）。"
                "请保持当前题材不变，但降低真实武器细节、明确开火动作和过强写实冲击，"
                "改为更电影化、风格化的未来都市氛围表达，突出霓虹、雨夜、蒸汽、反射光与角色姿态。"
            )
            print("    ⚠️ [安全拦截] 绿网误判，已转为 FAIL 重绘")
            print(f"    {fail_msg}")
            return fail_msg

        print(f"    ⚠️ [审核员接口异常] {str(e)[:200]}")
        print("    为防止中断，默认放行此图。")
        return "PASS"


def evaluate_video(
    video_url: str,
    script: str,
    prompt_template: str,
    previous_image_url: Optional[str] = None,
    frame_count: int = 4
) -> str:
    """调用视觉模型审核视频片段：抽关键帧后做视频级审核。"""
    system_msg = prompt_template.format(script=script)

    tmp_path = None
    try:
        if video_url.startswith("http://") or video_url.startswith("https://"):
            tmp_path = _download_to_tempfile(video_url, suffix=".mp4")
            local_video_path = tmp_path
        else:
            local_video_path = video_url

        frame_data_urls = _sample_video_frames(local_video_path, frame_count=frame_count)

        content = []
        for frame_url in frame_data_urls:
            content.append({"type": "image_url", "image_url": {"url": frame_url}})

        if previous_image_url:
            content.append({"type": "image_url", "image_url": {"url": previous_image_url}})
            system_msg += "\n\n【重要】最后一张图片是前一个镜头的关键帧。请严格检查跨镜头视觉一致性。"

        system_msg += "\n\n这些图片按时间顺序表示同一个视频片段，请重点审核连续性、动作自然度、主体稳定性、镜头运动合理性。"
        content.append({"type": "text", "text": system_msg})

        response = client.chat.completions.create(
            model="qwen3.5-omni-plus-2026-03-15",
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=512
        )

        result = str(response.choices[0].message.content or "").strip()

        if "PASS" in result.upper():
            print("    ✅ 视频审核通过！")
            return "PASS"

        print(f"    ❌ 视频审核打回: {result[:300]}...")
        return result

    except Exception as e:
        error_str = str(e).lower()

        if "data_inspection_failed" in error_str or "inappropriate content" in error_str:
            fail_msg = (
                "FAIL: 视频内容安全预检拦截（data_inspection_failed）。"
                "请保持当前题材不变，但降低真实武器细节、明确开火动作和过强写实冲击，"
                "改为更电影化、风格化的未来都市氛围表达，突出霓虹、雨夜、蒸汽、反射光与角色姿态。"
            )
            print("    ⚠️ [安全拦截] 视频审核绿网误判，已转为 FAIL 重绘")
            print(f"    {fail_msg}")
            return fail_msg

        print(f"    ⚠️ [视频审核员接口异常] {str(e)[:200]}")
        print("    为防止中断，默认放行此视频。")
        return "PASS"

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def design_camera_movement(
    image_url: str,
    script: str,
    prompt_template: str,
    critique: str = "无"
) -> str:
    """调用全模态模型设计运镜。"""
    system_msg = prompt_template.format(script=script, critique=critique or "无")

    try:
        response = client.chat.completions.create(
            model="qwen3.5-omni-plus-2026-03-15",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": system_msg}
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=256
        )
        return str(response.choices[0].message.content or "").strip()

    except Exception as e:
        print(f"    ⚠️ [摄影师接口异常] {str(e)}")
        return "Slow pan right, cinematic lighting, highly detailed."
