"""智谱 AI (Zhipu / BigModel) 免费视频生成后端。

免费模型：cogvideox-flash（model 级免费，无需信用卡，速率限制而非按金额计费）。
支持文生视频 / 图生视频（image_url，可传两张作首尾帧）。

接口（异步）：
  提交：POST https://open.bigmodel.cn/api/paas/v4/videos/generations
        body: { model, prompt, image_url, size, duration, with_audio, watermark_enabled, quality }
        -> 200: { id, task_status }
  轮询：GET  https://open.bigmodel.cn/api/paas/v4/async-result/{id}
        -> 200: { video_result: [ { url, cover_image_url } ] }

与 agnes.generate_video 保持相同签名，返回本地 mp4 绝对路径。
注意：CogVideoX 仅支持首尾帧（最多 2 张 image_url），不支持多参考图注入，
因此 reference_images 在本后端被忽略（仅首尾帧参与图生视频）。
"""
import os
import time
import base64
import mimetypes
import requests

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_ZHIPU_BASE = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
_ZHIPU_MODEL = os.getenv("ZHIPU_VIDEO_MODEL", "cogvideox-flash").strip()

_SUBMIT_TIMEOUT = int(os.getenv("ZHIPU_SUBMIT_TIMEOUT", "60"))
_POLL_TIMEOUT = int(os.getenv("ZHIPU_POLL_TIMEOUT", "60"))
_POLL_INTERVAL = int(os.getenv("ZHIPU_POLL_INTERVAL", "10"))
_POLL_MAX_WAIT = int(os.getenv("ZHIPU_POLL_MAX_WAIT", "600"))


class ZhipuQuotaError(RuntimeError):
    """智谱额度/限流/任务失败等可识别错误。"""
    pass


def _resolve_image(url_or_path: str) -> str:
    """把图像统一成可发送形式：http(s)/data: 原样返回；本地路径内联为 data URL。

    CogVideoX 仅支持 http(s) URL 或 base64 内联（≤5M，png/jpeg）。
    """
    if not url_or_path:
        return ""
    if url_or_path.startswith(("http://", "https://", "data:")):
        return url_or_path
    # 本地绝对路径 -> 内联 base64
    if os.path.isabs(url_or_path) and os.path.isfile(url_or_path):
        try:
            mime = mimetypes.guess_type(url_or_path)[0] or "image/png"
            with open(url_or_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            return f"data:{mime};base64,{b64}"
        except Exception as e:
            print(f"    [WARN][Zhipu] 首帧本地图转内联失败，跳过: {e}")
            return ""
    print(f"    [WARN][Zhipu] 无法解析的图像形式，跳过: {url_or_path}")
    return ""


def generate_video(
    prompt: str,
    reference_images: list = None,
    first_frame_url: str = "",
    last_frame_url: str = "",
    width: int = 1280,
    height: int = 720,
    num_frames: int = 81,
    frame_rate: int = 24,
    thread_id: str = "",
    timeout: int = 180,
) -> str:
    """调用智谱免费视频生成（cogvideox-flash），图生视频。

    仅首尾帧参与图生视频（最多 2 张 image_url）；reference_images 被忽略。
    返回本地 mp4 绝对路径。
    """
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise ZhipuQuotaError(
            "未配置 ZHIPU_API_KEY：请到 https://open.bigmodel.cn 注册并获取免费 API Key，"
            "写入 .env 的 ZHIPU_API_KEY。"
        )

    # 构造图生视频图像列表（首尾帧，最多 2 张）
    images = []
    ff = _resolve_image(first_frame_url)
    lf = _resolve_image(last_frame_url) if last_frame_url else ""
    if ff:
        images.append(ff)
    if lf and lf != ff:
        images.append(lf)

    # 尺寸对齐 16:9 标准（CogVideoX 用 size 字段，如 1280x720）
    size = "1280x720"
    for w, h in [(1920, 1080), (1280, 720), (1024, 576)]:
        if abs(w / h - (width / height if height else 16 / 9)) < 0.05:
            size = f"{w}x{h}"
            break

    # 时长：cogvideox-flash 支持 5 或 10，默认 5
    duration = int(os.getenv("ZHIPU_VIDEO_DURATION", "5"))
    if duration not in (5, 10):
        duration = 5

    body: dict = {
        "model": _ZHIPU_MODEL,
        "prompt": prompt,
        "size": size,
        "duration": duration,
        "fps": int(os.getenv("ZHIPU_VIDEO_FPS", "30")),
        "quality": os.getenv("ZHIPU_VIDEO_QUALITY", "speed"),  # 免费档默认 speed 更快
        "with_audio": os.getenv("ZHIPU_VIDEO_WITH_AUDIO", "false").lower() == "true",
        "watermark_enabled": os.getenv("ZHIPU_VIDEO_WATERMARK", "false").lower() == "true",
    }
    if images:
        # 智谱 image_url 可为字符串或数组；首尾帧传数组（首帧 + 尾帧）
        body["image_url"] = images if len(images) > 1 else images[0]
        print(f"    [Zhipu] 图生视频，注入 {len(images)} 张首尾帧 ({size}, {duration}s)")
    else:
        print(f"    [Zhipu] 无首尾帧，走纯文生视频 ({size}, {duration}s)")

    session = requests.Session()
    session.trust_env = False
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 提交（含 429 退避重试）
    resp = None
    for attempt in range(4):
        try:
            r = session.post(
                f"{_ZHIPU_BASE}/videos/generations",
                headers=headers,
                json=body,
                timeout=_SUBMIT_TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            print(f"    [WARN][Zhipu] 提交网络错误，3s 后重试: {e}")
            time.sleep(3)
            continue
        if r.status_code == 429:
            print(f"    [WARN][Zhipu] 限流 429，退避 {_POLL_INTERVAL}s（免费档速率限制）...")
            time.sleep(_POLL_INTERVAL)
            continue
        resp = r
        break

    if resp is None:
        raise ZhipuQuotaError("智谱视频提交失败：连续限流/网络错误")
    if resp.status_code != 200:
        # 余额/模型不可用等可识别错误统一抛 ZhipuQuotaError 触发上层 fallback
        raise ZhipuQuotaError(
            f"智谱视频提交失败 (HTTP {resp.status_code}): {resp.text[:300]}"
        )

    data = resp.json()
    task_id = data.get("id") or (data.get("data", {}) or {}).get("id")
    if not task_id:
        raise ZhipuQuotaError(f"智谱视频无任务 id: {str(data)[:200]}")

    print(f"    [Zhipu] 任务已提交 id={task_id}，等待生成（免费档较慢）...")

    # 轮询查询异步结果
    out_dir = PROJECT_ROOT / "output" / "clips" / (thread_id or "_generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"zhipu_vid_{int(time.time() * 1000)}.mp4"

    deadline = time.monotonic() + _POLL_MAX_WAIT
    while time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL)
        try:
            pr = session.get(
                f"{_ZHIPU_BASE}/async-result/{task_id}",
                headers=headers,
                timeout=_POLL_TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            print(f"    [WARN][Zhipu] 轮询网络错误，重试: {e}")
            continue
        if pr.status_code != 200:
            # 轮询接口偶发错误，继续等待
            print(f"    [WARN][Zhipu] 轮询 HTTP {pr.status_code}，继续等待...")
            continue
        presult = pr.json()
        # 兼容两种结构：顶层 video_result 或 data.video_result
        vres = presult.get("video_result") or (presult.get("data", {}) or {}).get("video_result")
        task_status = (
            presult.get("task_status")
            or (presult.get("data", {}) or {}).get("task_status")
            or ""
        )
        if vres and isinstance(vres, list) and vres:
            vid_url = vres[0].get("url") or (vres[0].get("cover_image_url") or "")
            if not vid_url:
                raise ZhipuQuotaError(f"智谱完成但无视频 URL: {str(presult)[:200]}")
            vr = session.get(vid_url, timeout=timeout)
            vr.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(vr.content)
            print(f"    [OK][Zhipu] 视频完成: {out_path}")
            return str(out_path)
        if task_status == "FAIL":
            raise ZhipuQuotaError(f"智谱视频任务失败: {str(presult)[:300]}")
        # PROCESSING / 其他：继续等
    raise ZhipuQuotaError(f"智谱视频轮询超时（>{_POLL_MAX_WAIT}s）")
