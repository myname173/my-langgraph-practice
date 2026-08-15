# src/agent/multimedia/tools/agnes.py
"""Agnes AI 免费视频生成后端。

官方：https://platform.agnes-ai.com  （完全免费、无需信用卡）
Endpoint: POST https://apihub.agnes-ai.com/v1/videos
鉴权:    Authorization: Bearer $AGNES_API_KEY
模型:    agnes-video-v2.0

能力（实测确认）：
- 文生视频: 仅传 prompt
- 图生视频: extra_body.image = [首帧]   (single image)
- 关键帧动画: extra_body.mode = "keyframes", extra_body.image = [首帧, 尾帧, ...]
- 多图参考: extra_body.image 支持多张 -> 可把「角色转面图 / 道具 / 环境参考图 /
  首尾帧关键帧」一起注入，实现素材参考图对视频生成的完整融入。

约束（实测确认）：
- 免费档速率限制：每分钟 1 次视频生成请求（429 + "allows 1 requests per 1 minute(s)"）。
- num_frames 须满足 8n+1 且 <= 441（约 5~18 秒 @24fps）。
- 尺寸会被服务端自动对齐（size_mapping.adjusted），故使用平台接受的 16:9 标准值。

本模块统一返回本地 mp4 绝对路径（与 jimeng / dashscope 分支一致），
供 graph.video_gen_node 本地化逻辑直接复用。
"""
import os
import time
import json
import base64
import requests
from pathlib import Path as _Path

from dotenv import load_dotenv

PROJECT_ROOT = _Path(__file__).resolve().parents[4]
load_dotenv()

_AGNES_BASE = os.getenv("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1").rstrip("/")
_AGNES_MODEL = os.getenv("AGNES_VIDEO_MODEL", "agnes-video-v2.0")

# 免费档限流：每分钟 1 次。提交遇 429 时按此退避串行重试。
_AGNES_RATE_LIMIT_WAIT = int(os.getenv("AGNES_RATE_LIMIT_WAIT", "65"))
_AGNES_SUBMIT_TIMEOUT = int(os.getenv("AGNES_SUBMIT_TIMEOUT", "60"))
_AGNES_POLL_TIMEOUT = int(os.getenv("AGNES_POLL_TIMEOUT", "30"))
_AGNES_POLL_MAX_WAIT = int(os.getenv("AGNES_POLL_MAX_WAIT", "600"))


class AgnesQuotaError(RuntimeError):
    """Agnes 服务端返回不可恢复错误（非限流）时抛出。"""


def _api_key() -> str:
    key = (os.getenv("AGNES_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("AGNES_API_KEY 未配置（Agnes 免费视频后端不可用）")
    return key


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }


def _resolve_image(url_or_path: str) -> str:
    """把参考图统一为 Agnes 可 fetch 的 http(s) URL。

    - 已是 http(s)：原样返回（Agnes 服务端会下载）。
    - data: URL：转 base64 内联（去掉前缀）。
    - 本地绝对路径：读文件转 base64 内联。
    - 项目自定义 /media/<rel> 前缀：还原为本地文件再内联。
    """
    if not url_or_path:
        return ""
    if url_or_path.startswith("data:image"):
        return url_or_path  # 已是内联 base64，直接可用
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        # localhost / 127.0.0.1 等回环地址：Agnes 云端无法回源到用户本机，
        # 先在同机下载为 base64 内联再发送，确保云端可解析参考图。
        _low = url_or_path.lower()
        if "localhost" in _low or "127.0.0.1" in _low:
            try:
                _r = requests.get(url_or_path, timeout=30)
                _r.raise_for_status()
                _ct = _r.headers.get("Content-Type", "image/jpeg")
                if not _ct.startswith("image/"):
                    _ct = "image/jpeg"
                _b64 = base64.b64encode(_r.content).decode("ascii")
                return f"data:{_ct};base64,{_b64}"
            except Exception as _e:
                print(f"    [WARN][Agnes] 回环地址参考图本地下载失败，跳过: {url_or_path} ({_e})")
                return ""
        return url_or_path
    if url_or_path.startswith("/media/"):
        # 本地路径（含 /media/ 还原）转 base64 内联发送
        local = str(PROJECT_ROOT / "output" / url_or_path[len("/media/"):])
        if os.path.isfile(local):
            with open(local, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            return f"data:image/png;base64,{b64}"
        print(f"    [WARN][Agnes] 参考图本地文件不存在，跳过: {local}")
        return ""
    # 其他无法识别的形式（非 data/http/回环/media）直接跳过
    print(f"    [WARN][Agnes] 参考图无法解析为可发送形式，跳过: {url_or_path}")
    return ""


def _aspect_size(width: int, height: int):
    """返回最接近的 16:9 标准尺寸（Agnes 会自动对齐，这里给推荐值减小抖动）。

    首选 1920x1080（1080p，遵循即梦/可灵 1080p 优先实践），
    回退 1280x720 / 1152x648 / 1024x576。
    """
    for w, h in [(1920, 1080), (1280, 720), (1152, 648), (1024, 576)]:
        if abs(w / h - width / height) < 0.05:
            return w, h
    return 1920, 1080


def generate_video(
    prompt: str,
    reference_images: list = None,
    first_frame_url: str = "",
    last_frame_url: str = "",
    width: int = 1920,
    height: int = 1080,
    num_frames: int = 81,
    frame_rate: int = 24,
    thread_id: str = "",
    timeout: int = 180,
) -> str:
    """调用 Agnes 免费视频生成。

    reference_images: 素材参考图列表（角色转面图 / 道具 / 环境参考图 URL 或本地路径）。
    first_frame_url / last_frame_url: 首尾帧关键帧（已带角色一致性）。
    所有图像统一走 keyframes 多图模式注入，使素材参考图完整融入视频生成。

    返回本地 mp4 绝对路径。
    """
    reference_images = reference_images or []

    # 二级防御：url 文件名含 turnaround/多视图网格关键词的图（即便上游漏过滤）
    # 一律剔除，避免网格残留在画面里。首尾帧始终保留。
    _TURNAROUND_HINTS = ("turnaround", "turntable", "three-view", "three_view",
                         "3-view", "3_view", "model_sheet", "reference_sheet",
                         "sheet", "转面", "三视图", "四视图")
    _fl_set = {first_frame_url, last_frame_url}

    def _is_turnaround(u: str) -> bool:
        # 解码 data URL 取元数据无意义，只对原始 url/path 做文件名关键词判定
        s = str(u).lower()
        if s.startswith("data:image"):
            return False  # 内联数据无法判定文件名，交由上游过滤负责
        return any(h in s for h in _TURNAROUND_HINTS)

    # 收集所有要注入的图像（首尾帧优先，其次素材参考图），去重
    img_pool = []
    for u in [first_frame_url, last_frame_url] + list(reference_images):
        if u and u not in _fl_set and _is_turnaround(u):
            # 整张多视图网格图：排除（二级防御，不注入视频生成）
            continue
        resolved = _resolve_image(u)
        if resolved and resolved not in img_pool:
            img_pool.append(resolved)

    # Agnes keyframes 模式硬性限制 2~3 张图（首尾帧 + 素材参考图共享此额度）。
    # 超过则精选：首尾帧优先保留，素材参考图按"角色/道具图优先于环境图"截断到上限，
    # 绝不超限（否则 API 报 invalid_request）。素材图在此仅作角色/场景一致性锚点，
    # 不当作首尾帧——首尾帧应是镜头专属构图（由生图模型产出），本环境即梦额度耗尽时留空。
    _AGNES_MAX_IMAGES = 3
    if len(img_pool) > _AGNES_MAX_IMAGES:
        print(f"    [Agnes] 注入图 {len(img_pool)} 张超过上限 {_AGNES_MAX_IMAGES}，"
              f"精选首尾帧优先 + 素材参考图截断")
        _fl = [x for x in img_pool if x in (_resolve_image(first_frame_url), _resolve_image(last_frame_url)) and x]
        _ref = [x for x in img_pool if x not in _fl]
        img_pool = (_fl + _ref)[:_AGNES_MAX_IMAGES]

    # keyframes 模式需要 2~3 张；不足 2 张（纯文生场景）则走无图 t2v。
    use_keyframes = 2 <= len(img_pool) <= 3

    w, h = _aspect_size(width, height)
    # 满足 8n+1
    if (num_frames - 1) % 8 != 0:
        num_frames = ((num_frames // 8) * 8) + 1
    num_frames = max(9, min(num_frames, 441))

    body = {
        "model": _AGNES_MODEL,
        "prompt": prompt,
        "width": w,
        "height": h,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
    }
    if use_keyframes:
        body["extra_body"] = {
            "mode": "keyframes",
            "image": img_pool,
        }
        print(f"    [Agnes] keyframes 多图模式，注入 {len(img_pool)} 张参考图"
              f"（首尾帧 + 素材参考图，上限 {_AGNES_MAX_IMAGES}）")
    else:
        print(f"    [Agnes] 注入图仅 {len(img_pool)} 张(<2)，走纯文生视频(t2v)")

    session = requests.Session()
    session.trust_env = False

    def _do_post():
        return session.post(
            f"{_AGNES_BASE}/videos",
            headers=_headers(),
            json=body,
            timeout=_AGNES_SUBMIT_TIMEOUT,
        )

    # 提交（含 429 限流退避重试）
    resp = None
    for attempt in range(4):
        try:
            r = _do_post()
        except requests.exceptions.RequestException as e:
            print(f"    [WARN][Agnes] 提交网络错误，3s 后重试: {e}")
            time.sleep(3)
            continue
        if r.status_code == 429:
            wait = _AGNES_RATE_LIMIT_WAIT
            print(f"    [WARN][Agnes] 限流 429，退避 {wait}s（免费档每分钟 1 次）...")
            time.sleep(wait)
            continue
        resp = r
        break

    if resp is None:
        raise RuntimeError("Agnes 视频提交失败：连续限流/网络错误")
    if resp.status_code != 200:
        raise AgnesQuotaError(
            f"Agnes 视频提交失败 (HTTP {resp.status_code}): {resp.text[:300]}"
        )

    data = resp.json()
    task_id = data.get("task_id") or (data.get("data", {}) or {}).get("task_id")
    if not task_id:
        raise AgnesQuotaError(f"Agnes 视频无 task_id: {str(data)[:200]}")

    print(f"    [Agnes] 任务已提交 task_id={task_id}，等待生成（免费档较慢）...")

    # 轮询
    out_dir = PROJECT_ROOT / "output" / "clips" / (thread_id or "_legacy")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"agnes_vid_{int(time.time() * 1000)}.mp4"

    deadline = time.monotonic() + _AGNES_POLL_MAX_WAIT
    while time.monotonic() < deadline:
        time.sleep(10)
        try:
            pr = session.get(
                f"{_AGNES_BASE}/videos/{task_id}",
                headers=_headers(),
                timeout=_AGNES_POLL_TIMEOUT,
            ).json()
        except requests.exceptions.RequestException as e:
            print(f"    [WARN][Agnes] 轮询网络错误，重试: {e}")
            continue
        status = pr.get("status") or (pr.get("data", {}) or {}).get("status")
        if status == "completed":
            # 视频地址真实位置（实测确认）：
            #   - 新结构：pr["metadata"]["url"]
            #   - 旧结构：pr["remixed_from_video_id"] / pr["data"]["remixed_from_video_id"]
            # 兼容两者，优先 metadata.url。
            _meta = pr.get("metadata") or {}
            vid_url = (
                _meta.get("url")
                or pr.get("remixed_from_video_id")
                or (pr.get("data", {}) or {}).get("remixed_from_video_id")
                or _meta.get("video_url")
            )
            if not vid_url:
                raise AgnesQuotaError(f"Agnes 完成但无视频 URL: {str(pr)[:200]}")
            vr = session.get(vid_url, timeout=timeout)
            vr.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(vr.content)
            print(f"    [OK][Agnes] 视频完成: {out_path}")
            return str(out_path)
        if status == "failed":
            raise AgnesQuotaError(f"Agnes 视频任务失败: {str(pr)[:300]}")
        # queued / running / 其他：继续等
    raise AgnesQuotaError(f"Agnes 视频轮询超时（>{_AGNES_POLL_MAX_WAIT}s）")
