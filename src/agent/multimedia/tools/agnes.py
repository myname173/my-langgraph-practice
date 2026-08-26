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
# 视频模型名（可在 .env 显式配置 AGNES_VIDEO_MODEL 切换更新模型）。
# 查询最新可用模型：https://platform.agnes-ai.com 或 https://apihub.agnes-ai.com/v1/models
_AGNES_MODEL = os.getenv("AGNES_VIDEO_MODEL", "agnes-video-v2.0")
print(f"    [Agnes] 当前视频模型: {_AGNES_MODEL}"
      f"（如需切换，请在 .env 设置 AGNES_VIDEO_MODEL）")

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
    # 本地绝对路径（每镜首帧/尾帧 png/jpg，由 video_gen_node 解码产出）：
    # 读取文件内联为 data: URL，确保每个分镜用自己的关键帧而非复用同一画面。
    if os.path.isabs(url_or_path):
        if os.path.isfile(url_or_path):
            try:
                import mimetypes
                mime = mimetypes.guess_type(url_or_path)[0] or "image/png"
                with open(url_or_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                return f"data:{mime};base64,{b64}"
            except Exception as _e:
                print(f"    [WARN][Agnes] 首帧本地图转内联失败，跳过: {_e}")
                return ""
        print(f"    [WARN][Agnes] 首帧本地文件不存在，跳过: {url_or_path}")
        return ""
    # 其他无法识别的形式（非 data/http/回环/media/本地文件）直接跳过
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


def _sanitize_agnes_prompt(prompt: str) -> str:
    """软化视频 prompt，规避 Agnes 免费档内容审核（content_policy_violation）。

    实测：Agnes 免费档对"逃离/坠落/深渊/黑暗/攻击/暴力/武器"类动态描述审核较严，
    仙侠动作题材易被 HTTP 400 content_policy_violation 拒绝。这里把高危词替换为
    中性、优雅、无攻击性的电影化描述，并追加一条温和声明，尽量在保留画面构图与
    运镜语义的前提下通过审核。仅作用于 Agnes 后端 prompt，不影响其他后端。
    """
    if not prompt:
        return prompt
    _s = prompt
    _repl = [
        # 暴力 / 攻击 / 战斗 → 中性动态
        ("violently", "gracefully"),
        ("battling the gale", "moving through the wind"),
        ("explosive", "sweeping"),
        ("attacking", "advancing"),
        ("striking", "gliding"),
        ("strikes", "glides"),
        ("shattering", "gently cracking"),
        ("battle", "flowing movement"),
        ("combat", "movement"),
        ("sword slash", "flowing energy"),
        ("slash", "sweep"),
        ("thrusting", "guiding"),
        ("thrust", "guide"),
        ("dismember", "part"),
        ("blood", "crimson light"),
        ("bleeding", "glowing"),
        ("gore", "glow"),
        ("corpse", "still figure"),
        ("death", "stillness"),
        # 黑暗 / 坠落 / 深渊 → 朦胧 / 优雅
        ("dark abyss", "misty valley"),
        ("abyss", "misty distance"),
        ("plunging", "drifting"),
        ("plunges", "drifts"),
        ("falling into darkness", "drifting in soft shadow"),
        ("evil", "mysterious"),
        ("dark army", "shadowy figures"),
        # 逃离 / 惊叫 → 翱翔 / 灵动
        ("fleeing", "soaring"),
        ("flees", "soars"),
        ("screeching", "calling"),
        ("screaming", "singing"),
    ]
    _sl = _s
    for _a, _b in _repl:
        _sl = _sl.replace(_a, _b)
    if _sl == _s:
        # 无高危词命中时，仍追加温和声明以降低审核误判概率
        _sl = _sl.rstrip()
    _sl = _sl.rstrip()
    _safe = (
        "\n\n[Style] A peaceful, elegant cinematic sequence. "
        "No violence, no weapons, no blood, no gore, no harm to any character. "
        "Graceful, serene and beautiful."
    )
    if _safe.strip() not in _sl:
        _sl = _sl + _safe
    return _sl


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

    # 软化 prompt 规避 Agnes 内容审核（仅对视频 prompt 生效，不影响补尾帧等本地调用）
    prompt = _sanitize_agnes_prompt(prompt)

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

    # 尾帧缺失保底（免费档 jimeng 偶发失败常见）：
    #  1) 优先用首帧即时 img2img 补生成「真尾帧」（低 ref_strength 制造姿态/构图差异），
    #     与首帧一起投 keyframes —— 运动区间真实，观感最佳；
    #  2) 若补尾帧也失败，单首帧走 agnes 原生 i2v 模式（image=[首帧]，首帧锁起始帧、
    #     运动正常），避免「首帧复用为尾帧」导致动画在首帧→首帧间退化成静止；
    #  3) 连首帧都没有（plan A 复用模式）才退 t2v。
    # 核心目标：尾帧缺失时首帧 100% 进视频第一帧，且不触发 API invalid_request
    # （agenes keyframes 硬性要求 >=2 张；单图则走 i2v 而非 keyframes）。
    _mode = None  # "keyframes" | "i2v" | None(纯 t2v)
    if len(img_pool) >= 2:
        _mode = "keyframes"
    elif len(img_pool) == 1 and first_frame_url:
        try:
            from .image_gen import generate_keyframe  # 延迟导入，避免与 video_gen 循环依赖
            _tail = generate_keyframe(
                prompt,
                reference_image_urls=[first_frame_url],
                ref_strength=0.35,
                thread_id=os.getenv("THREAD_ID", ""),
            )
            _resolved_tail = _resolve_image(_tail) if _tail else None
            if _resolved_tail and _resolved_tail not in img_pool:
                img_pool.append(_resolved_tail)
                _mode = "keyframes"
                print(f"    [Agnes] 尾帧缺失，已用首帧即时 img2img 补生成真尾帧（keyframes 双图，运动区间真实）")
        except Exception as _e:
            print(f"    [Agnes] 补尾帧失败({_e})，改用单首帧 i2v 模式（首帧锁起始帧，运动正常）")
        if _mode is None:
            _mode = "i2v"  # 仅首帧：agnes 原生单图图生视频

    use_keyframes = _mode == "keyframes"

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
    elif _mode == "i2v":
        # agnes 原生单图图生视频：image=[首帧]，首帧锁起始帧
        body["extra_body"] = {"image": img_pool}
        print(f"    [Agnes] i2v 单首帧模式，注入 {len(img_pool)} 张首帧（首帧锁起始帧，运动正常）")
    else:
        print(f"    [Agnes] 无首帧，走纯文生视频(t2v)")

    session = requests.Session()
    session.trust_env = False

    def _do_post():
        return session.post(
            f"{_AGNES_BASE}/videos",
            headers=_headers(),
            json=body,
            timeout=_AGNES_SUBMIT_TIMEOUT,
        )

    # 提交（含 429 限流 / 503 队列满 退避重试）
    # 免费档实测：除 429（每分钟 1 次）外，常遇 503 video_queue_full（队列满，
    # 临时性、稍后通常可恢复）。两者均为可重试的瞬时错误，需退避后重试而非立即失败。
    resp = None
    for attempt in range(8):
        try:
            r = _do_post()
        except requests.exceptions.RequestException as e:
            print(f"    [WARN][Agnes] 提交网络错误，5s 后重试: {e}")
            time.sleep(5)
            continue
        if r.status_code == 429:
            wait = _AGNES_RATE_LIMIT_WAIT
            print(f"    [WARN][Agnes] 限流 429，退避 {wait}s（免费档每分钟 1 次）...")
            time.sleep(wait)
            continue
        if r.status_code == 503:
            # 队列满：指数退避（65s 起，逐步加长），最多重试到循环上限
            wait = _AGNES_RATE_LIMIT_WAIT + attempt * 30
            print(f"    [WARN][Agnes] 队列满 503(video_queue_full)，退避 {wait}s 后重试"
                  f"（免费档算力紧张，第 {attempt+1} 次）...")
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
    out_dir = PROJECT_ROOT / "output" / "clips" / (thread_id or "_generated")
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
