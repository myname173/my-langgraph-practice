# src/agent/multimedia/tools/jimeng.py
"""
即梦 AI Free（jimeng-api）后端封装 — 白嫖即梦网页版每日免费积分（默认 ~66 积分/天）。


背景：
  用户希望摆脱百炼免费额度少、API 格式多变（各模型端点/payload 不一致）的麻烦。
  GitHub 上 `hedashuaiii/jimeng-api`（fork 自 `DylanWu92/jimeng-free-api-all`）
  把即梦官网每日赠送的免费积分暴露为 OpenAI 兼容的本地 API：
    - 生图：POST {base}/v1/images/generations
    - 视频：POST {base}/v1/videos/generations
  支持文生图、图生图（1-10 张）、文生视频、图生视频，以及 Seedance 2.0 多参考图。

    接入方式（用户侧）：
  1) 部署 jimeng-api：
       docker run -it -d --name jimeng-free-api-all -p 8090:8000 \
         -e TZ=Asia/Shanghai vinlic/jimeng-free-api:latest
     用网页版 sessionid 作为 Authorization: Bearer <sessionid> 鉴权
     （多个 sessionid 逗号分隔，轮换降低单号封禁风险）。
  2) .env 配置：
       JIMENG_BASE_URL=http://localhost:8090/v1
       JIMENG_SESSION_ID=sessionid1,sessionid2
  注意：仅个人自用；逆向 API 随官网改版可能失效，且免费额度无角色一致性硬锁定，
        需靠多参考图 / 尾帧衔接尽量维持。本项目将其作为百炼免费额度的补充兜底后端。

本模块只负责封装调用，不发起配置读取（由 config_loader.get_jimeng_config 提供）。
"""
from __future__ import annotations

import base64
import itertools
import os
import time
import requests
from pathlib import Path
from urllib.parse import quote

# 项目根目录：jimeng.py -> tools -> multimedia -> agent -> src -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# 注意：不要从 video_gen 导入，避免循环依赖（video_gen 已 import 本模块）。
# 这里自带轻量网络重试，复用项目既有的退避风格。
def _retry_request(fn, max_retries: int = 3, backoff: list = None, label: str = "", _check_body: bool = False):
    """fn 返回 requests.Response。_check_body=True 时，HTTP 200 但响应体含错误码也视为
    失败并重试（即梦代理常把「积分不足/权益不足」「队列满」等以 HTTP 200 + code=-2001/-1000
    返回，需当作可重试错误而非成功）。"""
    backoff = backoff or [2, 5, 10]
    last = None
    for i in range(max_retries + 1):
        try:
            resp = fn()
            if _check_body and resp.status_code == 200:
                # 解析响应体判断是否为代理错误（即梦常把「积分不足/权益不足」以
                # HTTP 200 + code=-2001/-1000 返回）。仅此解析容错，错误判定必须在 try 之外抛出，
                # 否则会被下面的 except 吞掉。
                _d = None
                try:
                    _d = resp.json()
                except Exception:
                    pass
                if isinstance(_d, dict):
                    _code = _d.get("code")
                    _has_url = bool(_d.get("data") and ("url" in str(_d.get("data"))))
                    # code 非 0/None 且无图片 URL → 视为错误，抛出以触发重试
                    if (_code not in (0, None)) and not _has_url:
                        raise RuntimeError(f"即梦错误码 {_code}: {_d.get('message')}")
            return resp
        except Exception as e:  # 网络级异常 或 错误体触发的重试
            last = e
            if i < max_retries:
                wait = backoff[min(i, len(backoff) - 1)]
                print(f"    [WARN] 即梦 {label} 重试 ({type(e).__name__}: {str(e)[:60]})，{wait}s 后重试 ({i+1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise
    if last:
        raise last

# 本地服务默认地址（jimeng-api docker 默认监听 8000，映射到宿主机 8000）
_DEFAULT_BASE_URL = "http://localhost:8000/v1"
# 单任务（视频）最长等待（秒），即梦免费层视频较慢
_VIDEO_MAX_WAIT = 600
# 单次轮询请求超时（秒），与项目其他后端保持一致
_POLL_TIMEOUT = 60


def _size_to_ratio(size: str) -> str:
    """把 '1024x1024' / '1920x1080' 这类 size 归一为即梦 ratio（如 1:1 / 16:9）。"""
    s = size.replace("*", "x").lower()
    table = {
        "1024x1024": "1:1", "1x1": "1:1", "square": "1:1",
        "1920x1080": "16:9", "1280x720": "16:9", "16x9": "16:9",
        "1080x1920": "9:16", "720x1280": "9:16", "9x16": "9:16",
        "1080x1350": "4:5", "1350x1080": "5:4",
    }
    return table.get(s, "1:1")


def _session_ids() -> list:
    raw = os.getenv("JIMENG_SESSION_ID", "").strip()
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _base_url() -> str:
    return os.getenv("JIMENG_BASE_URL", _DEFAULT_BASE_URL).strip().rstrip("/")


def _round_robin_header() -> dict:
    """从多个 sessionid 轮换取一个作为 Bearer。"""
    ids = _session_ids()
    if not ids:
        return {}
    # 用单调时间做简单轮转，避免每次都取第一个
    idx = int(time.monotonic() * 1000) % len(ids)
    return {"Authorization": f"Bearer {ids[idx]}"}


# ─────────────────────────────────────────────────────────────────────────────
# 生图
# ─────────────────────────────────────────────────────────────────────────────
def generate_image(prompt: str, size: str = "1920x1080",
                   reference_image_urls: list = None, ref_strength: float = 1.0,
                   timeout: int = 180, thread_id: str = "") -> str:
    """调用即梦生图（OpenAI 兼容 /v1/images/generations）。

    返回本地落盘的图片路径（与项目其他后端一致，下游 keyframe 审查可直接读图）。
    支持多参考图融合（img2img）：reference_image_urls 为 http(s) / data: / 本地路径列表，
    全部内联为 images 数组（1-10 张）做多参考融合，对齐 Vidu 资产多视图。
    即梦参考图无 IP-Adapter 硬锁定，ref_strength 近似等价。

    Raises:
        RuntimeError: 服务未配置 / 调用失败 / 无返回。
    """
    # ── 本地关键帧优先（KEYFRAME_LOCAL_DIR）──────────────────────────────────
    # 当设了 KEYFRAME_LOCAL_DIR 且该镜头序号的 PNG 已存在时，直接返回本地路径，
    # 跳过即梦调用。用于在即梦免费积分耗尽时，由外部（AI 生图 / 人工素材）预生成
    # 关键帧注入流水线，保证 full_video_run 不卡额度仍可推进。feature flag，默认关闭。
    local_dir = os.getenv("KEYFRAME_LOCAL_DIR", "").strip()
    if local_dir:
        ldir = Path(local_dir)
        # 用 prompt 中嵌入的镜头序号标记 [KF#idx] 定位本地文件；否则按 thread 目录顺序取
        import re as _re
        m = _re.search(r"\[KF#(\d+)\]", prompt)
        if m:
            idx = int(m.group(1))
            cand = ldir / f"kf_{idx}.png"
            if cand.exists():
                print(f"    [OK] 本地关键帧优先: {cand}")
                return str(cand)
    # ─────────────────────────────────────────────────────────────────────────

    ids = _session_ids()
    if not ids:
        raise RuntimeError("JIMENG_SESSION_ID 未配置（即梦免费后端不可用）")
    base = _base_url()
    url = f"{base}/images/generations"

    # 参考图统一为 base64（即梦接口更稳）；多张全部注入 images 做融合
    image_inputs = [_as_inline(u) for u in (reference_image_urls or []) if u]

    # 【修复-A】jimeng 免费接口对 resolution/ratio 取值较敏感；先归一为合法枚举，
    # 避免非法字符串触发 -2001 invalid parameter（关键帧 img2img 失败的根因之一）。
    _resolution = str(os.getenv("JIMENG_IMAGE_RESOLUTION", "2k")).strip().lower()
    if _resolution not in ("1k", "2k", "4k"):
        _resolution = "2k"
    _ratio = _size_to_ratio(size)
    # 【修复-D】img2img（图生图）模式下 2k/4k 分辨率会被代理判定 invalid parameter 并超时，
    # 实测 1k 稳定返回。纯文生图可保留 2k；仅当带参考图（img2img）时强制降到 1k。
    _is_img2img = bool(reference_image_urls)
    if _is_img2img and _resolution != "1k":
        _resolution = "1k"

    # 【修复-B】使用最新且稳定的模型 jimeng-5.0（本地 jimeng-api 代理支持，且对 img2img +
    # 本地素材图 URL + resolution=2K 输入实测 2/2 稳定返回真实图片 URL）。代理对模型名存在
    # 偶发抖动（intermittent -1000 socket disconnected / -2001），因此失败统一走下方退避重试，
    # 不依赖单一模型名一定成功。jimeng-4.0 / jimeng-4.5 / seedream-5.0 为可用的备选模型。
    _model = "jimeng-5.0"

    payload = {
        "model": _model,
        "prompt": prompt,
        "ratio": _ratio,
        "resolution": _resolution,
        "n": 1,
    }
    if image_inputs:
        # 图生图：images 数组（1-10 张），sample_strength 控制参考强度（近似角色一致）。
        # 【修复-E】sample_strength 不能等于 1.0（等于 1.0 传 images 会被代理判 invalid
        # parameter 并超时）；也不能过低（jimeng 实测 sample_strength < 0.1 会被代理判
        # invalid parameter -2001，尾帧 ref_strength 低至 0.02/0.03 时必触发）。clamp 到
        # [0.1, 0.95]：下限 0.1 仍为"低参考→大视觉差异"保留足够自由度，且合法。
        _ss = max(0.1, min(0.95, ref_strength))
        payload["images"] = image_inputs
        payload["sample_strength"] = _ss

    headers = {"Content-Type": "application/json", **_round_robin_header()}

    session = requests.Session()
    session.trust_env = False

    # 【修复-B/C】失败策略：优先退避重试（覆盖代理瞬时抖动 / 队列满 / 免费档权益不足）。
    # 关键：_check_body=True —— 即梦代理常把「积分不足/权益不足」以 HTTP 200 + code=-2001 返回，
    # 必须当作可重试错误（等免费权益恢复窗口）而非当成功。退避拉长到 8/20/40s 给权益恢复时间。
    # 不再遇到 -2001 就立刻剥离参考图降级为纯文生图（那会丢失素材锚定，违背用户
    # "关键帧要基于素材图"的需求）。仅当 img2img 路径持续失败时，才回退纯文生图一次作为保底，
    # 但纯文生图同样失败则直接抛错，交由 graph 层统一退避重试，而非静默降级为"无首帧"模式。
    def _post():
        return session.post(url, headers=headers, json=payload, timeout=timeout)

    resp = _retry_request(_post, max_retries=4, backoff=[8, 20, 40, 60],
                          label="即梦生图(img2img)" if image_inputs else "即梦生图",
                          _check_body=True)
    if resp.status_code != 200 and image_inputs:
        # 退避重试后仍失败：可能是参考图格式问题，剥离 images 以纯文生图再尝试一次
        _err = resp.text[:300]
        print(f"    [WARN] 即梦 img2img 退避重试后仍失败 (HTTP {resp.status_code}): {_err}")
        print(f"    [WARN] 剥离参考图，纯文生图保底重试一次（保留素材锚定优先，纯文仅兜底）")
        _txt_payload = {
            "model": _model,
            "prompt": prompt,
            "ratio": _ratio,
            "resolution": _resolution,
            "n": 1,
        }
        try:
            resp = _retry_request(lambda: session.post(url, headers=headers, json=_txt_payload, timeout=timeout),
                                  max_retries=1, label="即梦纯文生图保底", _check_body=True)
        except Exception as _e:
            raise RuntimeError(f"即梦生图失败 (img2img 与纯文生图均失败): {_err} | 纯文生图: {_e}")

    if resp.status_code != 200:
        raise RuntimeError(f"即梦生图失败 (HTTP {resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    # OpenAI 兼容返回 data[].url 或 data[].b64_json
    items = data.get("data") or []
    img_url = items[0].get("url") if items else None
    b64 = items[0].get("b64_json") if items else None

    import os as _os
    # 落盘到项目根 output/keyframes/<thread_id>/（无 thread_id 归入 _generated/），
    # 关键帧/尾帧只进 keyframes 目录，绝不进 reference_sheets/library（参考图素材库）。
    out_dir = PROJECT_ROOT / "output" / "keyframes" / (thread_id or "_generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"jimeng_kf_{int(time.time() * 1000)}.png"

    if b64:
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(b64))
    elif img_url:
        r = session.get(img_url, timeout=timeout)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
    else:
        raise RuntimeError(f"即梦生图无返回: {str(data)[:200]}")

    # 返回本地绝对路径：graph 会将其复制到按 thread 隔离的 clips 目录并最终写入 state，
    # 前端经静态服务的 root_marker 兜底（把绝对路径归一为 /media/<项目根名>/...）加载。
    print(f"    [OK] 即梦生图完成: {out_path}")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 生视频（图生视频 / 文生视频）
# ─────────────────────────────────────────────────────────────────────────────
def generate_video(prompt: str, first_frame_url: str = None,
                   duration: int = 5, timeout: int = 180, thread_id: str = "") -> str:
    """调用即梦生视频（OpenAI 兼容 /v1/videos/generations）。

    支持图生视频（first_frame_url 为参考首帧）/ 纯文生视频。
    本地首帧图自动转 base64 内联。返回本地 mp4 路径。

    Raises:
        RuntimeError: 调用失败 / 轮询超时 / 无返回。
    """
    ids = _session_ids()
    if not ids:
        raise RuntimeError("JIMENG_SESSION_ID 未配置（即梦免费后端不可用）")
    base = _base_url()
    url = f"{base}/videos/generations"

    payload = {
        "model": "jimeng-video-3.5-pro",  # 即梦免费层视频模型
        "prompt": prompt,
        "ratio": "16:9",
        "resolution": "720p",
        "duration": duration,
        "n": 1,
    }
    if first_frame_url:
        # 图生视频首帧：inline 为 http(s)/data: 时放入 file_paths；本地路径也转 base64
        inline = _as_inline(first_frame_url)
        payload["file_paths"] = [inline]

    headers = {"Content-Type": "application/json", **_round_robin_header()}
    session = requests.Session()
    session.trust_env = False

    def _do_post():
        return session.post(url, headers=headers, json=payload, timeout=timeout)

    resp = _retry_request(_do_post, label="即梦生视频")
    if resp.status_code != 200:
        raise RuntimeError(f"即梦生视频失败 (HTTP {resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    # 同步直出（部分实现）或返回 task_id 异步轮询
    items = data.get("data") or []
    if items and (items[0].get("url") or items[0].get("b64_json")):
        return _save_video(items[0], session, timeout, thread_id=thread_id)

    task_id = data.get("id") or (data.get("task_id"))
    if not task_id:
        raise RuntimeError(f"即梦生视频无 task_id/直出: {str(data)[:200]}")

    # 异步轮询：jimeng-api 常见两种任务查询路径，依次尝试
    poll_candidates = [
        f"{base}/videos/generations/tasks/{task_id}",
        f"{base}/tasks/{task_id}",
        f"{base.split('/v1')[0]}/v1/video/query",
    ]
    deadline = time.monotonic() + _VIDEO_MAX_WAIT
    while time.monotonic() < deadline:
        pd = None
        for poll_url in poll_candidates:
            def _do_get(url=poll_url):
                # 部分 query 接口需要 POST + task_id body
                if url.endswith("/video/query"):
                    return session.post(url, headers=headers,
                                        json={"task_id": task_id}, timeout=_POLL_TIMEOUT)
                return session.get(url, headers=headers, timeout=_POLL_TIMEOUT)
            try:
                p = _retry_request(_do_get, label="即梦视频轮询")
                pd = p.json()
            except Exception:
                continue
            if pd:
                break
        if not pd:
            time.sleep(5)
            continue
        status = pd.get("status") or pd.get("task_status") or (pd.get("data", {}) or {}).get("status")
        if status in ("succeeded", "SUCCEEDED", "completed", "COMPLETED"):
            item = pd.get("data", {}).get("video") or (pd.get("data") or {})
            if isinstance(item, list):
                item = item[0] if item else {}
            return _save_video(item, session, timeout, thread_id=thread_id)
        if status in ("failed", "FAILED", "error"):
            raise RuntimeError(f"即梦视频任务失败: {str(pd)[:300]}")
        time.sleep(5)
    raise RuntimeError(f"即梦视频轮询超时（>{_VIDEO_MAX_WAIT}s）")


def _save_video(item: dict, session: requests.Session, timeout: int,
                thread_id: str = "") -> str:
    # 落盘到项目根 output/clips/<thread_id>/（无 thread_id 归入 _generated），按会话隔离
    out_dir = PROJECT_ROOT / "output" / "clips" / (thread_id or "_generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"jimeng_vid_{int(time.time() * 1000)}.mp4"
    b64 = item.get("b64_json")
    url = item.get("url")
    if b64:
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(b64))
    elif url:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
    else:
        raise RuntimeError(f"即梦视频无返回内容: {str(item)[:200]}")
    # 返回本地绝对路径：graph 会将其本地化并写入 state，前端经 static_server 兜底加载。
    print(f"    [OK] 即梦视频完成: {out_path}")
    return str(out_path)


def _as_inline(url_or_path: str) -> str:
    """统一为 http(s) 或 data:base64（即梦接口更接受这两种）。本地路径转 base64。

    注意：参考图可能被持久化改写为本项目自定义的 ``/media/<rel>`` 前缀（仅供前端
    static_server 访问），即梦无法据此 fetch。这里把 ``/media/`` 还原为本地绝对
    路径（PROJECT_ROOT/output/<rel>），使下游能读到本地文件并内联为 base64。
    """
    if not url_or_path:
        return url_or_path
    if url_or_path.startswith("data:"):
        return url_or_path
    if url_or_path.startswith(("http://", "https://")):
        from urllib.parse import urlparse
        _host = (urlparse(url_or_path).hostname or "").lower()
        # 即梦服务端无法回源下载任何远程 URL（既包括本机 static_server 的
        # localhost 图，也包括即梦自身返回的 byteimg 等带签名/跨域的远程 URL）。
        # 这里统一由本机 fetch 后内联为 base64 再下发，避免 "invalid parameter"(-2001)。
        try:
            import mimetypes
            import urllib.request as _ur
            req = _ur.Request(url_or_path, headers={"User-Agent": "Mozilla/5.0"})
            with _ur.urlopen(req, timeout=30) as _r:
                _b = _r.read()
            _mime = mimetypes.guess_type(url_or_path)[0] or "image/png"
            return f"data:{_mime};base64,{base64.b64encode(_b).decode('ascii')}"
        except Exception as _e:
            # 媒体静态服务（如 localhost:8900）宕机时，远程下载失败。此时若 URL 指向
            # 本项目 /media/<rel> 静态资源，直接还原为本地文件（PROJECT_ROOT/output/<rel>）
            # 再内联 base64，避免回退成即梦无法回源 fetch 的 localhost URL 而触发 -2001。
            # 这样 jimeng 关键帧/尾帧不依赖媒体服务器存活，离线锚定素材图。
            if _host in ("localhost", "127.0.0.1") and "/media/" in url_or_path:
                _media_rel = url_or_path.partition("/media/")[2]
                _local = str(PROJECT_ROOT / "output" / _media_rel)
                try:
                    _b = open(_local, "rb").read()
                    _mime = mimetypes.guess_type(_local)[0] or "image/png"
                    _du = _downscale_inline_if_needed(_b, _mime)
                    if _du:
                        return _du
                    return f"data:{_mime};base64,{base64.b64encode(_b).decode('ascii')}"
                except Exception as _e2:
                    print(f"    [WARN] 即梦 localhost 媒体宕机，本地还原 /media/{_media_rel} 也失败: {_e2}")
            print(f"    [WARN] 即梦远程参考图下载失败({_host})，回退原值: {_e}")
            return url_or_path
    # 还原 /media/<rel> -> PROJECT_ROOT/output/<rel>
    candidate = url_or_path
    if url_or_path.startswith("/media/"):
        rel = url_or_path[len("/media/"):]
        candidate = str(PROJECT_ROOT / "output" / rel)
    try:
        import mimetypes
        mime = mimetypes.guess_type(candidate)[0] or "image/png"
        b = open(candidate, "rb").read()
        # 【修复-A】即梦免费接口对超大/非标准参考图易报 -2001 invalid parameter。
        # 内联前将最长边限制到 1024（保持纵横比），转 JPEG，显著降低拒收概率，
        # 让素材图锚点（img2img）真正生效，消除"关键帧未基于素材图"。PIL 不可用时跳过。
        data_url = _downscale_inline_if_needed(b, mime)
        if data_url:
            return data_url
        return f"data:{mime};base64,{base64.b64encode(b).decode('ascii')}"
    except Exception as e:
        print(f"    [WARN] 即梦参考图转 base64 失败，回退原值: {e}")
        return url_or_path


def _downscale_inline_if_needed(raw: bytes, mime: str, max_edge: int = 1024):
    """参考图内联前下采样到 max_edge，避免即梦 -2001 invalid parameter。无 PIL 返回 None。"""
    try:
        from io import BytesIO
        from PIL import Image
    except Exception:
        return None
    try:
        im = Image.open(BytesIO(raw))
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) <= max_edge:
            return None
        scale = max_edge / float(max(w, h))
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception as _e:
        print(f"    [WARN] 即梦参考图下采样失败（跳过）: {_e}")
        return None
