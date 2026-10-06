# src/agent/multimedia/tools/frame_extract.py
"""P0（跨镜连贯性）：真实末帧接力 —— 从上一镜成片视频抽取物理最后一帧。

跨镜连续性的业界标准手法是「尾帧接力」（last-frame relay）：把上一镜**实际输出的
最后一帧**作为下一镜 FLF 的首帧输入，让镜头链传递"现实"而非"计划"。

本项目此前的承接用的是「上一镜计划尾帧图」（end_frame_director 生成的目标静图），
与视频实际落点存在漂移——实测真实末帧 vs 计划尾帧图相关性 CC=-0.326（近乎无关），
这正是接缝断裂（观感"无连贯"）的根因。

本模块用项目已有的 imageio_ffmpeg 自带 ffmpeg 二进制抽取真实末帧，落盘到
``output/keyframes/<thread_id>/linked_last_*.png`` 并返回 ``/media`` URL。
任何失败（视频不可达 / ffmpeg 缺失 / 解码失败）均返回 None，由调用方回退到计划
尾帧图——绝不阻断主流程。
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from .media_paths import local_to_media_url, media_url_to_local, output_dir

_RUN_TIMEOUT = 120
_ENV_RELAY = "MULTIMEDIA_FLF_RELAY"  # 'video'(默认，真实末帧) | 'keyframe'(回退计划尾帧图)


def relay_mode() -> str:
    """承接帧来源模式：'video'（默认，真实末帧接力）/ 'keyframe'（计划尾帧图）。"""
    return (os.getenv(_ENV_RELAY, "video") or "video").strip().lower()


def _ffmpeg_exe() -> Optional[str]:
    """取 imageio_ffmpeg 自带的 ffmpeg 二进制（与 video_post.py 同款，零新增依赖）。"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def resolve_local_video(video: "str | None") -> Optional[str]:
    """把 本地 mp4 路径 / ``/media`` URL 解析为存在可读的本地文件路径；否则 None。"""
    if not video or not isinstance(video, str):
        return None
    if os.path.isfile(video):
        return video
    p = media_url_to_local(video)
    if p and Path(p).is_file():
        return str(p)
    return None


def extract_last_frame(video: "str | None", *, thread_id: str = "", tag: str = "") -> Optional[str]:
    """从视频抽取**物理最后一帧**，落盘并返回 ``/media`` URL；失败返回 None。

    - ``video``：本地 mp4 路径 或 ``/media/...`` URL（可被 :func:`resolve_local_video` 解析）。
    - 输出：``<root>/output/keyframes/<thread_id>/linked_last_<tag>_<ts>.png``。
    - 抽取策略：先用 ``-sseof -1 -update 1`` 让 ffmpeg 覆写写出最后 1 秒的每一帧，
      最终文件即视频真实末帧；失败再退 ``-sseof -0.05 -frames:v 1``。
      注：勿用 ``-0.15`` 之类的较大回看窗口——实测会落到倒数第 2~4 帧（低对比度
      过渡帧），与真实末帧互相关仅 -0.475，会引入错误锚点。
    """
    local = resolve_local_video(video)
    if not local:
        return None
    exe = _ffmpeg_exe()
    if not exe:
        return None

    out_dir = output_dir() / "keyframes" / (thread_id or "_shared")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    stamp = int(time.time() * 1000)
    safe_tag = "".join(c for c in (tag or "frame") if c.isalnum() or c in "_-") or "frame"
    out_png = out_dir / f"linked_last_{safe_tag}_{stamp}.png"

    attempts = (
        [exe, "-y", "-sseof", "-1", "-i", local, "-update", "1", "-q:v", "2", str(out_png)],
        [exe, "-y", "-sseof", "-0.05", "-i", local, "-frames:v", "1", "-q:v", "2", str(out_png)],
    )
    for cmd in attempts:
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=_RUN_TIMEOUT)
        except Exception:
            continue
        try:
            if r.returncode == 0 and out_png.is_file() and out_png.stat().st_size > 0:
                return local_to_media_url(out_png)
        except OSError:
            continue
    return None
