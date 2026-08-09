# src/agent/multimedia/tools/subtitle.py
"""
字幕工具 —— 复用 GitHub 现成组件 pysrt (byroot/pysrt) 处理 SubRip(SRT)，
字幕烧录复用项目已安装的 moviepy，不自造 SRT 解析 / 视频字幕渲染算法。

功能：
  - build_srt(scenes, ...): 由分镜台词生成标准 .srt 文件（pysrt）。
  - burn_subtitles(video_path, srt_path): 用 moviepy SubtitlesClip 把字幕烧录进视频。
"""
import os
import tempfile
from typing import Any, Dict, List, Optional

import pysrt

# moviepy 2.x 移除了 moviepy.editor 聚合入口，做兼容导入
try:
    from moviepy.editor import (
        VideoFileClip, TextClip, CompositeVideoClip,
    )
    from moviepy.video.tools.subtitles import SubtitlesClip
except Exception:
    from moviepy import VideoFileClip, TextClip, CompositeVideoClip
    from moviepy.video.tools.subtitles import SubtitlesClip


def build_srt(
    scenes: List[Dict[str, Any]],
    per_scene_seconds: float = 6.0,
    out_path: Optional[str] = None,
) -> str:
    """由分镜 scenes 的 script 字段生成 SRT 字幕文件，返回路径。

    时间轴策略：
      - 优先 P1 精确对齐：若 scene 含真实字段 "video_duration"（秒，由 stitch 阶段回写），
        则按该场景实际视频时长排布字幕，避免异构片段导致的"字幕偏大提前结束"。
      - 兜底（无 video_duration 时）：按 per_scene_seconds 均分（P0 行为，向后兼容）。
    """
    subs = pysrt.SubRipFile()

    t = 0.0
    idx = 0
    for scene in scenes:
        text = (scene.get("script") or "").strip()
        if not text:
            continue
        idx += 1
        dur = scene.get("video_duration")
        # 真实时长须为正且有限，否则回退均分，保证字幕时间轴不出现 0/负值
        if isinstance(dur, (int, float)) and dur and dur > 0.1:
            seg = float(dur)
        else:
            seg = per_scene_seconds
        start = t
        end = t + seg
        item = pysrt.SubRipItem(index=idx, text=text)
        item.start.seconds = start
        item.end.seconds = end
        subs.append(item)
        t = end

    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".srt", prefix="subs_")
        os.close(fd)
    subs.save(out_path, encoding="utf-8")
    return out_path


def burn_subtitles(
    video_path: str,
    srt_path: str,
    font_size: int = 28,
    out_path: Optional[str] = None,
) -> str:
    """把 SRT 字幕烧录进视频，返回带字幕的视频路径。"""
    if not os.path.exists(srt_path):
        return video_path

    video = VideoFileClip(video_path)
    generator = lambda txt: TextClip(
        txt,
        font="Microsoft-YaHei",
        fontsize=font_size,
        color="white",
        stroke_color="black",
        stroke_width=1.2,
        size=(video.w * 0.9, None),
        method="caption",
    )
    subtitles = SubtitlesClip(srt_path, generator)
    result = CompositeVideoClip([video, subtitles.set_position(("center", "bottom"))])

    if out_path is None:
        base, ext = os.path.splitext(video_path)
        out_path = f"{base}_subs{ext}"

    result.write_videofile(
        out_path,
        codec="libx264",
        audio_codec="aac",
    )
    video.close()
    result.close()
    return out_path
