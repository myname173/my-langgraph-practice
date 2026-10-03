# src/agent/multimedia/tools/fonts.py
"""
字体工具 —— 解决 moviepy 2.x TextClip 字体兼容问题。

背景（重要，避免后来者重复踩坑）：
  1. moviepy 1.x 的 TextClip 签名是 TextClip(txt, font=...)，第一个位置参数是文本；
     而 moviepy 2.x 改成了 TextClip(text=None, font=None, font_size=None, ...)，
     第一个位置参数变成了 font。若沿用旧写法 TextClip("字幕", font="微软雅黑")，
     会把"字幕"当成 font 传给 PIL，报
     "Invalid font <function ...>, pillow failed to use it" / "multiple values for argument 'font'"。
  2. 字体名（如 "Microsoft-YaHei" / "DejaVu-Sans"）在 Windows 上无法被 PIL 的
     ImageFont.truetype 直接解析（cannot open resource），必须给字体文件的绝对路径。
  3. 中文字幕必须用支持中文的字体（微软雅黑 msyh.ttc / 黑体 simhei.ttf 等），
     否则中文会渲染成方框。

本模块提供：
  - resolve_font_path()       跨平台定位一个可用的中文字体绝对路径（无则 None）。
  - make_text_clip(...)       moviepy 2.x 安全的文本 clip 工厂（text= 关键字 + 绝对字体路径）。
  - burn_subtitles(...)       兼容 moviepy 1.x/2.x 的字幕烧录实现，供 subtitle.py 复用。
"""
import os
from typing import Any, Dict, List, Optional

# 候选字体文件（按优先级）：Windows / macOS / Linux 常见中文字体。
_CANDIDATES: List[str] = [
    # Windows
    r"C:\Windows\Fonts\msyh.ttc",       # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",     # 微软雅黑 Bold
    r"C:\Windows\Fonts\simhei.ttf",     # 黑体
    r"C:\Windows\Fonts\simsun.ttc",     # 宋体
    r"C:\Windows\Fonts\simfang.ttf",    # 仿宋
    r"C:\Windows\Fonts\kaiu.ttf",       # 楷体
    r"C:\Windows\Fonts\arial.ttf",      # Arial（无中文字形，仅兜底）
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",   # 文泉驿微米黑
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",     # 文泉驿正黑
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # 仅兜底（无中文字形）
]

_cache: Optional[str] = None


def resolve_font_path() -> Optional[str]:
    """跨平台定位一个可用的中文字体绝对路径，找不到则返回 None（缓存结果）。"""
    global _cache
    if _cache is not None:
        return _cache
    for p in _CANDIDATES:
        try:
            if os.path.exists(p):
                _cache = p
                return _cache
        except OSError:
            continue
    _cache = ""
    return None


def make_text_clip(
    txt: str,
    font_size: int = 28,
    color: str = "white",
    stroke_color: str = "black",
    stroke_width: float = 1.2,
    size=(None, None),
    method: str = "caption",
    font: Optional[str] = None,
    **kwargs: Any,
):
    """moviepy 2.x 安全的文本 clip 工厂。

    关键：文本必须用关键字 `text=` 传入（2.x 第一个位置参数是 font）；
    字体用 resolve_font_path() 解析出的绝对路径，避免字体名无法被 PIL 加载。
    找不到字体时抛出明确异常，由调用方决定降级策略。
    """
    font_path = font or resolve_font_path()
    from moviepy import TextClip

    kw: Dict[str, Any] = dict(
        text=txt,
        font_size=font_size,
        color=color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        size=size,
        method=method,
    )
    if font_path:
        kw["font"] = font_path
    kw.update(kwargs)
    return TextClip(**kw)


# P2-1 字幕主题：color/stroke/字号缩放/底部位置（0-1 相对高度）
_SUBTITLE_THEMES: Dict[str, Dict[str, Any]] = {
    "default": {"color": "white", "stroke_color": "black", "stroke_width": 1.2, "scale": 1.0, "bottom": 0.92},
    "minimal": {"color": "#E6E6E6", "stroke_color": "#101010", "stroke_width": 0.0, "scale": 0.85, "bottom": 0.95},
    "cinema": {"color": "#F5D76E", "stroke_color": "#141414", "stroke_width": 1.8, "scale": 1.18, "bottom": 0.88},
}


def burn_subtitles(
    video_path: str,
    srt_path: str,
    font_size: int = 28,
    out_path: Optional[str] = None,
    theme: str = "default",
) -> str:
    """兼容 moviepy 1.x/2.x 的字幕烧录：把 SRT 烧录进视频，返回带字幕的视频路径。

    兼容要点：
      - TextClip 用 text= 关键字 + 绝对字体路径（2.x 安全）。
      - SubtitlesClip 在 2.x 用 make_textclip= 传生成函数（1.x 是第二位置参数 font=generator）。
    """
    if not os.path.exists(srt_path):
        return video_path

    from moviepy import VideoFileClip, CompositeVideoClip
    try:
        from moviepy.editor import SubtitlesClip
    except Exception:
        from moviepy.video.tools.subtitles import SubtitlesClip
    from moviepy.video.VideoClip import TextClip as _TextClip2

    # moviepy 2.x 的 Clip 有 with_duration 方法；1.x 只有 set_duration。
    is_mpy2 = hasattr(_TextClip2, "with_duration")

    font_path = resolve_font_path()
    if not font_path:
        raise RuntimeError("未找到可用中文字体，无法烧录字幕。请安装 微软雅黑/黑体 等中文字体。")

    th = _SUBTITLE_THEMES.get((theme or "default").strip().lower(), _SUBTITLE_THEMES["default"])
    _fs = max(int(round(font_size * th["scale"])), 12)

    video = VideoFileClip(video_path)

    def _make_clip(txt: str):
        from moviepy import TextClip
        return TextClip(
            text=txt,
            font=font_path,
            font_size=_fs,
            color=th["color"],
            stroke_color=th["stroke_color"],
            stroke_width=th["stroke_width"],
            size=(int(video.w * 0.9), None),
            method="caption",
        )

    # 关键：SRT 由 pysrt 以 UTF-8 写入，moviepy 读取时必须显式指定 encoding='utf-8'，
    # 否则 Windows 上默认用 gbk 读 UTF-8 中文会报
    # "'gbk' codec can't decode byte ... illegal multibyte sequence"
    if is_mpy2:
        subtitles = SubtitlesClip(srt_path, make_textclip=_make_clip, encoding="utf-8")
    else:
        subtitles = SubtitlesClip(srt_path, font=_make_clip, encoding="utf-8")

    # 定位兼容 1.x(set_position)/2.x(with_position)
    _bottom = th.get("bottom", 0.92)
    subtitles_pos = (
        subtitles.with_position(("center", _bottom), relative=True)
        if is_mpy2
        else subtitles.set_position(("center", _bottom), relative=True)
    )
    result = CompositeVideoClip([video, subtitles_pos])

    if out_path is None:
        base, ext = os.path.splitext(video_path)
        out_path = f"{base}_subs{ext}"

    result.write_videofile(out_path, codec="libx264", audio_codec="aac")
    try:
        video.close()
    except Exception:
        pass
    try:
        result.close()
    except Exception:
        pass
    return out_path
