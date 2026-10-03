# src/agent/multimedia/tools/video_pack.py
"""P2-1 成片包装层：片头片尾卡 + 轻量色彩预设。

设计约束：
  - 复用 fonts.make_text_clip（moviepy 2.x 兼容 + 中文字体绝对路径）。
  - **失败绝不阻断**：字体缺失/渲染异常一律返回原片与 0 偏移，调用方无需特判。
  - 片头卡会改变成片时间轴，attach_title_cards 返回 intro 时长，由 stitcher_node
    把所有 scene.segment_start 统一后移，保证字幕/配音对齐不漂移。
  - 色彩预设用 256 级 LUT 查表（numpy fancy indexing），不做 3D LUT 解析；
    noir 为灰度映射单独处理。
"""
import os
from typing import Dict, Any, Optional, Tuple

try:
    from moviepy import (
        VideoFileClip, ColorClip, CompositeVideoClip,
        concatenate_videoclips, vfx,
    )
except ImportError:  # moviepy 1.x 兜底
    from moviepy.editor import (
        VideoFileClip, ColorClip, CompositeVideoClip,
        concatenate_videoclips, vfx,
    )

from .fonts import make_text_clip

_CARD_BG = (10, 10, 18)
_PRESETS = ("warm", "cool", "noir")


def _lut_for_preset(preset: str):
    """构造 256 级 RGB LUT；noir 返回 None（走灰度分支）。"""
    import numpy as np
    x = np.arange(256, dtype="float32")

    if preset == "warm":
        lut = np.stack([
            np.clip(x * 1.13 + 9, 0, 255),   # R 提升
            x * 0.99,                        # G 微降
            np.clip(x * 0.86, 0, 255),       # B 压低
        ], axis=-1)
    elif preset == "cool":
        lut = np.stack([
            np.clip(x * 0.90, 0, 255),       # R 压低
            x * 0.99,                        # G 不动
            np.clip(x * 1.10 + 6, 0, 255),   # B 提升
        ], axis=-1)
    else:
        return None
    return lut.astype("uint8")


def _img_transform(clip, fn):
    """moviepy 1.x/2.x 的逐帧变换兼容。"""
    if hasattr(clip, "image_transform"):
        return clip.image_transform(fn)
    return clip.fl_image(fn)


def apply_color_preset(clip, preset: str):
    """对视频 clip 应用轻量色彩预设；未知预设原样返回。

    warm/cool 用 256 级通道 LUT 查表（单帧 O(N)）；noir 为灰度+对比映射。
    """
    preset = (preset or "").strip().lower()
    if preset not in _PRESETS:
        return clip
    import numpy as np

    if preset == "noir":
        coef = np.array([0.299, 0.587, 0.114], dtype="float32")
        contrast = 1.22
        pivot = 96.0

        def _noir(frame):
            g = frame.astype("float32") @ coef
            g = np.clip((g - pivot) * contrast + pivot, 0, 255).astype("uint8")
            return np.stack([g, g, g], axis=-1)

        return _img_transform(clip, _noir)

    lut = _lut_for_preset(preset)

    def _lut_frame(frame):
        return lut[frame]

    return _img_transform(clip, _lut_frame)


def build_title_card(
    title: str,
    subtitle: str = "",
    duration: float = 2.2,
    size: Tuple[int, int] = (1280, 720),
    mode: str = "title",
):
    """构造一张片头/片尾卡：深色底 + 居中标题（+ 副标题），带淡入淡出。

    mode="title"：标题在上 1/3 处，副标题在其下（片头）。
    mode="end"  ：整组垂直居中（片尾）。
    字体缺失等异常由调用方（attach_title_cards）兜底降级。
    """
    w, h = int(size[0]), int(size[1])
    bg = ColorClip((w, h), color=_CARD_BG).with_duration(duration)

    is_title = mode == "title"
    title_size = int(h * (0.075 if is_title else 0.062))
    title_clip = make_text_clip(
        txt=title,
        font_size=max(title_size, 18),
        color="#F2F2F2",
        stroke_width=0,
        size=(int(w * 0.86), None),
        method="caption",
        text_align="center",
    )
    layers = [bg]
    if is_title:
        title_clip = title_clip.with_position(("center", 0.40), relative=True)
    else:
        title_clip = title_clip.with_position(("center", 0.46), relative=True)
    layers.append(title_clip)

    if subtitle:
        sub_clip = make_text_clip(
            txt=subtitle,
            font_size=max(int(h * 0.038), 12),
            color="#9aa0b5",
            stroke_width=0,
            size=(int(w * 0.7), None),
            method="caption",
            text_align="center",
        )
        sub_clip = sub_clip.with_position(("center", 0.56 if is_title else 0.60), relative=True)
        layers.append(sub_clip)

    if mode == "end":
        end_mark = make_text_clip(
            txt="— FIN —",
            font_size=max(int(h * 0.030), 10),
            color="#6d7387",
            stroke_width=0,
            size=(int(w * 0.4), None),
            method="caption",
            text_align="center",
        )
        end_mark = end_mark.with_position(("center", 0.80), relative=True)
        layers.append(end_mark)

    card = CompositeVideoClip(layers).with_duration(duration)
    return card.with_effects([vfx.FadeIn(0.45), vfx.FadeOut(0.6)])


def attach_title_cards(
    video_path: str,
    title: str,
    subtitle: str = "",
    out_path: Optional[str] = None,
    card_duration: float = 2.2,
) -> Tuple[str, float, float]:
    """把片头卡与片尾卡接到成片首尾，返回 ``(新路径, intro时长, outro时长)``。

    任何失败（字体缺失/编码异常/磁盘问题）都返回 ``(原路径, 0.0, 0.0)``，
    绝不阻断拼接主流程；调用方用 intro 对 segment_start 做统一偏移。
    """
    if not video_path or not os.path.exists(video_path) or not (title or "").strip():
        return video_path, 0.0, 0.0

    video = None
    final = None
    try:
        video = VideoFileClip(video_path)
        w, h = int(video.w), int(video.h)
        intro_card = build_title_card(title, subtitle, card_duration, (w, h), mode="title")
        outro_card = build_title_card(title, "", card_duration, (w, h), mode="end")

        final = concatenate_videoclips(
            [intro_card.without_audio(), video.without_audio(), outro_card.without_audio()],
            method="compose",
        )

        if out_path is None:
            base, ext = os.path.splitext(video_path)
            out_path = f"{base}_packed{ext}"

        final.write_videofile(
            out_path, codec="libx264", fps=24, preset="medium",
            ffmpeg_params=["-movflags", "+faststart"], logger=None,
        )
        print(f"    [P2-1] 包装卡已附加：片头/片尾各 {card_duration:.1f}s → {os.path.basename(out_path)}")
        return out_path, float(card_duration), float(card_duration)
    except Exception as e:  # noqa: BLE001
        print(f"    [WARN] 片头片尾卡生成失败，使用原成片(不阻断): {str(e)[:150]}")
        return video_path, 0.0, 0.0
    finally:
        for c in (final, video):
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass
