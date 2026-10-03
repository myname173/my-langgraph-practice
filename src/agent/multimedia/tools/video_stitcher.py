# src/agent/multimedia/tools/video_stitcher.py
import os
import re
import time
import shutil
import tempfile
import requests
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

# 兼容 MoviePy 1.x / 2.x
try:
    from moviepy.editor import (
        VideoFileClip, concatenate_videoclips,
        CompositeVideoClip, ColorClip,
    )
except ImportError:
    from moviepy import (
        VideoFileClip, concatenate_videoclips,
        CompositeVideoClip, ColorClip,
    )

from moviepy.video.fx import Resize, Margin


# ==============================
# P2-3: Color Grading (亮度统一)
# ==============================
def _get_clip_brightness(clip: VideoFileClip) -> float:
    """采样 clip 多帧亮度均值（0-1），单帧易受明暗突变误导，取 3 帧均值更稳。"""
    try:
        import numpy as np
        d = clip.duration
        sample_times = [d * f for f in (0.25, 0.5, 0.75)] if d > 0.5 else [d / 2]
        vals = []
        for tt in sample_times:
            frame = clip.get_frame(min(tt, max(0.0, d - 0.01)))
            gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
            vals.append(gray.mean() / 255.0)
        return float(np.mean(vals))
    except Exception as e:
        print(f"      [WARN] brightness sampling failed: {e}")
        return 0.5


def _color_grade_clips(clips: list) -> list:
    """
    轻量级色彩统一：以第一个 clip 的亮度为基准，
    调整后续 clip 的亮度使其与基准对齐。

    调整幅度限制在 +/- 20% 内，避免过度校正导致画质劣化。
    """
    if len(clips) <= 1:
        return clips

    reference_brightness = _get_clip_brightness(clips[0])
    print(f"    [*] Color grading: reference brightness = {reference_brightness:.3f} (clip 1)")

    graded = [clips[0]]  # 第一个 clip 不调整

    for i, clip in enumerate(clips[1:], start=1):
        clip_brightness = _get_clip_brightness(clip)
        if clip_brightness < 0.01:
            graded.append(clip)
            continue

        ratio = reference_brightness / clip_brightness
        # 限制调整幅度在 0.8 ~ 1.2
        ratio = max(0.8, min(1.2, ratio))

        if abs(ratio - 1.0) > 0.03:  # 只在差异 > 3% 时调整
            try:
                # moviepy 2.x: MultiplyColor; moviepy 1.x: ColorX
                try:
                    from moviepy.video.fx import MultiplyColor
                    graded_clip = clip.with_effects([MultiplyColor(factor=ratio)])
                except ImportError:
                    from moviepy.video.fx import ColorX
                    graded_clip = ColorX(factor=ratio).apply(clip)
                graded.append(graded_clip)
                print(f"    [*] Clip {i+1}: brightness {clip_brightness:.3f} -> adjusted x{ratio:.2f}")
            except Exception as e:
                print(f"    [WARN] Clip {i+1}: color adjust failed ({e}), using original")
                graded.append(clip)
        else:
            graded.append(clip)

    return graded


# ==============================
# 转场类型定义
# ==============================
_TRANSITION_TYPES = {
    "hard_cut": {
        "description": "instant switch, no transition effect",
    },
    "dissolve": {
        "description": "crossfade blend between clips, time passage feel",
    },
    "fade_through_black": {
        "description": "fade to black then fade in, chapter break / dramatic pause",
    },
    "whip_pan": {
        "description": "fast motion-blur transition, energetic scene change",
    },
    "smash_cut": {
        "description": "abrupt cut with brief brightness flash, dramatic contrast",
    },
    "camera_carry": {
        "description": "extended dissolve preserving motion direction, continuous camera feel",
    },
}

# 默认转场时长（秒）
_DEFAULT_TRANSITION_DURATION = 0.4
# whip_pan 专用更短时长
_WHIP_PAN_DURATION = 0.25


# ==============================
# 下载模块（带重试）
# ==============================
def _has_moov(path: str) -> bool:
    """快速检查 mp4 是否含 moov 原子（无 moov = 索引缺失，播放器无法解析）。"""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
            if b"moov" in head:
                return True
            # moov 可能在文件靠后位置，粗扫尾部
            f.seek(max(0, os.path.getsize(path) - 2_000_000))
            tail = f.read()
            return b"moov" in tail
    except Exception:
        return False


def download_video(url: str, filename: str, retries: int = 3):
    # ── SSRF 防护：仅允许 http/https，拒绝 file:// 等本地/内网协议 ──
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"非法视频 URL 协议 '{scheme}'，仅允许 http/https")

    print(f"    [下载] 获取片段: {url[-60:]}...")

    for attempt in range(retries):
        try:
            from . import net as _net
            response = _net.fetch_response(url, stream=True, timeout=120, label="片段下载")

            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            if os.path.getsize(filename) < 50_000:
                raise Exception("文件过小，疑似损坏")

            # ── 完整性校验：mp4 必须含 moov 原子，否则是残缺/下载不全的片段，
            #    带病拼接会导致最终成片无法播放（moov 缺失）。 ──
            if not _has_moov(filename):
                raise Exception("片段缺少 moov 原子，下载可能不完整（URL 已过期/被截断）")

            print(f"    [OK] Download complete: {filename}")
            return

        except Exception as e:
            print(f"    [WARN] Download failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(2)

    raise Exception(f"Video download failed after {retries} retries: {url}")


# ==============================
# clip 标准化
# ==============================
def normalize_clip(clip: VideoFileClip) -> VideoFileClip:
    target_width = 1920
    target_height = 1080

    # --- Resize ---
    try:
        if hasattr(clip, "resized"):
            clip = clip.resized(width=target_width)
        else:
            clip = Resize(width=target_width).apply(clip)
    except Exception as e:
        print(f"      [WARN] Resize 失败: {e}")
        raise

    # --- FPS ---
    try:
        if hasattr(clip, "set_fps"):
            clip = clip.set_fps(24)
        elif hasattr(clip, "with_fps"):
            clip = clip.with_fps(24)
    except Exception:
        pass

    # --- Padding ---
    if abs(clip.h - target_height) > 2:
        top = max(0, (target_height - clip.h) // 2)
        bottom = max(0, target_height - clip.h - top)

        try:
            clip = Margin(
                top=top,
                bottom=bottom,
                left=0,
                right=0,
                color=(0, 0, 0)
            ).apply(clip)
        except Exception as e:
            print(f"      [WARN] Padding 失败: {e}")
            raise

    return clip


# ==============================
# 转场效果工具函数（兼容 MoviePy 1.x / 2.x）
# ==============================
def _safe_fadein(clip, duration: float):
    """安全地为片段添加淡入效果。"""
    try:
        if hasattr(clip, "fadein"):
            return clip.fadein(duration)
        from moviepy.video.fx import FadeIn
        return clip.with_effects([FadeIn(duration)])
    except Exception as e:
        print(f"      [WARN] fadein 效果不可用: {e}")
        return clip


def _safe_fadeout(clip, duration: float):
    """安全地为片段添加淡出效果。"""
    try:
        if hasattr(clip, "fadeout"):
            return clip.fadeout(duration)
        from moviepy.video.fx import FadeOut
        return clip.with_effects([FadeOut(duration)])
    except Exception as e:
        print(f"      [WARN] fadeout 效果不可用: {e}")
        return clip


def _safe_crossfadein(clip, duration: float):
    """安全地为片段添加交叉淡入效果（用于 dissolve）。"""
    try:
        if hasattr(clip, "crossfadein"):
            return clip.crossfadein(duration)
        from moviepy.video.fx import CrossFadeIn
        return clip.with_effects([CrossFadeIn(duration)])
    except Exception as e:
        print(f"      [WARN] crossfadein 不可用，降级为 fadein: {e}")
        return _safe_fadein(clip, duration)


def _safe_crossfadeout(clip, duration: float):
    """安全地为片段添加交叉淡出效果（用于 dissolve）。"""
    try:
        if hasattr(clip, "crossfadeout"):
            return clip.crossfadeout(duration)
        from moviepy.video.fx import CrossFadeOut
        return clip.with_effects([CrossFadeOut(duration)])
    except Exception as e:
        print(f"      [WARN] crossfadeout 不可用，降级为 fadeout: {e}")
        return _safe_fadeout(clip, duration)


def _safe_brightness_flash(clip, duration: float, peak_factor: float = 2.0):
    """
    在 clip 的开头或结尾添加亮度闪烁效果（用于 whip_pan / smash_cut）。
    peak_factor: 亮度峰值倍数（2.0 = 2倍亮度）
    """
    try:
        # moviepy 2.x: MultiplyColor; moviepy 1.x: ColorX
        try:
            from moviepy.video.fx import MultiplyColor
            _apply_color = lambda c, f: c.with_effects([MultiplyColor(factor=f)])
        except ImportError:
            from moviepy.video.fx import ColorX
            _apply_color = lambda c, f: ColorX(factor=f).apply(c)
        # 对前/后 duration 秒应用亮度提升
        if hasattr(clip, "subclip"):
            total_dur = clip.duration
            if total_dur <= duration * 2:
                return clip  # clip 太短，跳过效果
            # 只对最后 duration 秒做亮度提升
            main_part = clip.subclip(0, total_dur - duration)
            flash_part = clip.subclip(total_dur - duration, total_dur)
            flash_part = _apply_color(flash_part, peak_factor)
            from moviepy import concatenate_videoclips as _concat
            return _concat([main_part, flash_part])
        else:
            return _apply_color(clip, peak_factor)
    except Exception as e:
        print(f"      [WARN] brightness flash 不可用: {e}")
        return clip


# ==============================
# 转场效果应用
# ==============================
def _apply_transition_effects(
    clip,
    fade_in: bool,
    fade_out: bool,
    transition_in: str,
    transition_out: str,
    duration: float,
):
    """
    为单个 clip 应用转场效果。
    - dissolve: 交叉淡入淡出（用于 CompositeVideoClip 重叠）
    - fade_through_black: 普通淡入淡出 + 黑色背景
    - whip_pan: 交叉淡入淡出 + 亮度闪烁（快速能量转场）
    - smash_cut: 亮度闪烁 + 硬切（突兀对比感）
    - camera_carry: 交叉淡入淡出（延续运镜感，比 dissolve 更长）
    - hard_cut / 其他: 无效果
    """
    if fade_in and transition_in in _TRANSITION_TYPES:
        if transition_in == "dissolve":
            clip = _safe_crossfadein(clip, duration)
        elif transition_in == "fade_through_black":
            clip = _safe_fadein(clip, duration)
        elif transition_in == "whip_pan":
            clip = _safe_crossfadein(clip, _WHIP_PAN_DURATION)
        elif transition_in == "camera_carry":
            # camera_carry 使用比 dissolve 更长的交叉淡化
            clip = _safe_crossfadein(clip, duration * 1.5)
        # smash_cut 入向不需要淡入（保持硬切的突兀感）

    if fade_out and transition_out in _TRANSITION_TYPES:
        if transition_out == "dissolve":
            clip = _safe_crossfadeout(clip, duration)
        elif transition_out == "fade_through_black":
            clip = _safe_fadeout(clip, duration)
        elif transition_out == "whip_pan":
            clip = _safe_crossfadeout(clip, _WHIP_PAN_DURATION)
            clip = _safe_brightness_flash(clip, _WHIP_PAN_DURATION, peak_factor=1.8)
        elif transition_out == "smash_cut":
            # smash_cut 出向：亮度闪烁 + 无淡出（硬切）
            clip = _safe_brightness_flash(clip, duration * 0.5, peak_factor=2.5)
        elif transition_out == "camera_carry":
            clip = _safe_crossfadeout(clip, duration * 1.5)

    return clip


def _build_with_transitions(
    clips: list,
    transitions: list,
    transition_duration: float = _DEFAULT_TRANSITION_DURATION,
):
    """
    使用 CompositeVideoClip 构建带转场效果的最终视频。
    支持 6 种转场: dissolve, fade_through_black, whip_pan, smash_cut, camera_carry, hard_cut。

    Args:
        clips: 标准化后的视频片段列表
        transitions: 转场类型列表，transitions[i] 是 clips[i] 和 clips[i+1] 之间的转场
        transition_duration: 转场时长（秒）
    """
    n = len(clips)

    # 为每个 clip 计算时间轴位置
    t = 0.0
    clip_starts = []
    clip_ends = []
    for i in range(n):
        clip_starts.append(t)
        clip_ends.append(t + clips[i].duration)
        if i < n - 1:
            trans = transitions[i] if i < len(transitions) else "hard_cut"
            if trans == "whip_pan":
                t += clips[i].duration - _WHIP_PAN_DURATION
            elif trans == "camera_carry":
                t += clips[i].duration - transition_duration * 1.5
            elif trans in ("dissolve", "fade_through_black"):
                t += clips[i].duration - transition_duration
            else:
                # hard_cut, smash_cut: no overlap
                t += clips[i].duration

    # 为每个 clip 应用淡入淡出效果
    processed_clips = []
    for i in range(n):
        trans_in = transitions[i - 1] if i > 0 and (i - 1) < len(transitions) else ""
        trans_out = transitions[i] if i < len(transitions) else ""

        # 判断是否需要入/出效果
        _FADE_IN_TRANSITIONS = {"dissolve", "fade_through_black", "whip_pan", "camera_carry"}
        _FADE_OUT_TRANSITIONS = {"dissolve", "fade_through_black", "whip_pan", "smash_cut", "camera_carry"}
        needs_fade_in = trans_in in _FADE_IN_TRANSITIONS
        needs_fade_out = trans_out in _FADE_OUT_TRANSITIONS

        processed = _apply_transition_effects(
            clips[i], needs_fade_in, needs_fade_out,
            trans_in, trans_out, transition_duration
        )
        processed = processed.with_start(clip_starts[i])
        processed_clips.append(processed)

    # 检查是否有任何转场效果需要黑色背景
    has_fade_through_black = any(
        t == "fade_through_black" for t in transitions
    )

    # 构建 CompositeVideoClip
    composite_clips = list(processed_clips)

    if has_fade_through_black:
        # 添加黑色背景层（仅在有 fade_through_black 时需要）
        try:
            black = ColorClip(
                size=(1920, 1080),
                color=(0, 0, 0),
                duration=clip_ends[-1],
            )
            black = black.with_start(0)
            composite_clips = [black] + composite_clips
        except Exception as e:
            print(f"    [WARN] 无法创建黑色背景层: {e}，fade_through_black 将降级为硬切")

    try:
        final = CompositeVideoClip(composite_clips, size=(1920, 1080))
        return final
    except Exception as e:
        print(f"    [WARN] CompositeVideoClip 失败: {e}，降级为硬切拼接")
        return concatenate_videoclips(clips, method="compose")


# ==============================
# 动态文件名生成
# ==============================
def _generate_filename(task: str = "", timestamp: Optional[datetime] = None) -> str:
    """
    根据任务描述和时间戳生成唯一的视频文件名。
    替代硬编码的 'epic_game_trailer.mp4'。
    """
    ts = timestamp or datetime.now()
    ts_str = ts.strftime("%Y%m%d_%H%M%S")

    if not task:
        return f"video_{ts_str}.mp4"

    # 提取前 30 个字符，转小写，空格转下划线，过滤非安全字符
    slug = task.strip()[:30].lower().strip()
    slug = re.sub(r'\s+', '_', slug)
    slug = re.sub(r'[^\w\u4e00-\u9fff]', '', slug)

    if not slug:
        slug = "video"

    return f"{slug}_{ts_str}.mp4"


# ==============================
# 主拼接逻辑
# ==============================
def _compute_segment_starts(
    clips: list,
    transitions: list,
    transition_duration: float = _DEFAULT_TRANSITION_DURATION,
) -> list:
    """计算每个片段在成片中的真实起始时间（秒）。

    与 _build_with_transitions 的重叠规则严格一致：转场会让前后片段重叠，
    成片总长 = Σ片段时长 − Σ重叠。字幕时间轴必须用这里的 start 排布，
    否则「时长累加」会让越靠后的字幕前移偏差越大、末条超出片尾。

    Args:
        clips: 已构建的片段列表
        transitions: 相邻片段之间的转场类型列表（长度 n-1）
        transition_duration: 转场时长

    Returns:
        每个片段的起始秒数列表（长度与 clips 相同）
    """
    n = len(clips)
    starts = []
    t = 0.0
    for i in range(n):
        starts.append(t)
        if i >= n - 1:
            break
        trans = transitions[i] if i < len(transitions) else "hard_cut"
        dur = clips[i].duration or 0.0
        if trans == "whip_pan":
            t += dur - _WHIP_PAN_DURATION
        elif trans == "camera_carry":
            t += dur - transition_duration * 1.5
        elif trans in ("dissolve", "fade_through_black"):
            t += dur - transition_duration
        else:
            # hard_cut / smash_cut: 无重叠
            t += dur
    return starts


def stitch_videos(
    video_urls: List[str],
    output_filename: str = "",
    transitions: Optional[List[str]] = None,
    transition_duration: float = _DEFAULT_TRANSITION_DURATION,
    beat_grid: Optional[float] = None,
) -> str:
    """
    下载并拼接视频片段，支持 6 种转场效果和自动色彩统一。

    Args:
        video_urls: 视频 URL 列表
        output_filename: 输出文件名（为空时自动生成）
        transitions: 转场类型列表（transitions[i] 是第 i 和第 i+1 个片段之间的转场）。
                     支持: 'dissolve', 'fade_through_black', 'whip_pan', 'smash_cut',
                     'camera_carry', 'hard_cut'。其他值均视为 hard_cut。
        transition_duration: 转场时长（秒），默认 0.4s

    Returns:
        生成的最终视频文件绝对路径
    """
    # 自动生成文件名
    if not output_filename:
        output_filename = _generate_filename()

    print(f"\n--- [Editor] Starting video stitching -> {output_filename} ---")
    _EFFECT_TRANSITIONS = {"dissolve", "fade_through_black", "whip_pan", "smash_cut", "camera_carry"}
    if transitions:
        effect_trans = [t for t in transitions if t in _EFFECT_TRANSITIONS]
        if effect_trans:
            from collections import Counter
            trans_counts = Counter(effect_trans)
            detail = ", ".join(f"{count} {name}" for name, count in trans_counts.items())
            print(f"    [*] Transition plan: {len(effect_trans)} effects ({detail})")
        else:
            print(f"    [*] All hard cuts")

    temp_files: List[str] = []
    clips: List[VideoFileClip] = []
    valid_indices: List[int] = []  # 记录成功下载的原始 URL 索引
    # 每次拼接使用独立临时目录，避免多请求并发时固定文件名互相覆盖
    temp_dir = tempfile.mkdtemp(prefix="stitch_")

    # ── 片段持久化缓存：把每个成功下载的片段存到 output/clips/，
    #    便于排错、以及 URL 过期后仍能重新拼接（不再依赖 DashScope 临时链接）。 ──
    clips_cache_dir = os.path.join("output", "clips")
    os.makedirs(clips_cache_dir, exist_ok=True)

    try:
        # ==========================
        # 下载 + 构建 clip
        # ==========================
        for i, url in enumerate(video_urls):
            if not url:
                print(f"    [WARN] 跳过空 URL (镜头 {i+1})")
                continue

            # 【修复】二级防御：本地占位黑场文件（make_placeholder_clip 产物，文件名含
            # "placeholder"）直接跳过，不拼入成片，避免上游漏标 shot_failed 时黑屏漏入。
            if isinstance(url, str) and os.path.isfile(url) and "placeholder" in os.path.basename(url).lower():
                print(f"    ⏭️ 镜头 {i+1} 为本地占位黑场，跳过（二级防御，避免黑屏）")
                continue

            temp_file = os.path.join(temp_dir, f"temp_scene_{i:02d}.mp4")
            cached_file = os.path.join(clips_cache_dir, f"scene_{i:02d}.mp4")

            try:
                # 本地占位黑场（降级产物）：直接拷贝，不走 download_video，
                # 保持 SSRF 防护仅约束 http(s) URL 的语义不变。
                if os.path.isfile(url):
                    shutil.copy2(url, temp_file)
                else:
                    download_video(url, temp_file)
                temp_files.append(temp_file)

                # 缓存到本地（拷贝），URL 过期后仍可重拼
                try:
                    shutil.copy2(temp_file, cached_file)
                except Exception:
                    pass

                clip = VideoFileClip(temp_file)

                # --- 防止 duration=0 ---
                if not clip.duration or clip.duration < 0.3:
                    print(f"    [WARN] 无效视频（duration={clip.duration}），跳过")
                    clip.close()
                    continue

                print(f"    [处理] 镜头 {i+1} 标准化")

                clip = normalize_clip(clip)

                # P1-2：卡点对齐（可选，MULTIMEDIA_BEAT_GRID 开启）——
                # 把每段微调到节拍网格（只剪不补，最多剪 0.6s），让切点落在
                # 节拍线上。segment_durations/starts 基于修剪后的 clips 计算，
                # 字幕/配音对齐自动跟随，无需额外处理。
                if beat_grid and beat_grid >= 0.2:
                    import math as _math
                    _d = float(clip.duration or 0.0)
                    _k = int(_math.floor(_d / beat_grid))
                    if _k >= 2:
                        _target = _k * beat_grid
                        if _d - _target >= 0.08:
                            try:
                                clip = clip.subclipped(0, _target)
                                print(f"    [P1-2] 镜头 {i+1} 卡点对齐: {_d:.2f}s → {_target:.2f}s (grid={beat_grid}s)")
                            except Exception as _e:
                                print(f"    [WARN] 卡点对齐失败，保留原时长: {_e}")

                clips.append(clip)
                valid_indices.append(i)

            except Exception as e:
                print(f"    [ERROR] Shot {i+1} processing failed: {e}")
                continue

        # ==========================
        # P2-3: Color Grading (亮度统一)
        # ==========================
        if len(clips) > 1:
            print("    [*] Color grading: normalizing brightness across clips...")
            clips = _color_grade_clips(clips)

        # ==========================
        # 兜底检查
        # ==========================
        if len(clips) == 0:
            raise Exception("没有可用的视频片段，无法生成最终视频")

        if len(clips) == 1:
            print("    [WARN] 仅有一个片段，直接导出")
            final_clip = clips[0]
        else:
            # 根据 valid_indices 提取有效的转场列表
            clip_transitions: List[str] = []
            if transitions and len(transitions) > 0:
                for ci in range(len(valid_indices) - 1):
                    orig_idx = valid_indices[ci]
                    if orig_idx < len(transitions):
                        clip_transitions.append(transitions[orig_idx])
                    else:
                        clip_transitions.append("hard_cut")

            has_transitions = any(
                t in _EFFECT_TRANSITIONS for t in clip_transitions
            )

            if has_transitions:
                print(f"    [剪辑] 带转场拼接 {len(clips)} 个片段...")
                final_clip = _build_with_transitions(
                    clips, clip_transitions, transition_duration
                )
            else:
                print(f"    [剪辑] 硬切拼接 {len(clips)} 个片段...")
                final_clip = concatenate_videoclips(clips, method="compose")

        # ==========================
        # 导出
        # ==========================
        print(f"    [渲染] 导出最终视频 → {output_filename}")

        try:
            final_clip.write_videofile(
                output_filename,
                codec="libx264",
                audio_codec="aac",
                fps=24,
                preset="medium",
                threads=4,
                ffmpeg_params=["-movflags", "+faststart"],
                logger=None
            )
        except Exception as e:
            print(f"    [ERROR] Render failed, trying fallback: {e}")

            # fallback：降低线程 + 去掉 audio
            final_clip.write_videofile(
                output_filename,
                codec="libx264",
                fps=24,
                preset="ultrafast",
                threads=1,
                logger=None
            )

        # 导出后完整性校验：mp4 必须含 moov 原子，否则播放器无法解析
        if not _has_moov(output_filename):
            raise Exception(
                "导出文件缺少 moov 原子，成片可能损坏不可播放（请检查源片段完整性）"
            )

        abs_path = os.path.abspath(output_filename)
        print(f"    [OK] Final video complete: {abs_path}")

        # P1 字幕精确对齐：返回各有效片段的真实时长（秒），顺序与 valid_indices 对应。
        # 供 stitcher_node 写回 scene["video_duration"]，使 build_srt 按真实时长对齐字幕。
        segment_durations = [c.duration for c in clips]
        # 【修复】同时回写每个片段在成片中的真实起始时间（已扣除转场重叠）。
        # 成片总长 = Σ片段时长 − Σ转场重叠，若字幕仍用「时长累加」排布，
        # 越靠后的字幕前移偏差越大、末条会超出片尾。有了真实 start 即可精确对齐。
        segment_starts = _compute_segment_starts(
            clips, clip_transitions if len(clips) > 1 else [], transition_duration
        )
        return abs_path, valid_indices, segment_durations, segment_starts

    except Exception as e:
        print(f"    [ERROR] Stitching failed: {str(e)}")
        raise e

    finally:
        print("    [清理] 释放资源...")

        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass

        # 整目录清理，连带其中所有临时片段一并删除
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"    [WARN] 临时目录清理失败: {temp_dir} ({e})")
