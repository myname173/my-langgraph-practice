# src/agent/multimedia/tools/subtitle.py
"""
字幕工具 —— 复用 GitHub 现成组件 pysrt (byroot/pysrt) 处理 SubRip(SRT)，
字幕烧录复用项目已安装的 moviepy，不自造 SRT 解析 / 视频字幕渲染算法。

功能：
  - build_srt(scenes, ...): 由分镜台词生成标准 .srt 文件（pysrt）。
  - burn_subtitles(video_path, srt_path): 用 moviepy SubtitlesClip 把字幕烧录进视频。
       （实现委托给 tools/fonts.py 的 burn_subtitles，统一解决 moviepy 2.x 的
         TextClip 字体/API 兼容问题。）
"""
import os
import tempfile
from typing import Any, Dict, List, Optional

import pysrt

# 字幕烧录委托给 fonts.py（moviepy 2.x 兼容：text= 关键字 + 绝对字体路径）
from .fonts import burn_subtitles as _burn_subtitles


def build_srt(
    scenes: List[Dict[str, Any]],
    per_scene_seconds: float = 6.0,
    out_path: Optional[str] = None,
    voice_timings: Optional[Dict[int, Tuple[float, float]]] = None,
) -> str:
    """由分镜 scenes 的 script 字段生成 SRT 字幕文件，返回路径。

    时间轴策略（优先级从高到低）：
      - **P2 配音对齐（最高）**：传入 voice_timings（逐镜头配音的真实 [start, end]，
        秒）时直接采用。这是唯一能保证「字幕与配音严格同步」的方式——字幕不再
        依赖视频时长猜测，而是用 TTS 的真实发声区间，彻底消除"台词还没念完
        字幕就消失/字幕与声音对不上"。
      - P1 精确对齐：若 scene 含真实字段 "video_duration"（秒，由 stitch 阶段回写），
        则按该场景实际视频时长排布字幕，避免异构片段导致的"字幕偏大提前结束"。
      - 兜底（无 video_duration 时）：按 per_scene_seconds 均分（P0 行为，向后兼容）。

    Args:
        voice_timings: {scene_index: (start_sec, end_sec)}，由 audio_mixer 逐镜头
            TTS 后按各镜头起点放置得到；缺失的镜头自动回退 P1/P0 策略。
    """
    subs = pysrt.SubRipFile()

    t = 0.0
    idx = 0
    # 收集有效片段的「真实起始时间」（由 stitcher 回写，已扣除转场重叠）。
    # 只要存在任意 segment_start，就整体改用真实起点排布，彻底消除累加漂移。
    has_real_start = any(
        isinstance(s.get("segment_start"), (int, float))
        and s.get("segment_start") is not None
        and s.get("segment_start") >= 0
        and (s.get("dialogue") or "").strip()
        and not s.get("shot_failed")
        for s in scenes
    )

    for si, scene in enumerate(scenes):
        # 仅使用对话/台词字段，不使用分镜剧本(script)作为字幕，避免把镜头描述烧录成字幕
        text = (scene.get("dialogue") or "").strip()
        if not text:
            continue
        # 【修复】被跳过/失败的镜头没有画面，若仍生成字幕，会凭空多出一条
        # 并把后续所有字幕整体推移（成片里根本没有这段画面）。
        if scene.get("shot_failed"):
            continue
        if has_real_start and not isinstance(scene.get("segment_start"), (int, float)):
            # 真实起点模式下，该镜头未参与成片 → 不产生字幕
            continue

        idx += 1
        # 【P2 配音对齐】该镜头若有逐镜头 TTS 的真实发声区间，直接采用：
        # 字幕与配音同源，杜绝"台词未念完字幕已消失 / 声画错位"。
        vt = (voice_timings or {}).get(si)
        if vt:
            v_start, v_end = float(vt[0]), float(vt[1])
            if v_end <= v_start:
                v_end = v_start + 0.5
            item = pysrt.SubRipItem(index=idx, text=text)
            item.start.seconds = max(v_start, 0.0)
            item.end.seconds = v_end
            subs.append(item)
            continue
        if has_real_start:
            start = float(scene["segment_start"])
            dur = scene.get("video_duration")
            if isinstance(dur, (int, float)) and dur and dur > 0.1:
                end = start + float(dur)
            else:
                # 时长缺失时，用下一条字幕的起点作为本条终点，避免 0 长度或溢出
                end = start + float(per_scene_seconds)
        else:
            dur = scene.get("video_duration")
            # 真实时长须为正且有限，否则回退均分，保证字幕时间轴不出现 0/负值
            if isinstance(dur, (int, float)) and dur and dur > 0.1:
                seg = float(dur)
            else:
                # 【修复】兜底优先用导演估算的 duration_seconds，而非固定 6.0s
                est = scene.get("duration_seconds")
                if isinstance(est, (int, float)) and est and est > 0.1:
                    seg = float(est)
                else:
                    seg = per_scene_seconds
            start = t
            end = t + seg
            t = end

        # 防止 0/负值或倒挂导致 SRT 非法
        if end <= start:
            end = start + 0.5
        item = pysrt.SubRipItem(index=idx, text=text)
        item.start.seconds = start
        item.end.seconds = end
        subs.append(item)

    # 【修复】去重叠：segment_start 已扣除转场重叠，而 video_duration 是片段完整时长，
    # 二者相加会让相邻字幕互相交叠（实测重叠约 0.4s），表现为两句字幕同时挂在屏幕上。
    # 这里把每条的结束时间收紧到下一条的开始时间，保证同一时刻只显示一条字幕。
    # 注意：必须用 ordinal（毫秒整数）赋值。pysrt 的 .seconds setter 会保留原毫秒
    # 分量（实测 15.6 → 赋 15.2 后变成 15.8），导致收紧结果错误。
    for i in range(len(subs) - 1):
        cur_end_ms = subs[i].end.ordinal
        next_start_ms = subs[i + 1].start.ordinal
        if cur_end_ms > next_start_ms:
            subs[i].end.ordinal = max(next_start_ms, subs[i].start.ordinal)

    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".srt", prefix="subs_")
        os.close(fd)
    subs.save(out_path, encoding="utf-8")
    return out_path


def parse_srt_to_entries(srt_path: str) -> List[Dict[str, Any]]:
    """把 SRT 解析为可序列化的字幕条目列表，供前端逐句时间轴展示。

    与 build_srt 同源（均用 pysrt），零额外依赖。返回:
      [{"index": int, "start": float(秒), "end": float(秒), "text": str}, ...]
    用于前端「字幕生成在第几秒」的可观测性。
    """
    try:
        subs = pysrt.open(srt_path, encoding="utf-8")
    except Exception as e:
        print(f"    [WARN] 字幕解析失败（前端时间轴不可用）: {e}")
        return []
    entries: List[Dict[str, Any]] = []
    for sub in subs:
        entries.append(
            {
                "index": sub.index,
                "start": round(float(sub.start.ordinal) / 1000.0, 3),
                "end": round(float(sub.end.ordinal) / 1000.0, 3),
                "text": (sub.text or "").replace("\n", " ").strip(),
            }
        )
    return entries


def burn_subtitles(
    video_path: str,
    srt_path: str,
    font_size: int = 28,
    out_path: Optional[str] = None,
    theme: str = "default",
) -> str:
    """把 SRT 字幕烧录进视频，返回带字幕的视频路径。

    实现委托给 tools/fonts.burn_subtitles，以兼容 moviepy 2.x 的 TextClip
    签名变化（text= 关键字 + 绝对字体路径），避免旧写法把字幕文本误当 font。
    """
    return _burn_subtitles(
        video_path,
        srt_path,
        font_size=font_size,
    theme=theme,
        out_path=out_path,
    )
