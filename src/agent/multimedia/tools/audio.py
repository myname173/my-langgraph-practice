# src/agent/multimedia/tools/audio.py
"""
音频合成工具 —— 复用现成组件，不自造音频生成/混流算法。

  - mix_audio(video, voiceover, bgm): 音视频混流，复用项目已装的 moviepy。
  - generate_bgm(mood, duration): 背景音乐获取。

BGM 引擎选型说明（重要，避免后来者重复踩坑）：
  当前采用「预置曲库」方案 —— 按 mood 从 assets/bgm/ 选取免版权音频。
  之所以不用在线 API：阿里云 dashscope 只提供语音能力
  (asr / tts / qwen_omni)，并**没有**音乐生成模型，不存在可用的 BGM 接口。
  之所以不默认用 audiocraft(MusicGen)：其依赖要求 Python <3.12 且需要
  torchaudio，与本项目 Python 3.13 环境不兼容；CPU 推理也需数十秒到数分钟，
  会显著拖慢流水线迭代速度。

  本函数签名保持稳定：未来若要换成 MusicGen 或第三方音乐 API，
  只需替换内部实现，graph.py 等上层调用方无需任何改动。
"""
import os
import math
import random
import tempfile
from typing import Optional, Tuple, Dict

# moviepy 2.x 移除了 moviepy.editor 聚合入口，做兼容导入
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
except Exception:
    from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

from moviepy import afx

# BGM 情绪 → 自然语言文字描述（喂给 MusicGen / API 的文本条件）
BGM_MOOD_PROMPTS = {
    "ambient": "calm ambient background music, soft pads, no drums",
    "tense": "tense cinematic suspense music, low strings, slow pulse",
    "upbeat": "upbeat cheerful pop background music, light drums, bright",
    "epic": "epic orchestral trailer music, big drums, soaring brass",
    "romantic": "gentle romantic piano music, warm strings, slow",
    "tech": "futuristic electronic background music, synths, minimal beat",
}

# 场景情绪(emotion) → BGM mood 映射，供 audio_mixer_node 按当前场景情绪选曲
# 复用项目 scenes 已有的 emotion 字段（如 calm/tense/joyful/epic/romantic/tech 等）
BGM_MOOD_BY_EMOTION = {
    "calm": "ambient",
    "neutral": "ambient",
    "peaceful": "ambient",
    "tense": "tense",
    "suspense": "tense",
    "fear": "tense",
    "anxious": "tense",
    "joy": "upbeat",
    "joyful": "upbeat",
    "happy": "upbeat",
    "excited": "upbeat",
    "epic": "epic",
    "grand": "epic",
    "heroic": "epic",
    "romantic": "romantic",
    "love": "romantic",
    "tender": "romantic",
    "tech": "tech",
    "futuristic": "tech",
    "cyber": "tech",
}

# 预置 BGM 曲库目录：assets/bgm/<mood>/*.mp3|wav（相对项目根目录）
# 可用环境变量 BGM_LIBRARY_DIR 覆盖，便于指向自己的私有曲库。
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
BGM_LIBRARY_DIR = os.getenv("BGM_LIBRARY_DIR", os.path.join(_PROJECT_ROOT, "assets", "bgm"))

# 曲库可接受的音频扩展名
_BGM_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".flac")


def resolve_bgm_mood(emotion: Optional[str], fallback: str = "ambient") -> str:
    """由场景情绪解析 BGM mood，未知情绪回退 fallback。"""
    if not emotion:
        return fallback
    return BGM_MOOD_BY_EMOTION.get(emotion.strip().lower(), fallback)


def _list_bgm_candidates(mood: str) -> list:
    """列出某个 mood 目录下的候选音频文件（绝对路径，已排序保证可复现）。"""
    mood_dir = os.path.join(BGM_LIBRARY_DIR, mood)
    if not os.path.isdir(mood_dir):
        return []
    return sorted(
        os.path.join(mood_dir, f)
        for f in os.listdir(mood_dir)
        if f.lower().endswith(_BGM_EXTS)
    )


def generate_bgm(
    mood: str = "ambient",
    duration: float = 30.0,
    seed: Optional[int] = None,
) -> Optional[str]:
    """按情绪获取一段背景音乐，返回本地音频路径；无可用曲目时返回 None。

    从预置曲库 assets/bgm/<mood>/ 中挑选音频。找不到对应 mood 时回退到
    ambient 目录，仍找不到则返回 None —— 调用方（audio_mixer_node）会
    跳过配乐，不影响成片产出。

    Args:
        mood: BGM 情绪标签，见 BGM_MOOD_PROMPTS 的 key。
        duration: 期望时长（秒）。此处不做裁剪，实际对齐由 mix_audio 按
            视频长度循环/截断处理；保留该参数是为了与未来的生成式实现同签名。
        seed: 可选随机种子，固定后同一 mood 每次选中同一首，便于复现调试。

    Returns:
        音频文件绝对路径，或 None。
    """
    candidates = _list_bgm_candidates(mood)

    # mood 目录缺失/为空 → 回退 ambient
    if not candidates and mood != "ambient":
        candidates = _list_bgm_candidates("ambient")
        if candidates:
            print(f"    [BGM] 无 '{mood}' 曲目，回退 ambient")

    if not candidates:
        print(
            f"    [BGM] 曲库为空（{BGM_LIBRARY_DIR}），跳过配乐。"
            f"可放入音频到 assets/bgm/<mood>/ 启用，详见 assets/bgm/README.md"
        )
        return None

    picker = random.Random(seed) if seed is not None else random
    chosen = picker.choice(candidates)
    print(f"    [BGM] mood={mood} → {os.path.basename(chosen)}")
    return chosen


def _apply_bgm_duck(
    bgm,
    duration: float,
    speaking_ranges,
    duck_factor: float = 0.35,
    bgm_volume: float = 0.25,
):
    """P1-5：BGM 自动闪避 —— 配音有声段把 BGM 压低 duck_factor 倍（≈ -9dB），无声段恢复。

    实现：把 BGM 渲染成采样数组，按配音真实区间 [start,end] 构建音量包络
    （50fps 网格 + 滑动平均平滑防爆音），逐采样相乘后包回 AudioArrayClip。

    Args:
        bgm: 已 loop/对齐时长后的 BGM AudioFileClip。
        duration: 成片总时长（秒）。
        speaking_ranges: [(start_sec, end_sec), ...] 配音有声区间。
        duck_factor: 有声段音量倍率（0.35 ≈ -9dB）。
        bgm_volume: 原 BGM 恒定音量（duck 版本把它乘进数组）。

    Returns:
        闪避后的 AudioArrayClip。任何异常由调用方兜底回退恒定音量。
    """
    import numpy as np
    try:
        from moviepy import AudioArrayClip
    except ImportError:  # 旧版 moviepy 路径
        from moviepy.audio.io.AudioArrayClip import AudioArrayClip

    sr = 22050
    arr = bgm.to_soundarray(fps=sr)
    if arr.ndim == 1:
        arr = arr[:, None]

    fps_env = 50.0
    n_env = max(int(duration * fps_env), 2)
    step = 1.0 / fps_env
    gain = np.ones(n_env, dtype="float32")
    for _seg, (s, e) in (speaking_ranges or {}).items():
        i0 = max(int(float(s) / step) - 3, 0)   # 提前 ~60ms 开始压，避免第一个音节漏出
        i1 = min(int(float(e) / step) + 5, n_env)  # 结束后 ~100ms 恢复
        if i1 > i0:
            gain[i0:i1] = duck_factor
    # 滑动平均平滑包络（约 140ms 过渡带），避免音量阶跃产生可闻爆点
    gain = np.convolve(gain, np.ones(7, dtype="float32") / 7.0, mode="same")

    # 逐采样插值并乘入 BGM 数组（同时把恒定 bgm_volume 乘进来）
    env = np.interp(np.arange(arr.shape[0]), np.arange(n_env) * step * sr, gain).astype("float32")
    out = arr.astype("float32") * (env[:, None] * bgm_volume)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak * 0.98
    return AudioArrayClip(out, fps=sr).with_duration(duration)


def _load_sfx_events(sfx_events):
    """P1-5：把 SFX 事件列表 [{path,start,volume}] 加载为音频 clip 列表。

    单条加载失败只告警跳过，绝不阻断混音主流程。
    """
    loaded = []
    for ev in sfx_events or []:
        p = ev.get("path") or ""
        if not p or not os.path.exists(p):
            continue
        try:
            sc = AudioFileClip(p)
            sc = sc.with_volume_scaled(float(ev.get("volume", 0.6)))
            sc = sc.with_start(max(float(ev.get("start") or 0.0), 0.0))
            loaded.append(sc)
        except Exception as e:  # noqa: BLE001
            print(f"    [WARN] SFX 加载失败（跳过 {os.path.basename(p)}）: {e}")
    return loaded


def mix_audio(
    video_path: str,
    voiceover_path: str,
    bgm_path: Optional[str] = None,
    bgm_volume: float = 0.25,
    out_path: Optional[str] = None,
    voice_timings: Optional[Dict[int, Tuple[float, float]]] = None,
    duck_factor: float = 0.35,
    sfx_events: Optional[list] = None,
) -> str:
    """把配音（与可选 BGM）混入视频，返回带音轨的成片路径。"""
    if not voiceover_path or not os.path.exists(voiceover_path):
        return video_path

    video = VideoFileClip(video_path)
    voice = AudioFileClip(voiceover_path)

    # ── 配音长于视频时延长视频（末帧冻结），而不是截断配音 ──
    # 背景：台词时长按中文 3.5 字/秒 + 停顿估算，成片常短于配音总长。
    # 旧逻辑 `CompositeAudioClip(...).with_duration(video.duration)` 会把配音硬截断到
    # 视频时长，直接表现为「台词还没念完画面就结束了」（用户反馈的核心问题之一）。
    # 这里改为：配音更长时把最后一帧冻结补足到配音长度，保证台词完整播完。
    if voice.duration and voice.duration > video.duration + 0.05:
        extra = float(voice.duration) - float(video.duration)
        print(f"    [AUDIO] 配音({voice.duration:.2f}s) 长于视频({video.duration:.2f}s)，"
              f"末帧冻结延长 {extra:.2f}s 以保证台词完整")
        try:
            from moviepy import concatenate_videoclips as _concat
            freeze = video.to_ImageClip(t=max(video.duration - 0.04, 0)).with_duration(extra)
            video = _concat([video, freeze])
        except Exception as _e:
            print(f"    [WARN] 末帧冻结延长失败，回退为慢放: {_e}")
            try:
                from moviepy import vfx as _vfx
                _ratio = float(voice.duration) / float(video.duration)
                video = video.with_effects([_vfx.MultiplySpeed(factor=1.0 / _ratio)])
            except Exception as _e2:
                print(f"    [WARN] 慢放也失败，保持原时长（配音尾部可能被截断）: {_e2}")

    clips = [voice]
    bgm = None
    sfx_clips = []
    if bgm_path and os.path.exists(bgm_path):
        bgm = AudioFileClip(bgm_path)
        # BGM 短于视频时循环铺满，长于视频时截断，并加淡入淡出避免突兀
        effects = []
        if bgm.duration < video.duration:
            effects.append(afx.AudioLoop(duration=video.duration))
        effects.extend([afx.AudioFadeIn(1.0), afx.AudioFadeOut(1.5)])
        bgm = (
            bgm.with_effects(effects)
            .with_duration(video.duration)
        )
        # P1-5：BGM 自动闪避 —— 有逐镜头配音真实区间时，配音段压低、间隙恢复；
        # 失败（如 numpy/内存异常）回退恒定音量旧行为，绝不阻断混音。
        _ducked = False
        if voice_timings and duck_factor < 1.0:
            try:
                bgm = _apply_bgm_duck(bgm, float(video.duration), voice_timings,
                                      duck_factor=duck_factor, bgm_volume=bgm_volume)
                _ducked = True
                try:
                    _db = 20 * math.log10(max(duck_factor, 1e-6))
                except Exception:
                    _db = -9.0
                print(f"    [P1-5] BGM 闪避启用：配音段压低 {_db:.0f}dB（{len(voice_timings)} 段配音区间）")
            except Exception as e:  # noqa: BLE001
                print(f"    [WARN] BGM 闪避失败，回退恒定音量: {e}")
        if not _ducked:
            bgm = bgm.with_volume_scaled(bgm_volume)
        clips.append(bgm)

    # P1-5：SFX 音效轨（转场切点铺设，事件由 audio_mixer_node 按转场类型挑选）
    sfx_clips = _load_sfx_events(sfx_events)
    if sfx_clips:
        print(f"    [P1-5] SFX: 铺设 {len(sfx_clips)} 条音效")
        clips.extend(sfx_clips)

    merged = CompositeAudioClip(clips).with_duration(video.duration)
    final = video.with_audio(merged)

    if out_path is None:
        base, ext = os.path.splitext(video_path)
        out_path = f"{base}_audio{ext}"

    try:
        final.write_videofile(out_path, codec="libx264", audio_codec="aac")
    finally:
        # 确保句柄释放，避免 Windows 上临时文件被占用无法删除
        for c in (final, video, voice, bgm):
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass
        for c in sfx_clips:
            try:
                c.close()
            except Exception:
                pass
    return out_path


def build_segmented_voiceover(
    segments: list,
    voice: str = "xiaoxiao",
    work_dir: Optional[str] = None,
) -> Tuple[Optional[str], Dict[int, Tuple[float, float]]]:
    """逐镜头配音并合成一条「与画面时间轴对齐」的配音轨。

    背景（修复的核心缺陷）：
      原实现把所有镜头台词拼成**一整段**文本一次性 TTS，配音从 t=0 连续播放；
      而字幕是按各镜头视频时长（video_duration）排布的。二者时长来源完全不同
      （TTS 真实语速 vs 生视频模型时长），必然错位——表现为「台词还没念完画面
      就结束了」「字幕与声音对不上」。

    本函数改为：
      1. 对每个镜头单独 TTS，拿到该段配音的**真实时长**；
      2. 把每段配音放到它在成片中的真实起点（segment_start，已扣除转场重叠）；
      3. 返回合并后的配音轨 + 每段真实 [start, end]，供 build_srt 直接排字幕，
         使字幕与配音同源，严格同步。

    Args:
        segments: [{"index": int, "text": str, "start": float}, ...]
                  start 为该镜头在成片中的起始秒数（stitcher 回写的 segment_start）。
        voice: 音色 key。
        work_dir: 中间产物目录，默认系统临时目录。

    Returns:
        (配音轨路径或 None, {scene_index: (start_sec, end_sec)})
    """
    from .tts import generate_voiceover

    timings: Dict[int, Tuple[float, float]] = {}
    if not segments:
        return None, timings

    work_dir = work_dir or tempfile.gettempdir()
    os.makedirs(work_dir, exist_ok=True)

    clips = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        si = seg.get("index")
        start = float(seg.get("start") or 0.0)
        if not text or si is None:
            continue

        out_path = os.path.join(work_dir, f"vo_{si:03d}.mp3")
        path = generate_voiceover(text, voice=voice, out_path=out_path)
        if not path or not os.path.exists(path):
            print(f"    [WARN] 镜头 {si+1} 配音生成失败，该段无配音")
            continue

        try:
            ac = AudioFileClip(path)
        except Exception as e:
            print(f"    [WARN] 镜头 {si+1} 配音读取失败: {e}")
            continue

        dur = float(ac.duration or 0.0)
        if dur <= 0.01:
            try:
                ac.close()
            except Exception:
                pass
            continue

        # 放到该镜头在成片中的真实起点
        clips.append(ac.with_start(max(start, 0.0)))
        timings[int(si)] = (round(max(start, 0.0), 3), round(max(start, 0.0) + dur, 3))
        print(f"    ✓ 镜头 {si+1} 配音 {dur:.2f}s → 放置于 {start:.2f}s")

    if not clips:
        return None, timings

    # 总长 = 最后一段的结束时刻，保证台词完整不被截断
    total = max(end for _, end in timings.values())
    merged = CompositeAudioClip(clips).with_duration(total)

    out_path = os.path.join(work_dir, "voiceover_merged.mp3")
    try:
        merged.write_audiofile(out_path, logger=None)
    except Exception as e:
        print(f"    [WARN] 配音轨合并失败: {e}")
        out_path = None
    finally:
        for c in clips:
            try:
                c.close()
            except Exception:
                pass

    return (out_path if out_path and os.path.exists(out_path) else None), timings


# P1-5：SFX 曲库目录 assets/sfx/<类别>/*.mp3|wav（相对项目根，可用 env 覆盖）
_SFX_LIBRARY_DIR = os.getenv("SFX_LIBRARY_DIR", os.path.join(_PROJECT_ROOT, "assets", "sfx"))

# 转场类型 → SFX 类别回退链（曲库有哪类用哪类，全缺则跳过该切点）
_TRANSITION_SFX = {
    "smash_cut": ("impact", "hit"),
    "whip_pan": ("whoosh", "swish"),
    "camera_carry": ("whoosh", "swish"),
    "dissolve": ("soft", "transition"),
    "fade_through_black": ("deep", "transition"),
}


def _pick_sfx(cats):
    """按类别回退链从 SFX 曲库挑一条音效；曲库为空返回 None。"""
    for c in cats:
        d = os.path.join(_SFX_LIBRARY_DIR, c)
        if os.path.isdir(d):
            files = sorted(
                os.path.join(d, f) for f in os.listdir(d)
                if f.lower().endswith(_BGM_EXTS)
            )
            if files:
                return random.choice(files)
    return None


def build_sfx_events(scenes, lead: float = 0.15, volume: float = 0.6) -> list:
    """P1-5：按跨场景转场类型，在成片切点排布 SFX 事件。

    与 stitcher_node 完全同源：对相邻有效镜头用 select_cross_scene_transition
    重算转场（确定性纯函数），非硬切的转场在其切点前 lead 秒铺一条音效。

    Args:
        scenes: 各镜头 dict（消费 segment_start / beat / emotion / shot_failed）。
        lead: 音效相对切点的提前量（秒）。
        volume: 音效音量倍率。

    Returns:
        [{"path": 音效路径, "start": 起始秒, "volume": 音量}, ...]；
        曲库为空 / 全是硬切时返回空列表（调用方静默跳过）。
    """
    used = [
        s for s in (scenes or [])
        if isinstance(s.get("segment_start"), (int, float))
        and not s.get("shot_failed")
        and (s.get("final_video_url") or s.get("raw_video_url"))
    ]
    if len(used) < 2:
        return []

    # 延迟导入：sequence_orchestrator 属上层包，避免 tools→上层 的模块级循环依赖
    try:
        from ..sequence_orchestrator import select_cross_scene_transition
    except Exception as e:  # noqa: BLE001
        print(f"    [WARN] SFX 排布跳过（无法导入转场选择器）: {e}")
        return []

    events = []
    for i in range(1, len(used)):
        prev, curr = used[i - 1], used[i]
        try:
            t = select_cross_scene_transition(
                prev_beat=prev.get("beat", ""),
                curr_beat=curr.get("beat", ""),
                prev_emotion=prev.get("emotion", ""),
                curr_emotion=curr.get("emotion", ""),
            )
        except Exception:
            continue
        cats = _TRANSITION_SFX.get(t)
        if not cats:
            continue  # hard_cut：不加音效
        start = float(curr["segment_start"]) - lead
        if start <= 0.3:
            continue  # 片头附近不铺
        path = _pick_sfx(cats)
        if path:
            events.append({"path": path, "start": round(start, 3), "volume": volume})
    return events


def has_audio_track(video_path: str) -> bool:
    """判断视频文件是否自带音轨（音频流）。

    背景：即梦 Seedance 1.5 Pro（视频3.5 Pro）等模型会生成带原生音轨
    （环境音效/人声/音乐）的视频；而 dashscope 等 t2v 返回的视频通常是静音。
    本函数供 audio_mixer_node 在「原生音轨优先」模式下判断是否跳过本地 TTS 混音。

    实现：用 moviepy 自带的 ffmpeg binary 探测音轨（系统无需安装 ffmpeg），
    通过解析 `ffmpeg -i file` 的 stderr 判断是否存在 Audio 流。

    Returns:
        存在音频流返回 True；探测失败/异常时返回 False（保守视为无音轨）。
    """
    if not video_path or not os.path.exists(video_path):
        return False
    try:
        import subprocess
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        p = subprocess.run(
            [exe, "-i", video_path],
            capture_output=True,
            text=True,
            errors="ignore",
            timeout=30,
        )
        stderr = (p.stderr or "") + (p.stdout or "")
        return "Audio:" in stderr
    except Exception as e:
        print(f"    [WARN] 音轨检测失败（视为无音轨）: {e}")
        return False
