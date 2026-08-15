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
import random
from typing import Optional

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


def mix_audio(
    video_path: str,
    voiceover_path: str,
    bgm_path: Optional[str] = None,
    bgm_volume: float = 0.25,
    out_path: Optional[str] = None,
) -> str:
    """把配音（与可选 BGM）混入视频，返回带音轨的成片路径。"""
    if not voiceover_path or not os.path.exists(voiceover_path):
        return video_path

    video = VideoFileClip(video_path)
    voice = AudioFileClip(voiceover_path)

    clips = [voice]
    bgm = None
    if bgm_path and os.path.exists(bgm_path):
        bgm = AudioFileClip(bgm_path)
        # BGM 短于视频时循环铺满，长于视频时截断，并加淡入淡出避免突兀
        effects = []
        if bgm.duration < video.duration:
            effects.append(afx.AudioLoop(duration=video.duration))
        effects.extend([afx.AudioFadeIn(1.0), afx.AudioFadeOut(1.5)])
        bgm = (
            bgm.with_effects(effects)
            .with_volume_scaled(bgm_volume)
            .with_duration(video.duration)
        )
        clips.append(bgm)

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
    return out_path


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
