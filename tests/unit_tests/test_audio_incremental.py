"""
增量功能测试 —— BGM 选取与音画合成。

不触发完整 pipeline（无需 DashScope 额度、不跑 LangGraph 编排），
直接驱动 src/agent/multimedia/tools/audio.py 的纯本地工具链：
  - resolve_bgm_mood: 场景情绪 → BGM mood（含未知情绪回退）
  - generate_bgm:     按 mood 从 assets/bgm/ 预置曲库选曲（含回退 ambient / 空库返回 None）
  - mix_audio:        配音 + BGM 混入视频，产出带音轨成片（BGM 短则循环铺满、长则截断 + 淡入淡出）

设计约束（见 plan）：
  - 零外部网络依赖：仅用 moviepy 自造无声样本视频 + 复现 assets/bgm 曲库作为 BGM/配音占位。
  - 不污染业务产物：所有临时文件落在系统临时目录，测试结束清理。
  - 不修改主链路代码，仅做只读验证。

运行：
    python -m pytest tests/unit_tests/test_audio_incremental.py -v
或
    python tests/unit_tests/test_audio_incremental.py
"""
import os
import sys
import tempfile
import shutil
import unittest

# 把项目 src 加入 import 路径（脚本直跑时也可用）。
# 项目采用 src 布局，包根在 <root>/src， pytest 通过 pyproject 的 pythonpath 配置
# 已能解析 `agent`，脚本直跑时需手动把 src 加进来。
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from moviepy import VideoFileClip, AudioFileClip, ColorClip  # noqa: E402

from agent.multimedia.tools.audio import (  # noqa: E402
    resolve_bgm_mood,
    generate_bgm,
    mix_audio,
    BGM_LIBRARY_DIR,
    BGM_MOOD_BY_EMOTION,
)


def _make_silent_video(path: str, duration: float = 2.0, size=(320, 180)):
    """用 moviepy 合成一段无声彩条视频，作为 mix_audio 的视频样本。"""
    clip = ColorClip(size=size, color=(30, 60, 120), duration=duration)
    clip.fps = 24
    clip.write_videofile(path, codec="libx264", audio=False)
    clip.close()


def _first_candidate(mood: str) -> str:
    """返回某 mood 目录下第一个真实存在的音频文件绝对路径。"""
    mood_dir = os.path.join(BGM_LIBRARY_DIR, mood)
    if not os.path.isdir(mood_dir):
        raise FileNotFoundError(f"BGM 曲库目录缺失: {mood_dir}")
    for f in sorted(os.listdir(mood_dir)):
        if f.lower().endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac")):
            return os.path.join(mood_dir, f)
    raise FileNotFoundError(f"BGM mood '{mood}' 目录为空")


class TestResolveBgmMood(unittest.TestCase):
    def test_known_emotions_map(self):
        for emotion, expected in BGM_MOOD_BY_EMOTION.items():
            with self.subTest(emotion=emotion):
                self.assertEqual(resolve_bgm_mood(emotion), expected)

    def test_unknown_emotion_falls_back(self):
        self.assertEqual(resolve_bgm_mood("mystery"), "ambient")
        self.assertEqual(resolve_bgm_mood("未知情绪"), "ambient")

    def test_none_and_empty_use_fallback(self):
        self.assertEqual(resolve_bgm_mood(None), "ambient")
        self.assertEqual(resolve_bgm_mood(""), "ambient")
        self.assertEqual(resolve_bgm_mood("  ", fallback="epic"), "epic")

    def test_case_insensitive(self):
        self.assertEqual(resolve_bgm_mood("TENSE"), "tense")
        self.assertEqual(resolve_bgm_mood("Joyful"), "upbeat")


class TestGenerateBgm(unittest.TestCase):
    def test_seed_reproducible_and_returns_existing_file(self):
        path1 = generate_bgm(mood="tense", seed=42)
        path2 = generate_bgm(mood="tense", seed=42)
        self.assertIsNotNone(path1)
        self.assertEqual(path1, path2)  # 同 seed 必选同一首
        self.assertTrue(os.path.exists(path1), "[BGM] 选曲应存在")

    def test_missing_mood_falls_back_to_ambient(self):
        # 找一个真实存在但不在曲库中的 mood 名：用不存在的目录
        path = generate_bgm(mood="__nonexistent_mood__", seed=1)
        self.assertIsNotNone(path)
        self.assertIn("ambient", path.replace("\\", "/"), "应回退到 ambient")

    def test_empty_library_returns_none(self):
        import agent.multimedia.tools.audio as audio_mod

        # 临时把曲库指向一个空临时目录，验证空库返回 None
        tmp = tempfile.mkdtemp(prefix="bgm_empty_")
        try:
            orig = audio_mod.BGM_LIBRARY_DIR
            audio_mod.BGM_LIBRARY_DIR = tmp
            self.assertIsNone(generate_bgm(mood="ambient"))
        finally:
            audio_mod.BGM_LIBRARY_DIR = orig
            shutil.rmtree(tmp, ignore_errors=True)


class TestMixAudio(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="audio_inc_")
        self.video = os.path.join(self.tmp, "sample_silent.mp4")
        _make_silent_video(self.video, duration=2.0)
        # 用真实 BGM 文件同时充当配音占位与 BGM（验证混流 + 循环铺满 + 淡入淡出）
        self.bgm = _first_candidate("tense")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_mix_audio_produces_output_with_audio_track(self):
        out = os.path.join(self.tmp, "final_movie_with_audio.mp4")
        result = mix_audio(
            video_path=self.video,
            voiceover_path=self.bgm,  # 配音占位
            bgm_path=self.bgm,        # BGM（短于视频 → 应循环铺满）
            bgm_volume=0.3,
            out_path=out,
        )
        self.assertTrue(os.path.exists(result), "应产出带音轨成片")
        self.assertGreater(os.path.getsize(result), 0, "产物不应为空")

        # 验证产物确实带音轨，且时长与视频对齐
        with AudioFileClip(result) as a:
            self.assertGreater(a.duration, 0, "产物应含音轨")
        # mix_audio 不裁剪视频长度，时长应与原视频一致
        with VideoFileClip(result) as v:
            self.assertAlmostEqual(v.duration, 2.0, delta=0.2)

    def test_mix_audio_skips_when_voiceover_missing(self):
        # 配音缺失时应直接返回原视频路径（不报错）
        result = mix_audio(
            video_path=self.video,
            voiceover_path="",  # 空 → 跳过混音
            bgm_path=self.bgm,
        )
        self.assertEqual(result, self.video)


if __name__ == "__main__":
    unittest.main(verbosity=2)
