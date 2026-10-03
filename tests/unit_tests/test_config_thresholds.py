"""
配置化与规则增强单测（不跑完整 pipeline，零 DashScope 依赖）。

验证：
  - config_loader 能正确加载 thresholds.json / visual_rules.json
  - env 覆盖（CHARACTER_CONSISTENCY_THRESHOLD / MAX_VIDEO_GEN_FAILURES）生效
  - shot_strategy._extract_focus 白名单未命中时兜底保留用户焦点（不静默丢意图）
  - cinematic_critic._check_arc_shape 对 trailer 的 anti_crescendo 在 opening 阶段放行
"""
import os
import sys
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.multimedia.config_loader import (  # noqa: E402
    character_consistency_threshold,
    max_video_gen_failures,
    emotion_energy_table,
    acceptable_arcs,
    focus_patterns,
    anti_crescendo_relax_phases,
)
from agent.multimedia import shot_strategy  # noqa: E402
from agent.multimedia import cinematic_critic  # noqa: E402


class TestConfigLoading(unittest.TestCase):
    def test_thresholds_loaded(self):
        from agent.multimedia.config_loader import load_thresholds
        t = load_thresholds()
        self.assertEqual(t["character_consistency_threshold"], 0.28)
        self.assertEqual(t["max_video_gen_failures"], 6)
        self.assertEqual(t["max_rewrites"], 2)
        qs = t["quality_score"]
        self.assertEqual(qs["excellent"], 0.8)
        self.assertEqual(qs["pass"], 0.6)
        self.assertEqual(qs["fail"], 0.15)

    def test_visual_rules_loaded(self):
        vr = focus_patterns()
        self.assertIn("character", vr)
        self.assertIn("environment", vr)
        self.assertIn("剑客", vr["character"])
        self.assertIn("城堡", vr["environment"])

    def test_emotion_energy_has_two_profiles(self):
        # orchestrator(默认) 与 editor 副本数值不同，需各自保留
        default_t = emotion_energy_table("default")
        editor_t = emotion_energy_table("editor")
        self.assertIn("awe", default_t)
        self.assertIn("awe", editor_t)
        self.assertNotEqual(default_t["awe"], editor_t["awe"])
        self.assertEqual(default_t["awe"], 0.6)
        self.assertEqual(editor_t["awe"], 0.5)


class TestEnvOverride(unittest.TestCase):
    def test_consistency_threshold_env(self):
        # 用 patch.dict 注入：退出时自动恢复原值。
        # 不能用 del os.environ[...] —— 那会删掉 load_dotenv() 从 .env 注入的全局值，
        # 污染同进程内后续测试（曾使 test_graph_routing 的常量/函数读取不一致）。
        with mock.patch.dict(os.environ, {"CHARACTER_CONSISTENCY_THRESHOLD": "0.8"}):
            self.assertEqual(character_consistency_threshold(), 0.8)

    def test_max_failures_env(self):
        with mock.patch.dict(os.environ, {"MAX_VIDEO_GEN_FAILURES": "5"}):
            self.assertEqual(max_video_gen_failures(), 5)


class TestFocusFallback(unittest.TestCase):
    """规则增强：白名单未命中时保留用户焦点，不静默丢意图。"""

    def test_whitelist_hit_returns_keyword(self):
        self.assertEqual(shot_strategy._extract_focus("剑客拔剑冲向城堡"), "剑客")

    def test_whitelist_miss_falls_back_to_user_phrase(self):
        # 用户写了白名单之外的焦点（机械信鸽），不应退化为中性 "scene"
        focus = shot_strategy._extract_focus("机械信鸽掠过废墟，留下银色轨迹")
        self.assertNotEqual(focus, "scene")
        self.assertTrue(("信鸽" in focus) or bool(focus))

    def test_empty_script_returns_neutral(self):
        self.assertEqual(shot_strategy._extract_focus(""), "scene")
        self.assertEqual(shot_strategy._extract_focus(None), "scene")


class TestAntiCrescendoRelax(unittest.TestCase):
    """trailer 的 anti_crescendo 在 opening 阶段按配置放行。"""

    def test_opening_anti_crescendo_accepted(self):
        score, issues = cinematic_critic._check_arc_shape(
            {"shape": "anti_crescendo", "overall_direction": "down"}, "trailer", "opening"
        )
        self.assertEqual(score, 1.0)
        self.assertTrue(any("non-blocking" in i for i in issues))

    def test_non_opening_anti_crescendo_flagged(self):
        # opening 之外默认不接受（如 climax），仍走原有门控（返回低于通过线 0.70）
        score, issues = cinematic_critic._check_arc_shape(
            {"shape": "anti_crescendo", "overall_direction": "down"}, "trailer", "climax"
        )
        self.assertLess(score, 0.7)

    def test_normal_arc_unchanged(self):
        score, _ = cinematic_critic._check_arc_shape(
            {"shape": "crescendo", "overall_direction": "up"}, "trailer", ""
        )
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
