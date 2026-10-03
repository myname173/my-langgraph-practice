# tests/unit_tests/test_narrative.py
"""P2-5 叙事结构参数化单测：镜头数反推、模板伸缩不变量、提示生成、剧集编排。

运行（需 pytest）：
    pytest tests/unit_tests/test_narrative.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent.multimedia.tools import narrative as N  # noqa: E402


# ── plan_scene_count ────────────────────────────────────────────────────
@pytest.mark.parametrize("dur,ct,expected", [
    (15, "trailer", 3),
    (60, "trailer", 8),
    (40, "short_film", 5),
    (120, "short_film", 12),
    (180, "music_video", 20),
    (24, "unknown", 3),
    (100000, "music_video", 20),
])
def test_plan_scene_count(dur, ct, expected):
    assert N.plan_scene_count(dur, ct) == expected


def test_plan_scene_count_missing_duration_uses_lower_bound():
    assert N.plan_scene_count(None, "documentary") == 4
    assert N.plan_scene_count(0, "short_film") == 3


def test_plan_scene_count_custom_shot_seconds():
    assert N.plan_scene_count(64, "short_film", 16.0) == 4


# ── expand_beats 不变量 ─────────────────────────────────────────────────
@pytest.mark.parametrize("key", list(N.NARRATIVE_TEMPLATES.keys()))
@pytest.mark.parametrize("n", range(1, 16))
def test_expand_beats_invariants(key, n):
    beats = N.expand_beats(key, n)
    assert len(beats) == n
    if n >= 2:
        assert beats[0] == "setup"
        assert beats[-1] == "resolution"


def test_expand_beats_preserves_climax_when_room():
    assert "climax" in N.expand_beats("crescendo", 6)
    assert "climax" in N.expand_beats("arc", 3)


def test_expand_beats_unknown_key_falls_back_to_arc():
    assert N.expand_beats("does_not_exist", 5) == N.expand_beats("arc", 5)


def test_expand_beats_empty():
    assert N.expand_beats("arc", 0) == []
    assert N.expand_beats("arc", 1) == ["setup"]


def test_templates_all_contain_develop():
    """模板 anchors 必须含 develop，否则模板回退自身过不了三幕校验。"""
    for key, tpl in N.NARRATIVE_TEMPLATES.items():
        assert "develop" in tpl["anchors"], key


# ── scene_count_hint ────────────────────────────────────────────────────
def test_scene_count_hint_duration_driven():
    n, hint = N.scene_count_hint(32, "short_film")
    assert n == 4
    assert "4 个" in hint and "32" in hint


def test_scene_count_hint_max_shots_priority():
    n, hint = N.scene_count_hint(999, "trailer", max_shots=2)
    assert n == 2 and "2 个镜头" in hint


def test_scene_count_hint_without_duration_is_soft():
    n, hint = N.scene_count_hint(None, "trailer")
    assert "叙事结构偏好" in hint
    assert "恰好" not in hint


# ── resolve_* ───────────────────────────────────────────────────────────
def test_resolve_target_duration():
    assert N.resolve_target_duration({"target_duration": 30}) == 30.0
    assert N.resolve_target_duration({}) is None
    assert N.resolve_target_duration({"target_duration": "abc"}) is None


def test_resolve_episode():
    assert N.resolve_episode({}) == 1
    assert N.resolve_episode({"episode": 3}) == 3
    assert N.resolve_episode({"episode": -2}) == 1


# ── 模板映射 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("ct,key", [
    ("trailer", "crescendo"),
    ("short_film", "arc"),
    ("documentary", "flat_wave"),
    ("music_video", "rhythmic"),
    ("commercial", "single_peak"),
    ("tutorial", "ascending_steps"),
])
def test_template_for_content_type(ct, key):
    assert N.template_for_content_type(ct) == key


def test_template_for_content_type_unknown():
    assert N.template_for_content_type("zzz") == "arc"


# ── 剧集提示与编排 ─────────────────────────────────────────────────────
def test_episode_beats_hint():
    assert N.episode_beats_hint(1) == ""
    assert "第 2 集" in N.episode_beats_hint(2)


def test_format_episode_anchors_hint():
    anchors = {
        "intent": "对峙收束", "ambient_context": "night", "color_direction": "cool",
        "camera_momentum": "slow push-in", "emotion_energy": 0.83,
        "character_identity": {"has_character": True, "subject": "白衣剑客", "appearance": ["青衫"]},
        "last_frame_url": "http://x/tail.png",
    }
    h = N.format_episode_anchors_hint(anchors)
    for token in ("对峙收束", "0.83", "白衣剑客", "尾帧"):
        assert token in h
    assert N.format_episode_anchors_hint(None) == ""


def test_export_episode_assets():
    ea = N.export_episode_assets(
        {"characters": {"hero": {"url": "u"}}, "props": {}, "environments": {}},
        {"intent": "x"},
    )
    assert ea["series_assets"]["characters"]["hero"]["url"] == "u"
    assert ea["episode_anchors"]["intent"] == "x"
    assert N.export_episode_assets({}, None)["series_assets"] is None


def test_build_series_inputs():
    ins = N.build_series_inputs("主线任务", ["第二集任务", "第三集任务"], 3, 40)
    assert len(ins) == 3
    assert ins[0]["task"] == "主线任务" and ins[0]["episode"] == 1
    assert ins[1]["task"] == "第二集任务" and ins[1]["episode"] == 2
    assert all(i.get("target_duration") == 40 for i in ins)
