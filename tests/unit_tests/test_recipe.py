"""P2-4：镜头配方卡（Shot Recipe）单元测试。

覆盖 tools/recipe.py 的纯逻辑：
- build_shot_recipe / build_task_recipe：从 scene + render_params 提炼完整配方
- recipe_to_markdown：可读渲染
- apply_shot_patch：白名单深合并、空值不覆盖、纯函数、越界报错
- shot_patch_to_scene_fields：补丁 → scene 字段翻译
- save_task_recipe / load_task_recipe：落盘 / 读取往返

零网络、零额度；不依赖 langgraph（仅导入 tools.recipe）。
"""
import json
import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

recipe = pytest.importorskip("agent.multimedia.tools.recipe")


def _scene():
    return {
        "script": "少年握剑立于崖边",
        "action_beat": "抬剑 → 回身",
        "image_prompt": "a young swordsman on a cliff",
        "last_image_prompt": "the swordsman mid-turn",
        "video_prompt": "smooth camera push-in",
        "duration_seconds": 5.0,
        "image_url": "/media/keyframes/t/scene_00.jpg",
        "last_image_url": "/media/keyframes/t/scene_00_last.jpg",
        "final_video_url": "/media/clips/scene_00.mp4",
        "fix_type": "keep_first_frame",
        "render_params": {
            "backend": "agnes",
            "quality_tier": "standard",
            "duration": 5.0,
            "video_prompt": "smooth camera push-in",
            "use_first_last_frame": True,
            "image": {
                "backend": "jimeng",
                "prompt": "a young swordsman on a cliff",
                "size": "2K",
                "ref_images": ["/media/ref/a.jpg"],
                "ref_strength": 0.6,
            },
            "end_frame": {
                "prompt": "the swordsman mid-turn",
                "ref_strength": 0.45,
                "ref_images": ["/media/keyframes/t/scene_00.jpg"],
            },
        },
    }


def _state():
    return {
        "thread_id": "t-demo",
        "task": "仙侠短片",
        "global_setting": "古代仙侠",
        "visual_style": "xianxia",
        "style_description": "cinematic ancient china",
        "style_suffix": ", ink painting",
        "title_card": True,
        "quality_tier": "standard",
        "use_first_last_frame": True,
        "scenes": [_scene(), {"script": "第二镜", "image_url": "/media/k1.jpg"}],
    }


# ── build_shot_recipe ────────────────────────────────────────────────────
def test_build_shot_recipe_full():
    sh = recipe.build_shot_recipe(_scene(), 0, _state())
    assert sh["index"] == 0
    assert sh["image"]["backend"] == "jimeng"
    assert sh["image"]["size"] == "2K"
    assert sh["image"]["ref_strength"] == 0.6
    assert sh["image"]["ref_images"] == ["/media/ref/a.jpg"]
    assert sh["end_frame"]["prompt"] == "the swordsman mid-turn"
    assert sh["end_frame"]["ref_images"] == ["/media/keyframes/t/scene_00.jpg"]
    assert sh["video"]["use_first_last_frame"] is True
    assert sh["video"]["quality_tier"] == "standard"
    assert sh["fix"]["fix_type"] == "keep_first_frame"
    assert sh["artifacts"]["final_video_url"] == "/media/clips/scene_00.mp4"


def test_build_shot_recipe_tolerates_empty_scene():
    sh = recipe.build_shot_recipe({"script": "空镜"}, 5, {})
    assert sh["index"] == 5
    assert sh["image"]["prompt"] == ""
    assert sh["video"]["use_first_last_frame"] is False


# ── build_task_recipe / markdown ─────────────────────────────────────────
def test_build_task_recipe():
    tr = recipe.build_task_recipe(_state())
    assert tr["schema"] == recipe.RECIPE_SCHEMA
    assert tr["thread_id"] == "t-demo"
    assert len(tr["shots"]) == 2
    assert tr["style"]["visual_style"] == "xianxia"
    assert tr["post"]["title_card"] is True


def test_recipe_to_markdown():
    md = recipe.recipe_to_markdown(recipe.build_task_recipe(_state()))
    for token in ("镜头配方卡", "镜头 1", "a young swordsman on a cliff", "keep_first_frame"):
        assert token in md


# ── apply_shot_patch ─────────────────────────────────────────────────────
def test_apply_shot_patch_whitelist_and_purity():
    tr = recipe.build_task_recipe(_state())
    patch = {
        "image": {"prompt": "NEW", "ref_strength": 0.3, "bogus": 1},
        "video": {"duration": 8, "prompt": "NEWV"},
        "fix": {"fix_type": "reimage"},
        "end_frame": {"prompt": ""},  # 空值不覆盖
    }
    patched = recipe.apply_shot_patch(tr, 0, patch)
    p0 = patched["shots"][0]
    assert p0["image"]["prompt"] == "NEW"
    assert p0["image"]["ref_strength"] == 0.3
    assert "bogus" not in p0["image"]
    assert p0["video"]["duration"] == 8
    assert p0["video"]["prompt"] == "NEWV"
    assert p0["fix"]["fix_type"] == "reimage"
    assert p0["end_frame"]["prompt"] == "the swordsman mid-turn"
    assert "recipe_patched_at" in p0["fix"]
    # 纯函数：原配方不被改动
    assert tr["shots"][0]["image"]["prompt"] == "a young swordsman on a cliff"


def test_apply_shot_patch_missing_shot_raises():
    tr = recipe.build_task_recipe(_state())
    with pytest.raises(KeyError):
        recipe.apply_shot_patch(tr, 99, {})


# ── shot_patch_to_scene_fields ───────────────────────────────────────────
def test_shot_patch_to_scene_fields():
    patch = {"image": {"prompt": "P"}, "video": {"prompt": "V", "duration": 8}, "fix": {"fix_type": "reimage"}}
    fields = recipe.shot_patch_to_scene_fields(patch)
    assert fields["image_prompt"] == "P"
    assert fields["video_prompt"] == "V"
    assert fields["duration_seconds"] == 8.0
    assert fields["fix_type"] == "reimage"
    assert "image" in fields["render_params"] and "video" in fields["render_params"]
    assert "last_image_prompt" not in fields


# ── save / load ──────────────────────────────────────────────────────────
def test_save_and_load_roundtrip(tmp_path):
    tr = recipe.build_task_recipe(_state())
    saved = recipe.save_task_recipe(tr, out_dir=str(tmp_path))
    assert os.path.isfile(saved["latest"])
    loaded = recipe.load_task_recipe("t-demo", out_dir=str(tmp_path))
    assert loaded and loaded["task"] == "仙侠短片"
    assert recipe.load_task_recipe("nope", out_dir=str(tmp_path)) is None
    # JSON 可解析（可分享）
    json.loads(open(saved["latest"], encoding="utf-8").read())
