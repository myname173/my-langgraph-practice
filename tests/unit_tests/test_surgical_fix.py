"""P1-3：镜头级外科手术（fix_type 修正粒度）零额度纯逻辑单测。

验证 `_normalize_fix_type` 归一化、`_normalize_decision` 透传，以及两处路由
（decide_image_quality / decide_video_quality）按 fix_type 选择的重试粒度。

依赖：导入 graph 会带入 langgraph 等重型依赖；缺失则整体 skip，不破坏 CI。
"""
import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

graph = pytest.importorskip("agent.multimedia.graph")


# ── _normalize_fix_type ────────────────────────────────────────────────
@pytest.mark.parametrize("raw,expect", [
    ("prompt_only", "prompt_only"),
    ("edit_prompt", "prompt_only"),
    ("改词", "prompt_only"),
    ("keep_first_frame", "keep_first_frame"),
    ("保首帧", "keep_first_frame"),
    ("reimage", "reimage"),
    ("重绘", "reimage"),
    ("extend", "extend"),
    ("延长", "extend"),
    ("prolong", "extend"),
    (None, None),
    ("nonsense", None),
    ("", None),
])
def test_normalize_fix_type(raw, expect):
    assert graph._normalize_fix_type(raw) == expect


def test_normalize_decision_passthrough():
    d = graph._normalize_decision({"action": "rewrite", "fix_type": "改词"})
    assert d["action"] == "rewrite" and d["fix_type"] == "prompt_only"
    d2 = graph._normalize_decision({"action": "rewrite", "fix": "extend"})
    assert d2["fix_type"] == "extend"
    # 未提供 -> 不注入 fix_type 键
    d3 = graph._normalize_decision({"action": "approve"})
    assert "fix_type" not in d3


# ── decide_video_quality ───────────────────────────────────────────────
def _vstate(fix_type=None, iterations=0):
    scene = {
        "video_is_perfect": False,
        "video_iterations": iterations,
        "image_url": "x.png",
        "video_prompt": "pan left",
        "duration_seconds": 8,
    }
    if fix_type is not None:
        scene["fix_type"] = fix_type
    return {"current_scene_index": 0, "scenes": [scene]}


def test_video_quality_prompt_only_skips_videographer():
    assert graph.decide_video_quality(_vstate("prompt_only")) == "video_generator"


def test_video_quality_extend_goes_direct_to_generator():
    assert graph.decide_video_quality(_vstate("extend")) == "video_generator"


def test_video_quality_reimage_returns_director():
    assert graph.decide_video_quality(_vstate("reimage")) == "director"


def test_video_quality_keep_first_frame_returns_videographer():
    assert graph.decide_video_quality(_vstate("keep_first_frame")) == "videographer"


def test_video_quality_default_returns_videographer():
    assert graph.decide_video_quality(_vstate(None)) == "videographer"


def test_video_quality_passed_advances():
    st = _vstate("reimage")
    st["scenes"][0]["video_is_perfect"] = True
    assert graph.decide_video_quality(st) == "advance_scene"


def test_video_quality_iterations_exhausted_advances():
    assert graph.decide_video_quality(_vstate("reimage", iterations=3)) == "advance_scene"


def test_video_quality_reviewer_unavailable_advances():
    st = _vstate("reimage")
    st["reviewer_unavailable"] = True
    assert graph.decide_video_quality(st) == "advance_scene"


# ── decide_image_quality ───────────────────────────────────────────────
def _istate(fix_type=None, passed=False, use_flf=True, iterations=0):
    scene = {
        "is_perfect": passed,
        "iterations": iterations,
        "image_url": "kf.png",
        "portrait_similarity": None,
        "script": "a wide shot",
        "last_image_is_perfect": False,
        "last_image_iterations": 0,
    }
    if fix_type is not None:
        scene["fix_type"] = fix_type
    return {
        "current_scene_index": 0,
        "scenes": [scene],
        "use_first_last_frame": use_flf,
        "reviewer_unavailable": False,
    }


def test_image_quality_reimage_returns_director():
    assert graph.decide_image_quality(_istate("reimage")) == "director"


def test_image_quality_keep_first_frame_skips_reimage_to_video():
    # 保留关键帧 -> 跳过重绘，直接进入视频侧（flf 开启走 end_frame_director）
    assert graph.decide_image_quality(_istate("keep_first_frame", use_flf=True)) == "end_frame_director"
    assert graph.decide_image_quality(_istate("prompt_only", use_flf=False)) == "videographer"


def test_image_quality_keep_image_requires_image_present():
    st = _istate("keep_first_frame")
    st["scenes"][0]["image_url"] = ""
    # 无图无法保留 -> 回退重绘
    assert graph.decide_image_quality(st) == "director"


def test_image_quality_passed_normal_path():
    assert graph.decide_image_quality(_istate(None, passed=True, use_flf=True)) == "end_frame_director"
