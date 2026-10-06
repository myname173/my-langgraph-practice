"""P2 接缝平滑层（零额度单测）。

覆盖：
- match_cut 接入真实短溶解（修复“计划 match_cut → 实际硬切”的静默降级）；
- 接缝平滑模式/时长/冻结 的 env 读取与钳制；
- `_seam_is_hard` 判定；
- `_apply_seam_smoothing` 三模式（dissolve / freeze / off）行为；
- 溶解使成片总长按重叠缩短（证明真实生效）。

用 moviepy ColorClip 构造假片段，不依赖 ffmpeg 生成素材。
"""
import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

vs = pytest.importorskip("agent.multimedia.tools.video_stitcher")

try:
    from moviepy import ColorClip
except ImportError:  # moviepy 1.x
    from moviepy.editor import ColorClip


def _clip(color=(200, 30, 30), duration=1.0):
    return ColorClip(size=(320, 240), color=color, duration=duration).with_fps(24)


# ── 常量 / match_cut 接线 ───────────────────────────────────────────────
def test_match_cut_in_effect_transitions():
    assert "match_cut" in vs._EFFECT_TRANSITIONS


def test_hard_seam_types():
    assert vs._seam_is_hard("hard_cut") is True
    assert vs._seam_is_hard("") is True
    assert vs._seam_is_hard("match_cut") is False
    assert vs._seam_is_hard("dissolve") is False
    assert vs._seam_is_hard("camera_carry") is False


# ── 模式 / 时长 / 冻结 ──────────────────────────────────────────────────
def test_mode_default_dissolve(monkeypatch):
    monkeypatch.delenv("MULTIMEDIA_SEAM_SMOOTH", raising=False)
    assert vs.seam_smooth_mode() == "dissolve"


def test_mode_env_override(monkeypatch):
    monkeypatch.setenv("MULTIMEDIA_SEAM_SMOOTH", "freeze")
    assert vs.seam_smooth_mode() == "freeze"
    monkeypatch.setenv("MULTIMEDIA_SEAM_SMOOTH", "off")
    assert vs.seam_smooth_mode() == "off"


def test_mode_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("MULTIMEDIA_SEAM_SMOOTH", "banana")
    assert vs.seam_smooth_mode() == "dissolve"


def test_duration_default_and_clamp(monkeypatch):
    monkeypatch.delenv("MULTIMEDIA_SEAM_DURATION", raising=False)
    assert vs.seam_smooth_duration() == 0.4
    monkeypatch.setenv("MULTIMEDIA_SEAM_DURATION", "99")
    assert vs.seam_smooth_duration() == 0.6
    monkeypatch.setenv("MULTIMEDIA_SEAM_DURATION", "0.001")
    assert vs.seam_smooth_duration() == 0.08


def test_freeze_default_and_clamp(monkeypatch):
    monkeypatch.delenv("MULTIMEDIA_SEAM_FREEZE", raising=False)
    assert vs.seam_freeze_seconds() == 0.2
    monkeypatch.setenv("MULTIMEDIA_SEAM_FREEZE", "9")
    assert vs.seam_freeze_seconds() == 0.5


# ── _apply_seam_smoothing ───────────────────────────────────────────────
def test_off_keeps_hard(monkeypatch):
    monkeypatch.setenv("MULTIMEDIA_SEAM_SMOOTH", "off")
    clips = [_clip(), _clip()]
    out_clips, trans, changed = vs._apply_seam_smoothing(clips, ["hard_cut"])
    assert changed is False
    assert trans == ["hard_cut"]
    assert out_clips is clips


def test_dissolve_converts_hard_seam(monkeypatch):
    monkeypatch.setenv("MULTIMEDIA_SEAM_SMOOTH", "dissolve")
    clips = [_clip(), _clip()]
    out_clips, trans, changed = vs._apply_seam_smoothing(clips, ["hard_cut"])
    assert changed is True
    assert trans == ["dissolve"]
    assert out_clips is clips  # dissolve 不改片段本身


def test_dissolve_leaves_match_cut(monkeypatch):
    # match_cut 由引擎自身处理，平滑层不再重复改写
    monkeypatch.setenv("MULTIMEDIA_SEAM_SMOOTH", "dissolve")
    clips = [_clip(), _clip()]
    _, trans, changed = vs._apply_seam_smoothing(clips, ["match_cut"])
    assert changed is False
    assert trans == ["match_cut"]


def test_dissolve_skips_short_clips(monkeypatch):
    monkeypatch.setenv("MULTIMEDIA_SEAM_SMOOTH", "dissolve")
    clips = [_clip(duration=0.1), _clip(duration=0.1)]  # 短于 2*dur → 跳过
    _, trans, changed = vs._apply_seam_smoothing(clips, ["hard_cut"])
    assert changed is False
    assert trans == ["hard_cut"]


def test_freeze_extends_outgoing_clip(monkeypatch):
    monkeypatch.setenv("MULTIMEDIA_SEAM_SMOOTH", "freeze")
    monkeypatch.setenv("MULTIMEDIA_SEAM_FREEZE", "0.2")
    clips = [_clip(duration=1.0), _clip(duration=1.0)]
    out, trans, changed = vs._apply_seam_smoothing(clips, ["hard_cut"])
    assert changed is True
    # 出镜片段被冻结延长；末镜不延长
    assert out[0].duration == pytest.approx(1.2, abs=0.05)
    assert out[1].duration == pytest.approx(1.0, abs=0.05)


# ── 溶解真实生效：成片总长按重叠缩短 ────────────────────────────────────
def test_dissolve_reduces_total_duration():
    clips = [_clip(duration=1.0), _clip(duration=1.0)]
    hard = vs._build_with_transitions(clips, ["hard_cut"], 0.4)
    dis = vs._build_with_transitions(clips, ["dissolve"], 0.4)
    assert hard.duration == pytest.approx(2.0, abs=0.05)
    assert dis.duration == pytest.approx(1.6, abs=0.05)  # 2.0 - 0.4 重叠
    hard.close()
    dis.close()


def test_match_cut_now_overlaps_like_dissolve():
    # 修复点：match_cut 现在与 dissolve 一样产生重叠（不再退化为硬切）
    clips = [_clip(duration=1.0), _clip(duration=1.0)]
    mc = vs._build_with_transitions(clips, ["match_cut"], 0.4)
    assert mc.duration == pytest.approx(1.6, abs=0.05)
    mc.close()
