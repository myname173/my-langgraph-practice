"""P1-1：FLF2V 首尾帧链式衔接 × 并行共存（零额度纯逻辑单测）。

验证 graph 的链切分与扇出路由：
- `_flf_chain_plan`：连续链切分。**默认连续性优先**（单链，全部镜头串行承接尾帧）；
  设 `MULTIMEDIA_FLF_CHAINS=N` 可切成 N 条链（链间并行，链缝不承接）。
- `_fan_out_shots`：FLF 开启时扇出到 `shot_chain_seq`（链），关闭时回退逐镜 `shot_chain`。

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


def _state(n, use_flf=True, locked=None):
    return {
        "task": "t",
        "scenes": [{"script": f"镜头{i+1}", "final_video_url": ""} for i in range(n)],
        "current_scene_index": 0,
        "auto_mode": True,
        "use_first_last_frame": use_flf,
        "critic_eval": {"pass_gates": True, "overall_score": 0.9},
        "quality_tier": "fast",
        "locked_scenes": list(locked or []),
    }


# ── 链切分 ──────────────────────────────────────────────────────────────
def test_chain_plan_none_when_flf_off():
    assert graph._flf_chain_plan(_state(5, use_flf=False)) is None


def test_chain_plan_none_when_single_shot():
    assert graph._flf_chain_plan(_state(1, use_flf=True)) is None


def test_chain_plan_env_zero_disables(monkeypatch):
    monkeypatch.setenv("MULTIMEDIA_FLF_CHAINS", "0")
    assert graph._flf_chain_plan(_state(5)) is None


def test_chain_plan_default_is_continuity_first(monkeypatch):
    # 默认连续性优先：单链，全部镜头串行承接尾帧（无链缝）
    monkeypatch.delenv("MULTIMEDIA_FLF_CHAINS", raising=False)
    assert graph._flf_chain_plan(_state(5)) == [[0, 1, 2, 3, 4]]
    assert graph._flf_chain_plan(_state(4)) == [[0, 1, 2, 3]]
    assert graph._flf_chain_plan(_state(2)) == [[0, 1]]


def test_chain_plan_env_override(monkeypatch):
    # 显式链数：切 N 条连续链（链间并行）
    monkeypatch.setenv("MULTIMEDIA_FLF_CHAINS", "2")
    assert graph._flf_chain_plan(_state(5)) == [[0, 1, 2], [3, 4]]
    monkeypatch.setenv("MULTIMEDIA_FLF_CHAINS", "3")
    assert graph._flf_chain_plan(_state(5)) == [[0, 1], [2, 3], [4]]
    monkeypatch.setenv("MULTIMEDIA_FLF_CHAINS", "1")
    assert graph._flf_chain_plan(_state(4)) == [[0, 1, 2, 3]]


def test_chain_plan_k_ge_n_falls_back(monkeypatch):
    # 链数 >= 镜头数 → 每链 1 镜，与逐镜并行等价 → 不启用链模式
    monkeypatch.setenv("MULTIMEDIA_FLF_CHAINS", "5")
    assert graph._flf_chain_plan(_state(5)) is None


def test_chain_all_locked():
    scenes = [{"final_video_url": "a"}, {"final_video_url": ""}]
    assert graph._chain_all_locked([0], scenes, {0}) is True
    assert graph._chain_all_locked([1], scenes, {1}) is False  # 无成片
    assert graph._chain_all_locked([0, 1], scenes, {0}) is False  # 有未锁定


# ── 扇出路由 ────────────────────────────────────────────────────────────
def test_fan_out_default_single_chain(monkeypatch):
    monkeypatch.delenv("MULTIMEDIA_FLF_CHAINS", raising=False)
    monkeypatch.setenv("MULTIMEDIA_PARALLEL_SHOTS", "1")
    sends = graph._fan_out_shots(_state(5))
    assert isinstance(sends, list) and len(sends) == 1, "默认应为单链"
    assert sends[0].node == "shot_chain_seq"
    assert sends[0].arg["shot_indices"] == [0, 1, 2, 3, 4]


def test_fan_out_chain_parallel_when_env_set(monkeypatch):
    monkeypatch.setenv("MULTIMEDIA_FLF_CHAINS", "2")
    monkeypatch.setenv("MULTIMEDIA_PARALLEL_SHOTS", "1")
    sends = graph._fan_out_shots(_state(5))
    assert {s.node for s in sends} == {"shot_chain_seq"}
    covered = sorted(i for s in sends for i in s.arg["shot_indices"])
    assert covered == [0, 1, 2, 3, 4] and len(sends) == 2


def test_fan_out_per_shot_when_flf_off(monkeypatch):
    monkeypatch.setenv("MULTIMEDIA_PARALLEL_SHOTS", "1")
    sends = graph._fan_out_shots(_state(5, use_flf=False))
    assert isinstance(sends, list)
    assert {s.node for s in sends} == {"shot_chain"}  # 回退逐镜并行
    assert len(sends) == 5


def test_fan_out_serial_when_parallel_disabled(monkeypatch):
    monkeypatch.setenv("MULTIMEDIA_PARALLEL_SHOTS", "0")
    assert graph._fan_out_shots(_state(5)) == "director"
