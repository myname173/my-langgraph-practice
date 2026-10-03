"""
图路由函数单测（零 LLM / 零模型厂商依赖，只验证决策分支）。

验证 decide_* 路由函数（定义于 agent.multimedia.graph）在给定 state 下
返回正确的目标节点，从而佐证简历中所述的质量门 / 重写上限 / 角色一致性闭环
等行为是真实生效的，而非口头描述。

依赖：导入 graph 模块会带入 langgraph 等重型依赖；若运行环境缺失则自动 skip，
不破坏 CI。其余常量（QUALITY_THRESHOLDS / MAX_VIDEO_GEN_FAILURES / 角色一致性阈值）
从对应轻量模块导入，确保断言值与代码配置一致。
"""
import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# graph 模块顶部依赖 langgraph 等重型库；缺失时整体 skip。
graph = pytest.importorskip("agent.multimedia.graph")
from agent.multimedia.cinematic_critic import QUALITY_THRESHOLDS  # noqa: E402
from agent.multimedia.config_loader import (  # noqa: E402
    max_video_gen_failures,
    character_consistency_threshold,
)


# ----------------------------------------------------------------
# 测试夹具：构造最小可驱动路由函数的 state
# ----------------------------------------------------------------
def _base_scenes(n=1):
    """生成 n 个最小镜头字典，字段对齐 graph 路由读取路径。"""
    scenes = []
    for _ in range(n):
        scenes.append(
            {
                "is_perfect": False,
                "iterations": 0,
                "image_url": None,
                "portrait_similarity": None,
                "last_image_is_perfect": False,
                "last_image_iterations": 0,
                "last_image_url": None,
                "video_is_perfect": False,
                "video_iterations": 0,
                "raw_video_url": None,
                "video_gen_failures": 0,
                "script": "",
            }
        )
    return scenes


def _base_state(**overrides):
    state = {
        "current_scene_index": 0,
        "scenes": _base_scenes(1),
        "use_first_last_frame": False,
        "aborted": False,
        "rewrite_count": 0,
        "critic_eval": None,
    }
    state.update(overrides)
    return state


# ----------------------------------------------------------------
# decide_after_critic —— Phase 4 质量门控（简历核心量化点）
# ----------------------------------------------------------------
def test_decide_after_critic_pass_gates_goes_director():
    state = _base_state(
        critic_eval={"overall_score": 0.85, "pass_gates": True}
    )
    assert graph.decide_after_critic(state) == "director"


def test_decide_after_critic_fail_below_max_rewrites_triggers_rewrite():
    # 未达上限（默认 2）且未通过 → 走重写分支
    state = _base_state(
        rewrite_count=0,
        critic_eval={"overall_score": 0.50, "pass_gates": False},
    )
    assert graph.decide_after_critic(state) == "rewrite"


def test_decide_after_critic_fail_at_max_rewrites_forces_director():
    # 已达 max_rewrites（配置值） → 强制放行，绝不死循环
    max_rw = QUALITY_THRESHOLDS.get("max_rewrites", 2)
    state = _base_state(
        rewrite_count=max_rw,
        critic_eval={"overall_score": 0.40, "pass_gates": False},
    )
    assert graph.decide_after_critic(state) == "director"


def test_decide_after_critic_missing_eval_goes_director():
    assert graph.decide_after_critic(_base_state(critic_eval=None)) == "director"


# ----------------------------------------------------------------
# decide_image_quality —— 角色一致性闭环
# ----------------------------------------------------------------
def test_decide_image_quality_perfect_without_ff_goes_videographer():
    state = _base_state()
    state["scenes"][0]["is_perfect"] = True
    assert graph.decide_image_quality(state) == "videographer"


def test_decide_image_quality_perfect_with_ff_goes_end_frame_director():
    state = _base_state(use_first_last_frame=True)
    state["scenes"][0]["is_perfect"] = True
    assert graph.decide_image_quality(state) == "end_frame_director"


def test_decide_image_quality_low_portrait_sim_triggers_regeneration():
    # VLM 判 PASS，但角色肖像相似度低于阈值且仍有配额 → 强制重生一次
    thr = character_consistency_threshold()
    state = _base_state()
    scene = state["scenes"][0]
    scene["is_perfect"] = True
    scene["iterations"] = 0
    scene["portrait_similarity"] = thr - 0.1  # 明显低于阈值
    assert graph.decide_image_quality(state) == "director"


def test_decide_image_quality_low_sim_but_quota_exhausted_passes():
    # 一致性重试预算耗尽（consistency_attempts >= 3）→ 即便相似度不足也放行，避免白耗额度。
    # 该预算由 consistency_attempts 独立计数（P1-4 起与通用 iterations 分离）。
    thr = character_consistency_threshold()
    state = _base_state()
    scene = state["scenes"][0]
    scene["is_perfect"] = True
    scene["consistency_attempts"] = 3
    scene["portrait_similarity"] = thr - 0.1
    assert graph.decide_image_quality(state) == "videographer"


# ----------------------------------------------------------------
# decide_after_image_generation / video / end_frame —— 失败重试与跳过
# ----------------------------------------------------------------
def test_decide_after_image_generation_success_goes_reviewer():
    state = _base_state()
    state["scenes"][0]["image_url"] = "output/kf.png"
    assert graph.decide_after_image_generation(state) == "reviewer"


def test_decide_after_image_generation_fail_retry_within_quota():
    max_fail = max_video_gen_failures()
    state = _base_state()
    state["scenes"][0]["video_gen_failures"] = max_fail - 1
    assert graph.decide_after_image_generation(state) == "retry"


def test_decide_after_image_generation_fail_skip_at_quota():
    max_fail = max_video_gen_failures()
    state = _base_state()
    state["scenes"][0]["video_gen_failures"] = max_fail
    assert graph.decide_after_image_generation(state) == "skip"


def test_decide_after_image_generation_aborted_goes_abort():
    state = _base_state(aborted=True)
    assert graph.decide_after_image_generation(state) == "abort"


def test_decide_after_video_generation_success_goes_video_reviewer():
    state = _base_state()
    state["scenes"][0]["raw_video_url"] = "output/v.mp4"
    assert graph.decide_after_video_generation(state) == "video_reviewer"


def test_decide_after_video_generation_fail_skip_at_quota():
    max_fail = max_video_gen_failures()
    state = _base_state()
    state["scenes"][0]["video_gen_failures"] = max_fail
    assert graph.decide_after_video_generation(state) == "skip"


def test_decide_after_end_frame_generation_success_goes_review():
    state = _base_state()
    state["scenes"][0]["last_image_url"] = "output/ef.png"
    assert graph.decide_after_end_frame_generation(state) == "end_frame_review"


# ----------------------------------------------------------------
# decide_end_frame_quality / decide_video_quality —— 达标放行
# ----------------------------------------------------------------
def test_decide_end_frame_quality_perfect_goes_videographer():
    state = _base_state()
    state["scenes"][0]["last_image_is_perfect"] = True
    assert graph.decide_end_frame_quality(state) == "videographer"


def test_decide_end_frame_quality_fail_goes_end_frame_director():
    state = _base_state()
    state["scenes"][0]["last_image_is_perfect"] = False
    state["scenes"][0]["last_image_iterations"] = 0
    assert graph.decide_end_frame_quality(state) == "end_frame_director"


def test_decide_video_quality_perfect_goes_advance_scene():
    state = _base_state()
    state["scenes"][0]["video_is_perfect"] = True
    assert graph.decide_video_quality(state) == "advance_scene"


def test_decide_video_quality_fail_goes_videographer():
    state = _base_state()
    state["scenes"][0]["video_is_perfect"] = False
    state["scenes"][0]["video_iterations"] = 0
    assert graph.decide_video_quality(state) == "videographer"


# ----------------------------------------------------------------
# decide_next_scene —— 多镜头推进 / 收尾
# ----------------------------------------------------------------
def test_decide_next_scene_continues_when_more_scenes():
    state = _base_state()
    state["scenes"] = _base_scenes(3)
    state["current_scene_index"] = 0
    assert graph.decide_next_scene(state) == "continue"


def test_decide_next_scene_reaches_end_goes_stitcher():
    state = _base_state()
    state["scenes"] = _base_scenes(3)
    state["current_scene_index"] = 3  # 已超出，全部制作完毕
    assert graph.decide_next_scene(state) == "stitcher"


# ----------------------------------------------------------------
# decide_after_prompt_preview / end_frame_prompt_preview —— 人工审核后续路由
# ----------------------------------------------------------------
def test_decide_after_prompt_preview_perfect_goes_image_generator():
    state = _base_state()
    state["scenes"][0]["is_perfect"] = True
    assert graph.decide_after_prompt_preview(state) == "image_generator"


def test_decide_after_prompt_preview_not_perfect_goes_director():
    state = _base_state()
    state["scenes"][0]["is_perfect"] = False
    assert graph.decide_after_prompt_preview(state) == "director"


def test_decide_after_end_frame_prompt_preview_perfect_goes_generator():
    state = _base_state()
    state["scenes"][0]["last_image_is_perfect"] = True
    assert graph.decide_after_end_frame_prompt_preview(state) == "end_frame_generator"


def test_decide_after_end_frame_prompt_preview_not_perfect_goes_director():
    state = _base_state()
    state["scenes"][0]["last_image_is_perfect"] = False
    assert graph.decide_after_end_frame_prompt_preview(state) == "end_frame_director"
