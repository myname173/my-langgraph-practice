"""
Cinematic Memory Bank (Phase 5)
=================================
职责：维护跨场景的电影风格一致性。
      记录已处理场景的摄影风格、光照风格、剪辑风格，
      供后续场景的专业 agent 参考。

架构位置：
    全局状态（非节点），被 film_studio_orchestrator 读写。

设计原则：
    - 轻量级 dict 结构，无向量数据库
    - 按场景索引存储风格快照
    - 提供 get_current_style() 聚合最近风格趋势
    - 新 trailer 开始时 reset
"""

from typing import Dict, List, Any, Optional


# ============================================================
# 1. 全局记忆库（模块级单例）
# ============================================================

_memory_bank: Dict[int, Dict[str, Any]] = {}


# ============================================================
# 2. 记忆操作函数
# ============================================================

def record_scene_style(
    scene_index: int,
    cinematography: Dict[str, Any],
    lighting: Dict[str, Any],
    editing: Dict[str, Any],
) -> None:
    """
    记录一个场景的电影风格快照。

    Args:
        scene_index: 场景索引
        cinematography: cinematography_agent 的 style_summary
        lighting: lighting_agent 的 style_summary
        editing: editor_agent 的 style_summary
    """
    _memory_bank[scene_index] = {
        "scene_index": scene_index,
        "cinematography": cinematography.get("style_summary", {}),
        "lighting": lighting.get("style_summary", {}),
        "editing": editing.get("style_summary", {}),
        "coherence_scores": {
            "camera": cinematography.get("consistency_score", 1.0),
            "lighting": lighting.get("consistency_score", 1.0),
            "pacing": editing.get("pacing_score", 1.0),
        },
    }


def get_previous_scene_style(
    scene_index: int,
) -> Optional[Dict[str, Any]]:
    """
    获取前一个场景的风格记录。

    Returns:
        前一场景的风格快照 dict，或 None（如果没有记录）
    """
    prev_index = scene_index - 1
    return _memory_bank.get(prev_index)


def get_all_recorded_styles() -> List[Dict[str, Any]]:
    """获取所有已记录场景的风格快照列表。"""
    return [_memory_bank[i] for i in sorted(_memory_bank.keys())]


def get_current_style_state() -> Dict[str, Any]:
    """
    获取当前全局风格状态（所有已记录场景的聚合）。

    用于生成"全局电影风格报告"。
    """
    if not _memory_bank:
        return {
            "scenes_recorded": 0,
            "established_style": "none",
            "camera_consistency": 1.0,
            "lighting_consistency": 1.0,
            "pacing_consistency": 1.0,
        }

    records = get_all_recorded_styles()
    n = len(records)

    # 聚合一致性评分
    cam_scores = [r["coherence_scores"]["camera"] for r in records]
    light_scores = [r["coherence_scores"]["lighting"] for r in records]
    pace_scores = [r["coherence_scores"]["pacing"] for r in records]

    # 收集所有使用过的 shot types 和 transitions
    all_shot_types = set()
    all_transitions = set()
    for r in records:
        all_shot_types.update(r.get("cinematography", {}).get("shot_types", []))
        all_transitions.update(r.get("editing", {}).get("transitions", []))

    return {
        "scenes_recorded": n,
        "established_style": "established" if n >= 2 else "forming",
        "camera_consistency": round(sum(cam_scores) / n, 2),
        "lighting_consistency": round(sum(light_scores) / n, 2),
        "pacing_consistency": round(sum(pace_scores) / n, 2),
        "all_shot_types": sorted(all_shot_types),
        "all_transitions": sorted(all_transitions),
    }


def reset_memory() -> None:
    """清空记忆库（新 trailer 开始时调用）。"""
    _memory_bank.clear()


# ============================================================
# 3. 全局电影意识报告（Global Film Brain）
# ============================================================

def generate_film_brain_report(
    global_setting: str,
    total_scenes: int,
) -> str:
    """
    生成全局电影意识报告。
    这是一个"导演意图"摘要，包含整个 trailer 的：
    - 全局基调
    - 视觉身份
    - 已建立的节奏曲线
    - 情绪弧目标

    可注入后续场景的 director prompt，确保全局一致性。
    """
    state = get_current_style_state()

    lines = [
        f"[Global Film Brain — {state['scenes_recorded']}/{total_scenes} scenes processed]",
        f"Style Status: {state['established_style']}",
        f"Camera Consistency: {state['camera_consistency']:.0%}",
        f"Lighting Consistency: {state['lighting_consistency']:.0%}",
        f"Pacing Consistency: {state['pacing_consistency']:.0%}",
    ]

    if state.get("all_shot_types"):
        lines.append(f"Established Shot Vocabulary: {', '.join(state['all_shot_types'])}")
    if state.get("all_transitions"):
        lines.append(f"Established Transition Palette: {', '.join(state['all_transitions'])}")

    if state["scenes_recorded"] >= 2:
        lines.append("")
        lines.append("[Directorial Note]: Maintain established visual language.")
        lines.append("New scenes should feel like they belong in the same film —")
        lines.append("consistent camera grammar, lighting mood, and editing rhythm.")
    elif state["scenes_recorded"] == 1:
        lines.append("")
        lines.append("[Directorial Note]: First scene established the visual tone.")
        lines.append("Next scene should build on this foundation.")

    return "\n".join(lines)
