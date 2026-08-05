"""
Editor Agent (Phase 5)
========================
职责：专注于剪辑系统的质量优化。
      sequence_graph 优化、transition 强化、
      pacing 调整、emotional arc 平滑。

架构位置：
    lighting_agent → 【editor_agent】→ debate

设计原则：
    - 纯规则驱动，无 LLM 调用
    - 只修改 sequence-level 属性（transitions, pacing, arc）
    - 可以 flag 其他 agent 的工作（供 debate 使用）
    - 关注全局节奏和流畅度，而非单镜头细节
"""

from typing import Dict, List, Any


# ============================================================
# 1. 剪辑规则库
# ============================================================

# Pacing phase → 推荐的 transition 强度
_TRANSITION_STRENGTH: Dict[str, List[str]] = {
    "opening":    ["dissolve", "dissolve", "match_cut", "dissolve"],
    "build_up":   ["hard_cut", "camera_carry", "hard_cut", "match_cut"],
    "climax":     ["smash_cut", "whip_pan", "hard_cut", "hard_cut"],
    "resolution": ["fade_through_black", "dissolve", "camera_carry", "dissolve"],
}

# 基于能量差的 transition 优化
_ENERGY_TRANSITION_MAP: Dict[str, Dict[str, str]] = {
    "large_increase": {
        "recommended": "smash_cut",
        "reason": "energy surge requires explosive transition",
    },
    "small_increase": {
        "recommended": "hard_cut",
        "reason": "moderate energy build with clean cut",
    },
    "lateral": {
        "recommended": "camera_carry",
        "reason": "maintained energy level with seamless bridge",
    },
    "small_decrease": {
        "recommended": "dissolve",
        "reason": "gentle energy reduction with smooth blend",
    },
    "large_decrease": {
        "recommended": "fade_through_black",
        "reason": "significant energy drop requires dramatic pause",
    },
}

# Emotion energy levels (shared with sequence_orchestrator)
_EMOTION_ENERGY: Dict[str, float] = {
    "anticipation": 0.4, "adrenaline": 0.8, "intensity": 0.9, "triumph": 0.7,
    "awe": 0.5, "grandeur": 0.6, "insignificance": 0.3, "mystery": 0.35,
    "presence": 0.5, "intimacy": 0.4, "power": 0.75, "vulnerability": 0.25,
    "tenderness": 0.3, "isolation": 0.2, "hope": 0.45, "loss": 0.15,
    "neutral": 0.3,
}

# Pacing rhythm templates (weight distributions for different pacing phases)
_PACING_TEMPLATES: Dict[str, List[float]] = {
    "opening":    [0.20, 0.25, 0.30, 0.25],
    "build_up":   [0.20, 0.30, 0.35, 0.15],
    "climax":     [0.15, 0.30, 0.40, 0.15],
    "resolution": [0.25, 0.30, 0.25, 0.20],
}

# 剪辑节奏检测：连续高权重镜头过多 = 视觉疲劳
_MAX_CONSECUTIVE_HIGH_WEIGHT = 2  # 最多连续 2 个高权重镜头


# ============================================================
# 2. 核心优化函数
# ============================================================

def optimize_editing(
    shot_plan: Dict[str, Any],
    sequence_graph: Dict[str, Any],
    previous_style: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    剪辑师 Agent 核心函数。

    对 sequence_graph + shot_plan 的剪辑系统进行专业级优化：
    1. Transition 强化 — 根据 pacing phase 和能量变化优化转场
    2. Pacing 节奏调整 — 确保节奏变化符合 pacing phase 模板
    3. Emotional arc 平滑 — 检测并修复情绪曲线的不自然跳变
    4. 剪辑节奏检查 — 防止视觉疲劳（连续高权重过多）
    5. 跨场景节奏一致性 — 与前一场景的节奏基线对齐

    Returns:
        {
            "modifications": List[str],
            "editing_style": str,
            "pacing_score": float,
            "agent_concerns": List[Dict],
            "refined_transitions": List[str],  # 优化后的 transition 列表
        }
    """
    shots = shot_plan.get("shots", [])
    pacing_phase = shot_plan.get("pacing_phase", "opening")
    connections = sequence_graph.get("connections", [])
    arc = sequence_graph.get("emotional_arc", {})
    modifications: List[str] = []
    agent_concerns: List[Dict] = []

    refined_transitions = []

    # ── 1. Transition 强化 ──
    transition_template = _TRANSITION_STRENGTH.get(pacing_phase, [])

    for i, conn in enumerate(connections):
        src_emotion = shots[conn["from_shot"]]["emotion"] if conn["from_shot"] < len(shots) else ""
        tgt_emotion = shots[conn["to_shot"]]["emotion"] if conn["to_shot"] < len(shots) else ""

        src_energy = _EMOTION_ENERGY.get(src_emotion, 0.5)
        tgt_energy = _EMOTION_ENERGY.get(tgt_emotion, 0.5)
        energy_diff = tgt_energy - src_energy

        # 确定能量变化类别
        if energy_diff > 0.3:
            energy_cat = "large_increase"
        elif energy_diff > 0.1:
            energy_cat = "small_increase"
        elif energy_diff < -0.3:
            energy_cat = "large_decrease"
        elif energy_diff < -0.1:
            energy_cat = "small_decrease"
        else:
            energy_cat = "lateral"

        # 从能量映射获取推荐转场
        energy_rec = _ENERGY_TRANSITION_MAP.get(energy_cat, {})
        recommended_transition = energy_rec.get("recommended", conn["transition"])

        # 与 pacing phase 模板交叉验证
        if i < len(transition_template):
            template_transition = transition_template[i]
            # 优先使用能量映射（更精确），模板作为备选
            final_transition = recommended_transition
        else:
            final_transition = recommended_transition

        if final_transition != conn["transition"]:
            modifications.append(
                f"[Transition Upgrade] '{conn['from_name']}' → '{conn['to_name']}': "
                f"'{conn['transition']}' → '{final_transition}' "
                f"({energy_rec.get('reason', 'pacing optimization')})"
            )

        refined_transitions.append(final_transition)

    # ── 2. Pacing 节奏调整 ──
    pacing_template = _PACING_TEMPLATES.get(pacing_phase, [])
    current_weights = [s.get("pacing_weight", 0.25) for s in shots]

    if pacing_template and len(pacing_template) == len(current_weights):
        # 计算当前节奏与模板的偏差
        deviations = []
        for i, (curr, target) in enumerate(zip(current_weights, pacing_template)):
            dev = abs(curr - target)
            if dev > 0.10:
                deviations.append((i, curr, target))

        if deviations:
            agent_concerns.append({
                "agent": "editor",
                "type": "pacing_deviation",
                "concern": f"Pacing weights deviate from '{pacing_phase}' template: "
                           f"{', '.join(f'shot #{d[0]+1} ({d[1]:.2f} vs {d[2]:.2f})' for d in deviations)}",
                "suggestion": "consider aligning pacing with phase template for rhythm coherence",
            })

    # ── 3. Emotional arc 平滑 ──
    energy_curve = arc.get("energy_curve", [])
    if len(energy_curve) >= 3:
        # 检测不自然的跳变（连续两个大跳变）
        jumps = []
        for i in range(1, len(energy_curve)):
            jump = abs(energy_curve[i] - energy_curve[i - 1])
            if jump > 0.4:
                jumps.append((i, jump))

        if len(jumps) >= 2:
            agent_concerns.append({
                "agent": "editor",
                "type": "arc_jagged",
                "concern": f"Emotional arc has {len(jumps)} large energy jumps "
                           f"(>{0.4}): {', '.join(f'at shot #{j[0]+1} (Δ{j[1]:.2f})' for j in jumps)}",
                "suggestion": "smooth arc by adjusting intermediate shot emotions or pacing",
            })

    # ── 4. 剪辑节奏检查（视觉疲劳防护） ──
    high_weight_count = 0
    max_consecutive = 0
    for weight in current_weights:
        if weight >= 0.30:
            high_weight_count += 1
            max_consecutive = max(max_consecutive, high_weight_count)
        else:
            high_weight_count = 0

    if max_consecutive > _MAX_CONSECUTIVE_HIGH_WEIGHT:
        agent_concerns.append({
            "agent": "editor",
            "type": "visual_fatigue",
            "concern": f"{max_consecutive} consecutive high-intensity shots "
                       f"(>{_MAX_CONSECUTIVE_HIGH_WEIGHT}) — risk of viewer fatigue",
            "suggestion": "insert a breathing room shot (lower weight) between high-intensity moments",
        })

    # ── 5. 跨场景节奏一致性 ──
    pacing_score = 1.0
    if previous_style:
        prev_avg_weight = previous_style.get("avg_pacing_weight", 0.25)
        curr_avg_weight = sum(current_weights) / len(current_weights) if current_weights else 0.25
        weight_drift = abs(curr_avg_weight - prev_avg_weight)

        if weight_drift > 0.15:
            pacing_score = max(0.5, 1.0 - weight_drift * 2)
            agent_concerns.append({
                "agent": "editor",
                "type": "pacing_drift",
                "concern": f"Average pacing weight drifted {weight_drift:.2f} "
                           f"from previous scene ({prev_avg_weight:.2f} → {curr_avg_weight:.2f})",
                "suggestion": "consider gradual pacing transition from previous scene",
            })

    # ── 6. 生成剪辑风格总结 ──
    transitions = [c["transition"] for c in connections]
    unique_trans = set(transitions) if transitions else set()
    editing_style = (
        f"Editing System: {len(unique_trans)} transition types | "
        f"Pacing: {pacing_phase} (avg weight {sum(current_weights)/len(current_weights):.2f}) | "
        f"Arc: {arc.get('shape', 'unknown')} ({arc.get('overall_direction', 'unknown')})"
    )

    return {
        "modifications": modifications,
        "editing_style": editing_style,
        "pacing_score": round(pacing_score, 2),
        "agent_concerns": agent_concerns,
        "refined_transitions": refined_transitions,
        "style_summary": {
            "transitions": transitions,
            "avg_pacing_weight": round(sum(current_weights) / len(current_weights), 2) if current_weights else 0.25,
            "arc_shape": arc.get("shape", "unknown"),
        },
    }
