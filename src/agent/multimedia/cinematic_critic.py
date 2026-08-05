"""
Cinematic Critic Layer (Phase 4)
==================================
职责：评估 shot_plan + sequence_graph 的电影级质量，
      提供量化评分、问题检测和优化建议。
      驱动 self-refining loop：质量不达标时自动触发重写。

架构位置：
    sequence_orchestrator → 【cinematic_critic】→ director (pass)
                                        ↘ shot_strategy (rewrite)

与 Phase 2/3 的关系：
    Phase 2 (shot_strategy) = "会设计镜头"
    Phase 3 (sequence_orch) = "会剪电影"
    Phase 4 (cinematic_critic) = "会评价和优化电影结构"

核心输出：
    critic_eval — 结构化评估报告，包含：
    - scores: 四维度量化评分 (visual_quality, coherence, cinematic_flow, emotion_consistency)
    - overall_score: 加权综合分
    - issues: 具体问题列表
    - suggestions: 优化建议列表
    - gates: 三级质量门结果

设计原则：
    - 纯规则驱动，无 LLM 调用（确定性、零延迟、零成本）
    - 多维度评估，每个维度有明确的通过/不通过阈值
    - 建议可被 shot_strategy.rewrite_strategy() 直接消费
    - Quality Gates 提供系统级约束（三级门控）
"""

from typing import Dict, List, Any, Tuple


# ============================================================
# 1. 质量阈值常量
# ============================================================

QUALITY_THRESHOLDS = {
    "shot_plan_min": 0.60,       # shot plan 最低通过分
    "sequence_min": 0.55,        # sequence consistency 最低通过分
    "overall_min": 0.70,         # 综合最低通过分（收紧：朝向/光照同质化必须触发重写）
    "max_rewrites": 2,           # 最大重写次数
}

# Genre-aware 评分权重：不同题材侧重不同维度
_GENRE_WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {
    #                        visual  coherence  flow  emotion
    "action":               {"vq": 0.20, "co": 0.15, "cf": 0.40, "ec": 0.25},
    "world_building":       {"vq": 0.35, "co": 0.25, "cf": 0.20, "ec": 0.20},
    "character_introduction":{"vq": 0.25, "co": 0.20, "cf": 0.20, "ec": 0.35},
    "emotional":            {"vq": 0.15, "co": 0.20, "cf": 0.25, "ec": 0.40},
    "cyberpunk":            {"vq": 0.30, "co": 0.20, "cf": 0.30, "ec": 0.20},
    "gothic":               {"vq": 0.25, "co": 0.20, "cf": 0.25, "ec": 0.30},
    "fantasy":              {"vq": 0.30, "co": 0.20, "cf": 0.30, "ec": 0.20},
}
_DEFAULT_WEIGHTS = {"vq": 0.25, "co": 0.20, "cf": 0.30, "ec": 0.25}


# ============================================================
# 2. 问题检测器
# ============================================================

# 镜头级问题
SHOT_ISSUES = {
    "camera_incomplete": "shot '{name}' has incomplete camera specification",
    "lighting_flat": "shot '{name}' has generic/flat lighting (setup < 15 chars)",
    "emotion_redundant": "shots '{a}' and '{b}' share emotion '{e}' — no emotional progression",
    "pacing_monotone": "all shots have similar pacing_weight — no rhythm variation",
    "focus_static": "all shots focus on '{f}' — no visual diversity",
    "orientation_monotone": "all shots use orientation '{o}' — no subject orientation diversity",
}

# 序列级问题
SEQUENCE_ISSUES = {
    "redundant_camera": "shots '{a}' -> '{b}' use identical camera type '{t}' — visual redundancy",
    "weak_hero": "hero shot #{idx} '{name}' has low pacing_weight ({w:.2f}) — weak focal point",
    "transition_weak": "transition '{t}' between '{a}' -> '{b}' may not match energy levels",
    "lighting_clash": "lighting contrast jump '{c1}' -> '{c2}' between '{a}' -> '{b}' without transition buffer",
    "weak_arc": "emotional arc is '{s}' — lacks dramatic shape for a trailer",
    "no_emotional_progression": "emotional energy stays flat across all shots — no narrative momentum",
    "spatial_discontinuity": "spatial focus jumps from '{f1}' to '{f2}' without establishing context",
}

# 问题 → 重写策略映射（供 rewrite_strategy 消费）
ISSUE_REWRITE_STRATEGY: Dict[str, str] = {
    "camera_incomplete": "complete_camera",
    "redundant_camera": "diversify_camera",
    "weak_hero": "strengthen_hero",
    "lighting_flat": "enhance_lighting",
    "lighting_clash": "smooth_lighting",
    "weak_arc": "reshape_arc",
    "emotion_redundant": "diversify_emotion",
    "pacing_monotone": "reshape_pacing",
    "focus_static": "diversify_focus",
    "orientation_monotone": "diversify_orientation",
    "no_emotional_progression": "add_progression",
    "transition_weak": "upgrade_transition",
    "weak_end_shot": "strengthen_end",
    "spatial_discontinuity": "add_establishing",
}


# ============================================================
# 3. 评估辅助函数
# ============================================================

def _check_camera_completeness(shots: List[Dict]) -> Tuple[float, List[str]]:
    """
    检查每个 shot 的 camera 规格是否完整。
    完整 camera 需要: type, movement, angle, lens 四个字段。
    """
    issues = []
    complete_count = 0
    required_keys = {"type", "movement", "angle", "lens"}

    for shot in shots:
        cam = shot.get("camera", {})
        present = required_keys.intersection(cam.keys())
        if len(present) == len(required_keys):
            complete_count += 1
        else:
            missing = required_keys - present
            issues.append(f"shot '{shot['shot_name']}' missing camera: {missing}")

    score = complete_count / len(shots) if shots else 0.0
    return score, issues


def _check_camera_variety(shots: List[Dict]) -> Tuple[float, List[str]]:
    """
    检查镜头间 camera type 是否有变化。
    连续两个相同 camera type = 视觉冗余。
    """
    if len(shots) < 2:
        return 1.0, []

    issues = []
    unique_pairs = set()
    consecutive_same = 0

    for i, shot in enumerate(shots):
        cam_type = shot["camera"]["type"]
        movement = shot["camera"]["movement"]
        unique_pairs.add((cam_type, movement))
        if i > 0 and cam_type == shots[i - 1]["camera"]["type"]:
            consecutive_same += 1
            issues.append(
                f"shots '{shots[i-1]['shot_name']}' -> '{shot['shot_name']}' "
                f"use identical camera type '{cam_type}' — visual redundancy"
            )

    variety_score = len(unique_pairs) / len(shots)
    # 惩罚连续重复
    penalty = consecutive_same * 0.15
    score = max(0.0, min(1.0, variety_score - penalty))
    return score, issues


def _check_emotional_progression(shots: List[Dict]) -> Tuple[float, List[str]]:
    """
    检查情绪是否有变化和推进。
    全部相同情绪 = 无叙事动力。
    """
    if len(shots) < 2:
        return 1.0, []

    issues = []
    emotions = [s.get("emotion", "") for s in shots]
    unique_emotions = set(emotions)

    if len(unique_emotions) == 1:
        issues.append(
            f"all {len(shots)} shots share emotion '{emotions[0]}' "
            f"— no emotional progression"
        )
        return 0.3, issues

    # 检查连续重复情绪
    consecutive_same = 0
    for i in range(1, len(emotions)):
        if emotions[i] == emotions[i - 1]:
            consecutive_same += 1
            issues.append(
                f"shots '{shots[i-1]['shot_name']}' and '{shots[i]['shot_name']}' "
                f"share emotion '{emotions[i]}' — no emotional progression"
            )

    diversity = len(unique_emotions) / len(shots)
    penalty = consecutive_same * 0.1
    score = max(0.0, min(1.0, diversity - penalty))
    return score, issues


def _check_lighting_coherence(shots: List[Dict]) -> Tuple[float, List[str]]:
    """
    检查光照是否有变化且不过于单一。
    """
    if len(shots) < 2:
        return 1.0, []

    issues = []
    setups = [s.get("lighting", {}).get("setup", "") for s in shots]
    unique_setups = set(setups)

    if len(unique_setups) == 1:
        issues.append("all shots use identical lighting setup — no visual depth")
        return 0.2, issues

    # 检查过于简短/泛化的光照描述
    flat_count = sum(1 for s in setups if len(s) < 15)
    if flat_count > len(shots) // 2:
        issues.append(
            f"{flat_count}/{len(shots)} shots have generic/flat lighting descriptions"
        )

    coherence = min(1.0, len(unique_setups) / (len(shots) * 0.75))
    penalty = flat_count * 0.05
    score = max(0.0, min(1.0, coherence - penalty))
    return score, issues


def _check_pacing_rhythm(shots: List[Dict]) -> Tuple[float, List[str]]:
    """
    检查节奏权重是否有变化（trailer 需要节奏起伏）。
    """
    if len(shots) < 2:
        return 1.0, []

    issues = []
    weights = [s.get("pacing_weight", 0.25) for s in shots]
    weight_range = max(weights) - min(weights)

    if weight_range < 0.10:
        issues.append(
            "all shots have similar pacing_weight "
            f"(range={weight_range:.2f}) — no rhythm variation"
        )
        return 0.4, issues

    # 理想的 trailer 节奏：有低有高，range > 0.15
    score = min(1.0, weight_range / 0.25)
    return score, issues


def _check_focus_diversity(shots: List[Dict]) -> Tuple[float, List[str]]:
    """
    检查视觉焦点是否有变化。
    所有镜头聚焦同一对象 = 视觉单调。
    """
    if len(shots) < 2:
        return 1.0, []

    issues = []
    focuses = [s.get("focus_object", "environment") for s in shots]
    unique_focus = set(focuses)

    if len(unique_focus) == 1:
        issues.append(
            f"all shots focus on '{focuses[0]}' — no visual diversity"
        )
        return 0.5, issues

    score = min(1.0, len(unique_focus) / 2.0)
    return score, issues


def _check_orientation_variety(shots: List[Dict]) -> Tuple[float, List[str]]:
    """
    检查主体朝向是否有变化。
    所有镜头都用 facing_camera = 视觉单调，虚假感重。
    """
    if len(shots) < 2:
        return 1.0, []

    issues = []
    orientations = [
        s.get("camera", {}).get("orientation", "facing_camera")
        for s in shots
    ]
    unique_orientations = set(orientations)

    if len(unique_orientations) == 1:
        issues.append(
            f"all {len(shots)} shots use orientation '{orientations[0]}' "
            f"— no subject orientation diversity, appears artificial"
        )
        return 0.2, issues

    # 检查连续重复朝向
    consecutive_same = 0
    for i in range(1, len(orientations)):
        if orientations[i] == orientations[i - 1]:
            consecutive_same += 1
            issues.append(
                f"shots '{shots[i-1]['shot_name']}' and '{shots[i]['shot_name']}' "
                f"share orientation '{orientations[i]}' — reduce front-facing monotony"
            )

    diversity = len(unique_orientations) / len(shots)
    penalty = consecutive_same * 0.15
    score = max(0.0, min(1.0, diversity - penalty))
    return score, issues


def _check_hero_strength(
    shots: List[Dict],
    hero_index: int,
) -> Tuple[float, List[str]]:
    """
    检查 hero shot 是否足够强（高权重 + 高能量情绪）。
    """
    issues = []
    hero = shots[hero_index]
    hero_weight = hero.get("pacing_weight", 0.25)

    # Hero shot 应该是序列中权重最高或次高的
    max_weight = max(s.get("pacing_weight", 0.25) for s in shots)
    if hero_weight < max_weight * 0.8:
        issues.append(
            f"hero shot #{hero_index+1} '{hero['shot_name']}' "
            f"has sub-optimal pacing_weight ({hero_weight:.2f} vs max {max_weight:.2f})"
        )
        return 0.5, issues

    return 1.0, issues


def _check_transition_quality(connections: List[Dict]) -> Tuple[float, List[str]]:
    """
    检查转场选择是否合理。
    """
    if not connections:
        return 1.0, []

    issues = []
    # 检查是否全部使用同一种转场
    transitions = [c["transition"] for c in connections]
    unique_trans = set(transitions)

    if len(unique_trans) == 1 and len(transitions) > 1:
        issues.append(
            f"all {len(transitions)} transitions use '{transitions[0]}' — "
            f"no transition variety"
        )
        return 0.5, issues

    score = min(1.0, len(unique_trans) / max(1, len(transitions) * 0.6))
    return score, issues


# ── 各内容类型对应的"好"弧形状（模块级，供 _check_arc_shape 和 _gate_trailer_flow 共享）──
_ACCEPTABLE_ARCS: Dict[str, set] = {
    "trailer":      {"crescendo", "arc", "ascent"},
    "commercial":   {"crescendo", "single_peak", "arc"},
    "short_film":   {"arc", "crescendo", "gentle_undulation", "ascent", "anti_crescendo", "descent"},
    "documentary":  {"gentle_undulation", "arc", "flat", "ascending_steps"},
    "music_video":  {"rhythmic", "crescendo", "arc"},
    "tutorial":     {"ascending_steps", "flat", "gentle_undulation"},
}


def _check_arc_shape(arc: Dict, content_type: str = "trailer") -> Tuple[float, List[str]]:
    """
    检查情绪曲线形状是否适合当前内容类型。
    trailer/commercial: 需要 crescendo, arc, ascent
    documentary/tutorial: 允许 flat, gentle_undulation, ascending_steps
    music_video: 允许 rhythmic, crescendo
    short_film: 允许 arc, crescendo, gentle_undulation
    """
    issues = []
    shape = arc.get("shape", "unknown")
    direction = arc.get("overall_direction", "unknown")

    acceptable = _ACCEPTABLE_ARCS.get(content_type, _ACCEPTABLE_ARCS["trailer"])

    if shape in acceptable:
        return 1.0, issues

    # 检查弱形状（仅对 trailer/commercial 严格惩罚）
    weak_shapes = {"flat", "static", "single"}
    if shape in weak_shapes and content_type in ("trailer", "commercial"):
        issues.append(
            f"emotional arc is '{shape}' — lacks dramatic shape for a {content_type}"
        )
        return 0.3, issues

    # 检查是否有明确的峰值
    peak_idx = arc.get("peak_index", 0)
    valley_idx = arc.get("valley_index", 0)
    if peak_idx == valley_idx:
        issues.append("no clear peak-valley contrast in emotional arc")
        return 0.5, issues

    # 不在可接受列表中但也不是弱形状
    return 0.75, issues


# ============================================================
# 4. 核心评估引擎
# ============================================================

def evaluate_cinematic_quality(
    shot_plan: Dict[str, Any],
    sequence_graph: Dict[str, Any],
    content_type: str = "trailer",
) -> Dict[str, Any]:
    """
    核心评估函数：对 shot_plan + sequence_graph 进行多维度质量评估。

    Returns:
        {
            "scores": {
                "visual_quality": float,       # 视觉规格完整度
                "coherence": float,            # 镜头间连贯性
                "cinematic_flow": float,       # 电影级流程质量
                "emotion_consistency": float,  # 情绪一致性/弧形状
            },
            "overall_score": float,            # 加权综合分
            "pass_gates": bool,               # 是否通过所有质量门
            "issues": List[str],              # 具体问题列表
            "suggestions": List[str],         # 优化建议列表
            "gates": {
                "shot_quality": {...},
                "sequence_consistency": {...},
                "trailer_flow": {...},
            },
            "rewrite_strategies": List[str],  # 可执行的重写策略名
        }
    """
    shots = shot_plan.get("shots", [])
    hero_index = shot_plan.get("hero_shot_index", 0)
    connections = sequence_graph.get("connections", [])
    arc = sequence_graph.get("emotional_arc", {})

    all_issues: List[str] = []
    all_suggestions: List[str] = []
    rewrite_strategies: List[str] = []

    # ── Visual Quality: camera 完整度 + 多样性 + 朝向多样性 ──
    cam_completeness, cam_issues = _check_camera_completeness(shots)
    cam_variety, variety_issues = _check_camera_variety(shots)
    orient_variety, orient_issues = _check_orientation_variety(shots)
    all_issues.extend(cam_issues)
    all_issues.extend(variety_issues)
    all_issues.extend(orient_issues)
    if cam_issues:
        all_suggestions.append("complete camera specs (type, movement, angle, lens) for all shots")
        rewrite_strategies.append("complete_camera")
    if variety_issues:
        all_suggestions.append("diversify camera types between consecutive shots")
        rewrite_strategies.append("diversify_camera")
    if orient_issues:
        all_suggestions.append("diversify subject orientation across shots — avoid all-facing-camera monotony")
        rewrite_strategies.append("diversify_orientation")

    visual_quality = cam_completeness * 0.35 + cam_variety * 0.35 + orient_variety * 0.30

    # ── Coherence: 光照一致性 + 焦点多样性 ──
    light_coherence, light_issues = _check_lighting_coherence(shots)
    focus_diversity, focus_issues = _check_focus_diversity(shots)
    all_issues.extend(light_issues)
    all_issues.extend(focus_issues)
    if light_issues:
        all_suggestions.append("enhance lighting descriptions with scene-specific detail")
        rewrite_strategies.append("enhance_lighting")
    if focus_issues:
        all_suggestions.append("vary focus_object across shots for visual diversity")
        rewrite_strategies.append("diversify_focus")

    coherence = light_coherence * 0.6 + focus_diversity * 0.4

    # ── Cinematic Flow: 节奏变化 + 转场质量 + hero 强度 ──
    pacing_rhythm, pacing_issues = _check_pacing_rhythm(shots)
    trans_quality, trans_issues = _check_transition_quality(connections)
    hero_strength, hero_issues = _check_hero_strength(shots, hero_index)
    all_issues.extend(pacing_issues)
    all_issues.extend(trans_issues)
    all_issues.extend(hero_issues)
    if pacing_issues:
        all_suggestions.append("reshape pacing weights to create rhythm variation")
        rewrite_strategies.append("reshape_pacing")
    if trans_issues:
        all_suggestions.append("diversify transition types between shots")
        rewrite_strategies.append("upgrade_transition")
    if hero_issues:
        all_suggestions.append(f"strengthen hero shot selection (currently #{hero_index+1})")
        rewrite_strategies.append("strengthen_hero")

    cinematic_flow = pacing_rhythm * 0.35 + trans_quality * 0.35 + hero_strength * 0.30

    # ── Emotion Consistency: 情绪推进 + 弧形状 ──
    emo_progression, emo_issues = _check_emotional_progression(shots)
    arc_shape, arc_issues = _check_arc_shape(arc, content_type)
    all_issues.extend(emo_issues)
    all_issues.extend(arc_issues)
    if emo_issues:
        all_suggestions.append("diversify emotions across shots for narrative momentum")
        rewrite_strategies.append("diversify_emotion")
    if arc_issues:
        all_suggestions.append("reshape emotional arc to have a clear dramatic shape")
        rewrite_strategies.append("reshape_arc")

    emotion_consistency = emo_progression * 0.5 + arc_shape * 0.5

    # ── 综合评分（genre-aware 加权） ──
    # 从 shot_plan 中提取 intent 和 genres 来选择合适的权重
    intent = shot_plan.get("intent", "")
    genres = shot_plan.get("genres", [])
    weights = _DEFAULT_WEIGHTS
    # 优先用 intent 匹配，其次用 genre 匹配
    for key in [intent] + genres:
        if key in _GENRE_WEIGHT_PROFILES:
            weights = _GENRE_WEIGHT_PROFILES[key]
            break

    overall_score = (
        visual_quality * weights["vq"] +
        coherence * weights["co"] +
        cinematic_flow * weights["cf"] +
        emotion_consistency * weights["ec"]
    )

    # ── Quality Gates ──
    gates = {
        "shot_quality": _gate_shot_quality(cam_completeness, cam_variety, pacing_rhythm),
        "sequence_consistency": _gate_sequence_consistency(coherence, arc),
        "trailer_flow": _gate_trailer_flow(connections, arc, trans_quality, content_type),
    }

    pass_gates = all(g["pass"] for g in gates.values())

    # 额外检查：综合分也需达标
    if overall_score < QUALITY_THRESHOLDS["overall_min"]:
        pass_gates = False

    return {
        "scores": {
            "visual_quality": round(visual_quality, 2),
            "coherence": round(coherence, 2),
            "cinematic_flow": round(cinematic_flow, 2),
            "emotion_consistency": round(emotion_consistency, 2),
        },
        "overall_score": round(overall_score, 2),
        "pass_gates": pass_gates,
        "issues": all_issues,
        "suggestions": all_suggestions,
        "gates": gates,
        "rewrite_strategies": list(set(rewrite_strategies)),
    }


# ============================================================
# 5. Quality Gates（三级质量门控）
# ============================================================

def _gate_shot_quality(
    camera_completeness: float,
    camera_variety: float,
    pacing_rhythm: float,
) -> Dict[str, Any]:
    """
    Gate 1: Shot Quality Gate
    检查镜头规格完整度、镜头间多样性、节奏变化。
    """
    score = (
        camera_completeness * 0.4 +
        camera_variety * 0.35 +
        pacing_rhythm * 0.25
    )
    passed = score >= QUALITY_THRESHOLDS["shot_plan_min"]
    return {
        "pass": passed,
        "score": round(score, 2),
        "details": {
            "camera_completeness": round(camera_completeness, 2),
            "camera_variety": round(camera_variety, 2),
            "pacing_rhythm": round(pacing_rhythm, 2),
        },
    }


def _gate_sequence_consistency(
    coherence: float,
    arc: Dict,
) -> Dict[str, Any]:
    """
    Gate 2: Sequence Consistency Gate
    检查镜头间连贯性和情绪弧一致性。
    """
    arc_score = 1.0 if arc.get("shape") not in {"flat", "static", "single"} else 0.3
    score = coherence * 0.6 + arc_score * 0.4
    passed = score >= QUALITY_THRESHOLDS["sequence_min"]
    return {
        "pass": passed,
        "score": round(score, 2),
        "details": {
            "coherence": round(coherence, 2),
            "arc_shape": arc.get("shape", "unknown"),
            "arc_direction": arc.get("overall_direction", "unknown"),
        },
    }


def _gate_trailer_flow(
    connections: List[Dict],
    arc: Dict,
    transition_quality: float,
    content_type: str = "trailer",
) -> Dict[str, Any]:
    """
    Gate 3: Sequence Flow Gate
    检查整体序列是否有良好的节奏变化和情绪弧。
    根据 content_type 调整弧形状标准（documentary/tutorial 允许平缓弧）。
    """
    issues = []

    # 足够的镜头连接
    min_connections = 2 if content_type in ("trailer", "commercial", "music_video") else 1
    if len(connections) < min_connections:
        issues.append(f"too few shot connections for {content_type} flow")

    # 弧形状检查（content_type-aware）
    shape = arc.get("shape", "unknown")
    acceptable = _ACCEPTABLE_ARCS.get(content_type, _ACCEPTABLE_ARCS["trailer"])
    if shape not in acceptable and shape != "unknown":
        issues.append(f"arc shape '{shape}' not ideal for {content_type} (expected: {', '.join(sorted(acceptable))})")

    # 转场多样性（trailer/commercial 要求更高）
    transitions = [c["transition"] for c in connections]
    if content_type in ("trailer", "commercial", "music_video"):
        if len(set(transitions)) <= 1 and len(transitions) > 1:
            issues.append("no transition variety — monotonous editing")

    base_score = 1.0 - len(issues) * 0.25
    score = max(0.0, min(1.0, base_score * transition_quality))
    passed = score >= QUALITY_THRESHOLDS["sequence_min"] and len(issues) == 0
    return {
        "pass": passed,
        "score": round(score, 2),
        "issues": issues,
        "details": {
            "connection_count": len(connections),
            "arc_shape": shape,
            "unique_transitions": len(set(transitions)),
        },
    }


# ============================================================
# 6. 重写建议解析器
# ============================================================

def get_rewrite_strategies(
    critic_eval: Dict[str, Any],
) -> List[str]:
    """
    从 critic_eval 中提取可执行的重写策略名称列表。
    这些策略名将被 rewrite_strategy() 函数消费。
    """
    return critic_eval.get("rewrite_strategies", [])


# ============================================================
# 7. 评估报告格式化器
# ============================================================

def format_critic_report(critic_eval: Dict[str, Any]) -> str:
    """
    将评估结果格式化为人类可读的诊断报告。
    用于控制台日志输出和调试。
    """
    s = critic_eval["scores"]
    lines = [
        f"[Cinematic Critic Report]",
        f"  Overall: {critic_eval['overall_score']:.2f} "
        f"({'PASS' if critic_eval['pass_gates'] else 'FAIL'})",
        "",
        "  Scores:",
        f"    visual_quality:       {s['visual_quality']:.2f}",
        f"    coherence:            {s['coherence']:.2f}",
        f"    cinematic_flow:       {s['cinematic_flow']:.2f}",
        f"    emotion_consistency:  {s['emotion_consistency']:.2f}",
        "",
        "  Quality Gates:",
    ]

    for gate_name, gate in critic_eval["gates"].items():
        status = "PASS" if gate["pass"] else "FAIL"
        lines.append(f"    {gate_name}: {status} ({gate['score']:.2f})")
        if gate.get("issues"):
            for issue in gate["issues"]:
                lines.append(f"      - {issue}")

    if critic_eval["issues"]:
        lines.append("")
        lines.append("  Issues:")
        for issue in critic_eval["issues"]:
            lines.append(f"    - {issue}")

    if critic_eval["suggestions"]:
        lines.append("")
        lines.append("  Suggestions:")
        for suggestion in critic_eval["suggestions"]:
            lines.append(f"    + {suggestion}")

    return "\n".join(lines)
