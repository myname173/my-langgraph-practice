"""
Cinematography Agent (Phase 5)
================================
职责：专注于摄影系统的质量优化。
      camera movement 优化、lens 一致性、framing 连贯性、
      visual grammar 一致性。

架构位置：
    film_studio_orchestrator → 【cinematography_agent】→ lighting_agent → editor_agent

设计原则：
    - 纯规则驱动，无 LLM 调用
    - 只修改 camera 相关字段，不触碰 lighting/emotion
    - 输出修改日志（modifications）供 debate 机制使用
    - 支持跨场景风格一致性（通过 cinematic_memory_bank）
"""

from typing import Dict, List, Any, Tuple


# ============================================================
# 1. 摄影视觉语法规则库
# ============================================================

# Shot type → 正确的镜头焦距验证
_LENS_VALIDATION: Dict[str, Dict[str, Any]] = {
    "extreme wide shot": {
        "correct_lens": "16mm ultra-wide",
        "acceptable": ["16mm", "14mm", "ultra-wide"],
        "fov": "maximum environmental scope",
    },
    "wide shot": {
        "correct_lens": "24mm wide",
        "acceptable": ["24mm", "28mm", "35mm", "wide"],
        "fov": "environmental context with subject",
    },
    "medium shot": {
        "correct_lens": "50mm standard",
        "acceptable": ["50mm", "35mm", "standard"],
        "fov": "balanced subject-environment",
    },
    "close-up": {
        "correct_lens": "85mm telephoto",
        "acceptable": ["85mm", "100mm", "70mm", "telephoto"],
        "fov": "subject isolation and compression",
    },
    "extreme close-up": {
        "correct_lens": "100mm macro",
        "acceptable": ["100mm", "135mm", "macro"],
        "fov": "intimate detail and shallow DOF",
    },
    "tracking shot": {
        "correct_lens": "35mm",
        "acceptable": ["35mm", "50mm", "24mm"],
        "fov": "dynamic subject-environment balance",
    },
}

# Intent → 推荐的 camera movement 叙事模式
_MOVEMENT_NARRATIVE_PATTERNS: Dict[str, Dict[str, str]] = {
    "action": {
        "establish": "slow dolly in with subtle tension build",
        "movement": "fast lateral tracking with dynamic parallax",
        "impact": "handheld shake with slow-mo impact frame",
        "aftermath": "slow crane ascending revealing scope",
    },
    "world_building": {
        "panorama": "slow horizontal pan with stable horizon",
        "sweep": "slow crane ascending revealing vertical scale",
        "scale_reveal": "slow dolly in with foreground parallax",
        "atmosphere": "static locked with subtle organic drift",
    },
    "character_introduction": {
        "silhouette": "static locked with subject backlit",
        "reveal": "slow dolly in with rack focus to face",
        "detail": "micro dolly with shallow DOF shift",
        "stance": "slow orbit revealing power pose",
    },
    "emotional": {
        "close_up": "slow push in with intimate breathing room",
        "stillness": "static locked with micro subject movement",
        "subtle_motion": "gentle handheld with organic drift",
        "fade": "slow pull out with widening isolation",
    },
    "dialogue": {
        "two_shot": "static balanced framing with subtle drift",
        "over_shoulder": "slow push in from behind speaker",
        "reaction": "static with micro reframing on expression change",
        "exchange": "slow lateral track following conversation flow",
    },
    "montage": {
        "flash": "whip pan with kinetic energy transfer",
        "match_cut": "smooth dolly in matching visual rhythm",
        "crescendo": "accelerating dolly out building visual climax",
    },
    "reveal": {
        "anticipation": "slow dolly in with obstruction framing",
        "breakthrough": "push through foreground element revealing subject",
        "hero_reveal": "crane ascending from detail to full reveal",
    },
}

# Camera angle → 叙事功能映射
_ANGLE_NARRATIVE: Dict[str, str] = {
    "low angle": "power, dominance, heroic presence",
    "high angle": "vulnerability, scope, overview",
    "eye level": "neutrality, relatability, grounded",
    "dutch angle": "unease, tension, disorientation",
    "low angle to high": "ascending power reveal",
    "bird's eye": "god perspective, fate, scale",
}

# Subject orientation → 叙事功能映射
_ORIENTATION_NARRATIVE: Dict[str, str] = {
    "facing_camera": "confrontation, directness, audience engagement",
    "three_quarter": "naturalism, depth, dimensional presence",
    "side_profile": "movement, contemplation, cinematic elegance",
    "back_to_camera": "mystery, isolation, leading into environment",
    "over_shoulder": "intimacy, surveillance, narrative perspective",
    "three_quarter_reveal": "dramatic reveal, character introduction",
}

# Camera tracking mode → 运镜跟随叙事功能
_TRACKING_NARRATIVE: Dict[str, str] = {
    "track_parallel": "camera moves alongside subject — immersion in motion",
    "push_toward_subject": "camera advances toward subject — increasing intensity",
    "orbit_around": "camera circles subject — 360° spatial reveal",
    "follow_behind": "camera follows subject from behind — journey, pursuit",
    "ascend_behind": "camera rises behind subject — scale expansion",
    "observe_from_distance": "camera observes from afar — establishing context",
    "drift_behind": "camera drifts behind subject — atmospheric mystery",
    "orbit_to_face": "camera orbits from back to face — character reveal",
    "static_observe": "camera remains still — contemplation, isolation",
    "gentle_follow": "camera gently follows subject — intimate tracking",
    "intimate_push": "camera pushes in close — emotional intensity",
    "pull_away": "camera retreats from subject — loss, isolation",
    "rise_above": "camera rises above subject — overview, aftermath",
    "observe_panorama": "camera surveys the environment — world building",
}

# 连续镜头间的 camera type 变化规则（避免跳切违和）
_CAMERA_JUMP_RULES: Dict[Tuple[str, str], str] = {
    ("extreme wide shot", "close-up"): "jarring jump — add medium bridge shot or match cut justification",
    ("close-up", "extreme wide shot"): "disorienting pull — ensure spatial context established",
    ("extreme wide shot", "extreme close-up"): "extreme jump — only for shock/surprise moments",
}


# ============================================================
# 2. 核心优化函数
# ============================================================

def optimize_cinematography(
    shot_plan: Dict[str, Any],
    visual_context: Dict[str, Any],
    previous_style: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    摄影指导 Agent 核心函数。

    对 shot_plan 的 camera 系统进行专业级优化：
    1. Lens 一致性验证 — 确保 focal length 与 shot type 匹配
    2. Movement 叙事一致性 — 确保 movement 服务于叙事意图
    3. Angle 连贯性检查 — 确保 angle 变化有叙事功能
    4. 视觉语法检查 — 检查镜头间 camera 跳切是否合理
    5. 跨场景风格一致性 — 与前一场景的摄影风格对齐

    Args:
        shot_plan: Phase 2/4 的 shot plan
        visual_context: Phase 1 的 visual context
        previous_style: cinematic_memory_bank 中的前场景摄影风格

    Returns:
        {
            "modifications": List[str],      # 修改日志
            "camera_grammar": str,            # 摄影语法总结
            "consistency_score": float,       # 风格一致性评分
            "agent_concerns": List[Dict],     # 供 debate 使用的关注点
        }
    """
    shots = shot_plan.get("shots", [])
    intent = shot_plan.get("intent", "world_building")
    modifications: List[str] = []
    agent_concerns: List[Dict] = []

    # Deep copy shots for modification
    refined_shots = []
    for shot in shots:
        s = {k: (v.copy() if isinstance(v, dict) else v) for k, v in shot.items()}
        refined_shots.append(s)

    # ── 1. Lens 一致性验证 ──
    for shot in refined_shots:
        cam = shot.get("camera", {})
        shot_type = cam.get("type", "")
        current_lens = cam.get("lens", "")
        lens_rule = _LENS_VALIDATION.get(shot_type)

        if lens_rule:
            correct_lens = lens_rule["correct_lens"]
            acceptable = lens_rule["acceptable"]
            # 检查当前 lens 是否在可接受范围内
            lens_ok = any(acc in current_lens.lower() for acc in acceptable)
            if not lens_ok and current_lens:
                cam["lens"] = correct_lens
                modifications.append(
                    f"[Lens Fix] '{shot['shot_name']}': "
                    f"'{current_lens}' → '{correct_lens}' "
                    f"(shot type '{shot_type}' requires {lens_rule['fov']})"
                )

    # ── 2. Movement 叙事一致性 ──
    movement_patterns = _MOVEMENT_NARRATIVE_PATTERNS.get(intent, {})
    for shot in refined_shots:
        shot_name = shot.get("shot_name", "")
        current_movement = shot.get("camera", {}).get("movement", "")
        recommended = movement_patterns.get(shot_name)

        if recommended and current_movement:
            # 检查 movement 是否包含推荐动作的关键元素
            rec_keywords = set(recommended.lower().split())
            cur_keywords = set(current_movement.lower().split())
            overlap = rec_keywords & cur_keywords

            if len(overlap) < 2 and len(rec_keywords) > 3:
                # Movement 与推荐差异较大，增强它
                shot["camera"]["movement"] = recommended
                modifications.append(
                    f"[Movement Enhance] '{shot_name}': "
                    f"'{current_movement}' → '{recommended}' "
                    f"(intent '{intent}' requires this movement pattern)"
                )

    # ── 3. Angle 叙事功能检查 ──
    angles_used = []
    for shot in refined_shots:
        angle = shot.get("camera", {}).get("angle", "")
        angles_used.append(angle)
        narrative = _ANGLE_NARRATIVE.get(angle, "")
        if narrative:
            # 记录 angle 的叙事功能
            shot.setdefault("camera", {})["_angle_narrative"] = narrative

    # 检查是否所有镜头都用同一 angle（缺乏变化）
    unique_angles = set(angles_used)
    if len(unique_angles) == 1 and len(refined_shots) > 1:
        agent_concerns.append({
            "agent": "cinematography",
            "type": "angle_monotony",
            "concern": f"All {len(refined_shots)} shots use '{angles_used[0]}' — "
                       f"no angle variation for narrative dynamics",
            "suggestion": "introduce at least 2 different camera angles for visual variety",
        })

    # ── 3b. Orientation 朝向多样性检查 ──
    orientations_used = []
    for shot in refined_shots:
        orientation = shot.get("camera", {}).get("orientation", "facing_camera")
        orientations_used.append(orientation)
        orient_narrative = _ORIENTATION_NARRATIVE.get(orientation, "")
        if orient_narrative:
            shot.setdefault("camera", {})["_orientation_narrative"] = orient_narrative

    unique_orientations = set(orientations_used)
    if len(unique_orientations) == 1 and len(refined_shots) > 1:
        agent_concerns.append({
            "agent": "cinematography",
            "type": "orientation_monotony",
            "concern": f"All {len(refined_shots)} shots use orientation "
                       f"'{orientations_used[0]}' — no subject orientation diversity, "
                       f"appears artificial and front-facing",
            "suggestion": "introduce varied orientations: side_profile, back_to_camera, "
                          "three_quarter, over_shoulder to create cinematic depth",
        })
    elif len(unique_orientations) <= 2 and len(refined_shots) >= 4:
        agent_concerns.append({
            "agent": "cinematography",
            "type": "orientation_low_variety",
            "concern": f"Only {len(unique_orientations)} orientations across "
                       f"{len(refined_shots)} shots — consider more diversity",
            "suggestion": "aim for at least 3 different orientations in a 4+ shot sequence",
        })

    # ── 3c. Camera tracking 运镜跟随叙事标注 ──
    for shot in refined_shots:
        tracking = shot.get("camera", {}).get("tracking", "")
        tracking_narrative = _TRACKING_NARRATIVE.get(tracking, "")
        if tracking_narrative:
            shot.setdefault("camera", {})["_tracking_narrative"] = tracking_narrative

    # ── 4. 镜头间 camera 跳切检查 ──
    for i in range(1, len(refined_shots)):
        prev_type = refined_shots[i - 1].get("camera", {}).get("type", "")
        curr_type = refined_shots[i].get("camera", {}).get("type", "")
        jump_key = (prev_type, curr_type)

        if jump_key in _CAMERA_JUMP_RULES:
            agent_concerns.append({
                "agent": "cinematography",
                "type": "camera_jump",
                "concern": f"'{prev_type}' → '{curr_type}': "
                           f"{_CAMERA_JUMP_RULES[jump_key]}",
                "suggestion": "editor should add match_cut or dissolve transition to bridge",
            })

    # ── 5. 跨场景风格一致性 ──
    consistency_score = 1.0
    if previous_style:
        prev_movements = previous_style.get("movements", [])
        prev_angles = previous_style.get("angles", [])

        # 检查 movement 风格是否延续
        curr_movements = [s.get("camera", {}).get("movement", "") for s in refined_shots]
        if prev_movements:
            # 简单的风格延续度：movement 关键词重叠率
            prev_words = set()
            for m in prev_movements:
                prev_words.update(m.lower().split())
            curr_words = set()
            for m in curr_movements:
                curr_words.update(m.lower().split())
            if prev_words and curr_words:
                overlap = prev_words & curr_words
                consistency_score = min(1.0, len(overlap) / max(1, len(prev_words) * 0.4))

        if consistency_score < 0.5:
            agent_concerns.append({
                "agent": "cinematography",
                "type": "style_drift",
                "concern": f"Camera style consistency with previous scene: "
                           f"{consistency_score:.0%} — risk of visual drift",
                "suggestion": "align movement patterns with established scene style",
            })

    # ── 6. 生成摄影语法总结 ──
    cam_types = [s.get("camera", {}).get("type", "") for s in refined_shots]
    cam_moves = [s.get("camera", {}).get("movement", "") for s in refined_shots]
    cam_trackings = [s.get("camera", {}).get("tracking", "") for s in refined_shots]
    camera_grammar = (
        f"Camera System: {len(set(cam_types))} shot types, "
        f"{len(set(cam_moves))} movement patterns, "
        f"{len(unique_angles)} angles, "
        f"{len(unique_orientations)} orientations, "
        f"{len(set(cam_trackings))} tracking modes | "
        f"Primary: {cam_types[0] if cam_types else '?'}, "
        f"{cam_moves[0] if cam_moves else '?'}"
    )

    # 清理内部字段
    for shot in refined_shots:
        shot.get("camera", {}).pop("_angle_narrative", None)
        shot.get("camera", {}).pop("_orientation_narrative", None)
        shot.get("camera", {}).pop("_tracking_narrative", None)

    return {
        "modifications": modifications,
        "camera_grammar": camera_grammar,
        "consistency_score": round(consistency_score, 2),
        "agent_concerns": agent_concerns,
        "style_summary": {
            "shot_types": list(set(cam_types)),
            "movements": cam_moves,
            "angles": angles_used,
            "orientations": orientations_used,
            "trackings": cam_trackings,
        },
    }
