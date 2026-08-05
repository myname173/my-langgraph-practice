"""
Lighting/VFX Agent (Phase 5)
==============================
职责：专注于光影和视觉氛围的质量优化。
      lighting 连续性、mood 增强、atmospheric 效果、
      style 强化、contrast 进展控制。

架构位置：
    cinematography_agent → 【lighting_agent】→ editor_agent

设计原则：
    - 纯规则驱动，无 LLM 调用
    - 只修改 lighting 相关字段，不触碰 camera/emotion
    - 输出修改日志供 debate 机制使用
    - 支持跨场景光照风格一致性
"""

from typing import Dict, List, Any


# ============================================================
# 1. 光影规则库
# ============================================================

# Emotion → mood lighting overlays
_MOOD_LIGHTING: Dict[str, Dict[str, str]] = {
    "anticipation": {
        "mood_layer": "low-key ambient with directional tease light",
        "color_temp": "cool (6500K) with warm accent pools",
        "atmosphere": "subtle volumetric haze building tension",
    },
    "adrenaline": {
        "mood_layer": "high-contrast strobe-like directional with fast shadow play",
        "color_temp": "neutral (5600K) with saturated accent",
        "atmosphere": "kinetic light sources with motion-reactive particles",
    },
    "intensity": {
        "mood_layer": "dramatic single-source with deep shadow and hot highlight",
        "color_temp": "warm (3200K) with cool rim separation",
        "atmosphere": "dense volumetric with light shafts and particulate",
    },
    "triumph": {
        "mood_layer": "golden backwash with lens flare and atmospheric bloom",
        "color_temp": "warm (3500K) with golden highlight",
        "atmosphere": "celebratory particle effects with diffused warmth",
    },
    "awe": {
        "mood_layer": "expansive god rays with volumetric light columns",
        "color_temp": "warm golden (4000K) with cool shadow fill",
        "atmosphere": "majestic fog with distant light source diffusion",
    },
    "grandeur": {
        "mood_layer": "layered atmospheric with depth-graded lighting",
        "color_temp": "neutral warm (4500K) with depth falloff",
        "atmosphere": "multi-layer volumetric with scale-revealing haze",
    },
    "mystery": {
        "mood_layer": "selective pool lighting with deep surrounding shadow",
        "color_temp": "cool (7000K) with isolated warm source",
        "atmosphere": "heavy fog with obscured light origins",
    },
    "insignificance": {
        "mood_layer": "overwhelming ambient with subject underexposed",
        "color_temp": "cool flat (6000K) with distant warm horizon",
        "atmosphere": "vast atmospheric perspective with aerial haze",
    },
    "presence": {
        "mood_layer": "strong rim light with controlled fill defining silhouette",
        "color_temp": "neutral (5600K) with accent edge light",
        "atmosphere": "clean volumetric with subject-emphasizing haze",
    },
    "intimacy": {
        "mood_layer": "soft key light with warm wrap-around fill",
        "color_temp": "warm (3800K) with gentle falloff",
        "atmosphere": "minimal atmosphere, shallow DOF emphasis",
    },
    "power": {
        "mood_layer": "low-angle dramatic light with upward shadow cast",
        "color_temp": "warm-cool split (3500K/6500K)",
        "atmosphere": "controlled volumetric with ground-level haze",
    },
    "vulnerability": {
        "mood_layer": "high soft light with exposed shadow areas",
        "color_temp": "cool daylight (6000K) with minimal fill",
        "atmosphere": "sparse atmosphere emphasizing isolation",
    },
}

# Pacing phase → contrast 进展规则
_CONTRAST_PROGRESSION: Dict[str, List[str]] = {
    "opening":    ["low", "low", "medium", "low"],
    "build_up":   ["medium", "high", "medium", "high"],
    "climax":     ["high", "high", "extreme", "high"],
    "resolution": ["medium", "medium", "low", "low"],
}

# Atmospheric layer system (3 layers that can be combined)
_ATMOSPHERE_LAYERS: Dict[str, List[str]] = {
    "base":    ["clear air", "light haze", "moderate fog", "heavy fog"],
    "mid":     ["dust particles", "rain streaks", "smoke wisps", "heat shimmer"],
    "overlay": ["lens flare", "light bloom", "chromatic aberration", "film grain"],
}

# Pacing phase → atmosphere intensity
_ATMOSPHERE_INTENSITY: Dict[str, str] = {
    "opening":    "base",     # 仅基础层
    "build_up":   "mid",      # 加入中层
    "climax":     "overlay",  # 全层叠加
    "resolution": "mid",      # 回归中层
}


# ============================================================
# 2. 核心优化函数
# ============================================================

def optimize_lighting(
    shot_plan: Dict[str, Any],
    visual_context: Dict[str, Any],
    previous_style: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    光影/VFX Agent 核心函数。

    对 shot_plan 的 lighting 系统进行专业级优化：
    1. Mood lighting 增强 — 根据 emotion 叠加情绪光照层
    2. Contrast 进展控制 — 确保 contrast 匹配 pacing phase 进展
    3. Atmospheric 层系统 — 根据 pacing 叠加氛围效果
    4. Lighting 连续性平滑 — 平滑相邻镜头的光照跳变
    5. 跨场景风格一致性 — 与前一场景的光影风格对齐

    Returns:
        {
            "modifications": List[str],
            "lighting_style": str,
            "consistency_score": float,
            "agent_concerns": List[Dict],
        }
    """
    shots = shot_plan.get("shots", [])
    pacing_phase = shot_plan.get("pacing_phase", "opening")
    modifications: List[str] = []
    agent_concerns: List[Dict] = []

    # Deep copy shots for modification
    refined_shots = []
    for shot in shots:
        s = {k: (v.copy() if isinstance(v, dict) else v) for k, v in shot.items()}
        refined_shots.append(s)

    # ── 1. Mood lighting 增强 ──
    for shot in refined_shots:
        emotion = shot.get("emotion", "")
        lighting = shot.get("lighting", {})
        mood_rule = _MOOD_LIGHTING.get(emotion)

        if mood_rule:
            current_setup = lighting.get("setup", "")
            mood_layer = mood_rule["mood_layer"]

            # 如果 setup 中没有 mood 层信息，叠加上去
            if "mood" not in current_setup.lower() and len(current_setup) < 80:
                enhanced = f"{current_setup}; mood: {mood_layer}"
                lighting["setup"] = enhanced
                lighting["mood_color_temp"] = mood_rule["color_temp"]
                modifications.append(
                    f"[Mood Enhance] '{shot['shot_name']}' ({emotion}): "
                    f"added '{mood_layer[:40]}...'"
                )

    # ── 2. Contrast 进展控制 ──
    contrast_progression = _CONTRAST_PROGRESSION.get(pacing_phase, [])
    for i, shot in enumerate(refined_shots):
        lighting = shot.get("lighting", {})
        current_contrast = lighting.get("contrast", "medium")

        if i < len(contrast_progression):
            target_contrast = contrast_progression[i]
            if current_contrast != target_contrast:
                lighting["contrast"] = target_contrast
                modifications.append(
                    f"[Contrast Adjust] '{shot['shot_name']}': "
                    f"'{current_contrast}' → '{target_contrast}' "
                    f"(pacing '{pacing_phase}' position {i+1})"
                )

    # ── 3. Atmospheric 层系统 ──
    atmosphere_level = _ATMOSPHERE_INTENSITY.get(pacing_phase, "base")
    base_layers = _ATMOSPHERE_LAYERS.get("base", [])
    mid_layers = _ATMOSPHERE_LAYERS.get("mid", [])
    overlay_layers = _ATMOSPHERE_LAYERS.get("overlay", [])

    for shot in refined_shots:
        lighting = shot.get("lighting", {})
        atmosphere_effects = [base_layers[0]]  # 默认 base

        if atmosphere_level in ("mid", "overlay"):
            atmosphere_effects.append(mid_layers[0])
        if atmosphere_level == "overlay":
            atmosphere_effects.append(overlay_layers[0])

        lighting["atmosphere_layers"] = atmosphere_effects

    # ── 4. Lighting 连续性平滑 ──
    for i in range(1, len(refined_shots)):
        prev_contrast = refined_shots[i - 1].get("lighting", {}).get("contrast", "medium")
        curr_contrast = refined_shots[i].get("lighting", {}).get("contrast", "medium")

        # 检测极端 contrast 跳变
        contrast_levels = {"low": 1, "medium": 2, "high": 3, "extreme": 4}
        prev_level = contrast_levels.get(prev_contrast, 2)
        curr_level = contrast_levels.get(curr_contrast, 2)

        if abs(prev_level - curr_level) >= 3:
            agent_concerns.append({
                "agent": "lighting",
                "type": "contrast_jump",
                "concern": f"Extreme contrast jump between "
                           f"'{refined_shots[i-1]['shot_name']}' ({prev_contrast}) → "
                           f"'{refined_shots[i]['shot_name']}' ({curr_contrast})",
                "suggestion": "editor should add dissolve transition to smooth lighting shift",
            })

    # ── 5. 跨场景风格一致性 ──
    consistency_score = 1.0
    if previous_style:
        prev_contrasts = previous_style.get("contrasts", [])
        curr_contrasts = [s.get("lighting", {}).get("contrast", "") for s in refined_shots]

        if prev_contrasts:
            # 检查 contrast 基调是否延续
            prev_dominant = max(set(prev_contrasts), key=prev_contrasts.count)
            curr_dominant = max(set(curr_contrasts), key=curr_contrasts.count)
            if prev_dominant != curr_dominant:
                consistency_score = 0.6
                agent_concerns.append({
                    "agent": "lighting",
                    "type": "style_drift",
                    "concern": f"Lighting contrast baseline shifted: "
                               f"'{prev_dominant}' → '{curr_dominant}'",
                    "suggestion": "consider aligning contrast baseline with previous scene",
                })

    # ── 6. 生成光影风格总结 ──
    contrasts = [s.get("lighting", {}).get("contrast", "") for s in refined_shots]
    lighting_style = (
        f"Lighting System: {pacing_phase} pacing | "
        f"Contrast range: {min(contrasts) if contrasts else '?'}-{max(contrasts) if contrasts else '?'} | "
        f"Atmosphere: {atmosphere_level} layers"
    )

    return {
        "modifications": modifications,
        "lighting_style": lighting_style,
        "consistency_score": round(consistency_score, 2),
        "agent_concerns": agent_concerns,
        "style_summary": {
            "contrasts": contrasts,
            "atmosphere_level": atmosphere_level,
            "pacing_phase": pacing_phase,
        },
    }
