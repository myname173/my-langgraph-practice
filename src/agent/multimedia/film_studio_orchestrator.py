"""
Film Studio Orchestrator (Phase 5)
=====================================
职责：协调多个专业 Agent 协作，生成经过"电影工作室团队"
      审查和优化的镜头方案。

架构位置：
    sequence_orchestrator → 【film_studio_orchestrator】→ cinematic_critic

核心流程：
    1. Cinematography Agent 优化 camera 系统
    2. Lighting Agent 增强 mood 和 atmosphere
    3. Editor Agent 优化 sequence 流畅度
    4. Debate Phase 解决跨 agent 冲突
    5. 合成最终输出 + 更新记忆库

设计原则：
    - 所有 Agent 纯规则驱动，无 LLM 调用
    - 每个 Agent 只修改自己的领域，不跨界
    - Debate 机制通过规则检测跨领域冲突并自动解决
    - 记忆库维护跨场景风格一致性
"""

from typing import Dict, List, Any

from .cinematography_agent import optimize_cinematography
from .lighting_agent import optimize_lighting
from .editor_agent import optimize_editing
from .cinematic_memory import (
    record_scene_style,
    get_previous_scene_style,
    generate_film_brain_report,
)


# ============================================================
# 1. 跨 Agent 冲突检测规则
# ============================================================

# 每条规则检测两个 agent 之间的潜在冲突
_CROSS_AGENT_CONFLICT_RULES = [
    {
        "from": "cinematography",
        "to": "lighting",
        "type": "movement_atmosphere_clash",
        "check": lambda cine, light: _check_movement_atmosphere_clash(cine, light),
    },
    {
        "from": "editor",
        "to": "cinematography",
        "type": "pacing_camera_mismatch",
        "check": lambda edit, cine: _check_pacing_camera_mismatch(edit, cine),
    },
    {
        "from": "editor",
        "to": "lighting",
        "type": "pacing_lighting_mismatch",
        "check": lambda edit, light: _check_pacing_lighting_mismatch(edit, light),
    },
    {
        "from": "cinematography",
        "to": "editor",
        "type": "camera_jump_warning",
        "check": lambda cine, edit: _check_camera_jump_warning(cine, edit),
    },
]


def _check_movement_atmosphere_clash(
    cine_result: Dict,
    light_result: Dict,
) -> List[Dict]:
    """摄影-光影冲突：camera movement 与 atmosphere 效果是否矛盾。"""
    conflicts = []
    cine_style = cine_result.get("style_summary", {})
    light_style = light_result.get("style_summary", {})

    movements = cine_style.get("movements", [])
    atmosphere = light_style.get("atmosphere_level", "base")

    # 快速运动 + 重氛围 = 视觉噪音
    fast_keywords = {"fast", "tracking", "lateral", "dynamic"}
    has_fast = any(
        any(kw in m.lower() for kw in fast_keywords) for m in movements
    )
    heavy_atmo = atmosphere in ("overlay", "mid")

    if has_fast and heavy_atmo:
        conflicts.append({
            "from_agent": "cinematography",
            "to_agent": "lighting",
            "concern": (
                "Fast camera movement combined with heavy atmospheric effects "
                "may cause visual noise and reduce readability"
            ),
            "resolution": (
                "Compromise: use stable motion blur reference with "
                "reduced atmospheric density during dynamic shots"
            ),
        })

    return conflicts


def _check_pacing_camera_mismatch(
    edit_result: Dict,
    cine_result: Dict,
) -> List[Dict]:
    """剪辑-摄影冲突：pacing 节奏与 camera 动态是否匹配。"""
    conflicts = []
    edit_style = edit_result.get("style_summary", {})
    cine_style = cine_result.get("style_summary", {})

    avg_weight = edit_style.get("avg_pacing_weight", 0.25)
    movements = cine_style.get("movements", [])

    # 低 pacing + 快速 camera = 不匹配
    slow_keywords = {"static", "locked", "slow", "drift"}
    fast_keywords = {"fast", "tracking", "dynamic", "shake"}

    has_mostly_slow = sum(
        1 for m in movements
        if any(kw in m.lower() for kw in slow_keywords)
    ) > len(movements) * 0.6

    has_mostly_fast = sum(
        1 for m in movements
        if any(kw in m.lower() for kw in fast_keywords)
    ) > len(movements) * 0.6

    if avg_weight > 0.35 and has_mostly_slow:
        conflicts.append({
            "from_agent": "editor",
            "to_agent": "cinematography",
            "concern": (
                f"High pacing intensity ({avg_weight:.2f}) conflicts with "
                f"mostly static/slow camera work"
            ),
            "resolution": (
                "Resolution: intensify camera movement to match pacing energy, "
                "or reduce pacing weight for calmer visual treatment"
            ),
        })
    elif avg_weight < 0.20 and has_mostly_fast:
        conflicts.append({
            "from_agent": "editor",
            "to_agent": "cinematography",
            "concern": (
                f"Low pacing intensity ({avg_weight:.2f}) conflicts with "
                f"mostly dynamic/fast camera work"
            ),
            "resolution": (
                "Resolution: calm camera movement to match gentle pacing, "
                "or increase pacing for action-oriented treatment"
            ),
        })

    return conflicts


def _check_pacing_lighting_mismatch(
    edit_result: Dict,
    light_result: Dict,
) -> List[Dict]:
    """剪辑-光影冲突：pacing 与 contrast 是否匹配。"""
    conflicts = []
    edit_style = edit_result.get("style_summary", {})
    light_style = light_result.get("style_summary", {})

    avg_weight = edit_style.get("avg_pacing_weight", 0.25)
    contrasts = light_style.get("contrasts", [])

    # 高 pacing + 低 contrast = 不匹配
    low_contrast_count = sum(1 for c in contrasts if c in ("low",))
    if avg_weight > 0.35 and low_contrast_count > len(contrasts) * 0.6:
        conflicts.append({
            "from_agent": "editor",
            "to_agent": "lighting",
            "concern": (
                f"High pacing ({avg_weight:.2f}) with predominantly low contrast "
                f"lighting — lacks visual punch"
            ),
            "resolution": (
                "Resolution: increase contrast in high-energy shots to match pacing"
            ),
        })

    return conflicts


def _check_camera_jump_warning(
    cine_result: Dict,
    edit_result: Dict,
) -> List[Dict]:
    """摄影-剪辑警告：camera 跳切需要特殊转场。"""
    conflicts = []

    # 检查 cinematography agent 是否标记了 camera jump
    cine_concerns = cine_result.get("agent_concerns", [])
    jump_concerns = [c for c in cine_concerns if c.get("type") == "camera_jump"]

    if jump_concerns:
        conflicts.append({
            "from_agent": "cinematography",
            "to_agent": "editor",
            "concern": (
                f"{len(jump_concerns)} camera scale jump(s) detected — "
                f"editor should ensure appropriate transitions"
            ),
            "resolution": (
                "Resolution: use match_cut or dissolve at camera jump points "
                "to bridge scale differences smoothly"
            ),
        })

    return conflicts


# ============================================================
# 2. Debate 机制
# ============================================================

def _run_debate(
    cine_result: Dict,
    light_result: Dict,
    edit_result: Dict,
) -> List[Dict]:
    """
    Agent Debate 机制：检测并解决跨 Agent 冲突。

    流程：
    1. 收集所有 agent 的 concerns
    2. 运行跨 agent 冲突检测规则
    3. 对每个冲突生成 resolution
    4. 返回完整的 debate 记录

    Returns:
        debate 记录列表，每条包含：from_agent, to_agent, concern, resolution
    """
    debate: List[Dict] = []

    # 运行所有跨 agent 冲突检测规则
    for rule in _CROSS_AGENT_CONFLICT_RULES:
        from_agent = rule["from"]
        to_agent = rule["to"]

        # 获取对应 agent 的结果
        results = {
            "cinematography": cine_result,
            "lighting": light_result,
            "editor": edit_result,
        }

        from_result = results.get(from_agent, {})
        to_result = results.get(to_agent, {})

        # 运行检测
        conflicts = rule["check"](from_result, to_result)
        for conflict in conflicts:
            conflict["conflict_type"] = rule["type"]
            debate.append(conflict)

    return debate


# ============================================================
# 3. 核心编排函数
# ============================================================

def orchestrate_film_studio(
    shot_plan: Dict[str, Any],
    sequence_graph: Dict[str, Any],
    visual_context: Dict[str, Any],
    scene_index: int = 0,
    total_scenes: int = 1,
    global_setting: str = "",
    thread_id: str = "",
) -> Dict[str, Any]:
    """
    电影工作室编排器核心函数。

    协调 3 个专业 Agent 对 shot_plan + sequence_graph 进行
    多视角优化，通过 debate 机制解决冲突，输出经过"团队审查"的方案。

    Args:
        shot_plan: Phase 2/4 的 shot plan
        sequence_graph: Phase 3 的 sequence graph
        visual_context: Phase 1 的 visual context
        scene_index: 当前场景索引
        total_scenes: 总场景数
        global_setting: 全局视觉设定
        thread_id: 线程标识（P0-3：记忆库作用域 + 持久化键）

    Returns:
        {
            "studio_refined_plan": Dict,        # 经工作室优化的 shot_plan
            "cinematography": Dict,              # 摄影 agent 输出
            "lighting": Dict,                    # 光影 agent 输出
            "editing": Dict,                     # 剪辑 agent 输出
            "debate": List[Dict],                # debate 记录
            "film_brain": str,                   # 全局电影意识报告
            "studio_consensus": float,           # 团队共识评分
            "modifications": List[str],          # 所有修改的汇总
            "formatted_studio_report": str,      # 格式化工作室报告
        }
    """
    # 获取前场景风格（用于跨场景一致性）
    prev_style = get_previous_scene_style(thread_id, scene_index)

    # ── Step 1: Cinematography Agent ──
    cine_result = optimize_cinematography(
        shot_plan=shot_plan,
        visual_context=visual_context,
        previous_style=prev_style,
    )

    # ── Step 2: Lighting Agent ──
    light_result = optimize_lighting(
        shot_plan=shot_plan,
        visual_context=visual_context,
        previous_style=prev_style,
    )

    # ── Step 3: Editor Agent ──
    edit_result = optimize_editing(
        shot_plan=shot_plan,
        sequence_graph=sequence_graph,
        previous_style=prev_style,
    )

    # ── Step 4: Debate Phase ──
    debate = _run_debate(cine_result, light_result, edit_result)

    # ── Step 5: 记录风格到记忆库 ──
    record_scene_style(
        thread_id=thread_id,
        scene_index=scene_index,
        cinematography=cine_result,
        lighting=light_result,
        editing=edit_result,
    )

    # ── Step 6: 生成全局电影意识报告 ──
    film_brain = generate_film_brain_report(global_setting, total_scenes, thread_id=thread_id)

    # ── Step 7: 计算团队共识评分 ──
    scores = [
        cine_result.get("consistency_score", 1.0),
        light_result.get("consistency_score", 1.0),
        edit_result.get("pacing_score", 1.0),
    ]
    # Debate 冲突会降低共识分
    debate_penalty = len(debate) * 0.05
    studio_consensus = max(0.0, min(1.0,
        sum(scores) / len(scores) - debate_penalty
    ))

    # ── Step 8: 汇总所有修改 ──
    all_modifications = (
        cine_result.get("modifications", []) +
        light_result.get("modifications", []) +
        edit_result.get("modifications", [])
    )

    # ── Step 9: 构建 studio_refined_plan ──
    # 将 agent 的 refined transitions 注入 sequence_graph
    refined_seq = dict(sequence_graph)
    refined_transitions = edit_result.get("refined_transitions", [])
    if refined_transitions and "connections" in refined_seq:
        for i, conn in enumerate(refined_seq["connections"]):
            if i < len(refined_transitions):
                conn["transition"] = refined_transitions[i]

    studio_refined_plan = {
        "shot_plan": shot_plan,
        "sequence_graph": refined_seq,
    }

    # ── Step 10: 格式化工作室报告 ──
    report = _format_studio_report(
        cine_result, light_result, edit_result,
        debate, studio_consensus, film_brain,
    )

    return {
        "studio_refined_plan": studio_refined_plan,
        "cinematography": cine_result,
        "lighting": light_result,
        "editing": edit_result,
        "debate": debate,
        "film_brain": film_brain,
        "studio_consensus": round(studio_consensus, 2),
        "modifications": all_modifications,
        "formatted_studio_report": report,
    }


# ============================================================
# 4. 报告格式化器
# ============================================================

def _format_studio_report(
    cine_result: Dict,
    light_result: Dict,
    edit_result: Dict,
    debate: List[Dict],
    consensus: float,
    film_brain: str,
) -> str:
    """格式化完整的工作室报告，供控制台输出和 director prompt 注入。"""
    lines = [
        "=" * 60,
        "  FILM STUDIO ORCHESTRATION REPORT",
        "=" * 60,
        "",
    ]

    # 摄影报告
    lines.append(f"[Cinematography Agent] — Consistency: {cine_result['consistency_score']:.0%}")
    lines.append(f"  {cine_result['camera_grammar']}")
    for mod in cine_result.get("modifications", []):
        lines.append(f"  {mod}")
    if cine_result.get("agent_concerns"):
        for concern in cine_result["agent_concerns"]:
            lines.append(f"  ⚠ {concern['concern']}")
    lines.append("")

    # 光影报告
    lines.append(f"[Lighting/VFX Agent] — Consistency: {light_result['consistency_score']:.0%}")
    lines.append(f"  {light_result['lighting_style']}")
    for mod in light_result.get("modifications", []):
        lines.append(f"  {mod}")
    if light_result.get("agent_concerns"):
        for concern in light_result["agent_concerns"]:
            lines.append(f"  ⚠ {concern['concern']}")
    lines.append("")

    # 剪辑报告
    lines.append(f"[Editor Agent] — Pacing Score: {edit_result['pacing_score']:.0%}")
    lines.append(f"  {edit_result['editing_style']}")
    for mod in edit_result.get("modifications", []):
        lines.append(f"  {mod}")
    if edit_result.get("agent_concerns"):
        for concern in edit_result["agent_concerns"]:
            lines.append(f"  ⚠ {concern['concern']}")
    lines.append("")

    # Debate 记录
    if debate:
        lines.append(f"[Agent Debate] — {len(debate)} conflict(s) resolved:")
        for d in debate:
            lines.append(
                f"  {d['from_agent']} ↔ {d['to_agent']} ({d.get('conflict_type', '?')}):"
            )
            lines.append(f"    Concern: {d['concern']}")
            lines.append(f"    Resolution: {d['resolution']}")
    else:
        lines.append("[Agent Debate] — No conflicts detected. Full consensus.")
    lines.append("")

    # 全局共识
    lines.append(f"[Studio Consensus]: {consensus:.0%}")
    lines.append("")

    # 全局电影意识
    lines.append(film_brain)

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
