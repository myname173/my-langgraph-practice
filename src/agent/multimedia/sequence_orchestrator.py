# src/agent/multimedia/sequence_orchestrator.py
"""
Sequence Orchestration Layer (Phase 3)
=======================================
职责：将 Phase 2 的独立镜头列表升级为"连贯的电影级序列"。
      建模镜头间关系：转场、运动连续性、情绪轨迹、空间连续性。

架构位置：
    shot_strategy_builder → 【sequence_orchestrator】→ director

与 Phase 2 的关系：
    Phase 2 (shot_strategy) = "会拍镜头"（独立 shot 设计）
    Phase 3 (sequence_orch) = "会剪电影"（shot-to-shot 关系建模）

核心输出：
    sequence_graph — 镜头关系图，包含：
    - transition（转场类型）
    - motion_continuity（运动连续性指令）
    - emotional_delta（情绪变化方向）
    - spatial_continuity（空间连续性约束）
    - emotional_arc（全局情绪曲线）
"""

from typing import Dict, List, Any


# ============================================================
# 1. 转场规则系统
# ============================================================
# 基于 (source_emotion, target_emotion) 对 + shot type 变化
# 决定最佳转场方式

TRANSITION_TYPES = {
    "hard_cut": {
        "description": "instant switch, maximum energy transfer",
        "energy": "high",
    },
    "match_cut": {
        "description": "similar composition different subject, visual rhyme",
        "energy": "medium",
    },
    "whip_pan": {
        "description": "fast blur transition, speed and surprise",
        "energy": "very_high",
    },
    "smash_cut": {
        "description": "quiet-to-loud contrast, shock value",
        "energy": "explosive",
    },
    "dissolve": {
        "description": "gradual blend, time passage or dream",
        "energy": "low",
    },
    "fade_through_black": {
        "description": "dramatic pause, chapter break",
        "energy": "minimal",
    },
    "camera_carry": {
        "description": "continuous motion bridges two shots",
        "energy": "seamless",
    },
}


# ── 基于能量差值的转场选择算法 ──
# 用 _EMOTION_ENERGY（第 4 节）的能量差 delta = tgt - src 决定转场类型
# 覆盖所有 15×15 = 225 种情绪对，无需手动维护稀疏矩阵
#
# 设计逻辑：
#   大正向跳跃 (delta ≥ 0.35)  → smash_cut（静到爆，冲击力）
#   中正向跳跃 (0.2 ≤ d < 0.35) → hard_cut（能量攀升，标准动作剪辑）
#   大负向跌落 (delta ≤ -0.35) → fade_through_black（戏剧性终止，章节感）
#   中负向跌落 (-0.35 < d ≤ -0.2) → dissolve（渐弱，时间流逝感）
#   小变化 + 高能量区 (≥0.65)   → hard_cut（维持高强度节奏）
#   小变化 + 中能量区            → match_cut（视觉韵律，相似能量）
#   小变化 + 低能量区            → camera_carry（无缝衔接，沉浸感）

def _compute_emotion_transition(src_emotion: str, tgt_emotion: str) -> str:
    """
    基于情绪能量差值计算最佳转场类型。
    使用 _EMOTION_ENERGY 映射表（定义在本文件第 4 节）。

    Returns:
        转场类型字符串（hard_cut / match_cut / smash_cut / dissolve /
        fade_through_black / camera_carry / whip_pan）
    """
    src_energy = _EMOTION_ENERGY.get(src_emotion, 0.5)
    tgt_energy = _EMOTION_ENERGY.get(tgt_emotion, 0.5)
    delta = tgt_energy - src_energy
    abs_delta = abs(delta)

    # 大正向跳跃：安静→爆发
    if delta >= _ENERGY_JUMP_LARGE:
        return "smash_cut"
    # 大负向跌落：高潮→沉寂
    if delta <= -_ENERGY_JUMP_LARGE:
        return "fade_through_black"
    # 中等正向攀升
    if _ENERGY_JUMP_MEDIUM <= delta < _ENERGY_JUMP_LARGE:
        return "hard_cut"
    # 中等负向衰退
    if -_ENERGY_JUMP_LARGE < delta <= -_ENERGY_JUMP_MEDIUM:
        return "dissolve"

    # 小变化（|delta| < 0.2）：按能量区间选择
    avg_energy = (src_energy + tgt_energy) / 2
    if avg_energy >= 0.65:
        return "hard_cut"       # 高能量区维持快节奏
    if avg_energy >= 0.45:
        return "match_cut"      # 中能量区视觉韵律
    return "camera_carry"       # 低能量区无缝衔接

# ── 基于叙事功能（beat）的跨场景转场路由 ──
# 借鉴 STAGE / HoloCine 的"镜头间语义衔接"思想：转场类型不只是看情绪能量，
# 更要看上一个镜头的叙事功能与下一个镜头的功能之间的"承接关系"。
# 例：climax → resolution 表示高潮落幕，应用柔和的 dissolve / fade；
#     develop → turn 表示平稳到转折，用 hard_cut 制造冲击；
#     setup → inciting 表示建立到触发事件，用 camera_carry 保持沉浸。
_BEAT_TRANSITIONS: Dict[tuple, str] = {
    # (prev_beat, curr_beat) -> transition
    ("climax", "resolution"): "dissolve",
    ("climax", "fallout"): "fade_through_black",
    ("turn", "climax"): "smash_cut",
    ("turn", "resolution"): "dissolve",
    ("develop", "turn"): "hard_cut",
    ("develop", "climax"): "smash_cut",
    ("inciting", "develop"): "hard_cut",
    ("setup", "inciting"): "camera_carry",
    ("setup", "develop"): "camera_carry",
    ("fallout", "resolution"): "dissolve",
    ("resolution", "setup"): "fade_through_black",  # 环形/新篇章
}
# 与 beat 对应的情绪能量（用于无 emotion 时的兜底）
_BEAT_ENERGY: Dict[str, float] = {
    "setup": 0.4, "inciting": 0.55, "develop": 0.6,
    "turn": 0.8, "climax": 0.95, "fallout": 0.7, "resolution": 0.5,
}


def select_cross_scene_transition(
    prev_beat: str,
    curr_beat: str,
    prev_emotion: str = "",
    curr_emotion: str = "",
) -> str:
    """
    为两个相邻场景（跨场景拼接）选择转场类型。

    优先级：
        1. 命中 _BEAT_TRANSITIONS 语义路由（叙事功能承接）— 保证情节连贯的剪辑逻辑；
        2. 否则回退到情绪能量差算法 _compute_emotion_transition（与场景内转场保持一致）；
        3. 兜底 hard_cut。

    Returns:
        转场类型字符串（见 _TRANSITION_LIBRARY）。
    """
    pb = (prev_beat or "").strip().lower()
    cb = (curr_beat or "").strip().lower()

    # 1. 语义路由优先
    if (pb, cb) in _BEAT_TRANSITIONS:
        return _BEAT_TRANSITIONS[(pb, cb)]

    # 2. 回退到情绪能量差（兼容 beat 缺失场景）
    src_e = _BEAT_ENERGY.get(pb) or _EMOTION_ENERGY.get(prev_emotion, 0.5)
    tgt_e = _BEAT_ENERGY.get(cb) or _EMOTION_ENERGY.get(curr_emotion, 0.5)
    src_emotion = prev_emotion or ("intensity" if src_e >= 0.7 else "presence")
    tgt_emotion = curr_emotion or ("intensity" if tgt_e >= 0.7 else "presence")
    return _compute_emotion_transition(src_emotion, tgt_emotion)


# 基于 shot type 变化的转场修饰
_SHOT_TYPE_TRANSITIONS: Dict[tuple, str] = {
    ("wide shot", "close-up"):           "hard_cut",
    ("close-up",  "wide shot"):          "hard_cut",
    ("extreme close-up", "medium shot"): "match_cut",
    ("wide shot", "extreme wide shot"):  "camera_carry",
    ("medium shot", "medium shot"):      "camera_carry",
}

# ── 默认转场（当没有匹配规则时使用）──
DEFAULT_TRANSITION = "hard_cut"


def _select_transition(
    source_shot: Dict,
    target_shot: Dict,
) -> str:
    """
    根据两个相邻 shot 的运镜矢量、情绪和镜头类型，选择最佳转场方式。

    优先级：
    1. 运镜矢量匹配（同向连续运动 → camera_carry 无缝承接）
    2. 情绪能量差值计算（覆盖所有 225 种情绪对）
    3. 镜头类型匹配（当情绪无法区分时的修饰）
    4. 默认 hard_cut（兜底）
    """
    src_emotion = source_shot.get("emotion", "")
    tgt_emotion = target_shot.get("emotion", "")
    src_type = source_shot.get("camera", {}).get("type", "")
    tgt_type = target_shot.get("camera", {}).get("type", "")

    # ── 1. 运镜矢量匹配（Motion Vector Matching）──
    # 复用 _extract_motion_direction，让已有的运镜数据参与转场决策：
    # 同向连续运动时用 camera_carry 做真正的"运动承接"，比单纯叠化更自然；
    # 运动方向剧烈反转时明确避免 camera_carry（叠化会显得别扭），改用硬切。
    src_move = source_shot.get("camera", {}).get("movement", "")
    tgt_move = target_shot.get("camera", {}).get("movement", "")
    src_dir = _extract_motion_direction(src_move)
    tgt_dir = _extract_motion_direction(tgt_move)

    _REVERSAL_PAIRS = {("inward", "outward"), ("outward", "inward")}

    if src_dir != "unknown" and tgt_dir != "unknown":
        # 同向且均为真实运动（非静止）→ 运动矢量对齐，无缝承接
        if src_dir == tgt_dir and src_dir != "static":
            return "camera_carry"
        # 方向反转 → 禁用 camera_carry，用硬切表达明确的方向变化
        if (src_dir, tgt_dir) in _REVERSAL_PAIRS:
            return "hard_cut"

    # 2. 情绪能量差值计算
    if src_emotion and tgt_emotion:
        return _compute_emotion_transition(src_emotion, tgt_emotion)

    # 2. 镜头类型匹配（情绪缺失时的降级方案）
    type_key = (src_type, tgt_type)
    if type_key in _SHOT_TYPE_TRANSITIONS:
        return _SHOT_TYPE_TRANSITIONS[type_key]

    # 3. 兜底
    return DEFAULT_TRANSITION


# ============================================================
# 2. 运动连续性系统
# ============================================================
# 描述前一个 shot 的 camera movement 如何"传递"到下一个 shot
# 确保镜头之间的运动逻辑连贯

_MOTION_CARRY_RULES: Dict[tuple, str] = {
    # 同方向延续
    ("slow dolly in", "slow dolly in"):
        "continue forward push, deepen the approach",
    ("fast lateral tracking", "fast lateral tracking"):
        "maintain tracking direction and speed",

    # 方向反转 → 需要明确说明
    ("slow dolly in", "slow dolly out"):
        "reverse camera direction, pull back from subject",
    ("slow crane ascending", "slow dolly in"):
        "descend from overview, push into detail",

    # 从静态到动态
    ("static locked", "slow orbit"):
        "break stillness with gradual orbital reveal",
    ("static locked", "slow dolly in"):
        "activate from static, begin slow approach",
    ("static locked", "fast lateral tracking"):
        "explode from stillness into kinetic motion",

    # 从动态到静态
    ("fast lateral tracking", "static locked"):
        "arrest motion, lock into composed frame",
    ("handheld shake", "static locked"):
        "stabilize from chaos into resolved stillness",

    # 特殊连续性
    ("slow orbit", "slow dolly in"):
        "complete orbit arc, transition into direct approach",
    ("slow horizontal pan", "slow crane ascending"):
        "sweep concludes, camera rises to reveal scale",
    ("slow dolly out", "slow dolly out"):
        "continue withdrawal, increase emotional distance",
    ("very slow dolly in", "static locked"):
        "approach completes, hold on intimate moment",
    ("gentle handheld drift", "slow dolly out"):
        "release intimacy, begin emotional withdrawal",
}


def _compute_motion_continuity(
    source_shot: Dict,
    target_shot: Dict,
) -> str:
    """
    计算两个相邻 shot 之间的运动连续性指令。
    告诉 director 下一个 shot 的 camera motion 如何承接前一个。
    """
    src_move = source_shot.get("camera", {}).get("movement", "")
    tgt_move = target_shot.get("camera", {}).get("movement", "")

    # 精确匹配
    key = (src_move, tgt_move)
    if key in _MOTION_CARRY_RULES:
        return _MOTION_CARRY_RULES[key]

    # 模糊匹配：检测运动方向关键词
    src_direction = _extract_motion_direction(src_move)
    tgt_direction = _extract_motion_direction(tgt_move)

    if src_direction == tgt_direction and src_direction != "unknown":
        return f"maintain {src_direction} direction, smooth continuation"
    elif src_direction == "inward" and tgt_direction == "outward":
        return "reverse approach: camera pulls back from subject"
    elif src_direction == "outward" and tgt_direction == "inward":
        return "reverse withdrawal: camera re-engages subject"
    elif src_direction == "static" and tgt_direction != "static":
        return f"activate from stillness into {tgt_direction} motion"
    elif src_direction != "static" and tgt_direction == "static":
        return f"settle from {src_direction} motion into composed stillness"

    return "fresh camera setup, no motion carry-over required"


def _extract_motion_direction(movement: str) -> str:
    """从 movement 字符串提取运动方向。"""
    m = movement.lower()
    if "dolly in" in m or "push in" in m or "approach" in m:
        return "inward"
    if "dolly out" in m or "pull" in m or "withdraw" in m:
        return "outward"
    if "tracking" in m or "pan" in m or "orbit" in m:
        return "lateral"
    if "crane" in m or "ascending" in m or "descending" in m:
        return "vertical"
    if "static" in m or "locked" in m:
        return "static"
    if "handheld" in m or "drift" in m or "shake" in m:
        return "organic"
    return "unknown"


# ============================================================
# 3. 空间连续性系统
# ============================================================

def _compute_spatial_continuity(
    source_shot: Dict,
    target_shot: Dict,
) -> str:
    """
    计算空间连续性约束：
    - 主体在画面中的位置如何过渡
    - 景深如何变化
    - 视线方向是否一致
    """
    src_focus = source_shot.get("focus_object", "environment")
    tgt_focus = target_shot.get("focus_object", "environment")
    src_type = source_shot.get("camera", {}).get("type", "")
    tgt_type = target_shot.get("camera", {}).get("type", "")

    rules = []

    # 焦点对象一致性
    if src_focus == tgt_focus and src_focus == "character":
        rules.append("maintain subject screen position (180-degree rule)")
    elif src_focus != tgt_focus:
        rules.append(f"shift focus from {src_focus} to {tgt_focus}")

    # 景别跳跃处理
    wide_types = {"wide shot", "extreme wide shot"}
    close_types = {"close-up", "extreme close-up"}

    if src_type in wide_types and tgt_type in close_types:
        rules.append("dramatic scale jump: establish spatial context before cut")
    elif src_type in close_types and tgt_type in wide_types:
        rules.append("scale expansion: subject revealed in broader context")
    elif src_type == tgt_type:
        rules.append("consistent frame scale, smooth spatial transition")

    # 视角变化
    src_angle = source_shot.get("camera", {}).get("angle", "")
    tgt_angle = target_shot.get("camera", {}).get("angle", "")
    if "low" in src_angle and "high" in tgt_angle:
        rules.append("perspective shift: ground-level to overhead")
    elif "high" in src_angle and "low" in tgt_angle:
        rules.append("perspective shift: overhead to ground-level")

    return "; ".join(rules) if rules else "standard spatial continuity"


# ============================================================
# 4. 情绪轨迹建模
# ============================================================

# 情绪的"能量级别"映射（用于计算情绪变化方向和强度）
# 改从 visual_rules.json 读取（配置化、可调参），保留内置默认兜底保证缺失配置时行为不变。
from .config_loader import emotion_energy_table, transition_threshold

_EMOTION_ENERGY: Dict[str, float] = dict(emotion_energy_table("default")) or {
    "awe": 0.6, "grandeur": 0.65, "mystery": 0.4, "insignificance": 0.35,
    "anticipation": 0.5, "presence": 0.6, "intimacy": 0.55, "power": 0.75,
    "vulnerability": 0.45, "isolation": 0.3, "tenderness": 0.5, "loss": 0.35,
    "adrenaline": 0.85, "intensity": 0.9, "triumph": 0.8,
    "connection": 0.55, "tension": 0.65, "progression": 0.55, "wonder": 0.6,
}
# 转场能量跳变阈值（从 visual_rules.json 读取，保持默认 0.35/0.2 与原有逻辑一致）
_TR = transition_threshold()
_ENERGY_JUMP_LARGE = _TR.get("energy_jump_large", 0.4)
_ENERGY_JUMP_MEDIUM = _TR.get("energy_jump_small", 0.15)


def _compute_emotional_delta(
    source_shot: Dict,
    target_shot: Dict,
) -> Dict[str, Any]:
    """
    计算两个相邻 shot 之间的情绪变化：
    - direction: "escalation" / "de-escalation" / "lateral_shift"
    - intensity_change: 数值差
    - description: 可读描述
    """
    src_emotion = source_shot.get("emotion", "awe")
    tgt_emotion = target_shot.get("emotion", "awe")

    src_energy = _EMOTION_ENERGY.get(src_emotion, 0.5)
    tgt_energy = _EMOTION_ENERGY.get(tgt_emotion, 0.5)
    delta = tgt_energy - src_energy

    if delta > 0.1:
        direction = "escalation"
    elif delta < -0.1:
        direction = "de-escalation"
    else:
        direction = "lateral_shift"

    return {
        "from": src_emotion,
        "to": tgt_emotion,
        "direction": direction,
        "intensity_change": round(delta, 2),
        "description": f"{src_emotion} -> {tgt_emotion} ({direction})",
    }


def _compute_emotional_arc(shots: List[Dict]) -> Dict[str, Any]:
    """
    计算整个镜头序列的全局情绪曲线。
    """
    emotions = [s.get("emotion", "awe") for s in shots]
    energies = [_EMOTION_ENERGY.get(e, 0.5) for e in emotions]

    if not energies:
        return {"shape": "flat", "peak_index": 0, "valley_index": 0,
                "overall_direction": "static"}

    peak_idx = energies.index(max(energies))
    valley_idx = energies.index(min(energies))

    # 判断曲线形状
    n = len(energies)
    if n < 2:
        shape = "single"
    elif peak_idx == n - 1 or peak_idx == n - 2:
        shape = "crescendo"       # 情绪在最后达到高潮
    elif peak_idx == 0 or peak_idx == 1:
        shape = "anti_crescendo"  # 情绪在开始最高
    elif valley_idx == n - 1:
        shape = "decay"           # 情绪逐渐下降
    elif valley_idx == 0:
        shape = "ascent"          # 情绪逐渐上升
    else:
        shape = "arc"             # 中间高两端低，或中间低两端高

    # 总体方向
    if energies[-1] > energies[0] + 0.1:
        overall = "escalating"
    elif energies[-1] < energies[0] - 0.1:
        overall = "de-escalating"
    else:
        overall = "balanced"

    return {
        "shape": shape,
        "peak_index": peak_idx,
        "peak_emotion": emotions[peak_idx],
        "valley_index": valley_idx,
        "valley_emotion": emotions[valley_idx],
        "energy_curve": [round(e, 2) for e in energies],
        "overall_direction": overall,
    }


# ============================================================
# 5. 序列图格式化器
# ============================================================

def _format_sequence_graph(graph: Dict[str, Any]) -> str:
    """
    将结构化 sequence_graph 格式化为导演可读的指令块。
    将注入 director prompt，让 director 具备"序列感知"能力。
    """
    lines = [
        f"[Sequence Graph — {len(graph['connections'])} connections | Arc: {graph['emotional_arc']['shape']} ({graph['emotional_arc']['overall_direction']})]",
        "",
    ]

    # 全局情绪曲线
    arc = graph["emotional_arc"]
    curve_str = " -> ".join(
        f"{e}({v:.1f})" for e, v in zip(
            [c["from_name"] for c in graph["connections"]] + [graph["connections"][-1]["to_name"]]
            if graph["connections"] else [],
            arc.get("energy_curve", [])
        )
    )
    if curve_str:
        lines.append(f"[Emotional Arc]: {curve_str}")
        lines.append(f"[Arc Shape]: {arc['shape']} — peak at shot #{arc['peak_index']+1} ({arc['peak_emotion']}), valley at shot #{arc['valley_index']+1} ({arc['valley_emotion']})")
        lines.append("")

    # 镜头间关系
    lines.append("Shot-to-Shot Connections:")
    for conn in graph["connections"]:
        trans_info = TRANSITION_TYPES.get(conn["transition"], {})
        lines.append(f"  [{conn['from_shot']+1}] {conn['from_name']} --> [{conn['to_shot']+1}] {conn['to_name']}")
        lines.append(f"      Transition: {conn['transition']} — {trans_info.get('description', '')}")
        lines.append(f"      Motion carry: {conn['motion_continuity']}")
        lines.append(f"      Emotion: {conn['emotional_delta']['description']}")
        lines.append(f"      Spatial: {conn['spatial_continuity']}")
        lines.append("")

    return "\n".join(lines)


def _format_director_sequence_context(
    graph: Dict[str, Any],
    hero_index: int,
) -> str:
    """
    为 director 生成精简的序列上下文：
    - hero shot 的前后关系
    - 需要承接的运动和情绪
    """
    connections = graph["connections"]
    arc = graph["emotional_arc"]

    lines = [
        f"[Sequence Context for Director — Hero Shot #{hero_index + 1}]",
        f"[Emotional Arc: {arc['shape']}, direction: {arc['overall_direction']}]",
        "",
    ]

    # 找到 hero shot 相关的连接
    incoming = [c for c in connections if c["to_shot"] == hero_index]
    outgoing = [c for c in connections if c["from_shot"] == hero_index]

    if incoming:
        c = incoming[0]
        lines.append(f"  INCOMING from shot #{c['from_shot']+1} \"{c['from_name']}\":")
        lines.append(f"    Transition into hero: {c['transition']}")
        lines.append(f"    Motion carry: {c['motion_continuity']}")
        lines.append(f"    Emotional shift: {c['emotional_delta']['description']}")
        lines.append("")

    if outgoing:
        c = outgoing[0]
        lines.append(f"  OUTGOING to shot #{c['to_shot']+1} \"{c['to_name']}\":")
        lines.append(f"    Transition out: {c['transition']}")
        lines.append(f"    Motion handoff: {c['motion_continuity']}")
        lines.append(f"    Emotional target: {c['emotional_delta']['description']}")
        lines.append("")

    if not incoming and not outgoing:
        lines.append("  (Hero shot is the only shot — no sequence context)")
        lines.append("")

    lines.append("  [Directorial implication]: Your keyframe should feel like it belongs")
    lines.append("  in this sequence — consider the emotional momentum arriving at this")
    lines.append("  shot and the energy that needs to carry into the next.")

    return "\n".join(lines)


# ============================================================
# 6. 主入口
# ============================================================

def build_sequence_graph(shot_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    核心函数：消费 Phase 2 的 shot_plan，生成镜头关系图。

    Returns:
        {
            "connections": List[Dict],    # shot-to-shot 关系列表
            "emotional_arc": Dict,        # 全局情绪曲线
            "formatted_graph": str,       # 完整的镜头关系文本
            "director_context": str,      # 导演专用的精简序列上下文
        }
    """
    shots = shot_plan.get("shots", [])
    hero_index = shot_plan.get("hero_shot_index", 0)

    if len(shots) < 2:
        # 单镜头场景：返回最小化 graph
        return {
            "connections": [],
            "emotional_arc": {
                "shape": "single",
                "peak_index": 0,
                "peak_emotion": shots[0]["emotion"] if shots else "unknown",
                "valley_index": 0,
                "valley_emotion": shots[0]["emotion"] if shots else "unknown",
                "energy_curve": [_EMOTION_ENERGY.get(shots[0]["emotion"], 0.5)] if shots else [],
                "overall_direction": "static",
            },
            "formatted_graph": "[Sequence Graph — Single shot, no connections]",
            "director_context": "[Sequence Context — Single shot, no sequence relationships]",
        }

    # 构建 shot-to-shot 连接
    connections = []
    for i in range(len(shots) - 1):
        src = shots[i]
        tgt = shots[i + 1]

        transition = _select_transition(src, tgt)
        motion_cont = _compute_motion_continuity(src, tgt)
        spatial_cont = _compute_spatial_continuity(src, tgt)
        emo_delta = _compute_emotional_delta(src, tgt)

        connections.append({
            "from_shot": i,
            "to_shot": i + 1,
            "from_name": src["shot_name"],
            "to_name": tgt["shot_name"],
            "transition": transition,
            "motion_continuity": motion_cont,
            "emotional_delta": emo_delta,
            "spatial_continuity": spatial_cont,
        })

    # 计算全局情绪曲线
    emotional_arc = _compute_emotional_arc(shots)

    # 组装 sequence graph
    graph = {
        "connections": connections,
        "emotional_arc": emotional_arc,
    }

    # 格式化输出
    graph["formatted_graph"] = _format_sequence_graph(graph)
    graph["director_context"] = _format_director_sequence_context(graph, hero_index)

    return graph
