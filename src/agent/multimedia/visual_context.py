# src/agent/multimedia/visual_context.py
"""
Visual Context Retrieval Layer (Phase 1)
=========================================
职责：根据 scene script + global setting，生成结构化视觉上下文，
      让 director 具备"视觉规则决策"能力，而非单纯描述画面。

架构位置：
    showrunner → showrunner_review → 【visual_context_builder】→ director

设计原则：
    - 规则驱动 + 分类映射（无 embedding / vector DB）
    - 输出结构化 visual context，不做文本拼接
    - director 是"决策器"，不是"翻译器"
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# 1. 知识库加载器
# ============================================================

_KNOWLEDGE_DIR = Path(__file__).parent / "vision_knowledge"

# ============================================================
# 视觉规则配置（从 JSON 加载，支持热扩展）
# 配置文件：vision_knowledge/config/visual_rules.json
# 用户可通过编辑 JSON 文件添加新的 intent/genre/ambient 规则，无需修改代码
# ============================================================

_CONFIG_DIR = _KNOWLEDGE_DIR / "config"
_VISUAL_RULES_FILE = _CONFIG_DIR / "visual_rules.json"


def _load_visual_rules() -> dict:
    """从 JSON 配置文件加载视觉规则。文件缺失时抛出明确错误。"""
    if not _VISUAL_RULES_FILE.exists():
        raise FileNotFoundError(
            f"视觉规则配置文件缺失: {_VISUAL_RULES_FILE}\n"
            f"请确保 vision_knowledge/config/visual_rules.json 存在。"
        )
    with open(_VISUAL_RULES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# 模块级加载（首次 import 时执行一次）
_config = _load_visual_rules()

INTENT_VISUAL_RULES: Dict[str, Dict[str, List[str]]] = _config["intent_visual_rules"]
GENRE_MODIFIERS: Dict[str, Dict[str, List[str]]] = _config["genre_modifiers"]
TRAILER_PACING: Dict[str, Dict[str, str]] = _config["trailer_pacing"]
_INTENT_KEYWORDS: Dict[str, Dict[str, float]] = _config["intent_keywords"]
_GENRE_KEYWORDS: Dict[str, List[str]] = _config["genre_keywords"]
_AMBIENT_KEYWORDS: Dict[str, Dict[str, float]] = _config["ambient_keywords"]
AMBIENT_LIGHTING_OVERRIDES: Dict[str, List[str]] = _config["ambient_lighting_overrides"]
_AMBIENT_FILTER_KEYWORDS: Dict[str, List[str]] = _config["ambient_filter_keywords"]
_AMBIENT_WARNING_TEXT: Dict[str, str] = _config["ambient_warning_text"]


def detect_ambient_context(script: str, global_setting: str = "") -> Optional[str]:
    """
    从剧本和全局设定中检测环境光照上下文（昼夜 + 天气）。

    Returns:
        "night" | "rain" | "night+rain" | "dawn" | "dusk" | None
        如果同时检测到 night + rain，返回 "night+rain"（组合模式）
        如果什么都没检测到，返回 None（使用默认光照规则）
    """
    combined_text = (script + " " + global_setting).lower()
    scores: Dict[str, float] = {}

    for context, keywords in _AMBIENT_KEYWORDS.items():
        score = 0.0
        for kw, weight in keywords.items():
            if kw.lower() in combined_text:
                score += weight
        if score > 0:
            scores[context] = score

    if not scores:
        return None

    # 分离时间维度（night/dawn/dusk）和天气维度（rain）
    _TIME_CONTEXTS = {"night", "dawn", "dusk"}
    _WEATHER_CONTEXTS = {"rain"}
    _AMBIENT_THRESHOLD = 3.0  # 单一关键词即可触发的阈值

    time_hits = [ctx for ctx in _TIME_CONTEXTS if scores.get(ctx, 0) >= _AMBIENT_THRESHOLD]
    weather_hits = [ctx for ctx in _WEATHER_CONTEXTS if scores.get(ctx, 0) >= _AMBIENT_THRESHOLD]

    # 时间维度取最高分的（互斥：不可能同时是 night 和 dawn）
    best_time = max(time_hits, key=lambda x: scores[x]) if time_hits else None
    # 天气维度可以有多个，但目前只有 rain
    best_weather = weather_hits[0] if weather_hits else None

    # 组合逻辑：时间 + 天气
    if best_time and best_weather:
        combo = f"{best_time}+{best_weather}"
        # 如果组合在预设中有专用规则，使用它
        if combo in AMBIENT_LIGHTING_OVERRIDES:
            return combo
        # 否则用时间维度的规则 + 天气维度的规则叠加（动态生成）
        # 返回时间维度，但 _format_prompt_block 会补充天气警告
        return combo

    # 只有时间或只有天气
    if best_time:
        return best_time
    if best_weather:
        return best_weather

    # 有得分但都不够阈值 → 返回最高分的（弱匹配）
    best = max(scores, key=scores.get)
    return best


def filter_lighting_for_ambient(
    lighting_rules: List[str],
    ambient_context: str,
) -> List[str]:
    """
    根据环境上下文过滤掉不兼容的默认光照规则，
    并替换为环境适配的光照规则。

    Args:
        lighting_rules: 原始 INTENT_VISUAL_RULES 中的光照规则列表
        ambient_context: detect_ambient_context() 的返回值

    Returns:
        替换后的光照规则列表
    """
    if ambient_context is None:
        return lighting_rules

    # Step 1: 收集过滤关键词（支持组合拆分，如 "dawn+rain"）
    parts = ambient_context.split("+")
    filter_keywords = []
    for part in parts:
        filter_keywords.extend(_AMBIENT_FILTER_KEYWORDS.get(part.strip(), []))
    filter_keywords = list(set(filter_keywords))  # 去重

    # Step 1.5: 过滤掉不兼容的默认规则
    filtered = []
    for rule in lighting_rules:
        rule_lower = rule.lower()
        if any(kw in rule_lower for kw in filter_keywords):
            continue  # 跳过包含不兼容关键词的规则
        filtered.append(rule)

    # Step 2: 追加环境适配的光照规则（支持组合合并）
    override_rules = list(AMBIENT_LIGHTING_OVERRIDES.get(ambient_context, []))
    if not override_rules and len(parts) > 1:
        # 动态组合不在预设中 → 合并各子维度的规则
        seen = set()
        for part in parts:
            for rule in AMBIENT_LIGHTING_OVERRIDES.get(part.strip(), []):
                if rule not in seen:
                    override_rules.append(rule)
                    seen.add(rule)

    result = filtered + override_rules
    return result


def classify_scene_intent(script: str) -> str:
    """
    将场景剧本分类为 7 种意图之一。
    使用加权关键词匹配，返回得分最高的意图类型。

    Returns:
        "world_building" | "character_introduction" | "action" | "emotional"
        | "dialogue" | "montage" | "reveal"
    """
    script_lower = script.lower()
    scores: Dict[str, float] = {k: 0.0 for k in _INTENT_KEYWORDS}

    for intent, keywords in _INTENT_KEYWORDS.items():
        for keyword, weight in keywords.items():
            if keyword.lower() in script_lower:
                scores[intent] += weight

    # 取得分最高的意图；如果全为 0，默认 world_building（安全兜底）
    best_intent = max(scores, key=scores.get)
    if scores[best_intent] == 0.0:
        print("    [!] 意图分类：所有 intent 得分均为 0，降级为 world_building")
        return "world_building"

    # 置信度日志：打印前两名得分，弱匹配时警告
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top1_name, top1_score = sorted_scores[0]
    top2_name, top2_score = sorted_scores[1] if len(sorted_scores) > 1 else ("N/A", 0)
    if top1_score < 3.0:
        print(f"    ⚠️ 意图分类置信度低: {top1_name}={top1_score:.1f} vs {top2_name}={top2_score:.1f}（<3.0 为弱匹配）")
    else:
        print(f"    [*] 意图分类: {top1_name}（得分 {top1_score:.1f}，次高 {top2_name}={top2_score:.1f}）")

    return best_intent


def detect_global_genre(global_setting: str) -> List[str]:
    """
    从全局设定中检测风格类型（可多选）。

    Returns:
        匹配到的风格列表，如 ["cyberpunk", "gothic"]
    """
    setting_lower = global_setting.lower()
    detected = []

    for genre, keywords in _GENRE_KEYWORDS.items():
        matched = False
        for kw in keywords:
            kw_lower = kw.lower()
            # 英文关键词：使用词边界匹配避免子串误触发
            if all(ord(c) < 0x4e00 for c in kw):
                # 纯英文关键词
                pattern = r'\b' + re.escape(kw_lower) + r'\b'
                if re.search(pattern, setting_lower):
                    matched = True
                    break
            else:
                # 中文关键词：保持子串匹配
                if kw_lower in setting_lower:
                    matched = True
                    break
        if matched:
            detected.append(genre)

    if not detected:
        print("    [!] 未检测到匹配的视觉风格关键词，不叠加 genre modifier（而非默认 sci_fi）")
    return detected


def determine_pacing_phase(scene_index: int, total_scenes: int) -> str:
    """
    根据场景在预告片中的位置，判断节奏阶段。

    Returns:
        "opening" | "build_up" | "climax" | "resolution"
    """
    if total_scenes <= 1:
        return "climax"

    ratio = scene_index / max(total_scenes - 1, 1)

    if ratio < 0.2:
        return "opening"
    elif ratio < 0.6:
        return "build_up"
    elif ratio < 0.85:
        return "climax"
    else:
        return "resolution"


# ============================================================
# 3. 视觉上下文组装器
# ============================================================

def build_visual_context(
    script: str,
    global_setting: str,
    scene_index: int,
    total_scenes: int,
    prev_anchors: Optional[Dict] = None,
) -> Dict[str, any]:
    """
    核心函数：根据场景剧本 + 全局设定 → 生成结构化视觉上下文。

    Args:
        script: 当前场景剧本
        global_setting: 全局视觉设定
        scene_index: 当前场景索引（0-based）
        total_scenes: 总场景数
        prev_anchors: 上一场景的视觉锚点（由 advance_scene 提取），
                      用于生成场景间的视觉连续性约束。首场景为 None。

    Returns:
        {
            "intent": str,
            "camera_rules": List[str],
            "lighting_rules": List[str],
            "motion_rules": List[str],
            "composition_rules": List[str],
            "mood_keywords": List[str],
            "genre_modifiers": Dict[str, List[str]],
            "pacing": Dict[str, str],
            "formatted_prompt_block": str,
            "continuity_directives": Optional[Dict],  # 场景连续性指令
        }
    """
    # Step 1: 场景意图分类
    intent = classify_scene_intent(script)

    # Step 2: 从知识库获取意图对应的视觉规则
    # 容错：未知 intent 降级到 world_building（最通用的视觉规则）
    rules = INTENT_VISUAL_RULES.get(intent, INTENT_VISUAL_RULES.get("world_building", {}))
    if intent not in INTENT_VISUAL_RULES:
        print(f"    [!] 未知 intent '{intent}'，降级使用 world_building 视觉规则")

    # Step 2.5: 环境光照上下文检测（昼夜 + 天气）
    # 根据剧本内容过滤不兼容的默认光照规则（如夜间场景去掉 god rays）
    ambient_context = detect_ambient_context(script, global_setting)
    base_lighting = rules["lighting"]
    adjusted_lighting = filter_lighting_for_ambient(base_lighting, ambient_context)
    if ambient_context:
        print(f"    [*] 环境光照: {ambient_context} → 已替换 {len(base_lighting) - len(adjusted_lighting) + len(AMBIENT_LIGHTING_OVERRIDES.get(ambient_context, []))} 条光照规则")

    # Step 3: 全局风格检测 → 叠加修饰器
    genres = detect_global_genre(global_setting)
    genre_mods: Dict[str, List[str]] = {}
    for genre in genres:
        mods = GENRE_MODIFIERS.get(genre, {})
        for key, values in mods.items():
            genre_mods.setdefault(key, []).extend(values)

    # Step 4: 节奏阶段判断
    pacing_phase = determine_pacing_phase(scene_index, total_scenes)
    pacing = TRAILER_PACING[pacing_phase]

    # Step 5: 组装结构化视觉上下文
    context = {
        "intent": intent,
        "genres": genres,
        "camera_rules": rules["camera"],
        "lighting_rules": adjusted_lighting,  # 使用环境适配后的光照规则
        "motion_rules": rules["motion"],
        "composition_rules": rules["composition"],
        "mood_keywords": rules["mood"],
        "genre_modifiers": genre_mods,
        "pacing": pacing,
        "pacing_phase": pacing_phase,
        "ambient_context": ambient_context,  # 昼夜/天气上下文
    }

    # Step 5.5: 场景连续性指令（从上一场景的视觉锚点生成渐变约束）
    continuity = _build_continuity_directives(prev_anchors, context) if prev_anchors else None
    context["continuity_directives"] = continuity

    # Step 6: 生成导演可读的 prompt block
    context["formatted_prompt_block"] = _format_prompt_block(context, genres)

    return context


def _build_continuity_directives(
    prev_anchors: Dict,
    current_context: Dict,
) -> Dict[str, str]:
    """
    根据上一场景的视觉锚点和当前场景的上下文，生成场景间的连续性约束。

    约束类型：
    - color_transition: 色调渐变方向（warm→cool 需过渡，不可突变）
    - lighting_continuity: 光照衔接指令（保持/渐变/切换）
    - camera_continuity: 运镜惯性（延续上一场景的运动方向或从静止启动）
    - energy_transition: 能量过渡（上一场景结尾能量 → 当前场景起始能量的变化方向）
    """
    directives = {}

    # ── 色调渐变 ──
    prev_color = prev_anchors.get("color_direction", "neutral")
    # 从当前 mood 推断色调方向
    warm_moods = {"triumph", "power", "presence", "hope", "anticipation"}
    cool_moods = {"isolation", "mystery", "loss", "vulnerability", "awe"}
    curr_moods = set(current_context.get("mood_keywords", []))
    if curr_moods & warm_moods:
        curr_color = "warm"
    elif curr_moods & cool_moods:
        curr_color = "cool"
    else:
        curr_color = "neutral"

    if prev_color != curr_color and prev_color != "neutral" and curr_color != "neutral":
        directives["color_transition"] = (
            f"Shift color palette gradually from {prev_color} to {curr_color}. "
            f"Do not make an abrupt color temperature jump."
        )
    elif prev_color == curr_color and prev_color != "neutral":
        directives["color_transition"] = (
            f"Maintain {prev_color} color palette from the previous scene for visual cohesion."
        )

    # ── 光照衔接 ──
    prev_lighting = prev_anchors.get("lighting_anchor", "")
    prev_ambient = prev_anchors.get("ambient_context")
    curr_ambient = current_context.get("ambient_context")

    if prev_ambient and curr_ambient and prev_ambient != curr_ambient:
        directives["lighting_continuity"] = (
            f"Previous scene was set in {prev_ambient}; this scene is {curr_ambient}. "
            f"Transition the lighting atmosphere gradually — the first shot should carry "
            f"residual {prev_ambient} ambience before settling into {curr_ambient} lighting."
        )
    elif prev_ambient == curr_ambient and prev_ambient:
        directives["lighting_continuity"] = (
            f"Continue the {prev_ambient} ambient lighting from the previous scene."
        )

    # ── 运镜惯性 ──
    prev_momentum = prev_anchors.get("camera_momentum", "")
    if prev_momentum:
        # 提取运动方向关键词
        momentum_keywords = []
        if "dolly in" in prev_momentum or "forward" in prev_momentum:
            momentum_keywords.append("forward push")
        if "dolly out" in prev_momentum or "pull" in prev_momentum:
            momentum_keywords.append("pull back")
        if "tracking" in prev_momentum or "lateral" in prev_momentum:
            momentum_keywords.append("lateral movement")
        if "crane" in prev_momentum or "ascending" in prev_momentum:
            momentum_keywords.append("vertical rise")
        if "orbit" in prev_momentum:
            momentum_keywords.append("orbital motion")

        if momentum_keywords:
            directions = " / ".join(momentum_keywords)
            directives["camera_continuity"] = (
                f"Previous scene ended with {directions} camera momentum. "
                f"The opening shot of this scene should either continue this momentum "
                f"or deliberately arrest it for dramatic effect."
            )

    # ── 能量过渡 ──
    prev_energy = prev_anchors.get("emotion_energy", 0.5)
    # 从当前场景的第一个 mood 推断起始能量
    _ENERGY_LOOKUP = {
        "awe": 0.6, "grandeur": 0.65, "mystery": 0.4, "anticipation": 0.5,
        "presence": 0.6, "intimacy": 0.55, "power": 0.75, "vulnerability": 0.45,
        "isolation": 0.3, "tenderness": 0.5, "loss": 0.35, "adrenaline": 0.85,
        "intensity": 0.9, "triumph": 0.8, "insignificance": 0.35,
    }
    curr_moods = current_context.get("mood_keywords", [])
    curr_energy = _ENERGY_LOOKUP.get(curr_moods[0], 0.5) if curr_moods else 0.5
    energy_delta = curr_energy - prev_energy

    if abs(energy_delta) > 0.25:
        if energy_delta > 0:
            directives["energy_transition"] = (
                f"Energy rises sharply from the previous scene (low → high). "
                f"Open with a transitional beat before escalating to full intensity."
            )
        else:
            directives["energy_transition"] = (
                f"Energy drops sharply from the previous scene (high → low). "
                f"Begin with residual intensity and gradually settle into the calmer mood."
            )
    elif abs(energy_delta) <= 0.1:
        directives["energy_transition"] = (
            "Energy level is similar to the previous scene — maintain seamless flow."
        )

    # ── 角色外观一致性（identity lock）──
    char_id = prev_anchors.get("character_identity", {})
    if char_id.get("has_character"):
        subject = char_id.get("subject", "")
        appearance = char_id.get("appearance", [])
        if subject or appearance:
            parts = []
            if subject:
                parts.append(f"The character \"{subject}\" appeared in the previous scene")
            if appearance:
                # 去重并保持原始顺序
                seen = set()
                unique_appearance = []
                for a in appearance:
                    if a not in seen:
                        unique_appearance.append(a)
                        seen.add(a)
                parts.append(
                    f"with these visual traits: {', '.join(unique_appearance)}"
                )
            identity_desc = " ".join(parts) + "."
            directives["character_identity"] = (
                f"{identity_desc} "
                f"If the same character appears in this scene, maintain consistent "
                f"appearance — same clothing, hairstyle, equipment, and distinguishing features. "
                f"Do not alter the character's visual identity between scenes."
            )

    return directives


def _format_prompt_block(context: Dict, genres: List[str]) -> str:
    """
    将结构化视觉上下文格式化为一段简洁的导演指令块。
    这段文字将被注入 DIRECTOR_SYSTEM_PROMPT，
    让 LLM 以"导演决策"而非"画面描述"的方式思考。
    """
    intent = context["intent"]
    pacing = context["pacing"]

    # 镜头语言指令
    camera_lines = "\n".join(f"    - {c}" for c in context["camera_rules"])

    # 光影策略指令
    lighting_lines = "\n".join(f"    - {l}" for l in context["lighting_rules"])

    # 动态运动指令
    motion_lines = "\n".join(f"    - {m}" for m in context["motion_rules"])

    # 构图规则
    comp_lines = "\n".join(f"    - {c}" for c in context["composition_rules"])

    # 风格修饰器
    genre_section = ""
    if context["genre_modifiers"]:
        genre_parts = []
        for key, values in context["genre_modifiers"].items():
            vals = "; ".join(values)
            genre_parts.append(f"    {key}: {vals}")
        genre_section = "\n[Genre-Specific Modifiers]\n" + "\n".join(genre_parts)

    # 环境上下文提示（昼夜/天气）
    ambient_section = ""
    if context.get("ambient_context"):
        ambient_ctx = context['ambient_context']
        warning_text = _AMBIENT_WARNING_TEXT.get(
            ambient_ctx,
            f"All lighting must be consistent with the {ambient_ctx} setting. Avoid contradictory light sources."
        )
        ambient_section = (
            f"\n[Ambient Context: {ambient_ctx}]\n"
            f"    IMPORTANT: The scene takes place during {ambient_ctx}. "
            f"All lighting choices must be consistent with this ambient condition. "
            f"{warning_text}"
        )

    # 场景连续性约束（跨场景视觉锚点生成的渐变指令）
    continuity_section = ""
    continuity = context.get("continuity_directives")
    if continuity:
        cont_lines = []
        for key, instruction in continuity.items():
            label = key.replace("_", " ").title()
            cont_lines.append(f"    - [{label}]: {instruction}")
        continuity_section = (
            "\n[Cross-Scene Continuity — Maintain visual coherence with previous scene]\n"
            + "\n".join(cont_lines)
        )

    # 情绪关键词
    mood_str = ", ".join(context["mood_keywords"])

    block = f"""
[Visual Director Context — Scene Intent: {intent}]

[Camera Grammar — Choose the most fitting for this scene]
{camera_lines}

[Lighting Strategy — Apply based on mood and genre]
{lighting_lines}
{ambient_section}

[Motion & Pacing — Pacing phase: {context['pacing_phase']}]
{motion_lines}
    Pacing: {pacing['pacing']}
    Cut rhythm: {pacing['cut_rhythm']}

[Composition Rules]
{comp_lines}

[Target Mood]: {mood_str}{genre_section}{continuity_section}

[Director's Task]: Using the visual rules above, make deliberate cinematographic decisions:
    1. SELECT a specific camera shot type and movement
    2. CHOOSE a lighting setup that serves the mood
    3. DESIGN the motion and kinetic energy of the frame
    4. COMPOSE the frame with intentional subject placement
    5. UNIFY all choices into a single cohesive visual statement
"""
    return block.strip()
