# src/agent/multimedia/shot_strategy.py
"""
Shot Strategy Layer (Phase 2)
==============================
职责：将 Phase 1 的"视觉规则"转化为"镜头拍摄方案"。
      输出结构化的多镜头序列（shot plan），供 director 精确执行。

架构位置：
    visual_context_builder → 【shot_strategy_builder】→ director

与 Phase 1 的关系：
    Phase 1 (visual_context) = "懂视觉语言"
    Phase 2 (shot_strategy)  = "会设计镜头"

设计原则：
    - shot_strategy 不生成 prompt，只生成镜头计划（planning layer）
    - director 不再"思考镜头"，只执行 shot_strategy
    - 输出必须是结构化 shot list，不是自然语言描述
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from .visual_context import AMBIENT_LIGHTING_OVERRIDES


# ============================================================
# 镜头模板配置（从 JSON 加载，支持热扩展）
# 配置文件：vision_knowledge/config/shot_templates.json
# 用户可通过编辑 JSON 文件添加新的 intent 模板或调整镜头数量
# ============================================================

_CONFIG_DIR = Path(__file__).parent / "vision_knowledge" / "config"
_SHOT_TEMPLATES_FILE = _CONFIG_DIR / "shot_templates.json"


def _load_shot_config() -> dict:
    """从 JSON 配置文件加载镜头模板。文件缺失时抛出明确错误。"""
    if not _SHOT_TEMPLATES_FILE.exists():
        raise FileNotFoundError(
            f"镜头模板配置文件缺失: {_SHOT_TEMPLATES_FILE}\n"
            f"请确保 vision_knowledge/config/shot_templates.json 存在。"
        )
    with open(_SHOT_TEMPLATES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# 模块级加载
_shot_config = _load_shot_config()

SHOT_TEMPLATES: Dict[str, List[Dict[str, Any]]] = _shot_config["shot_templates"]
GENRE_LIGHTING_MODIFIERS: Dict[str, Dict[str, str]] = _shot_config["genre_lighting_modifiers"]
_HERO_POS_PREF = _shot_config["hero_position_preference"]


# ============================================================
# 3. 焦点对象提取（规则驱动，从 script 中提取视觉焦点）
# ============================================================

_FOCUS_PATTERNS: Dict[str, List[str]] = {
    "character": [
        "剑客", "战士", "骑士", "英雄", "角色", "主角",
        "身影", "面容", "眼神", "双眼", "脸庞",
        "hero", "warrior", "character", "figure", "knight",
        "eyes", "face", "silhouette", "protagonist",
    ],
    "vehicle": [
        "战马", "战车", "飞船", "机甲", "巨龙",
        "steed", "ship", "mech", "dragon", "vehicle", "machine",
    ],
    "weapon": [
        "剑", "枪", "弓", "刃", "锤", "盾",
        "sword", "gun", "blade", "weapon", "bow", "shield",
    ],
    "creature": [
        "龙", "兽", "怪物", "巨", "精灵", "恶魔",
        "dragon", "beast", "monster", "creature", "demon", "spirit",
    ],
    "environment": [
        "城堡", "城市", "天空", "山脉", "森林", "海洋",
        "废墟", "荒原", "都市", "星空", "天际",
        "castle", "city", "sky", "mountain", "forest", "ocean",
        "ruins", "landscape", "horizon", "storm", "thunder",
    ],
}


def _extract_focus(script: str) -> str:
    """从剧本中提取主要视觉焦点对象，返回实际匹配到的关键词而非类别名。"""
    script_lower = script.lower()
    category_scores: Dict[str, int] = {k: 0 for k in _FOCUS_PATTERNS}
    # 记录每个类别中匹配到的关键词及其出现位置（越靠前越重要）
    category_best_keyword: Dict[str, str] = {}

    for category, keywords in _FOCUS_PATTERNS.items():
        for kw in keywords:
            if kw.lower() in script_lower:
                category_scores[category] += 1
                # 只记录第一个命中的关键词（最具代表性）
                if category not in category_best_keyword:
                    category_best_keyword[category] = kw

    best_category = max(category_scores, key=category_scores.get)
    if category_scores[best_category] > 0:
        # 返回该类别中匹配到的实际关键词
        return category_best_keyword.get(best_category, best_category)
    return "scene"  # 通用默认值，比 "environment" 更中性


def _extract_camera_overrides(script: str) -> Dict[str, str]:
    """
    从剧本文本中提取用户指定的摄影机规格，用于覆盖模板默认值。
    剧本中写的摄影参数具有最高优先级。
    """
    script_lower = script.lower()
    overrides = {}

    # 视角检测
    angle_map = [
        ("鸟瞰", "bird's eye view"),
        ("俯瞰", "bird's eye view"),
        ("俯拍", "high angle overhead"),
        ("俯视", "high angle"),
        ("仰拍", "low angle"),
        ("仰角", "low angle"),
        ("高角度", "high angle"),
        ("低角度", "low angle"),
        ("平视", "eye level"),
        ("bird's eye", "bird's eye view"),
        ("overhead", "overhead"),
        ("high angle", "high angle"),
        ("low angle", "low angle"),
        ("dutch angle", "dutch angle"),
    ]
    for keyword, value in angle_map:
        if keyword in script_lower:
            overrides["angle"] = value
            break

    # 镜头运动检测
    movement_map = [
        ("静态", "static locked"),
        ("固定", "static locked"),
        ("跟随", "steadicam follow"),
        ("跟踪", "steadicam follow"),
        ("追踪", "fast lateral tracking"),
        ("推拉", "dolly in"),
        ("拉远", "dolly out"),
        ("推近", "slow dolly in"),
        ("环绕", "orbit"),
        ("旋转", "slow rotation"),
        ("弧线", "arc movement"),
        ("static", "static locked"),
        ("tracking", "steadicam follow"),
        ("dolly in", "slow dolly in"),
        ("dolly out", "slow dolly out"),
        ("orbit", "slow orbit"),
    ]
    for keyword, value in movement_map:
        if keyword in script_lower:
            overrides["movement"] = value
            break

    # 焦段检测
    lens_patterns = [
        ("16mm", "16mm ultra-wide"),
        ("24mm", "24mm wide"),
        ("35mm", "35mm standard wide"),
        ("50mm", "50mm standard"),
        ("85mm", "85mm telephoto"),
        ("100mm", "100mm macro"),
        ("135mm", "135mm telephoto"),
        ("200mm", "200mm telephoto"),
        ("超广角", "16mm ultra-wide"),
        ("广角", "24mm wide"),
        ("中长焦", "85mm telephoto"),
        ("长焦", "135mm telephoto"),
        ("微距", "100mm macro"),
    ]
    for keyword, value in lens_patterns:
        if keyword in script_lower:
            overrides["lens"] = value
            break

    # 景别检测（覆盖 camera type）
    shot_type_map = [
        ("特写", "close-up"),
        ("大特写", "extreme close-up"),
        ("中景", "medium shot"),
        ("全景", "wide shot"),
        ("远景", "extreme wide shot"),
        ("大远景", "extreme wide shot"),
        ("close-up", "close-up"),
        ("medium shot", "medium shot"),
        ("wide shot", "wide shot"),
        ("extreme wide", "extreme wide shot"),
    ]
    for keyword, value in shot_type_map:
        if keyword in script_lower:
            overrides["type"] = value
            break

    # 朝向检测：剧本中显式提到的主体朝向具有最高优先级
    orientation_map = [
        ("背影", "back_to_camera"),
        ("背面", "back_to_camera"),
        ("背对", "back_to_camera"),
        ("侧面", "side_profile"),
        ("侧身", "side_profile"),
        ("过肩", "over_shoulder"),
        ("四分之三", "three_quarter"),
        ("3/4", "three_quarter"),
        ("正面", "facing_camera"),
        ("面朝镜头", "facing_camera"),
        ("back view", "back_to_camera"),
        ("side profile", "side_profile"),
        ("over the shoulder", "over_shoulder"),
        ("three-quarter", "three_quarter"),
        ("facing camera", "facing_camera"),
    ]
    for keyword, value in orientation_map:
        if keyword in script_lower:
            overrides["orientation"] = value
            break

    return overrides


# ── 剧本情绪检测：从剧本关键词推导 emotion，覆盖模板硬编码值 ──
_SCRIPT_EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "triumph": ["胜利", "凯旋", "欢呼", " triumph", "victory", "triumphant", "cheer"],
    "despair": ["绝望", "崩溃", "哀嚎", "despair", "hopeless", "devastated", "collapse"],
    "rage": ["愤怒", "暴怒", "狂怒", "rage", "furious", "wrath", "enraged"],
    "serenity": ["宁静", "安详", "平和", "serene", "peaceful", "calm", "tranquil"],
    "tension": ["紧张", "警惕", "对峙", "tension", "alert", "standoff", "suspense"],
    "sorrow": ["悲伤", "哭泣", "告别", "sorrow", "grief", "tears", "farewell", "mourn"],
    "determination": ["决心", "誓言", "坚毅", "determination", "resolve", "vow", "steadfast"],
    "awe": ["震撼", "壮观", "敬畏", "awe", "majestic", "grandeur", "breathtaking"],
}


def _detect_script_emotion(script: str) -> Optional[str]:
    """从剧本中检测主导情绪。返回 emotion 字符串或 None（无明显信号时）。"""
    script_lower = script.lower()
    scores: Dict[str, int] = {}
    for emotion, keywords in _SCRIPT_EMOTION_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw.lower() in script_lower)
        if count > 0:
            scores[emotion] = count
    if not scores:
        return None
    best = max(scores, key=scores.get)
    if scores[best] >= 2:  # 至少 2 个关键词命中才算强信号
        return best
    return None


# ============================================================
# 4. 英雄镜头选择器
# ============================================================

# 每个 pacing phase 偏好的 shot_name
_PACING_PREFERRED_SHOTS: Dict[str, List[str]] = {
    "opening":    ["panorama", "silhouette", "establish", "close_up",
                   "two_shot", "exchange", "anticipation"],
    "build_up":   ["reveal", "movement", "sweep", "subtle_motion",
                   "over_shoulder", "reaction", "breakthrough"],
    "climax":     ["impact", "detail", "stance", "scale_reveal",
                   "flash", "match_cut", "crescendo", "hero_reveal"],
    "resolution": ["aftermath", "fade", "atmosphere", "stillness"],
}


def _select_hero_shot(shots: List[Dict], pacing_phase: str) -> int:
    """
    从镜头序列中选择最适合当前节奏阶段的"英雄镜头"。
    英雄镜头将作为该场景的首帧关键帧。

    评分规则：
      - pacing 匹配度（最高 +30）
      - 镜头权重 × 100（最高 +35）
      - 叙事位置偏好（中间偏前 +10）

    Returns:
        hero shot 在 shots 列表中的索引
    """
    preferred = _PACING_PREFERRED_SHOTS.get(pacing_phase, [])
    n = len(shots)
    scores = []

    for i, shot in enumerate(shots):
        score = 0.0
        name = shot["shot_name"]
        weight = shot["pacing_weight"]

        # pacing phase 匹配度
        if name in preferred:
            rank = preferred.index(name)
            score += (30 - rank * 8)

        # 镜头节奏权重
        score += weight * 100

        # 叙事位置偏好（偏向中间）
        if n > 1:
            position_ratio = i / (n - 1)
            if _HERO_POS_PREF["min_ratio"] <= position_ratio <= _HERO_POS_PREF["max_ratio"]:
                score += _HERO_POS_PREF["bonus_score"]

        scores.append(score)

    return scores.index(max(scores))


def _select_end_shot(shots: List[Dict], hero_index: int) -> int:
    """
    为双帧模式选择尾帧镜头。
    优先选择序列中最后一个与 hero 不同的 shot。
    """
    n = len(shots)
    # 从后往前找第一个不是 hero 的 shot
    for i in range(n - 1, -1, -1):
        if i != hero_index:
            return i
    # 如果只有一个 shot，返回 0
    return 0


# ============================================================
# 5. 镜头规格填充器
# ============================================================

_LENS_RULES = {
    "extreme wide shot": "16mm ultra-wide for maximum environmental scope",
    "wide shot":         "24mm wide-angle for environmental context",
    "medium shot":       "50mm standard for balanced subject-environment",
    "close-up":          "85mm telephoto for subject isolation and compression",
    "extreme close-up":  "100mm macro for intimate detail and shallow DOF",
}


_EMOTION_LIGHTING_MODIFIERS: Dict[str, str] = {
    "adrenaline":    "high-contrast dynamic lighting with sharp shadow edges",
    "intensity":     "dramatic chiaroscuro with concentrated key light",
    "triumph":       "warm golden accent highlights with subtle lens flare",
    "anger":         "harsh overhead shadows with red-warm color contrast",
    "sadness":       "cool desaturated fill light with blue color cast",
    "fear":          "flickering low-key light with deep shadow pools",
    "anticipation":  "building contrast with directional side light",
    "awe":           "soft diffused ambient glow with volumetric depth",
    "grandeur":      "layered volumetric light with rim accents",
    "mystery":       "low-key side lighting with deep negative space",
    "vulnerability": "soft intimate window light with warm undertone",
    "isolation":     "single harsh practical source with surrounding darkness",
    "tenderness":    "warm golden backlight with gentle diffusion",
    "loss":          "fading warm light overtaken by cool shadows",
    "power":         "strong directional key light with heroic rim",
    "presence":      "balanced cinematic lighting with subtle rim separation",
    "intimacy":      "close warm key light with shallow shadow falloff",
    # dialogue / montage / reveal emotions
    "connection":    "warm balanced two-subject lighting with matched key sources",
    "tension":       "split lighting with opposing shadow directions",
    "progression":   "evolving color temperature with increasing contrast",
    "wonder":        "ethereal volumetric light with soft bloom and sparkle",
}


def _diversify_lighting(
    shots: List[Dict],
    hero_index: int,
) -> None:
    """
    Per-shot lighting diversification (in-place).
    Based on emotion, pacing_weight, and hero status, append unique
    lighting modifiers so that no two shots have identical lighting descriptions.
    """
    for i, shot in enumerate(shots):
        light = shot.get("lighting", {})
        desc = light.get("full_description", light.get("setup", ""))
        emotion = shot.get("emotion", "")
        pacing_w = shot.get("pacing_weight", 0.25)
        additions = []

        # 1. Emotion-based modifier
        emo_mod = _EMOTION_LIGHTING_MODIFIERS.get(emotion, "")
        if emo_mod:
            additions.append(emo_mod)

        # 2. Hero shot gets dramatic emphasis
        if i == hero_index:
            additions.append("dramatic key light emphasis with enhanced volumetric atmosphere")

        # 3. Low-pacing (atmospheric) shots get subdued treatment
        elif pacing_w <= 0.15:
            additions.append("subdued ambient glow with desaturated atmospheric tones")

        if additions:
            light["full_description"] = f"{desc}; {', '.join(additions)}"


def _fill_shot_details(
    template_shot: Dict,
    script: str,
    genres: List[str],
    ambient_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    将模板 shot 填充为完整规格：
    - 添加 focus_object（从 script 提取）
    - 添加 genre 修饰的 lighting
    - 添加 lens feel 描述
    - 根据 ambient_context（昼夜/天气）替换不兼容的模板默认光照
    """
    shot = {k: (v.copy() if isinstance(v, dict) else v)
            for k, v in template_shot.items()}

    # 提取焦点对象
    shot["focus_object"] = _extract_focus(script)

    # 剧本情绪覆盖：当剧本有强情绪信号时，覆盖模板硬编码的 emotion
    script_emotion = _detect_script_emotion(script)
    if script_emotion:
        shot["emotion"] = script_emotion

    # 镜头焦距描述
    cam_type = shot["camera"]["type"]
    shot["lens_feel"] = _LENS_RULES.get(cam_type, "35mm standard")

    # 环境光照补充：保留模板原始 setup（维持 per-shot 差异），将环境光照作为补充追加
    if ambient_context and ambient_context in AMBIENT_LIGHTING_OVERRIDES:
        ambient_rules = AMBIENT_LIGHTING_OVERRIDES[ambient_context]
        original_setup = shot["lighting"]["setup"]
        shot["lighting"]["setup"] = f"{original_setup}; ambient: {'; '.join(ambient_rules)}"

    # genre 光照叠加
    genre_parts = []
    for genre in genres:
        mod = GENRE_LIGHTING_MODIFIERS.get(genre)
        if mod:
            genre_parts.append(f"{mod['overlay']}, {mod['atmosphere']}")

    if genre_parts:
        original = shot["lighting"]["setup"]
        shot["lighting"]["full_description"] = f"{original}; enhanced with {'; '.join(genre_parts)}"
    else:
        shot["lighting"]["full_description"] = shot["lighting"]["setup"]

    return shot


# ============================================================
# 6. 镜头计划格式化器
# ============================================================

def _format_shot_plan_for_prompt(shot_plan: Dict[str, Any]) -> str:
    """
    将结构化 shot_plan 格式化为导演可读的指令块。
    这段文字将注入 director prompt 中。
    """
    ambient = shot_plan.get("ambient_context")
    ambient_str = f" | Ambient: {ambient}" if ambient else ""
    lines = [
        f"[Shot Strategy — Intent: {shot_plan['intent']} | Genre: {', '.join(shot_plan['genres'])}{ambient_str}]",
        f"[Hero Shot: #{shot_plan['hero_shot_index'] + 1} \"{shot_plan['shots'][shot_plan['hero_shot_index']]['shot_name']}\" — Generate FIRST FRAME from this shot]",
        "",
        "Shot Sequence:",
    ]

    for i, shot in enumerate(shot_plan["shots"]):
        marker = " >>HERO<<" if i == shot_plan["hero_shot_index"] else ""
        cam = shot["camera"]
        light = shot["lighting"]

        lines.append(f"  [{i + 1}] {shot['shot_name'].upper()}{marker}")
        orientation = cam.get('orientation', 'facing_camera')
        tracking = cam.get('tracking', '')
        lines.append(f"      Camera: {cam['type']}, {cam['movement']}, {cam['angle']}, {cam['lens']}")
        lines.append(f"      Subject Orientation: {orientation}")
        if tracking:
            lines.append(f"      Camera Tracking: {tracking}")
        lines.append(f"      Lighting: {light.get('full_description', light['setup'])}")
        lines.append(f"      Emotion: {shot['emotion']}  |  Focus: {shot['focus_object']}  |  Pacing: {shot['pacing_weight']:.0%}")
        if shot.get("focus_hint"):
            lines.append(f"      Focus Hint: {shot['focus_hint']}")
        lines.append(f"      Lens: {shot['lens_feel']}")
        if shot.get("transition_preference"):
            lines.append(f"      -> Transition: {shot['transition_preference']}")
        if shot.get("next_shot_hint"):
            lines.append(f"      -> {shot['next_shot_hint']}")
        lines.append("")

    hero = shot_plan["shots"][shot_plan["hero_shot_index"]]
    hero_orientation = hero['camera'].get('orientation', 'facing_camera')
    hero_tracking = hero['camera'].get('tracking', '')
    lines.extend([
        "[Director Execution — HERO SHOT specs]",
        f"  Shot type: {hero['camera']['type']}",
        f"  Camera movement: {hero['camera']['movement']}",
        f"  Camera angle: {hero['camera']['angle']}",
        f"  Lens: {hero['camera']['lens']}",
        f"  Subject Orientation: {hero_orientation}",
        f"  Camera Tracking: {hero_tracking}" if hero_tracking else "",
        f"  Lighting: {hero['lighting'].get('full_description', hero['lighting']['setup'])}",
        f"  Emotion target: {hero['emotion']}",
        f"  Focus object: {hero['focus_object']}",
    ])

    # 如果剧本中有明确的摄影规格覆盖，显式标注给导演
    overrides = shot_plan.get("camera_overrides", {})
    if overrides:
        lines.append("")
        lines.append("[Script Camera Override — HIGHEST PRIORITY]")
        lines.append("  The script explicitly specifies the following camera parameters.")
        lines.append("  These MUST override any template defaults:")
        for key, value in overrides.items():
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)


# ============================================================
# 6.5 Phase 3: 邻接提示器（Adjacency Hints）
# ============================================================

# 基于情绪变化的默认转场偏好
_EMOTION_TRANSITION_HINTS: Dict[tuple, str] = {
    ("anticipation", "adrenaline"): "hard cut for energy burst",
    ("anticipation", "intensity"):  "hard cut into impact",
    ("adrenaline", "intensity"):    "continuous kinetic flow",
    ("intensity", "triumph"):       "dissolve into resolution",
    ("awe", "grandeur"):            "seamless camera carry",
    ("grandeur", "insignificance"): "match cut for scale contrast",
    ("insignificance", "mystery"):  "slow dissolve into atmosphere",
    ("presence", "intimacy"):       "camera carry into close range",
    ("intimacy", "power"):          "match cut to reveal stance",
    ("vulnerability", "isolation"): "gentle dissolve",
    ("isolation", "tenderness"):    "soft transition",
    ("tenderness", "loss"):         "slow dissolve into withdrawal",
}


def _add_adjacency_hints(shots: List[Dict]) -> None:
    """
    为每个 shot 添加邻接提示（in-place 修改）：
    - transition_preference: 该 shot 到下一个 shot 的转场偏好
    - next_shot_hint: 下一个 shot 的简要描述

    最后一个 shot 的 next_shot_hint 为 None，transition_preference 为 "sequence_end"。
    """
    for i in range(len(shots)):
        if i < len(shots) - 1:
            next_shot = shots[i + 1]
            src_emotion = shots[i].get("emotion", "")
            tgt_emotion = next_shot.get("emotion", "")

            # 查找转场偏好
            hint = _EMOTION_TRANSITION_HINTS.get(
                (src_emotion, tgt_emotion),
                "standard cut"
            )

            shots[i]["transition_preference"] = hint
            shots[i]["next_shot_hint"] = (
                f"next: {next_shot['shot_name']} "
                f"({next_shot['camera']['type']}, {next_shot['emotion']})"
            )
        else:
            shots[i]["transition_preference"] = "sequence_end"
            shots[i]["next_shot_hint"] = None


# ============================================================
# 7. 主入口
# ============================================================

def generate_shot_strategy(
    script: str,
    visual_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    核心函数：根据 scene script + Phase 1 的 visual_context
    生成结构化的镜头拍摄方案。

    Returns:
        {
            "intent": str,
            "genres": List[str],
            "pacing_phase": str,
            "shots": List[Dict],          # 完整镜头序列
            "hero_shot_index": int,       # 首帧使用的英雄镜头
            "end_shot_index": int,        # 尾帧使用的镜头（双帧模式）
            "formatted_plan": str,        # 导演可读的指令块
        }
    """
    intent = visual_context.get("intent", "world_building")
    pacing_phase = visual_context.get("pacing_phase", "opening")

    # 获取 genre 列表（优先使用 Phase 1 直接输出的 genre names）
    genre_list = visual_context.get("genres", [])
    if not genre_list:
        # 兜底：尝试从 genre_modifiers 推断
        genre_mods = visual_context.get("genre_modifiers", {})
        genre_list = list(genre_mods.keys()) if isinstance(genre_mods, dict) and genre_mods else []

    # 获取镜头模板
    templates = SHOT_TEMPLATES.get(intent, SHOT_TEMPLATES["world_building"])

    # 获取环境光照上下文（昼夜/天气），传递给 _fill_shot_details
    ambient_context = visual_context.get("ambient_context")

    # 填充每个 shot 的具体规格
    shots = []
    for template in templates:
        filled = _fill_shot_details(template, script, genre_list, ambient_context=ambient_context)
        shots.append(filled)

    # Phase 3: 添加 adjacency hints（镜头邻接提示）
    # 为 sequence_orchestrator 提供前置线索
    _add_adjacency_hints(shots)

    # 选择英雄镜头
    hero_index = _select_hero_shot(shots, pacing_phase)

    # 选择尾帧镜头
    end_index = _select_end_shot(shots, hero_index)

    # 光照去同质化：基于 emotion/pacing/hero 为每个镜头追加差异化光照修饰
    _diversify_lighting(shots, hero_index)

    # 从剧本中提取用户指定的摄影机规格，覆盖模板默认值
    camera_overrides = _extract_camera_overrides(script)
    if camera_overrides:
        print(f"    [*] 剧本摄影规格覆盖: {camera_overrides}")
        # Hero shot: 覆盖全部 camera 参数
        for key, value in camera_overrides.items():
            if key in ("type", "movement", "angle", "lens", "orientation"):
                shots[hero_index]["camera"][key] = value
        cam_type = shots[hero_index]["camera"]["type"]
        shots[hero_index]["lens_feel"] = _LENS_RULES.get(cam_type, shots[hero_index].get("lens_feel", "35mm standard"))
        # End shot 和其他 shots: 覆盖 angle 和 lens（保持各自模板的 type/movement 以维持叙事节奏）
        for i, shot in enumerate(shots):
            if i == hero_index:
                continue
            for key in ("angle", "lens"):
                if key in camera_overrides:
                    shot["camera"][key] = camera_overrides[key]

    # 智能默认：某些角度与运动组合不合理时自动修正
    hero_angle = shots[hero_index]["camera"]["angle"]
    hero_movement = shots[hero_index]["camera"]["movement"]
    if "movement" not in camera_overrides:  # 仅当用户没有显式指定运动时
        if hero_angle in ("bird's eye view", "overhead"):
            shots[hero_index]["camera"]["movement"] = "static locked"
            print(f"    [*] 智能默认: bird's eye view → static locked movement")
        elif hero_angle == "low angle" and "slow" in hero_movement:
            shots[hero_index]["camera"]["movement"] = "slow dolly in"
            print(f"    [*] 智能默认: low angle → slow dolly in movement")

    # 组装 shot plan
    plan = {
        "intent": intent,
        "genres": genre_list,
        "pacing_phase": pacing_phase,
        "shots": shots,
        "hero_shot_index": hero_index,
        "end_shot_index": end_index,
        "ambient_context": ambient_context,  # 传递环境上下文到格式化器
        "camera_overrides": camera_overrides,  # 剧本覆盖的摄影规格
    }

    # 格式化导演指令块
    plan["formatted_plan"] = _format_shot_plan_for_prompt(plan)

    return plan


# ============================================================
# 8. Phase 4: 重写策略引擎（Self-Refining Rewrite）
# ============================================================

# 重写策略的具体参数调整表
_REWRITE_CAMERA_SWAPS: Dict[str, Dict[str, str]] = {
    "establish": {"movement": "slow crane ascending", "angle": "low angle"},
    "panorama": {"movement": "slow crane ascending", "lens": "16mm ultra-wide"},
    "silhouette": {"movement": "slow dolly in", "angle": "low angle"},
    "movement": {"movement": "fast lateral tracking", "angle": "eye level"},
    "sweep": {"movement": "slow crane ascending", "angle": "low angle to high"},
    "reveal": {"movement": "slow dolly in", "lens": "85mm telephoto"},
    # dialogue intent
    "two_shot": {"movement": "static with subtle drift", "angle": "eye level"},
    "over_shoulder": {"movement": "slow push in", "lens": "85mm portrait"},
    "reaction": {"movement": "static", "angle": "slight low angle"},
    "exchange": {"movement": "slow lateral track", "angle": "eye level"},
    # montage intent
    "flash": {"movement": "whip pan", "angle": "dynamic angle"},
    "match_cut": {"movement": "dolly in", "lens": "50mm standard"},
    "crescendo": {"movement": "fast dolly out", "angle": "low angle"},
    # reveal intent
    "anticipation": {"movement": "slow dolly in", "angle": "eye level"},
    "breakthrough": {"movement": "push through", "angle": "low angle"},
    "hero_reveal": {"movement": "crane up", "angle": "low angle to eye level"},
}

_REWRITE_LIGHTING_ENHANCEMENTS: Dict[str, str] = {
    "high contrast": "high contrast with deep directional shadows and rim light accent",
    "medium": "medium contrast with layered fill light and atmospheric haze",
    "low": "low contrast with soft volumetric diffusion and color temperature gradient",
    "extreme": "extreme contrast with hard backlight silhouette and bloom spill",
}

_REWRITE_EMOTION_UPGRADES: Dict[str, Dict[str, Any]] = {
    "anticipation": {"pacing_weight": 0.20},
    "adrenaline": {"pacing_weight": 0.35},
    "intensity": {"pacing_weight": 0.40},
    "triumph": {"pacing_weight": 0.20},
    "awe": {"pacing_weight": 0.30},
    "mystery": {"pacing_weight": 0.15},
    # dialogue / montage / reveal emotions
    "connection": {"pacing_weight": 0.25},
    "intimacy": {"pacing_weight": 0.30},
    "tension": {"pacing_weight": 0.30},
    "progression": {"pacing_weight": 0.30},
    "wonder": {"pacing_weight": 0.35},
    "presence": {"pacing_weight": 0.20},
}

# 相邻镜头间推荐的 camera type 差异（避免重复）
_CAMERA_TYPE_ROTATION = [
    "wide shot", "medium shot", "close-up", "wide shot",
    "tracking shot", "extreme wide shot", "medium shot", "close-up",
    "medium two-shot", "over-the-shoulder", "close-up", "medium shot",
]


def rewrite_strategy(
    shot_plan: Dict[str, Any],
    suggestions: List[str],
    script: str,
) -> Dict[str, Any]:
    """
    Phase 4 核心函数：消费现有 shot_plan + critic 的优化建议，
    生成改进后的 shot_plan。

    重写策略：
    - complete_camera: 补全缺失的 camera 字段
    - diversify_camera: 确保相邻镜头使用不同 camera type
    - strengthen_hero: 重新选择 hero shot（偏好高权重镜头）
    - enhance_lighting: 丰富光照描述细节
    - diversify_emotion: 打破连续重复的情绪
    - reshape_pacing: 重新分配节奏权重
    - diversify_focus: 变化视觉焦点
    - reshape_arc: 通过调整情绪权重间接影响情绪弧
    - upgrade_transition: 标记 shot 需要更强的转场（由 sequence_orchestrator 消费）

    Returns:
        与 generate_shot_strategy 相同结构的 shot_plan dict
    """
    # Deep copy the existing plan
    refined = {
        "intent": shot_plan["intent"],
        "genres": list(shot_plan.get("genres", [])),
        "pacing_phase": shot_plan.get("pacing_phase", "opening"),
        "shots": [],
        "hero_shot_index": shot_plan.get("hero_shot_index", 0),
        "end_shot_index": shot_plan.get("end_shot_index", 0),
        "ambient_context": shot_plan.get("ambient_context"),  # 保留环境上下文
        "camera_overrides": shot_plan.get("camera_overrides", {}),  # 保留剧本摄影覆盖
    }

    # Deep copy shots
    shots = []
    for shot in shot_plan.get("shots", []):
        s = {k: (v.copy() if isinstance(v, dict) else v) for k, v in shot.items()}
        shots.append(s)

    strategy_set = set(suggestions)

    # ── Strategy: complete_camera ──
    if "complete_camera" in strategy_set:
        for shot in shots:
            cam = shot.setdefault("camera", {})
            cam.setdefault("type", "medium shot")
            cam.setdefault("movement", "static locked")
            cam.setdefault("angle", "eye level")
            cam.setdefault("lens", "50mm")

    # ── Strategy: diversify_camera ──
    if "diversify_camera" in strategy_set:
        for i in range(1, len(shots)):
            prev_type = shots[i - 1]["camera"]["type"]
            curr_type = shots[i]["camera"]["type"]
            if prev_type == curr_type:
                # 从 rotation 中选择一个不同的
                swap = _REWRITE_CAMERA_SWAPS.get(shots[i]["shot_name"])
                if swap:
                    shots[i]["camera"].update(swap)
                else:
                    # 通用兜底：切换到 medium shot
                    alt_types = ["medium shot", "close-up", "wide shot", "tracking shot"]
                    for alt in alt_types:
                        if alt != prev_type:
                            shots[i]["camera"]["type"] = alt
                            break

    # ── Strategy: enhance_lighting ──
    if "enhance_lighting" in strategy_set:
        for shot in shots:
            light = shot.get("lighting", {})
            contrast = light.get("contrast", "medium")
            enhancement = _REWRITE_LIGHTING_ENHANCEMENTS.get(contrast, "")
            if enhancement and len(light.get("setup", "")) < 30:
                light["setup"] = enhancement
                # 重新生成 full_description
                genres = refined["genres"]
                genre_parts = []
                for genre in genres:
                    mod = GENRE_LIGHTING_MODIFIERS.get(genre)
                    if mod:
                        genre_parts.append(f"{mod['overlay']}, {mod['atmosphere']}")
                if genre_parts:
                    light["full_description"] = f"{enhancement}; enhanced with {'; '.join(genre_parts)}"
                else:
                    light["full_description"] = enhancement

    # ── Strategy: diversify_emotion / reshape_arc ──
    if "diversify_emotion" in strategy_set or "reshape_arc" in strategy_set:
        # 打破连续重复的情绪：如果相邻 shot 情绪相同，微调后者
        for i in range(1, len(shots)):
            if shots[i]["emotion"] == shots[i - 1]["emotion"]:
                # 根据 shot_name 推断一个不同的情绪
                name_to_emotion = {
                    "establish": "anticipation",
                    "movement": "adrenaline",
                    "impact": "intensity",
                    "aftermath": "triumph",
                    "panorama": "awe",
                    "sweep": "grandeur",
                    "scale_reveal": "insignificance",
                    "atmosphere": "mystery",
                    "silhouette": "presence",
                    "reveal": "intimacy",
                    "detail": "vulnerability",
                    "stance": "power",
                    "close_up": "tenderness",
                    "stillness": "isolation",
                    "subtle_motion": "hope",
                    "fade": "loss",
                    # dialogue intent
                    "two_shot": "connection",
                    "over_shoulder": "intimacy",
                    "reaction": "tension",
                    "exchange": "presence",
                    # montage intent
                    "flash": "adrenaline",
                    "match_cut": "progression",
                    "crescendo": "intensity",
                    # reveal intent
                    "anticipation": "anticipation",
                    "breakthrough": "awe",
                    "hero_reveal": "wonder",
                }
                alt_emotion = name_to_emotion.get(shots[i]["shot_name"])
                if alt_emotion and alt_emotion != shots[i - 1]["emotion"]:
                    shots[i]["emotion"] = alt_emotion

    # ── Strategy: reshape_pacing ──
    if "reshape_pacing" in strategy_set:
        # 根据情绪重新分配 pacing_weight，确保有高低起伏
        for shot in shots:
            upgrade = _REWRITE_EMOTION_UPGRADES.get(shot.get("emotion", ""))
            if upgrade:
                shot["pacing_weight"] = upgrade["pacing_weight"]

        # 确保权重总和为 1.0
        total = sum(s.get("pacing_weight", 0.25) for s in shots)
        if total > 0 and abs(total - 1.0) > 0.01:
            for shot in shots:
                shot["pacing_weight"] = round(
                    shot.get("pacing_weight", 0.25) / total, 2
                )

    # ── Strategy: diversify_focus ──
    if "diversify_focus" in strategy_set:
        focus_rotation = ["environment", "character", "action", "environment"]
        for i, shot in enumerate(shots):
            if i < len(focus_rotation):
                cat = focus_rotation[i]
                shot["focus_category"] = cat
                # 保留原始 focus_object（由 _extract_focus 从剧本提取的真实关键词）
                # 不覆盖为类别名，而是通过 focus_hint 引导导演关注不同视觉维度
                if cat == "character":
                    shot["focus_hint"] = "emphasize character's face, expression, and body language"
                elif cat == "action":
                    shot["focus_hint"] = "emphasize dynamic motion, weapon interaction, and kinetic energy"
                # environment 类别不需要 hint，默认行为即可

    # ── Strategy: strengthen_hero ──
    hero_index = shot_plan.get("hero_shot_index", 0)
    if "strengthen_hero" in strategy_set:
        # 重新选择 hero：选 pacing_weight 最高的 shot
        hero_index = _select_hero_shot(shots, refined["pacing_phase"])

    # 选择 end shot
    end_index = _select_end_shot(shots, hero_index)

    # Phase 3: 重新添加 adjacency hints
    _add_adjacency_hints(shots)

    # 组装 refined plan
    refined["shots"] = shots
    refined["hero_shot_index"] = hero_index
    refined["end_shot_index"] = end_index
    refined["formatted_plan"] = _format_shot_plan_for_prompt(refined)

    return refined
