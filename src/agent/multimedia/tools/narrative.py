# src/agent/multimedia/tools/narrative.py
"""P2-5 叙事结构参数化（Narrative Parameterization）。

背景
----
此前 showrunner 的结构是**硬编码**的：prompt 里写死「包含 2 个连续镜头」，
content_type 注册表虽然定义了 `scene_count_range`，但**从未被任何代码消费**
（死字段）。结果是 15s 快闪与 3min 短剧集走完全相同的结构，且无法按目标时长伸缩。

本模块把「拆几个镜头、每个镜头承担什么叙事功能」从 prompt 里的常量，提升为
**由目标时长驱动、可自适应伸缩的模板**：

    target_duration(秒) ──plan_scene_count──▶ scene 数
    content_type ────────arc template───────▶ beat 骨架
                                               │
                              expand_beats(n) ─┘──▶ 恰好 n 个 beat（保形伸缩）

设计约束（与前序模块一致）：纯 stdlib、纯函数、零副作用、可单测。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ── 时长基准 ────────────────────────────────────────────────────────────
# 单镜基准时长（秒）。与 prompts.SHOWRUNNER_PROMPT 的 duration_seconds 约束
# （钳制 5~8s）一致：取上限 8s 作为「编片预算」的换算基准，实际成片更短也不浪费。
SHOT_SECONDS = 8.0

# 全局护栏：无论目标时长多极端，都不越过这两个界。
MIN_SCENES = 2
MAX_SCENES = 20


# ── 叙事弧模板（arc templates）─────────────────────────────────────────
# 每个模板 = 刚性锚点（anchors，尽量保序保留）+ 填充池（fillers，按需循环）。
# 模板 key 与 content_type 的 arc_expectation 对齐（见 common._CONTENT_TYPE_PRESETS）。
#
# 锚点刻意保持精简（3~4 个），使常见 scene 数下走「扩展」分支、模板形状得以保留。
BEAT_ENERGY: Dict[str, float] = {
    "setup": 0.40,
    "inciting": 0.55,
    "develop": 0.60,
    "turn": 0.80,
    "climax": 0.95,
    "fallout": 0.70,
    "resolution": 0.50,
}

#: 四个基础模板（roadmap 原文的「4 模板」）+ 两个派生模板。
#:   crescendo  预告片：高潮递进         arc        短片：起承转合
#:   flat_wave  纪录片：平稳起伏         rhythmic   MV：节奏驱动
#:   single_peak 广告：单峰收束          ascending_steps 教程：步骤递进
NARRATIVE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "crescendo": {
        "label": "高潮递进",
        "anchors": ["setup", "develop", "turn", "climax", "resolution"],
        "fillers": ["inciting", "develop", "fallout"],
    },
    "arc": {
        "label": "起承转合",
        "anchors": ["setup", "develop", "turn", "climax", "resolution"],
        "fillers": ["inciting", "fallout", "develop"],
    },
    "flat_wave": {
        "label": "平稳起伏",
        "anchors": ["setup", "develop", "develop", "resolution"],
        "fillers": ["inciting", "develop", "fallout"],
    },
    "rhythmic": {
        "label": "节奏驱动",
        "anchors": ["setup", "develop", "turn", "climax", "resolution"],
        "fillers": ["develop", "inciting", "develop"],
    },
    "single_peak": {
        "label": "单峰收束",
        "anchors": ["setup", "develop", "climax", "resolution"],
        "fillers": ["develop", "inciting"],
    },
    "ascending_steps": {
        "label": "步骤递进",
        "anchors": ["setup", "develop", "climax", "resolution"],
        "fillers": ["develop", "inciting", "develop"],
    },
}

#: content_type → 模板 key（无法识别时用 arc）。
CONTENT_TYPE_TEMPLATE: Dict[str, str] = {
    "trailer": "crescendo",
    "short_film": "arc",
    "documentary": "flat_wave",
    "music_video": "rhythmic",
    "commercial": "single_peak",
    "tutorial": "ascending_steps",
}

#: content_type → scene 数范围（与 common._CONTENT_TYPE_PRESETS 保持一致，
#: 此处复制一份常量以避免 tools → nodes 的反向依赖）。
CONTENT_TYPE_SCENE_RANGE: Dict[str, Tuple[int, int]] = {
    "trailer": (3, 8),
    "short_film": (3, 12),
    "documentary": (4, 15),
    "commercial": (3, 6),
    "music_video": (5, 20),
    "tutorial": (3, 10),
}

DEFAULT_TEMPLATE = "arc"


def template_for_content_type(content_type: Optional[str]) -> str:
    """把 content_type 映射到模板 key（未知类型回落 DEFAULT_TEMPLATE）。"""
    return CONTENT_TYPE_TEMPLATE.get((content_type or "").strip().lower(), DEFAULT_TEMPLATE)


def get_template(template_key: Optional[str]) -> Dict[str, Any]:
    return NARRATIVE_TEMPLATES.get(template_key or "", NARRATIVE_TEMPLATES[DEFAULT_TEMPLATE])


def plan_scene_count(
    target_duration: Optional[float],
    content_type: Optional[str] = None,
    shot_seconds: float = SHOT_SECONDS,
) -> int:
    """按目标时长反推镜头数。

    规则：
      - target_duration 缺失/非正 → 取该 content_type 的 scene 数范围下限（保守）。
      - 否则 scenes = ceil(target_duration / shot_seconds)，再钳制进
        content_type 的 scene_count_range，最后钳制进 [MIN_SCENES, MAX_SCENES]。
    """
    ct = (content_type or "").strip().lower()
    lo, hi = CONTENT_TYPE_SCENE_RANGE.get(ct, (MIN_SCENES, MAX_SCENES))
    try:
        dur = float(target_duration) if target_duration is not None else 0.0
    except (TypeError, ValueError):
        dur = 0.0
    if dur <= 0:
        return max(MIN_SCENES, min(hi, lo))
    shot = shot_seconds if shot_seconds and shot_seconds > 0 else SHOT_SECONDS
    import math
    n = int(math.ceil(dur / shot))
    n = max(lo, min(hi, n))
    n = max(MIN_SCENES, min(MAX_SCENES, n))
    return n


def expand_beats(template_key: Optional[str], n: int) -> List[str]:
    """把模板的 beat 骨架自适应伸缩为**恰好 n 个** beat。

    - n >= 锚点数：锚点按比例均匀落位，缝隙用 fillers 循环填充（保形）；
    - n < 锚点数：保留首/末，并优先保留 climax，其余按序补齐。

    不变量：返回长度 == n（n<=0 时为空）；首=setup、末=resolution（n>=2）。
    """
    tpl = get_template(template_key)
    anchors: List[str] = list(tpl["anchors"])
    fillers: List[str] = list(tpl["fillers"]) or ["develop"]
    n = int(n)
    if n <= 0:
        return []
    if n == 1:
        return [anchors[0]]

    m = len(anchors)
    if n >= m:
        out: List[Optional[str]] = [None] * n
        # 锚点均匀落位（首 0、末 n-1）
        pos = [round(i * (n - 1) / (m - 1)) for i in range(m)]
        for p, a in zip(pos, anchors):
            out[p] = a
        fi = 0
        for i in range(n):
            if out[i] is None:
                out[i] = fillers[fi % len(fillers)]
                fi += 1
        return [b for b in out if b]

    # n < m：压缩。保首、保 climax（若有）、保末。
    clash = "climax" if "climax" in anchors else None
    mid = [a for a in anchors[1:-1] if a != clash]
    need_mid = n - 2 - (1 if clash else 0)
    seq = [anchors[0]] + mid[: max(0, need_mid)]
    if clash and len(seq) < n - 0:
        seq.append(clash)
    seq.append(anchors[-1])
    if len(seq) > n:
        seq = [seq[0]] + seq[-(n - 1):]
    return seq[:n]


def scene_count_hint(
    target_duration: Optional[float],
    content_type: Optional[str] = None,
    shot_seconds: float = SHOT_SECONDS,
    max_shots: Optional[int] = None,
) -> Tuple[int, str]:
    """生成注入 showrunner prompt 的镜头数硬性约束文本。

    返回 (scene_count, hint_text)。hint_text 为空串表示无需附加约束。
    max_shots（smoke/显式上限）优先，用于免费额度下的截断保护。
    """
    if max_shots:
        n = int(max_shots)
        hint = (
            f"\n\n【硬性镜头数约束】本片严格限制生成恰好 {n} 个镜头，"
            "不得扩展为更多。这是用户明确指定的上限，请严格遵循，"
            f"把完整故事压缩进这 {n} 个镜头内（setup→climax 即可）。"
        )
        return n, hint

    tpl_key = template_for_content_type(content_type)
    tpl = get_template(tpl_key)
    n = plan_scene_count(target_duration, content_type, shot_seconds)
    if target_duration is None:
        # 未指定目标时长：不硬性约束数量，只给出结构与节奏倾向
        hint = (
            f"\n\n【叙事结构偏好】请按「{tpl['label']}」的节奏组织镜头"
            f"（功能序列示例：{' → '.join(tpl['anchors'])}），"
            f"镜头数在 {CONTENT_TYPE_SCENE_RANGE.get((content_type or '').lower(), (MIN_SCENES, MAX_SCENES))[0]}"
            f"~{CONTENT_TYPE_SCENE_RANGE.get((content_type or '').lower(), (MIN_SCENES, MAX_SCENES))[1]} 之间，"
            "以完整讲清故事为准。"
        )
        return n, hint

    est = round(n * shot_seconds, 1)
    hint = (
        f"\n\n【目标时长驱动的镜头数约束】本片目标总时长约 {target_duration:.0f} 秒，"
        f"按每镜约 {shot_seconds:.0f} 秒核算，应拆解为**恰好 {n} 个**连续镜头"
        f"（估算成片约 {est:.0f} 秒）。请严格按此数量拆解，"
        f"按「{tpl['label']}」的节奏组织，保证叙事完整。"
    )
    return n, hint


def resolve_target_duration(state: Dict[str, Any]) -> Optional[float]:
    """从 state 读取目标时长（秒）。支持 target_duration / 时长关键词兜底。"""
    for key in ("target_duration", "target_duration_seconds"):
        v = state.get(key)
        if v is None:
            continue
        try:
            f = float(v)
            if f > 0:
                return f
        except (TypeError, ValueError):
            continue
    return None


def resolve_episode(state: Dict[str, Any]) -> int:
    """从 state 读取集数（>=1）。缺省为 1。"""
    v = state.get("episode")
    try:
        i = int(v)
        return i if i >= 1 else 1
    except (TypeError, ValueError):
        return 1


def format_episode_anchors_hint(anchors: Optional[Dict[str, Any]]) -> str:
    """把上一集的 end_state 锚点格式化为注入本集 showrunner 的承接提示。

    anchors 来自上一集 advance_scene 产出的 scene_anchors 快照，关键字段：
    intent / ambient_context / color_direction / camera_momentum /
    emotion_energy / character_identity / last_frame_url。
    """
    if not anchors:
        return ""
    parts = []
    if anchors.get("intent"):
        parts.append(f"上集收束语境：{anchors['intent']}")
    if anchors.get("ambient_context"):
        parts.append(f"环境：{anchors['ambient_context']}")
    if anchors.get("color_direction"):
        parts.append(f"色调走向：{anchors['color_direction']}")
    if anchors.get("camera_momentum"):
        parts.append(f"运镜惯性：{anchors['camera_momentum']}")
    try:
        energy = float(anchors.get("emotion_energy")) if anchors.get("emotion_energy") is not None else None
    except (TypeError, ValueError):
        energy = None
    if energy is not None:
        parts.append(f"结尾能量：{energy:.2f}")
    cid = anchors.get("character_identity") or {}
    if cid.get("has_character"):
        subject = cid.get("subject") or "主角"
        appear = "、".join(cid.get("appearance") or [])
        parts.append(f"主角外观锚点：{subject}{('（' + appear + '）') if appear else ''}")
    if anchors.get("last_frame_url"):
        parts.append("上一集尾帧已作为本集首镜视觉参考")
    if not parts:
        return ""
    return (
        "\n\n【承接上一集】本集开场必须自然衔接上一集的结尾状态："
        + "；".join(parts)
        + "。禁止与本集设定矛盾。"
    )


def export_episode_assets(
    reference_sheets: Optional[Dict[str, Any]],
    scene_anchors: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """把本集产出整理为「下一集可复用」的资产 + 锚点快照。

    返回 {"series_assets": ..., "episode_anchors": ...}，可直接作为下一集
    初始 state 的对应字段注入（series_assets 复用参考图省额度，
    episode_anchors 保证跨集视觉/叙事连续性）。
    """
    sheets = reference_sheets or {}
    assets = {
        "characters": sheets.get("characters") or {},
        "props": sheets.get("props") or {},
        "environments": sheets.get("environments") or {},
    }
    # 无任何资产的空快照不返回，避免下游误判「已注入素材」
    if not (assets["characters"] or assets["props"] or assets["environments"]):
        assets = None
    return {"series_assets": assets, "episode_anchors": scene_anchors or None}


def build_series_inputs(
    base_task: str,
    episode_tasks: Optional[List[str]] = None,
    num_episodes: int = 1,
    target_duration: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """构造连续剧每集的初始 state 输入（供外部编排器逐集 invoke）。

    - 第 1 集：base_task；
    - 第 N 集（>1）：episode_tasks[N-2]（若提供）否则沿用 base_task，
      并由编排器在上一集结束后用 export_episode_assets 回填
      series_assets / episode_anchors。
    """
    n = max(1, int(num_episodes))
    tasks = list(episode_tasks or [])
    out: List[Dict[str, Any]] = []
    for i in range(n):
        ep = i + 1
        task = base_task if ep == 1 else (tasks[i - 1] if i - 1 < len(tasks) else base_task)
        entry: Dict[str, Any] = {"task": task, "episode": ep}
        if target_duration:
            entry["target_duration"] = target_duration
        out.append(entry)
    return out


def episode_beats_hint(episode: int) -> str:
    """跨集叙事提示：非首集需承接上集、并保留本集收束感。"""
    if episode <= 1:
        return ""
    return (
        f"\n\n【连续剧第 {episode} 集】本集是连续剧的第 {episode} 集，"
        "开场需自然承接上一集的结尾状态（人物处境/情绪/悬念），"
        "结尾需留下延续到下一集的钩子，同时保证本集自身叙事完整。"
    )
