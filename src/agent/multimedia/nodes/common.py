# src/agent/multimedia/nodes/common.py
"""跨 stage 共享的常量、纯工具，以及「风格 / prompt 工具箱」。

P2-3：由原 graph.py（单体）按 stage 机械拆分而来，逻辑逐字节保留；
仅新增本文件头注释与模块导入，函数体 / 常量 / 注释均未改动。
"""
import json
import os
import os as _os
import re
from typing import Any, Dict, List, Optional
from langgraph.types import interrupt
from .. import run_metrics
from .. import review_flywheel as _flywheel
from ..config_loader import character_consistency_threshold, max_video_gen_failures
from ..state import MultimediaState
from ..tools.text_llm import call_llm


APPROVE_ACTIONS = {"approve", "pass", "ok", "accept", "yes", "通过", "批准"}
REWRITE_ACTIONS = {"rewrite", "edit", "edit_prompt", "revise", "modify", "重写", "修改"}

# ── P1-3：镜头级外科手术的「修正类型」（fix_type）──
# 审核者的 rewrite 决策可携带修正类型，决定重试的**粒度**而非一律全量重 roll：
#   prompt_only      ：保关键帧，仅改运镜词后重滚视频（最便宜；跳过 videographer LLM）
#   keep_first_frame ：保首帧，重滚视频（重跑 videographer 重新设计运镜）
#   reimage          ：重生成关键帧（回 director，走完整图链路）
#   extend           ：保首帧，尾帧续接「延长时长」（bump duration 后重滚）
FIX_PROMPT_ONLY = "prompt_only"
FIX_KEEP_FIRST_FRAME = "keep_first_frame"
FIX_REIMAGE = "reimage"
FIX_EXTEND = "extend"
FIX_TYPES = (FIX_PROMPT_ONLY, FIX_KEEP_FIRST_FRAME, FIX_REIMAGE, FIX_EXTEND)

_FIX_ALIASES = {
    "prompt_only": FIX_PROMPT_ONLY, "prompt": FIX_PROMPT_ONLY, "edit_prompt": FIX_PROMPT_ONLY,
    "改词": FIX_PROMPT_ONLY, "仅改词": FIX_PROMPT_ONLY, "只改词": FIX_PROMPT_ONLY,
    "keep_first_frame": FIX_KEEP_FIRST_FRAME, "keep_first": FIX_KEEP_FIRST_FRAME,
    "first_frame": FIX_KEEP_FIRST_FRAME, "保首帧": FIX_KEEP_FIRST_FRAME, "重滚视频": FIX_KEEP_FIRST_FRAME,
    "reimage": FIX_REIMAGE, "re_image": FIX_REIMAGE, "regenerate_image": FIX_REIMAGE,
    "重绘": FIX_REIMAGE, "重生成关键帧": FIX_REIMAGE, "重新生图": FIX_REIMAGE,
    "extend": FIX_EXTEND, "extend_duration": FIX_EXTEND, "prolong": FIX_EXTEND,
    "延长": FIX_EXTEND, "延长时长": FIX_EXTEND, "续接": FIX_EXTEND,
}


def _normalize_fix_type(raw: Any) -> Optional[str]:
    """把任意输入归一为四种规范 fix_type 之一；无法识别返回 None（=沿用默认行为）。"""
    if raw is None:
        return None
    key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if key in FIX_TYPES:
        return key
    return _FIX_ALIASES.get(key) or _FIX_ALIASES.get(str(raw).strip())


# 单镜头视频生成失败（网络/超时/接口错误）的最大重试次数。
# 达到上限后生成占位黑场降级，继续生成其余镜头，避免单点失败拖垮整个任务。
# 改从 thresholds.json 读取（保持默认 2，MAX_VIDEO_GEN_FAILURES env 可覆盖）。

MAX_VIDEO_GEN_FAILURES = max_video_gen_failures()

# 角色一致性闭环阈值：关键帧与"源头角色肖像"的 embedding 余弦相似度下限。
# 低于该值视为角色漂移，即便 VLM 判 PASS 也触发一次重生（受 iterations 上限保护）。
#
# 【实测校准】生图后端为 jimeng（免费档）时，其 img2img 参考约束力上限有限，实测：
#   strength=0.5 → 相似度 0.2469（甚至低于纯文生图 0.2582，参考图等于无效）
#   strength=0.9 → 相似度 0.2995（该后端实测可达上限）
# 若沿用 0.5，则所有镜头都会判"漂移"并触发最多 3 次重生，白白烧掉 3 倍生图额度
# 且无法提升分数。故阈值下调至 0.28（略低于实测上限 0.2995）：仍能拦住真正失败的
# 产物（如与角色完全无关的画面），又不会把正常产物全部误判。
# 可用环境变量 CHARACTER_CONSISTENCY_THRESHOLD 覆盖；设为 0 关闭该闭环。
CHARACTER_CONSISTENCY_THRESHOLD = character_consistency_threshold()

# P1-4：VLM 成对一致性判断（embedding 失效时的兜底数据源）。
# 开关 env MULTIMEDIA_CONSISTENCY_VLM（默认 "1" 开启）；阈值 env
# MULTIMEDIA_VLM_CONSISTENCY_THRESHOLD（默认 0.6，语义为「同一角色」置信度下限）。
VLM_CONSISTENCY_THRESHOLD = float(os.getenv("MULTIMEDIA_VLM_CONSISTENCY_THRESHOLD", "0.6") or "0.6")
_CONSISTENCY_VLM_ENABLED = os.getenv("MULTIMEDIA_CONSISTENCY_VLM", "1") != "0"


# ── 视频生成失败降级：占位黑场 ──
# 达 MAX_VIDEO_GEN_FAILURES 上限时，生成本地占位黑场 .mp4（带静音底），
# 写入该镜头应有的 final_video_url 槽位，使 stitcher/audio_mixer 仍能拼出
# 时间轴完整的成片，避免"静默丢镜头"导致下游错位。占位文件为本地绝对路径。
# 实现放在 config_loader（与 graph 解耦，无重依赖），此处直接复用。


def safe_parse_json(response: str, partial: bool = False) -> dict:
    """解析 LLM 返回的 JSON。

    partial=True 时（用于 showrunner 等带自愈重试的节点）：解析失败不抛异常，
    而是尽量返回已完整的前缀部分（或空 dict），交由上游重试循环兜底；
    避免 LLM 输出被截断（超长/超时）时直接 FATAL 拖垮整个流程。
    """
    if partial:
        try:
            # 优先用 raw_decode 尽力解析截断 JSON 的前缀有效部分（如已完整的 scenes 前缀）
            return json.JSONDecoder().raw_decode(response.strip())[0]
        except Exception:
            pass
    cleaned = response.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    if partial:
        # 截断兜底：返回空 dict，让 showrunner 的 `if not raw_scenes: continue` 触发重试
        return {}
    raise Exception(f"总导演未按要求输出有效 JSON 对象！原始输出: {response[:500]}...")


# 审核结果判定：精确匹配 "PASS" 开头（允许后面跟空格、标点或结尾），
# 排除 "Passes"/"Passed"/"Passing" 等变体
_PASS_RE = re.compile(r"^\s*PASS(?:\s|[:.,;!\-]|$)", re.IGNORECASE)


def _is_review_pass(feedback: str) -> bool:
    """判断审核反馈是否为通过。比 .startswith('PASS') 更鲁棒。"""
    if not feedback:
        return False
    return bool(_PASS_RE.match(feedback.strip()))


def _normalize_action(raw_action: Any) -> str:
    if raw_action is None:
        return "approve"
    if isinstance(raw_action, bool):
        return "approve" if raw_action else "rewrite"
    action = str(raw_action).strip().lower()
    if action in {"通过", "批准"}:
        return "approve"
    if action in {"重写", "修改"}:
        return "rewrite"
    return action


def _normalize_decision(decision: Any, gate: str = "") -> Dict[str, Any]:
    if isinstance(decision, dict):
        out = dict(decision)
    elif isinstance(decision, str):
        out = {"action": decision}
    elif isinstance(decision, bool):
        out = {"action": "approve" if decision else "rewrite"}
    else:
        out = {"action": "approve"}
    out["action"] = _normalize_action(out.get("action", "approve"))
    # P1-3：透传并归一化「修正类型」（外科手术粒度）；未提供则保持 None。
    if out.get("fix_type") is not None or out.get("fix") is not None:
        out["fix_type"] = _normalize_fix_type(out.get("fix_type", out.get("fix")))
    # 运行指标埋点：记录人工审核卡被消费的 action（可观测性）
    if gate:
        run_metrics.record_human_gate(gate, out["action"])
    return out


def _decide(gate: str, payload: Dict[str, Any], state: MultimediaState) -> Dict[str, Any]:
    """统一的「人工审核闸口」决策入口。

    在 auto_mode（低干预模式）下跳过 interrupt()，直接以 approve 放行并标记
    auto_approved=True，供前端 NodeProgressTimeline 展示「自动」；否则走原有人审
    闭环 _normalize_decision(interrupt(payload), gate=...)。

    所有 6 处 interrupt 调用统一收口到此，便于后续扩展（如按 gate 配置跳过策略）。

    P1-6：决策消费后自动落入「审核飞轮」偏好对（rejected/chosen）。埋点零侵入、
    绝不阻断主流程；auto_mode 自动放行不构对但计入统计。详见 review_flywheel.py。
    """
    if state.get("auto_mode") or _os.getenv("AUTO_MODE") == "1":
        # 自动放行：记录指标但跳过人工卡
        run_metrics.record_human_gate(gate, "approve")
        _flywheel_record(gate, {"action": "approve", "auto_approved": True}, payload, state)
        return {"action": "approve", "auto_approved": True}
    decision = _normalize_decision(interrupt(payload), gate=gate)
    _flywheel_record(gate, decision, payload, state)
    return decision


def _flywheel_record(gate: str, decision: Dict[str, Any], payload: Dict[str, Any],
                     state: MultimediaState) -> None:
    """P1-6 审核飞轮埋点：把本次决策 + 被审产物落成偏好对（失败静默）。

    影子模式（MULTIMEDIA_AUTO_REVIEW 非 off）下同时记录自动审核员的建议，
    供离线评估「机器 vs 人」一致率，不改变任何主流程行为。
    """
    try:
        artifact = _flywheel.artifact_from_payload(gate, payload)
        _flywheel.record_decision(
            gate,
            decision,
            thread_id=str(state.get("thread_id") or ""),
            scene_index=state.get("current_scene_index"),
            rejected=artifact,
            context={
                "script": (payload.get("script") or "")[:400],
                "style_key": state.get("visual_style") or "",
                "content_type": state.get("content_type") or "",
                "quality_tier": state.get("quality_tier") or "",
                "global_setting": (state.get("global_setting") or "")[:300],
            },
        )
        if _flywheel.auto_review_enabled(gate):
            _flywheel.record_suggestion(gate, artifact)
    except Exception as e:  # 埋点绝不阻断主流程
        print(f"    [P1-6] 审核飞轮埋点失败（已忽略）: {e}")


def _normalize_scenes(scenes_value: Any) -> List[Dict[str, Any]]:
    if scenes_value is None:
        raise ValueError("scenes 不能为空")
    if isinstance(scenes_value, str):
        scenes_value = json.loads(scenes_value)
    if not isinstance(scenes_value, list):
        raise ValueError("scenes 必须是 list 或 JSON 数组字符串")
    normalized: List[Dict[str, Any]] =[]
    for item in scenes_value:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append({"script": item})
        else:
            normalized.append({"script": str(item)})
    return normalized


# 仅"纯质量/平台营销"词需要从 script 中剥离——这些词不表达任何视觉风格，
# 只会污染下游生图提示词且对画面无意义。
# 注意：合法风格词（watercolor / illustration / anime / cinematic / oil painting 等）
# 一律不在此列——它们可能是用户故事的真实风格意图（如"水彩童话"），误删会与风格系统自相矛盾。
_STYLE_TAIL_TOKENS = [
    "high-quality", "high quality", "best quality", "4k", "8k", "hd",
    "masterpiece", "trending on artstation", "artstation", "ultra-detailed",
    "octane render", "unreal engine", "ray tracing", "cgi", "3d render",
]


def _sanitize_script(script: str) -> str:
    """清洗 script：剥离 LLM 可能混入的英文风格/质量后缀，保证 script 纯净且仅含中文叙事。

    兜底措施，配合 SHOWRUNNER_PROMPT 的「script 纯净度硬性规则」双保险。
    """
    if not script:
        return script
    text = script.strip()
    # 1) 截掉从第一个英文字母片段开始的尾部（多为风格后缀段）
    #    若正文本身含英文（如镜头标注"全景镜头："是中文，但"Shot 2"这类），仅剥离末尾连续英文段。
    # 策略：从右向左，找到最后一个中文字符位置，截断其后的所有内容。
    last_cjk = -1
    for idx, ch in enumerate(text):
        if "\u4e00" <= ch <= "\u9fff" or ch in "，。：、！？…—（）《》":
            last_cjk = idx
    if last_cjk >= 0 and last_cjk < len(text) - 1:
        text = text[: last_cjk + 1]
    # 2) 显式剔除残留的风格 token（小写匹配，处理被中文句号分隔但仍混入的情况）
    low = text.lower()
    for tok in _STYLE_TAIL_TOKENS:
        if tok in low:
            # 删除该 token 及其可能的标点/空格残留
            text = re.sub(rf"\s*{re.escape(tok)}\b\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()
    # 3) 若截断后无结尾标点，补中文句号
    if text and not text[-1] in "。！？…":
        text += "。"
    return text


def _scene_base(script: str) -> Dict[str, Any]:
    return {
        "script": script,
        "dialogue": "",               # 适合配音的中文角色台词/旁白（与 script 视觉描述解耦）
        # ── 情节密度字段（Phase 0 总导演结构化输出，用于强化首尾帧可见内容）──
        "scale": "",                 # 景别
        "camera_note": "",           # 镜头运动与角度
        "action_beat": "",           # 镜头内主体连续可见动作（起始→结束）
        "visual_elements": [],       # 可辨识环境/道具/光影细节列表
        "emotion": "",               # 该镜头情绪
        "beat": "",                  # 叙事功能（setup/inciting/develop/turn/climax/fallout/resolution）
        "transition_in": "",         # 承接上一镜头的可见动作，保证因果连贯
        "iterations": 0,
        "critique": "无",
        "is_perfect": False,
        "image_prompt": "",
        "image_url": "",
        "embedding_similarity": None,   # 与上一镜头关键帧的相似度
        "portrait_similarity": None,    # 与源头角色肖像的相似度（角色一致性度量）
        "continuity_score": None,       # 连贯性可观测分数汇总
        "image_auto_feedback": "",
        
        # 尾帧相关字段
        "last_image_prompt": "",
        "last_image_url": "",
        "last_image_iterations": 0,
        "last_image_critique": "无",
        "last_image_is_perfect": False,
        "last_image_auto_feedback": "",

        "video_iterations": 0,      # 质量重试计数（审片不通过触发重新运镜+生成）
        "video_gen_failures": 0,    # 生成失败计数（网络/超时/接口错误），与质量重试分离
        "video_critique": "无",
        "video_is_perfect": False,
        "video_prompt": "",
        "raw_video_url": "",
        "final_video_url": "",
        "video_auto_feedback": "",
        # ── 分镜时长（秒）── 由总导演按台词长度估算（中文≈3.5字/秒+停顿buffer，钳制5~8s）
        # 用于驱动视频模型时长与字幕时间轴对齐，避免统一短时长导致台词念不完。
        "duration_seconds": 0.0,
    }


def _estimate_scene_duration(dialogue: str, explicit: Any = None) -> float:
    """按台词字符数估算分镜时长（秒）。

    规则（与 prompts.SHOWRUNNER_PROMPT 的 duration_seconds 约束一致）：
      - 中文约 3.5 字/秒；额外加 1.5s 停顿/气口 buffer；
      - 钳制在 [5, 8] 区间（短台词取 5，长台词取 8）；
      - 若 LLM 已显式给出合理时长（5~8s）则优先采用，否则按字符数估算；
      - 无对白镜头给 5s 下限。
    """
    try:
        if explicit is not None:
            ev = float(explicit)
            # LLM 给出时长：在 [5,8] 内直接采用；超出则钳制到边界，避免异常值
            if 0 < ev <= 8.0:
                return float(max(5.0, ev))
    except (TypeError, ValueError):
        pass
    text = (dialogue or "").strip()
    # 统计中文字符 + 其他可见字符（粗略按"字"计）
    n_chars = len([c for c in text if not c.isspace()])
    if n_chars <= 0:
        return 5.0
    seconds = n_chars / 3.5 + 1.5
    return float(max(5.0, min(8.0, round(seconds, 1))))


# ---- 安全网：截掉 director 可能附加的多镜头列表 ----

_SEQUENCE_PATTERNS = [
    r"(?i)\bsequence\s+includes\s*[:\-].*",
    r"(?i)\bshot\s+list\s*[:\-].*",
    r"(?i)\bsequence\s+overview\s*[:\-].*",
    r"(?i)\b(?:first[,\s].*then[,\s].*finally).*",
    r"(?i)\bvarying\s+focal\s+objects?\s*[:\-].*",
    r"(?i)\b(?:1\.\s|2\.\s|3\.\s).*shot.*(?:overhead|drone|close-up|low.angle|wide).*",
    r"(?i)\bfocus\s+shifts?\s+(?:to|from|between).*",  # NEW: catch "Focus shifts to X, then Y"
    r"(?i)\bcamera\s+then\s+.*(?:focus|detail|element).*",  # NEW: catch "camera then focuses"
]

# ============================================================
# 风格注册表（Style Registry）
# ============================================================
# 核心理念：genre（题材）和 style（视觉风格）是两个独立维度。
#   - genre: cyberpunk / fantasy / gothic / sci_fi ... — 决定内容和氛围
#   - style: aaa_cg / anime / watercolor / film_grain ... — 决定画面渲染风格
# 两者可自由组合：赛博朋克 + 动漫风 = 攻壳机动队感；奇幻 + 水彩 = 绘本感。
#
# 每个 preset 定义：
#   suffix          — 追加到图像 prompt 末尾的风格后缀
#   strip_keywords  — 当此风格被选中时，需要从 LLM 输出中清理的 AAA 专属术语
#   detect_keywords — 从用户 task 中检测此风格的中英文关键词
# ============================================================

# 通用冲突词：所有"非写实渲染"风格生图时都应剥离的写实CG术语。
# 抽出为单一来源，避免在每个 preset 的 strip_keywords 里重复维护一大串（解耦）。
_GENERIC_CONFLICT_KEYWORDS = [
    "unreal engine", "ue5", "ray tracing", "raytracing", "octane render",
    "aaa game", "aaa cg", "game cg", "3d cg", "cgi trailer", "photorealistic",
    "hyperrealistic", "nvidia rtx", "volumetric lighting", "subsurface scattering",
]


def _get_conflict_keywords(style_key: str) -> List[str]:
    """返回某风格生图时应剥离的冲突词。

    动态合并（解耦）：通用写实词 ∪ 该风格特有 strip_keywords，
    但**排除本风格自身 preset 里用到的词**（避免写实风格 aaa_cg 把自己
    的 "Unreal Engine" 也剥掉 —— 自相矛盾）。

    新增风格自动继承通用冲突词，无需在各 preset 重复写死。
    """
    preset = _STYLE_PRESETS.get(style_key, {})
    # 本风格自身用到的词（suffix + detect_keywords），不应作为冲突被剥除
    own_terms = set()
    for blob in (preset.get("suffix", ""), " ".join(preset.get("detect_keywords", []) or [])):
        for kw in _GENERIC_CONFLICT_KEYWORDS:
            if kw.lower() in blob.lower():
                own_terms.add(kw.lower())
    extra = preset.get("strip_keywords", []) or []
    merged = _GENERIC_CONFLICT_KEYWORDS + extra
    # 去重、保序，并剔除「本风格自己用到的词」
    seen = set()
    result = []
    for kw in merged:
        k = kw.lower()
        if k in seen or k in own_terms:
            continue
        seen.add(k)
        result.append(kw)
    return result


_STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "aaa_cg": {
        "suffix": "AAA game CG trailer, Unreal Engine 5 render, cinematic lighting, ray tracing, ultra-detailed, 8k resolution, masterpiece",
        "strip_keywords": [],
        "detect_keywords": [
            "CG", "cg", "游戏", "3A", "AAA", "虚幻引擎", "虚幻",
            "Unreal", "unreal engine", "光线追踪", "ray tracing",
            "游戏宣传片", "game trailer", "史诗", "epic",
        ],
    },
    "anime": {
        "suffix": "high-quality anime illustration, Studio Ghibli meets Makoto Shinkai, vibrant cel-shading, dynamic linework, painterly backgrounds, masterpiece",
        "strip_keywords": [
            "Unreal Engine", "unreal engine", "UE5", "ray tracing",
            "AAA game CG", "photorealistic", "8k resolution",
        ],
        "detect_keywords": [
            "动漫", "动画", "二次元", "anime", "manga", "赛璐璐",
            "吉卜力", "Ghibli", "新海诚", "Shinkai", "日系",
            "cel-shaded", "手绘风",
        ],
    },
    "watercolor": {
        "suffix": "delicate watercolor painting, soft pigment bleeding, visible brushstrokes, paper texture, ethereal and dreamlike atmosphere, fine art illustration",
        "strip_keywords": [
            "Unreal Engine", "unreal engine", "UE5", "ray tracing",
            "AAA game CG", "photorealistic", "8k resolution",
            "ultra-detailed", "cinematic lighting",
        ],
        "detect_keywords": [
            "水彩", "watercolor", "水墨", "ink wash", "手绘",
            "插画", "illustration", "绘本", "picture book",
        ],
    },
    "film_grain": {
        "suffix": "shot on 35mm film, natural grain texture, Kodak Portra 400 tones, soft lens vignetting, golden hour warmth, cinematic photography",
        "strip_keywords": [
            "Unreal Engine", "unreal engine", "UE5", "ray tracing",
            "AAA game CG", "8k resolution", "ultra-detailed",
        ],
        "detect_keywords": [
            "胶片", "film", "胶卷", "35mm", "柯达", "Kodak",
            "胶片感", "film grain", "复古摄影", "vintage photo",
            "portra", "颗粒感",
        ],
    },
    "documentary": {
        "suffix": "documentary photography, natural lighting, handheld camera feel, authentic and unposed, photojournalistic composition, raw and honest",
        "strip_keywords": [
            "Unreal Engine", "unreal engine", "UE5", "ray tracing",
            "AAA game CG", "8k resolution", "ultra-detailed",
            "masterpiece", "cinematic lighting",
        ],
        "detect_keywords": [
            "纪录片", "documentary", "纪实", "新闻", "journalism",
            "真实", "real", "采访", "interview",
        ],
    },
    "horror_cinematic": {
        "suffix": "atmospheric horror cinematography, deep shadows, unsettling color grading, desaturated tones, film noir influence, dread-inducing composition",
        "strip_keywords": [
            "Unreal Engine", "unreal engine", "UE5",
            "AAA game CG", "8k resolution", "masterpiece",
        ],
        "detect_keywords": [
            "恐怖", "horror", "惊悚", "thriller", "诡异",
            "阴森", "creepy", "dread", "haunted", "nightmare",
        ],
    },
    "pixel_art": {
        "suffix": "detailed pixel art, 16-bit era aesthetic, limited color palette, crisp sprites, retro game visual style, nostalgic digital art",
        "strip_keywords": [
            "Unreal Engine", "unreal engine", "UE5", "ray tracing",
            "AAA game CG", "photorealistic", "8k resolution",
            "ultra-detailed", "cinematic lighting", "masterpiece",
        ],
        "detect_keywords": [
            "像素", "pixel", "像素风", "pixel art", "8-bit",
            "16-bit", "复古游戏", "retro game", "sprite",
        ],
    },
    "comic_manga": {
        "suffix": "dynamic comic book panel art, bold ink outlines, halftone shading, high contrast, dramatic perspective, graphic novel style",
        "strip_keywords": [
            "Unreal Engine", "unreal engine", "UE5", "ray tracing",
            "AAA game CG", "photorealistic", "8k resolution",
        ],
        "detect_keywords": [
            "漫画", "comic", "manga", "美漫", "graphic novel",
            "连环画", "halftone", "ink outline",
        ],
    },
    "oil_painting": {
        "suffix": "classical oil painting, rich impasto texture, visible brushstrokes, Rembrandt lighting, gallery-quality fine art, museum masterpiece",
        "strip_keywords": [
            "Unreal Engine", "unreal engine", "UE5", "ray tracing",
            "AAA game CG", "photorealistic", "8k resolution",
            "ultra-detailed",
        ],
        "detect_keywords": [
            "油画", "oil painting", "古典", "classical", "伦勃朗",
            "Rembrandt", "厚涂", "impasto", "美术馆",
        ],
    },
    "minimalist": {
        "suffix": "minimalist composition, clean negative space, geometric simplicity, muted color palette, modern design aesthetic, elegant restraint",
        "strip_keywords": [
            "Unreal Engine", "unreal engine", "UE5", "ray tracing",
            "AAA game CG", "ultra-detailed", "8k resolution",
            "masterpiece", "epic",
        ],
        "detect_keywords": [
            "极简", "minimalist", "简约", "minimal", "留白",
            "negative space", "几何", "geometric",
        ],
    },
    "retro_vhs": {
        "suffix": "VHS tape aesthetic, analog distortion, scan lines, color bleeding, 80s retro-futurism, lo-fi nostalgia, CRT glow",
        "strip_keywords": [
            "Unreal Engine", "unreal engine", "UE5", "ray tracing",
            "AAA game CG", "8k resolution", "ultra-detailed",
        ],
        "detect_keywords": [
            "VHS", "vhs", "录像带", "复古", "retro", "80年代",
            "80s", "CRT", "扫描线", "scanline", "怀旧",
        ],
    },
    "stop_motion": {
        "suffix": "stop-motion animation style, tangible miniature sets, visible texture on puppet surfaces, warm practical lighting, Laika Studios quality",
        "strip_keywords": [
            "Unreal Engine", "unreal engine", "UE5", "ray tracing",
            "AAA game CG", "8k resolution", "ultra-detailed",
            "photorealistic",
        ],
        "detect_keywords": [
            "定格动画", "stop-motion", "stop motion", "黏土",
            "clay", "puppet", "微缩", "miniature", "Laika",
        ],
    },
    # 中性兜底风格：不偏向任何渲染流派（非写实 CG、非卡通、非水彩）。
    # 仅当「未识别到任何风格」时使用，避免静默把用户故事套成廉价 3D（上次画风崩坏的根因）。
    # 它提供一段「无流派偏向」的通用画面质量标准，让下游节点安全降级而非硬套 UE5。
    "neutral": {
        "suffix": "coherent illustrative visual style, clean readable composition, consistent lighting and palette",
        "strip_keywords": [],
        "detect_keywords": [],
    },
}

# AAA 后缀关键词列表——用于检测 prompt 中是否已有部分风格信息
_AAA_STYLE_KEYWORDS = ["cinematic", "8k", "masterpiece", "unreal engine", "ray tracing"]

# 向后兼容：默认风格（当没有检测到任何 style 关键词 / LLM 抽取失败时）。
# 注意：默认是「中性」而非「写实 CG」——避免未识别风格时静默套上廉价 3D 渲染，
# 违背「风格应顺用户输入」的原则。写实 CG(aaa_cg) 仅在用户输入明确命中其关键词时才生效。
_DEFAULT_STYLE_KEY = "neutral"

# ============================================================
# 风格质量标准（Quality Bar）
# ============================================================
# 每种风格对"好画面"的评判标准不同。
# 此字典供 reviewer 节点和 director prompt 使用，
# 让 LLM 知道当前风格下应该追求什么样的视觉品质。
# ============================================================

_STYLE_QUALITY_BAR: Dict[str, str] = {
    "aaa_cg": "AAA-level CGI quality, photorealistic detail, ray-traced lighting, Unreal Engine 5 fidelity, blockbuster game trailer standard",
    "anime": "Beautiful cel-shading, expressive character design, painterly backgrounds with detailed linework, high-quality anime key visual standard",
    "watercolor": "Harmonious color bleeding, elegant brushwork, emotional resonance through pigment texture, fine art illustration quality",
    "film_grain": "Authentic film texture, natural light quality, emotional depth through analog imperfection, professional photography standard",
    "documentary": "Authenticity and honesty, compelling real-world composition, natural lighting, photojournalistic impact",
    "horror_cinematic": "Atmospheric dread, masterful shadow play, unsettling composition, psychological tension through visual language",
    "pixel_art": "Pixel-perfect precision, effective limited palette, readable sprites at target resolution, retro game visual charm",
    "comic_manga": "Dynamic line energy, clear visual storytelling, impactful panel composition, professional comic art quality",
    "oil_painting": "Rich texture depth, masterful light and shadow, classical composition principles, gallery-quality fine art",
    "minimalist": "Elegant simplicity, balanced negative space, precise geometric relationships, refined visual restraint",
    "retro_vhs": "Authentic analog degradation, nostalgic color palette, convincing lo-fi texture, retro aesthetic coherence",
    "stop_motion": "Tangible material quality, charming miniature detail, warm practical lighting feel, handcrafted animation aesthetic",
    "neutral": "Coherent and clean visual quality, consistent lighting and palette, readable composition, no jarring rendering artifacts",
}


def _build_style_context(style_key: str) -> str:
    """
    为 LLM prompt 生成风格身份块。
    将风格的 角色身份 + 质量标准 + 风格后缀 整合为一段格式化文本，
    供 SHOWRUNNER_PROMPT / DIRECTOR_SYSTEM_PROMPT / REVIEWER 等模板
    通过 {style_context} 变量注入。

    参数：
      style_key: _STYLE_PRESETS 中的 key（如 'anime', 'watercolor'）
    返回：
      多行文本块（字符串）
    """
    preset = _STYLE_PRESETS.get(style_key, _STYLE_PRESETS[_DEFAULT_STYLE_KEY])
    quality = _STYLE_QUALITY_BAR.get(style_key, _STYLE_QUALITY_BAR[_DEFAULT_STYLE_KEY])

    return (
        f"VISUAL STYLE: {style_key.replace('_', ' ').title()}\n"
        f"Quality Standard: {quality}\n"
        f"Style Suffix (must append to every image prompt): {preset['suffix']}"
    )


# LLM 风格抽取专用 prompt：从用户输入（task + global_setting + story）识别视觉风格，
# 输出结构化 JSON。命中预设库时用 preset key；未命中时给出英文 style_prompt 由系统直接使用。
# 不维护任何硬编码风格表——"输入什么故事就什么风格"。
_VISUAL_STYLE_EXTRACT_PROMPT = """你是一个视频视觉风格分析器。根据用户提供的故事/任务描述，判断最适合的视觉呈现风格。

# 已知风格预设（若匹配请直接用其 key，可保证生图质量）：
{style_keys}

# 判断依据（综合阅读以下全部文本）：
{inputs}

# 输出要求（严格 JSON，不要多余文字）：
{{
  "style_key": "命中上面某个预设 key 则填它；都不匹配则填 \"custom\"",
  "label": "风格的中文短标签（如：水彩童话 / 赛博朋克CG / 水墨武侠）",
  "description": "一句话英文风格描述，供导演与审核节点理解风格基调",
  "style_prompt": "若 style_key 为 custom，给出一段英文生图风格后缀（逗号分隔，描述该风格的画面质感/笔触/光影/色调）；若命中预设则可留空"
}}

注意：风格必须忠实于用户输入的故事气质，不要强行套用预设，也不要替用户预设「某类题材就该用某风格」。
- 若输入已明确给出风格意图（如「水彩」「赛博朋克 CG」「水墨」），应尊重该意图选择/输出对应风格。
- 若输入未明确风格，则根据故事整体气质推断最贴切的风格（命中预设或输出 custom）。
- 不要为了匹配预设而扭曲用户输入的本意。"""


def _extract_visual_style(state: MultimediaState) -> Dict[str, str]:
    """LLM 驱动的风格抽取：从 task + global_setting + story 识别视觉风格。

    返回 dict: {"style_key", "style_suffix", "style_description"}。
    命中预设库 -> style_suffix 取预设 suffix（质量最优）；
    自定义风格 -> style_suffix 取 LLM 给的 style_prompt。
    LLM 失败则回退到关键词检测。
    """
    task = state.get("task", "") or ""
    global_setting = state.get("global_setting", "") or ""
    story = state.get("story", "") or state.get("script", "") or ""

    style_keys_block = "\n".join(f"  - {k}: {_STYLE_PRESETS[k]['suffix']}" for k in _STYLE_PRESETS)
    inputs_block = (
        f"[task]\n{task}\n\n"
        f"[global_setting]\n{global_setting}\n\n"
        f"[story]\n{story[:2000]}"
    )
    prompt = _VISUAL_STYLE_EXTRACT_PROMPT.format(
        style_keys=style_keys_block, inputs=inputs_block
    )

    # 兜底默认值：LLM 不可用时，使用「中性兜底」而非关键词猜测。
    # 关键点：不能让 _detect_visual_style 用关键词硬猜（如把"动画短片"误判成 anime），
    # 那会违背「风格顺用户输入、不随意发挥」的原则。中性兜底只提供通用画面质量标准，
    # 不偏向任何渲染流派，是最安全的降级（见 _DEFAULT_STYLE_KEY 设计注释）。
    fallback_key = _DEFAULT_STYLE_KEY
    result = {
        "style_key": fallback_key,
        "style_suffix": _STYLE_PRESETS.get(fallback_key, _STYLE_PRESETS[_DEFAULT_STYLE_KEY])["suffix"],
        "style_description": fallback_key.replace("_", " ").title(),
    }

    try:
        raw = call_llm(prompt, "风格抽取")
        parsed = _safe_json_loads(raw)
        if not parsed:
            return result
        key = (parsed.get("style_key") or "").strip()
        if key and key in _STYLE_PRESETS:
            result["style_key"] = key
            result["style_suffix"] = _STYLE_PRESETS[key]["suffix"]
        elif key == "custom" or (key and key not in _STYLE_PRESETS):
            # 自定义风格：使用 LLM 给出的英文 style_prompt
            custom_prompt = (parsed.get("style_prompt") or "").strip()
            result["style_key"] = key if key != "custom" else f"custom:{parsed.get('label', 'custom')}"
            result["style_suffix"] = custom_prompt or result["style_suffix"]
        result["style_description"] = parsed.get("description") or parsed.get("label") or result["style_description"]
    except Exception as e:
        print(f"    [风格抽取] LLM 失败，回退关键词检测: {e}")
    return result


def _resolve_style(state: MultimediaState) -> Dict[str, str]:
    """统一风格解析入口：优先读 state 缓存，否则抽取并写回。

    下游所有节点都应调用此函数，而非自行调用 _detect_visual_style，
    以保证整条流水线使用同一个风格身份（单一数据源，低耦合）。
    """
    if state.get("style_suffix") and state.get("visual_style"):
        return {
            "style_key": state["visual_style"],
            "style_suffix": state["style_suffix"],
            "style_description": state.get("style_description") or state["visual_style"],
        }
    extracted = _extract_visual_style(state)
    return extracted


def _safe_json_loads(text: str) -> Optional[dict]:
    """容错解析 LLM 返回的 JSON（兼容 ```json 包裹 / 前后多余文本）。"""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        # 去掉 ```json ... ``` 包裹
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


_TRANSLATE_PROMPT = """你是一个专业的图像/视频提示词翻译器。将下面的中文视觉描述翻译成地道、高质量的英文图像生成提示词。

要求：
- 保留所有视觉信息：景别、主体、动作、表情、服装、环境、光影、构图、镜头运动。
- 使用英文图像生成模型偏好的表达方式（具体名词、形容词，逗号分隔）。
- 不要添加原文没有的风格词或质量词（风格由系统统一注入）。
- 只输出翻译结果，不要解释、不要引号包裹。

中文描述：
{text}"""


def _translate_to_english(text: str, purpose: str = "image") -> str:
    """将中文提示词翻译为英文（生图/视频模型对英文理解更佳）。

    纯文本转换函数，与具体视觉风格无关，低耦合。
    LLM 失败时原样返回（中文直喂，不阻断主流程）。
    """
    if not text:
        return text
    # 若本就基本是英文（无中文字符），直接返回，省一次调用
    if not any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return text
    try:
        translated = call_llm(_TRANSLATE_PROMPT.format(text=text), "提示词翻译")
        translated = translated.strip().strip('"').strip("'")
        if translated:
            return translated
    except Exception as e:
        print(f"    [WARN] 提示词翻译失败，使用原文: {e}")
    return text


def _detect_visual_style(task: str) -> str:
    """
    从用户的 task 描述中检测视觉风格。
    返回 _STYLE_PRESETS 中的 key。
    未检测到任何关键词时返回默认值 "aaa_cg"。

    检测策略：逐 preset 计算命中的 detect_keywords 数量，
    命中最多的 preset 胜出（至少需要 1 个命中）。
    """
    if not task:
        return _DEFAULT_STYLE_KEY

    task_lower = task.lower()
    best_key = _DEFAULT_STYLE_KEY
    best_count = 0

    for key, preset in _STYLE_PRESETS.items():
        count = sum(1 for kw in preset["detect_keywords"] if kw.lower() in task_lower)
        if count > best_count:
            best_count = count
            best_key = key

    return best_key if best_count > 0 else _DEFAULT_STYLE_KEY


def _get_style_preset(state: MultimediaState) -> Dict[str, Any]:
    """
    从 state 获取当前风格 preset。
    优先使用 state["visual_style"]（如果已设置），
    否则从 task 重新检测。
    """
    style_key = state.get("visual_style")
    if not style_key or style_key not in _STYLE_PRESETS:
        style_key = _detect_visual_style(state.get("task", ""))
    return _STYLE_PRESETS[style_key]


def _get_style_suffix(state: MultimediaState) -> str:
    """根据 state 中的风格信息返回对应的 image prompt 后缀。"""
    return _get_style_preset(state)["suffix"]


# ============================================================
# Genre → 风格氛围修饰（叠加在 style suffix 之上）
# ============================================================
# genre 决定内容题材的"氛围词"，与 style 的"渲染后缀"互补。
# 例：genre=cyberpunk + style=anime → suffix 含 anime 渲染词 + "cyberpunk aesthetic, neon-lit"
_GENRE_ATMOSPHERE = {
    "cyberpunk": "cyberpunk aesthetic, neon-lit",
    "fantasy": "epic fantasy aesthetic",
    "gothic": "dark gothic aesthetic",
    "sci_fi": "sci-fi aesthetic",
    "post_apocalyptic": "post-apocalyptic aesthetic",
    "horror": "atmospheric horror aesthetic",
    "competitive": "esports competitive aesthetic",
}

# ============================================================
# 内容类型注册表（Content Type Registry）
# ============================================================
# 第三个正交维度：content_type 决定视频的结构形态和叙事节奏模式。
#   - trailer: 高潮递进，3-8 镜头，紧凑节奏
#   - short_film: 起承转合，灵活镜头数，自然节奏
#   - documentary: 平稳叙事，允许扁平弧线
#   - commercial: 展示→高潮→收束，3-5 镜头
#   - music_video: 节奏驱动，蒙太奇为主
#   - tutorial: 步骤递进，清晰结构
# ============================================================

_CONTENT_TYPE_PRESETS: Dict[str, Dict[str, Any]] = {
    "trailer": {
        "structure_guidance": "紧凑的预告片结构：开场建立→逐步升级→高潮爆发→余韵收束",
        "arc_expectation": "crescendo",
        "scene_count_range": (3, 8),
        "detect_keywords": [
            "预告片", "trailer", "宣传片", "promo", "teaser",
            "预告", "宣传", "先导片",
        ],
    },
    "short_film": {
        "structure_guidance": "短片叙事结构：起承转合，允许慢节奏和情感留白",
        "arc_expectation": "arc",
        "scene_count_range": (3, 12),
        "detect_keywords": [
            "短片", "short film", "微电影", "故事", "story",
            "叙事", "narrative", "剧情",
        ],
    },
    "documentary": {
        "structure_guidance": "纪录片结构：平稳叙事，允许信息性镜头和访谈式构图",
        "arc_expectation": "gentle_undulation",
        "scene_count_range": (4, 15),
        "detect_keywords": [
            "纪录片", "documentary", "纪实", "记录", "采访",
            "interview", "真实",
        ],
    },
    "commercial": {
        "structure_guidance": "广告结构：产品展示→卖点放大→品牌收束，简洁有力",
        "arc_expectation": "single_peak",
        "scene_count_range": (3, 6),
        "detect_keywords": [
            "广告", "commercial", "产品", "product", "品牌",
            "brand", "推广", "宣传",
        ],
    },
    "music_video": {
        "structure_guidance": "MV 结构：节奏驱动，视觉蒙太奇为主，允许风格化跳切",
        "arc_expectation": "rhythmic",
        "scene_count_range": (5, 20),
        "detect_keywords": [
            "MV", "音乐视频", "music video", "歌曲", "song",
            "配乐", "soundtrack",
        ],
    },
    "tutorial": {
        "structure_guidance": "教程结构：步骤清晰递进，每个镜头聚焦一个要点",
        "arc_expectation": "ascending_steps",
        "scene_count_range": (3, 10),
        "detect_keywords": [
            "教程", "tutorial", "教学", "讲解", "演示",
            "how to", "步骤",
        ],
    },
}

_DEFAULT_CONTENT_TYPE = "trailer"


def _detect_content_type(task: str) -> str:
    """
    从用户的 task 描述中检测内容类型。
    返回 _CONTENT_TYPE_PRESETS 中的 key。
    未检测到任何关键词时返回默认值 "trailer"。
    """
    if not task:
        return _DEFAULT_CONTENT_TYPE

    task_lower = task.lower()
    best_key = _DEFAULT_CONTENT_TYPE
    best_count = 0

    for key, preset in _CONTENT_TYPE_PRESETS.items():
        count = sum(1 for kw in preset["detect_keywords"] if kw.lower() in task_lower)
        if count > best_count:
            best_count = count
            best_key = key

    return best_key if best_count > 0 else _DEFAULT_CONTENT_TYPE


_CN_EN_TRANSLATIONS = {
    "蹲伏": "crouching", "飞檐": "flying eave", "建筑": "architecture",
    "机械": "mechanical", "马尾": "ponytail", "额带": "headband",
    "双瞳": "pupils", "萤火虫": "fireflies", "短刃": "short blades",
    "光": "light", "义肢": "prosthetic limb", "暗金": "dark gold",
    "幕": "curtain", "瓦": "tiles", "蒸汽": "steam",
    "全息": "holographic", "深绿": "dark green", "巡逻": "patrol",
    "机甲": "mech armor", "扫描光": "scanning light", "披风": "cloak",
    "能量": "energy", "碰撞": "collision", "火花": "sparks",
    "雨": "rain", "夜": "night", "赤红": "crimson red",
    "漆黑": "pitch black", "霓虹": "neon", "剪影": "silhouette",
    "合金": "alloy", "铠甲": "armor", "武士": "warrior",
    "追踪": "tracking", "俯拍": "overhead shot", "仰拍": "low angle shot",
    "全景": "panoramic", "中景": "medium shot", "特写": "close-up",
    "定格": "frozen moment", "瞬间": "instant",
    "弧线": "arc", "光迹": "light trail",
    "逆光": "backlit", "散景": "bokeh", "景深": "depth of field",
}

# 中文字符 Unicode 范围正则
_CN_CHAR_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]+')


def _strip_chinese_from_prompt(prompt: str) -> str:
    """
    后处理安全网：检测 image_prompt 中的中文字符，
    尝试用翻译表替换，无法翻译的则移除。
    """
    if not prompt:
        return prompt

    # 先检查是否有中文
    if not _CN_CHAR_RE.search(prompt):
        return prompt

    chinese_count = len(_CN_CHAR_RE.findall(prompt))
    print(f"    ⚠️ [安全网] 检测到 {chinese_count} 处中文字符，执行后处理替换...")

    # 用翻译表替换
    result = prompt
    for cn, en in _CN_EN_TRANSLATIONS.items():
        if cn in result:
            result = result.replace(cn, en)

    # 移除剩余的中文字符（无法翻译的）
    remaining = _CN_CHAR_RE.findall(result)
    if remaining:
        print(f"    ⚠️ [安全网] 仍有 {len(remaining)} 处无法翻译的中文: {remaining[:5]}...")
        # 移除中文字符，保留周围的空格和标点
        result = _CN_CHAR_RE.sub('', result)
        # 清理多余空格
        result = re.sub(r'\s{2,}', ' ', result)
        result = re.sub(r'\s+([,.;:!?])', r'\1', result)
        result = re.sub(r'([,.;:!?])\s*([,.;:!?])', r'\1', result)

    return result.strip()


def _visual_script(script: str) -> str:
    """从分镜 script 中剔除对话/台词原文，仅保留纯视觉叙述（景别+主体动作+环境）。

    用于受保护规格与环境锚点的抽取，避免把角色台词/对白文字渲染进画面。
    注意：这是对 script 的"视觉化"预处理，不影响 dialogue 字段（台词仍由 LLM 单独生成）。
    规则剥离，不调用 LLM，对纯视觉 script 幂等。
    """
    text = script or ""
    # 1) 去掉 "角色名: "台词"" / 角色名：'台词' 这类 说话人+引号对白 块
    text = re.sub(r"[一-龥A-Za-z0-9_·\-]+[\s]*[:：][\s]*[“\"'\"].*?[”\"'\"]", " ", text)
    # 2) 去掉残留的成对引号对白（中英文引号）
    text = re.sub(r"[“\"'\"].*?[”\"'\"]", " ", text)
    # 3) 去掉未带引号、但形如 "XX说/道/喊/问：" 的说话标记及其后到句号前的对白
    text = re.sub(r"[一-龥A-Za-z0-9_·\-]+[\s]*(说|道|喊|问|答|嘀咕|喃喃|沉声|冷笑)[\s]*[:：][^。！？]*[。！？]?", " ", text)
    # 4) 压缩多余空白
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _extract_protected_specs(script: str, global_setting: str) -> str:
    """
    从用户的 script 和 global_setting 中提取关键视觉规格，
    生成一个保护性规格列表（文本格式），供 director 和 end_frame_director 注入。

    这些规格来自用户原始输入，不经过任何 LLM 处理，
    下游节点必须将这些规格的英文翻译原文保留在输出 prompt 中。
    """
    specs = []

    # 1. 从 global_setting 提取角色身份锚点
    if global_setting:
        # 提取肤色描述
        skin_patterns = ["肤色", "皮肤", "skin"]
        for p in skin_patterns:
            if p in global_setting.lower():
                idx = global_setting.lower().index(p)
                start = max(0, global_setting.rfind("，", 0, idx) + 1)
                end = global_setting.find("，", idx)
                if end == -1:
                    end = global_setting.find("。", idx)
                if end == -1:
                    end = len(global_setting)
                specs.append(f"[Character/Skin] {global_setting[start:end].strip()}")
                break

        # 提取发型/发色描述
        hair_patterns = ["编发", "长发", "短发", "白发", "银发", "黑发", "金发", "红发",
                         "辫", "卷发", "直发", "发色", "发型", "braids", "hair"]
        for p in hair_patterns:
            if p in global_setting.lower():
                idx = global_setting.lower().index(p)
                start = max(0, global_setting.rfind("，", 0, idx) + 1)
                end = global_setting.find("，", idx)
                if end == -1:
                    end = global_setting.find("。", idx)
                if end == -1:
                    end = len(global_setting)
                specs.append(f"[Character/Hair] {global_setting[start:end].strip()}")
                break

        # 提取眼睛/瞳孔描述
        eye_patterns = ["眼", "瞳孔", "义眼", "虹膜", "双目", "eye", "iris", "pupil"]
        for p in eye_patterns:
            if p in global_setting.lower():
                idx = global_setting.lower().index(p)
                start = max(0, global_setting.rfind("，", 0, idx) + 1)
                end = global_setting.find("，", idx)
                if end == -1:
                    end = global_setting.find("。", idx)
                if end == -1:
                    end = len(global_setting)
                specs.append(f"[Character/Eyes] {global_setting[start:end].strip()}")
                break

        # 提取体型描述
        body_patterns = ["身材", "体型", "身高", "身姿", "build", "height", "figure"]
        for p in body_patterns:
            if p in global_setting.lower():
                idx = global_setting.lower().index(p)
                start = max(0, global_setting.rfind("，", 0, idx) + 1)
                end = global_setting.find("，", idx)
                if end == -1:
                    end = global_setting.find("。", idx)
                if end == -1:
                    end = len(global_setting)
                specs.append(f"[Character/Body] {global_setting[start:end].strip()}")
                break

        # 提取服装/装甲描述
        outfit_patterns = ["身穿", "穿着", "服装", "作战服", "战术服", "铠甲", "风衣", "长袍"]
        for p in outfit_patterns:
            if p in global_setting:
                idx = global_setting.index(p)
                start = max(0, global_setting.rfind("，", 0, idx) + 1)
                end = global_setting.find("，", idx)
                if end == -1:
                    end = global_setting.find("。", idx)
                if end == -1:
                    end = len(global_setting)
                specs.append(f"[Outfit] {global_setting[start:end].strip()}")
                break

        # 提取武器/道具描述
        weapon_patterns = ["手持", "携带", "配", "武器", "枪", "剑", "刀", "弓"]
        for p in weapon_patterns:
            if p in global_setting:
                idx = global_setting.index(p)
                start = max(0, global_setting.rfind("，", 0, idx) + 1)
                end = global_setting.find("，", idx)
                if end == -1:
                    end = global_setting.find("。", idx)
                if end == -1:
                    end = len(global_setting)
                specs.append(f"[Weapon/Prop] {global_setting[start:end].strip()}")
                break

        # 提取色彩方案
        color_patterns = ["色调", "色彩", "配色", "色系", "主色"]
        for p in color_patterns:
            if p in global_setting:
                idx = global_setting.index(p)
                start = max(0, global_setting.rfind("，", 0, idx) + 1)
                end = global_setting.find("，", idx)
                if end == -1:
                    end = global_setting.find("。", idx)
                if end == -1:
                    end = len(global_setting)
                specs.append(f"[Color] {global_setting[start:end].strip()}")
                break

    # 2. 从 script 提取具体视觉元素（环境物体、动作细节等）
    if script:
        # 提取环境物体
        env_patterns = [
            ("霓虹", "neon signs"), ("全息", "holographic"), ("投影", "projection"),
            ("废墟", "ruins"), ("玻璃", "glass"), ("钢铁", "steel"),
            ("雨水", "rain"), ("雾气", "fog"), ("火焰", "flames"),
        ]
        found_envs = []
        for cn, en in env_patterns:
            if cn in script:
                found_envs.append(f"{cn}({en})")
        if found_envs:
            specs.append(f"[Environment] {', '.join(found_envs)}")

        # 提取具体动作
        action_patterns = [
            ("奔跑", "running"), ("跳跃", "jumping"), ("蹲伏", "crouching"),
            ("挥剑", "swinging sword"), ("举枪", "raising gun"), ("转身", "turning"),
            ("站立", "standing"), ("行走", "walking"), ("飞行", "flying"),
        ]
        found_actions = []
        for cn, en in action_patterns:
            if cn in script:
                found_actions.append(f"{cn}({en})")
        if found_actions:
            specs.append(f"[Action] {', '.join(found_actions)}")

    if not specs:
        return "[No specific protected specs extracted — proceed with standard quality.]"

    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(specs))


def _extract_environment_anchors(global_setting: str, script: str) -> str:
    """
    从 global_setting 和 script 中提取环境/场景锚点。

    设计原则：使用结构性触发词（如"场景""风格""色调""整体"）定位环境描述所在的子句，
    而非逐个枚举环境物体名称。这些锚点会被注入 director prompt 的专用块中，
    强制每个 image_prompt 包含明确的环境描述，防止跨场景环境漂移。

    Returns:
        环境锚点文本块（可为空字符串）
    """
    anchors = []

    # ── 从 global_setting 提取环境子句 ──
    # 策略：用结构性标记词定位"场景级"描述子句（而非角色级描述）
    if global_setting:
        # 按中文标点分割为子句
        import re
        clauses = re.split(r'[，。；！？]', global_setting)

        # 结构性触发词——标识"这是场景/环境描述"而非"角色描述"的子句
        scene_triggers = [
            "场景", "风格", "色调", "整体", "氛围", "基调",
            "光线", "光影", "配色", "设定在", "发生在",
        ]

        for clause in clauses:
            clause = clause.strip()
            if not clause or len(clause) < 4:
                continue
            for trigger in scene_triggers:
                if trigger in clause:
                    anchors.append(clause)
                    break  # 一个子句只添加一次

    # ── 从 script 提取地点锚点 ──
    # 策略：检测地点类名词（通用场所词，非特定物体），提取所在短语
    if script:
        place_triggers = [
            "森林", "竹林", "城市", "废墟", "城堡", "洞穴", "神殿", "殿堂",
            "战场", "荒野", "沙漠", "雪地", "海岸", "山", "湖畔", "街道",
            "室内", "大殿", "密室", "走廊", "庭院", "花园", "码头", "港口",
            "forest", "city", "ruins", "castle", "temple", "arena",
        ]
        for trigger in place_triggers:
            if trigger in script:
                pos = script.index(trigger)
                # 向前扩展到最近的分隔符，向后扩展到 trigger 之后的修饰词
                start = pos
                for sep in ("，", "。", "：", " ", "、"):
                    idx = script.rfind(sep, 0, pos)
                    if idx >= 0 and pos - idx < start - (pos - start):
                        start = idx + 1
                        break
                else:
                    start = max(0, pos - 12)

                end = pos + len(trigger)
                # 向后吃修饰词（如"竹林深处"的"深处"）
                while end < len(script) and script[end] not in "，。；！？ ":
                    end += 1

                phrase = script[start:end].strip()
                if phrase and len(phrase) >= 2 and phrase not in anchors:
                    anchors.append(phrase)
                break  # 只取第一个匹配的地点

    if not anchors:
        return ""

    # 去重并格式化
    unique = []
    seen = set()
    for a in anchors:
        if a not in seen:
            unique.append(a)
            seen.add(a)

    return "\n".join(unique)


def _sanitize_single_frame_prompt(
    prompt: str,
    style_suffix: str = None,
    strip_keywords: list = None,
) -> str:
    """
    检测并截掉 image_prompt 中 LLM 自行附加的多镜头序列内容。
    如果检测到 'Sequence includes:' 等模式，从该位置截断。
    最后执行中文清理和 style suffix 注入。

    参数：
      prompt:          待清理的英文图像 prompt
      style_suffix:    风格后缀（从 _STYLE_PRESETS 获取），默认使用 aaa_cg
      strip_keywords:  需要从 prompt 中清理的风格不兼容术语列表
                       （例如 anime 风格时清理 "Unreal Engine", "ray tracing" 等）
    """
    if not prompt:
        return prompt

    # Step 1: 截掉多镜头序列尾巴
    for pattern in _SEQUENCE_PATTERNS:
        match = re.search(pattern, prompt)
        if match:
            cut_pos = match.start()
            original_tail = prompt[cut_pos:cut_pos+80]
            prompt = prompt[:cut_pos].rstrip(" .,;")
            print(f"    [!] 已截掉多镜头序列尾巴: \"{original_tail}...\"")
            break

    # Step 2: 清理中文字符（后处理安全网）
    prompt = _strip_chinese_from_prompt(prompt)

    # Step 3: 清理风格不兼容术语（非 AAA 风格时，移除 LLM 自动生成的 AAA 术语）
    if strip_keywords:
        stripped_any = False
        for kw in strip_keywords:
            # 不区分大小写替换
            pattern_re = re.compile(re.escape(kw), re.IGNORECASE)
            new_prompt = pattern_re.sub('', prompt)
            if new_prompt != prompt:
                stripped_any = True
                prompt = new_prompt
        if stripped_any:
            # 清理替换产生的多余逗号和空格
            prompt = re.sub(r',\s*,', ',', prompt)
            prompt = re.sub(r',\s*$', '', prompt.strip())
            prompt = re.sub(r'\s{2,}', ' ', prompt)
            print(f"    [*] 已清理 {len(strip_keywords)} 个风格不兼容术语")

    # Step 4: 注入 style suffix（风格感知）
    suffix = style_suffix or _STYLE_PRESETS[_DEFAULT_STYLE_KEY]["suffix"]
    if suffix.lower() not in prompt.lower():
        # 宽松检查：是否已有部分风格关键词（至少 2 个匹配才算已有部分风格）
        suffix_keywords = [w.strip() for w in suffix.split(",") if len(w.strip()) > 3]
        match_count = sum(1 for kw in suffix_keywords[:4] if kw.lower() in prompt.lower())
        has_partial = match_count >= 2
        if not has_partial:
            prompt = prompt.rstrip(" .,;") + ", " + suffix
            print(f"    [*] 已追加 style suffix")

    return prompt.strip()


def _finalize_single_frame_prompt(state: MultimediaState, prompt: str) -> str:
    """
    单帧 prompt 的统后置处理：风格安全网（截断/清理/注入 suffix）+ 翻译为英文。
    director 首帧与尾帧节点结尾复用，避免重复样板。
    """
    _preset = _get_style_preset(state)
    _style_key = _resolve_style(state)["style_key"]
    prompt = _sanitize_single_frame_prompt(
        prompt, _preset["suffix"], _get_conflict_keywords(_style_key)
    )
    # 翻译层：生图/视频模型对英文 prompt 出图质量显著更高，将中文叙事翻译为英文再喂模型
    return _translate_to_english(prompt, purpose="image")
