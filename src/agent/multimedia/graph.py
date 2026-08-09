# src/agent/multimedia/graph.py
import json
import os
import re
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt

from .state import MultimediaState
from . import run_metrics  # 运行指标埋点（可观测性，零依赖，可经 RUN_METRICS_ENABLED 关闭）
from .prompts import (
    SHOWRUNNER_PROMPT,
    DIRECTOR_SYSTEM_PROMPT,
    END_FRAME_DIRECTOR_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    VIDEOGRAPHER_PROMPT,
    VIDEOGRAPHER_DUAL_FRAME_PROMPT,
    VIDEO_REVIEWER_SYSTEM_PROMPT,
    SAFETY_PROMPT,
    DIRECTOR_REFINEMENT_PROMPT,
)
from .checkpointer import checkpointer, make_thread_config
from .tools.text_llm import call_llm
from .tools.image_gen import generate_keyframe
from .tools.video_gen import generate_video_from_image, FreeTierQuotaExhaustedError
from .tools.vision_eval import evaluate_image, evaluate_video, design_camera_movement
from .tools.video_stitcher import stitch_videos
# P0 音频闭环：配音 / 字幕 / 混音，全部复用 GitHub 现成组件
from .tools.tts import generate_voiceover
from .tools.subtitle import build_srt, burn_subtitles
from .tools.audio import generate_bgm, mix_audio, resolve_bgm_mood
from .visual_context import build_visual_context
from .shot_strategy import (
    generate_shot_strategy,
    rewrite_strategy,
    get_consistency_threshold,
)
from .sequence_orchestrator import build_sequence_graph, _EMOTION_ENERGY, select_cross_scene_transition
from .cinematic_critic import evaluate_cinematic_quality, format_critic_report, QUALITY_THRESHOLDS
from .film_studio_orchestrator import orchestrate_film_studio
from .cinematic_memory import reset_memory
from .rag.retriever import retrieve_rag_context
from .reference_gen import generate_reference_sheets

try:
    from .tools.embedding import compute_similarity, get_image_embedding
except Exception:
    compute_similarity = None
    get_image_embedding = None


APPROVE_ACTIONS = {"approve", "pass", "ok", "accept", "yes", "通过", "批准"}
REWRITE_ACTIONS = {"rewrite", "edit", "edit_prompt", "revise", "modify", "重写", "修改"}

# 单镜头视频生成失败（网络/超时/接口错误）的最大重试次数。
# 达到上限后生成占位黑场降级，继续生成其余镜头，避免单点失败拖垮整个任务。
# 改从 thresholds.json 读取（保持默认 2，MAX_VIDEO_GEN_FAILURES env 可覆盖）。
from .config_loader import max_video_gen_failures, character_consistency_threshold

MAX_VIDEO_GEN_FAILURES = max_video_gen_failures()

# 角色一致性闭环阈值：关键帧与"源头角色肖像"的 embedding 余弦相似度下限。
# 低于该值视为角色漂移，即便 VLM 判 PASS 也触发一次重生（受 iterations 上限保护）。
# 取值偏保守(0.5)，避免误杀正常的景别/构图差异；可用环境变量覆盖。
# 设为 0 可关闭该闭环（回退到纯 VLM 判定）。
CHARACTER_CONSISTENCY_THRESHOLD = character_consistency_threshold()


# ── 视频生成失败降级：占位黑场 ──
# 达 MAX_VIDEO_GEN_FAILURES 上限时，生成本地占位黑场 .mp4（带静音底），
# 写入该镜头应有的 final_video_url 槽位，使 stitcher/audio_mixer 仍能拼出
# 时间轴完整的成片，避免"静默丢镜头"导致下游错位。占位文件为本地绝对路径。
# 实现放在 config_loader（与 graph 解耦，无重依赖），此处直接复用。
from .config_loader import make_placeholder_clip as _make_placeholder_clip
from .config_loader import persist_studio_output as _persist_studio_output


def safe_parse_json(response: str) -> dict:
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
    # 运行指标埋点：记录人工审核卡被消费的 action（可观测性）
    if gate:
        run_metrics.record_human_gate(gate, out["action"])
    return out


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
    }


# ================= Phase 1: Visual Context Retrieval Layer =================
def visual_context_builder_node(state: MultimediaState):
    """
    Visual Context Retrieval Layer — 视觉上下文构建节点。
    不生成任何图片/视频，仅根据 scene script + global setting
    生成结构化视觉上下文，注入 state.visual_context 供 director 消费。
    """
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    global_setting = state.get("global_setting", "")
    total_scenes = len(state["scenes"])

    print(f"\n--- 🧠 [视觉上下文] 镜头 {idx+1}/{total_scenes} 意图分析 ---")

    # ── 跨镜头状态重置：防止上一镜头的残留数据污染当前镜头 ──
    scene_resets = {}
    for field in ("shot_plan", "sequence_graph", "critic_eval",
                  "studio_output", "studio_output_path", "studio_summary"):
        if state.get(field) is not None:
            scene_resets[field] = None
    if state.get("rewrite_count", 0) != 0:
        scene_resets["rewrite_count"] = 0
    if scene_resets:
        print(f"    [*] 已重置 {len(scene_resets)} 个跨镜头残留字段: {list(scene_resets.keys())}")

    visual_ctx = build_visual_context(
        script=scene["script"],
        global_setting=global_setting,
        scene_index=idx,
        total_scenes=total_scenes,
        prev_anchors=state.get("scene_anchors"),
    )

    # 连续性日志
    prev_anchors = state.get("scene_anchors")
    if prev_anchors and visual_ctx.get("continuity_directives"):
        n_directives = len(visual_ctx["continuity_directives"])
        print(f"    [*] 跨场景连续性: 已注入 {n_directives} 条渐变约束（来自镜头 {idx} 的视觉锚点）")

    print(f"    [*] 场景意图: {visual_ctx['intent']}")
    print(f"    [*] 节奏阶段: {visual_ctx['pacing_phase']}")
    print(f"    [*] 推荐镜头: {visual_ctx['camera_rules'][0][:50]}...")
    print(f"    [*] 推荐光影: {visual_ctx['lighting_rules'][0][:50]}...")

    # Phase RAG: 语义检索知识库，获取参考知识
    rag_result = retrieve_rag_context(
        script=scene["script"],
        global_setting=global_setting,
        intent=visual_ctx.get("intent", ""),
        verbose=True,
    )

    # 将 RAG 格式化文本注入 visual_context 供下游可选消费
    if rag_result:
        visual_ctx["rag_references"] = rag_result.get("formatted_rag_block", "")
        print(f"    [*] RAG 检索: {rag_result.get('total_retrieved', 0)} 条参考")
    else:
        visual_ctx["rag_references"] = ""
        print("    [*] RAG 检索: 无结果（静默降级）")

    result = {"visual_context": visual_ctx, "rag_context": rag_result}
    result.update(scene_resets)  # 注入跨镜头状态重置
    return result


# ================= Phase 2: Shot Strategy Layer =================
def shot_strategy_builder_node(state: MultimediaState):
    """
    Shot Strategy Layer — 镜头规划系统节点。
    消费 Phase 1 的 visual_context，生成结构化的多镜头拍摄方案（shot plan），
    注入 state.shot_plan 供 director / end_frame_director 精确执行。

    Phase 4 扩展：当 critic_eval 存在重写建议时，使用 rewrite_strategy 优化现有 plan。
    """
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    visual_ctx = state.get("visual_context")
    total_scenes = len(state["scenes"])

    # Phase 4: 检查是否有 critic 重写建议
    critic_eval = state.get("critic_eval")
    existing_plan = state.get("shot_plan")
    rewrite_count = state.get("rewrite_count", 0)

    if critic_eval and critic_eval.get("rewrite_strategies") and existing_plan:
        # ── Rewrite Mode: 基于 critic 建议优化现有 plan ──
        strategies = critic_eval["rewrite_strategies"]
        print(f"\n--- 🔄 [镜头规划] 镜头 {idx+1}/{total_scenes} 重写优化 (rewrite #{rewrite_count + 1}) ---")
        print(f"    [*] 应用策略: {', '.join(strategies)}")

        shot_plan = rewrite_strategy(
            shot_plan=existing_plan,
            suggestions=strategies,
            script=scene["script"],
        )
        rewrite_count += 1
    else:
        # ── Generate Mode: 从头生成 shot plan ──
        print(f"\n--- 🎬 [镜头规划] 镜头 {idx+1}/{total_scenes} 拍摄方案设计 ---")

        if not visual_ctx:
            from .visual_context import build_visual_context as _build_vc
            visual_ctx = _build_vc(
                script=scene["script"],
                global_setting=state.get("global_setting", ""),
                scene_index=idx,
                total_scenes=total_scenes,
                prev_anchors=state.get("scene_anchors"),
            )
            print("    [!] visual_context 缺失，已临时构建")

        shot_plan = generate_shot_strategy(
            script=scene["script"],
            visual_context=visual_ctx,
        )

    hero = shot_plan["shots"][shot_plan["hero_shot_index"]]
    end = shot_plan["shots"][shot_plan["end_shot_index"]]
    print(f"    [*] 镜头序列: {len(shot_plan['shots'])} shots")
    print(f"    [*] Hero Shot: #{shot_plan['hero_shot_index']+1} \"{hero['shot_name']}\" — {hero['camera']['type']}, {hero['camera']['movement']}")
    print(f"    [*] End Shot:  #{shot_plan['end_shot_index']+1} \"{end['shot_name']}\" — {end['camera']['type']}, {end['camera']['movement']}")
    print(f"    [*] 情绪曲线: {' → '.join(s['emotion'] for s in shot_plan['shots'])}")

    return {"shot_plan": shot_plan, "rewrite_count": rewrite_count}


# ================= Phase 3: Sequence Orchestration Layer =================
def sequence_orchestrator_node(state: MultimediaState):
    """
    Sequence Orchestration Layer — 电影剪辑系统节点。
    消费 Phase 2 的 shot_plan，构建镜头关系图（sequence_graph），
    包含转场、运动连续性、情绪轨迹。注入 state.sequence_graph。
    """
    idx = state["current_scene_index"]
    shot_plan = state.get("shot_plan")
    total_scenes = len(state["scenes"])

    print(f"\n--- 🎞️ [序列编排] 镜头 {idx+1}/{total_scenes} 剪辑设计 ---")

    if not shot_plan:
        # 兜底：无 shot_plan 时返回空 graph
        print("    [!] shot_plan 缺失，跳过序列编排")
        return {"sequence_graph": None}

    seq_graph = build_sequence_graph(shot_plan)

    connections = seq_graph.get("connections", [])
    arc = seq_graph.get("emotional_arc", {})

    print(f"    [*] 镜头连接数: {len(connections)}")
    if arc:
        print(f"    [*] 情绪曲线形状: {arc.get('shape', 'unknown')}")
        print(f"    [*] 总体方向: {arc.get('overall_direction', 'unknown')}")
        print(f"    [*] 能量峰值: shot #{arc.get('peak_index', 0)+1} ({arc.get('peak_emotion', '?')})")

    for conn in connections:
        print(f"    [*] {conn['from_name']} --[{conn['transition']}]--> {conn['to_name']}  |  {conn['emotional_delta']['description']}")

    return {"sequence_graph": seq_graph}


# ================= Phase 4: Self-Refining Cinematic Director System =================
def cinematic_critic_node(state: MultimediaState):
    """
    Cinematic Critic Layer — 电影质量评估节点。
    评估 shot_plan + sequence_graph 的质量，
    决定是否通过质量门控或触发重写循环。
    """
    idx = state["current_scene_index"]
    shot_plan = state.get("shot_plan")
    seq_graph = state.get("sequence_graph")
    rewrite_count = state.get("rewrite_count", 0)
    total_scenes = len(state["scenes"])

    print(f"\n--- 🔍 [电影评审] 镜头 {idx+1}/{total_scenes} 结构质量评估 (rewrite #{rewrite_count}) ---")

    if not shot_plan:
        print("    [!] shot_plan 缺失，跳过评估")
        return {
            "critic_eval": None,
            "quality_gates": None,
        }

    if not seq_graph:
        print("    [!] sequence_graph 缺失，使用空图评估")
        seq_graph = {"connections": [], "emotional_arc": {"shape": "unknown", "overall_direction": "unknown"}}

    # 执行多维度评估（传入 content_type 以适配不同内容类型的评分标准）
    content_type = state.get("content_type", "trailer")
    eval_result = evaluate_cinematic_quality(shot_plan, seq_graph, content_type=content_type)

    # 格式化诊断报告
    report = format_critic_report(eval_result)
    print(report)

    # 检查是否超过最大重写次数
    max_rewrites = QUALITY_THRESHOLDS.get("max_rewrites", 2)
    if rewrite_count >= max_rewrites and not eval_result["pass_gates"]:
        print(f"    [!] 已达最大重写次数 ({max_rewrites})，强制通过")
        eval_result["pass_gates"] = True

    gates = eval_result.get("gates", {})
    print(f"\n    [*] 综合评分: {eval_result['overall_score']:.2f} ({'PASS' if eval_result['pass_gates'] else 'FAIL'})")

    return {
        "critic_eval": eval_result,
        "quality_gates": gates,
    }


def director_refine_node(state: MultimediaState):
    """
    Director Refinement — 导演精修节点。
    消费 critic 评估结果，用 LLM 精修 image_prompt。
    仅当存在 critic suggestions 时执行精修，否则透传。
    """
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    critic_eval = state.get("critic_eval")

    image_prompt = scene.get("image_prompt", "")
    if not image_prompt:
        print(f"    [!] 无 image_prompt 可精修，跳过")
        return {"scenes": state["scenes"]}

    # 如果有评估结果且有建议，执行 LLM 精修
    if critic_eval and critic_eval.get("suggestions"):
        scores = critic_eval["scores"]
        issues_text = "\n".join(f"- {i}" for i in critic_eval["issues"]) if critic_eval["issues"] else "No specific issues found."
        suggestions_text = "\n".join(f"- {s}" for s in critic_eval["suggestions"])

        # 注入 shot_plan 和 RAG 上下文，让精修 LLM 了解原始拍摄意图
        shot_plan = state.get("shot_plan", {})
        context_block = ""
        if shot_plan and shot_plan.get("formatted_plan"):
            context_block += f"\n\n[Original Shot Plan — Do NOT contradict these specs]\n{shot_plan['formatted_plan'][:600]}"
        rag_ctx = state.get("rag_context")
        if rag_ctx and isinstance(rag_ctx, dict):
            rag_text = rag_ctx.get("formatted_rag_block", "")
            if rag_text:
                context_block += f"\n\n[RAG References]\n{rag_text[:400]}"
        global_setting = state.get("global_setting", "")
        if global_setting:
            context_block += f"\n\n[Global Visual Setting]\n{global_setting[:300]}"

        # ── 环境锚点（精修阶段也需保护）──
        env_anchor = _extract_environment_anchors(global_setting, scene.get("script", ""))
        if env_anchor:
            context_block += f"\n\n[Environment Anchor — Do NOT alter or omit these setting/location details]\n{env_anchor}"

        print(f"\n--- 🔧 [镜头 {idx+1}] 导演精修 image_prompt (score={critic_eval['overall_score']:.2f}) ---")

        protected_specs_text = _extract_protected_specs(
            scene.get("script", ""), state.get("global_setting", "")
        )

        # 风格后缀（统一从 state 解析，单一数据源）
        style = _resolve_style(state)
        style_key = style["style_key"]
        style_suffix = style["style_suffix"]

        prompt = DIRECTOR_REFINEMENT_PROMPT.format(
            image_prompt=image_prompt,
            overall_score=critic_eval["overall_score"],
            visual_quality=scores["visual_quality"],
            coherence=scores["coherence"],
            cinematic_flow=scores["cinematic_flow"],
            emotion_consistency=scores["emotion_consistency"],
            issues=issues_text,
            suggestions=suggestions_text,
            protected_specs=protected_specs_text,
            style_suffix=style_suffix,
        )
        # 追加上下文信息（DIRECTOR_REFINEMENT_PROMPT 模板不含这些变量，所以直接拼接）
        if context_block:
            prompt += context_block
            print(f"    [*] 已注入 Shot Plan + RAG + Global Setting 上下文")

        refined_prompt = call_llm(prompt, "导演精修")

        scenes = state["scenes"].copy()
        scenes[idx]["image_prompt"] = refined_prompt
        print(f"    [*] 精修完成 ({len(refined_prompt)} chars)")
        return {"scenes": scenes}
    else:
        print(f"    [*] 无精修建议，保持原始 image_prompt")
        return {"scenes": state["scenes"]}


# ================= Phase 5: Multi-Agent Film Studio System =================
def film_studio_node(state: MultimediaState):
    """
    Film Studio Orchestrator — 多智能体电影工作室节点。

    协调 3 个专业 Agent 协作优化 shot_plan + sequence_graph：
    1. Cinematography Agent — 摄影系统优化
    2. Lighting/VFX Agent — 光影氛围增强
    3. Editor Agent — 剪辑流畅度优化
    4. Debate — 跨 Agent 冲突解决

    输出经过"团队审查"的方案供下游 cinematic_critic 评估。
    """
    idx = state["current_scene_index"]
    shot_plan = state.get("shot_plan")
    seq_graph = state.get("sequence_graph")
    visual_ctx = state.get("visual_context")
    total_scenes = len(state["scenes"])
    global_setting = state.get("global_setting", "")

    print(f"\n--- 🎬 [电影工作室] 镜头 {idx+1}/{total_scenes} 多 Agent 协作优化 ---")

    if not shot_plan:
        print("    [!] shot_plan 缺失，跳过工作室优化")
        return {"studio_output": None}

    if not seq_graph:
        seq_graph = {"connections": [], "emotional_arc": {"shape": "unknown", "overall_direction": "unknown"}}

    if not visual_ctx:
        visual_ctx = {"intent": "world_building", "pacing_phase": "opening"}

    # 运行工作室编排器
    studio_result = orchestrate_film_studio(
        shot_plan=shot_plan,
        sequence_graph=seq_graph,
        visual_context=visual_ctx,
        scene_index=idx,
        total_scenes=total_scenes,
        global_setting=global_setting,
    )

    # 打印工作室报告
    report = studio_result.get("formatted_studio_report", "")
    print(report)

    print(f"\n    [*] 工作室共识: {studio_result['studio_consensus']:.0%}")
    print(f"    [*] 总修改数: {len(studio_result.get('modifications', []))}")
    print(f"    [*] Debate 冲突: {len(studio_result.get('debate', []))} resolved")

    # 将工作室优化结果注入 shot_plan（让下游 director 消费优化后的版本）
    refined_plan = studio_result.get("studio_refined_plan", {})
    refined_shot_plan = refined_plan.get("shot_plan", shot_plan)
    refined_seq_graph = refined_plan.get("sequence_graph", seq_graph)

    # 将优化后的 shot_plan 和 sequence_graph 注入 director prompt 可见的 formatted_plan
    if refined_shot_plan.get("formatted_plan"):
        print(f"    [*] Shot plan refined by studio team")
    if refined_seq_graph.get("formatted_graph"):
        print(f"    [*] Sequence graph refined by editor agent")

    # 大 state 瘦身：studio_result 是嵌套 3 Agent 结果 + 大报告字符串，若直接随 state
    # 入库会显著拉长 checkpointer 持锁时长、加剧死锁。改为落盘存路径指针，state 仅留
    # 摘要级字段（共识度/修改数/冲突数）+ 路径，下游需要时按路径读回。
    studio_output_path = _persist_studio_output(studio_result, idx)
    studio_summary = {
        "consensus": studio_result.get("studio_consensus"),
        "modification_count": len(studio_result.get("modifications", [])),
        "debate_count": len(studio_result.get("debate", [])),
        "studio_output_path": studio_output_path,
    }

    return {
        "studio_output": None,             # 剥离大对象，避免随 state 入库
        "studio_output_path": studio_output_path,
        "studio_summary": studio_summary,
        "shot_plan": refined_shot_plan,
        "sequence_graph": refined_seq_graph,
    }


# ================= P2-1: Narrative Arc Builder =================
def _build_narrative_arc(scenes: List[Dict]) -> Dict[str, Any]:
    """
    从 showrunner 生成的 scenes 自动推导全局叙事弧。
    不需要额外的 LLM 调用——纯规则驱动。

    输出结构：
    {
        "emotion_trajectory": [(index, energy_level, label), ...],
        "scene_functions": ["opening", "building", ...],
        "causal_chain": ["Scene 1: ...", "Scene 2: ...", ...],
        "missing_beats": [...],   # 三幕弧线缺失的环节（空=完整）
        "formatted_block": str,  # 直接注入 director prompt 的格式化文本
    }
    """
    n = len(scenes)
    if n == 0:
        return {"formatted_block": "", "missing_beats": [], "scene_functions": []}

    # ── 1. 为每个场景分配叙事功能 ──
    # 优先使用 showrunner 显式给出的 beat（真实情节功能），否则回退到序号规则
    _VALID_BEATS = {"setup", "inciting", "develop", "turn", "climax", "fallout", "resolution"}
    scene_functions = []
    used_beats = set()
    for i, scene in enumerate(scenes):
        raw_beat = (scene.get("beat") or "").strip().lower()
        if raw_beat in _VALID_BEATS:
            scene_functions.append(raw_beat)
            used_beats.add(raw_beat)
        else:
            # 回退规则：保证每个镜头都有明确叙事功能，避免全片平铺直叙
            if i == 0:
                scene_functions.append("setup")
            elif i == n - 1:
                scene_functions.append("resolution")
            elif i == n - 2 and n > 3:
                scene_functions.append("climax")
            elif n > 4 and i == (n // 2):
                scene_functions.append("turn")
            else:
                scene_functions.append("develop")

    # ── 2. 情绪轨迹（叙事功能 → 能量值）──
    _ARC_ENERGY = {
        "setup": 0.4,
        "inciting": 0.55,
        "develop": 0.6,
        "turn": 0.8,
        "climax": 0.95,
        "fallout": 0.7,
        "resolution": 0.5,
    }
    emotion_trajectory = []
    for i, func in enumerate(scene_functions):
        energy = _ARC_ENERGY.get(func, 0.5)
        emotion_trajectory.append((i + 1, energy, func))

    # ── 3. 因果链（承接上一镜的可见动作 + 本镜动作，保证镜头间连贯）──
    causal_chain = []
    for i, scene in enumerate(scenes):
        script = scene.get("script", "")[:80]
        func = scene_functions[i]
        transition = scene.get("transition_in", "")
        action = scene.get("action_beat", "")
        if i == 0:
            causal_chain.append(f"Scene 1 [{func}]: {script}")
        else:
            causal_chain.append(
                f"Scene {i+1} [{func}]: 承接上一镜「{transition}」→ 本镜动作「{action}」"
            )

    # ── 4. 格式化输出块 ──
    func_lines = []
    for i, func in enumerate(scene_functions):
        script_preview = scenes[i].get("script", "")[:50]
        func_lines.append(f"    Scene {i+1} ({func}): \"{script_preview}...\"")
    # ── 5. 完整故事三幕弧线校验（借鉴三幕式 beat sheet）──
    # 一个完整故事需覆盖：起(setup/inciting) → 承(develop) → 转(turn) → 合(climax) → 收(resolution)
    _arc_required = {
        "setup": ("setup", "inciting"),
        "develop": ("develop",),
        "turn": ("turn",),
        "climax": ("climax",),
        "resolution": ("resolution", "fallout"),
    }
    _missing = []
    for label, keys in _arc_required.items():
        if not (used_beats & set(keys)):
            _missing.append(label)
    if _missing and n > 1:
        func_lines.append(
            f"    [!] 故事不完整：缺少三幕弧线上的环节 {_missing}。"
            f"请补出对应镜头，使其构成 起(setup/inciting)→承(develop)→转(turn)→合(climax)→收(resolution) 的完整闭环。"
        )

    energy_curve = " -> ".join(f"{e:.1f}" for _, e, _ in emotion_trajectory)

    formatted = (
        "[Narrative Arc — Each scene must serve the overall story progression]\n"
        f"    Emotion curve: {energy_curve}\n"
        "    Scene functions:\n"
        + "\n".join(func_lines)
        + "\n    Causal chain:\n"
        + "\n".join(f"    -> {c}" for c in causal_chain)
    )

    return {
        "emotion_trajectory": emotion_trajectory,
        "scene_functions": scene_functions,
        "causal_chain": causal_chain,
        "missing_beats": _missing if n > 1 else [],
        "formatted_block": formatted,
    }


# ================= 节点定义 =================
def showrunner_node(state: MultimediaState):
    print("\n--- 👑[节点1: 总导演] 正在拆解长视频分镜剧本 ---")

    # 运行指标埋点：开启一条 run record（可观测性）
    run_metrics.start_run(state.get("task", ""), len(state.get("scenes", [])))

    # 清空跨运行累积的风格记忆单例，防止脏数据污染 Global Film Brain 报告
    reset_memory()

    # 抽取视觉风格（LLM 驱动，从 task+global_setting+story 识别；存入 state 供下游复用）
    style = _resolve_style(state)
    style_key = style["style_key"]
    state["visual_style"] = style_key
    state["style_suffix"] = style["style_suffix"]
    state["style_description"] = style["style_description"]
    style_context = _build_style_context(style_key)
    print(f"    [*] 检测到视觉风格: {style_key}（{style['style_description']}）")
    if style_key == "neutral":
        print("    [WARN] 未能从输入识别出明确视觉风格，已用中性兜底（非写实 CG）。"
              " 若需指定风格，请在 task / global_setting 中描述（如「水彩」「赛博朋克 CG」）。")

    # 检测内容类型
    content_type = _detect_content_type(state["task"])
    print(f"    [*] 检测到内容类型: {content_type}")

    prompt = SHOWRUNNER_PROMPT.format(
        task=state["task"],
        style_context=style_context,
    )

    # ── 故事完整性自愈循环（借鉴 LangGraph 多智能体 Reviewer 校验→重跑）──
    # 若总导演生成的剧本缺三幕弧线环节（如缺 turn/climax/resolution），
    # 把缺失环节作为补全指令追加进 prompt 重新生成，最多重试 2 次，保证"情节一定完整"。
    raw_scenes = []
    global_setting = "无全局设定"
    _MAX_SHOWRUNNER_RETRIES = 2
    for _attempt in range(_MAX_SHOWRUNNER_RETRIES + 1):
        response = call_llm(prompt, "总导演")
        parsed_data = safe_parse_json(response)
        global_setting = parsed_data.get("global_setting", global_setting)
        raw_scenes = parsed_data.get("scenes", [])
        if not raw_scenes:
            print("    [!] 总导演未返回有效 scenes，重试...")
            continue
        # 用已解析的 scene 字段做三幕弧线校验
        _probe = [{
            "script": s.get("script", s.get("description", "")),
            "beat": s.get("beat", ""),
        } for s in raw_scenes]
        _arc = _build_narrative_arc(_probe)
        _missing = _arc.get("missing_beats", [])
        if _missing:
            print(f"    [!] 故事不完整，缺三幕环节 {_missing}，要求总导演补全后重试 "
                  f"({_attempt+1}/{_MAX_SHOWRUNNER_RETRIES})...")
            prompt = SHOWRUNNER_PROMPT.format(
                task=state["task"],
                style_context=style_context,
            ) + (
                f"\n\n[校验反馈-必须修正] 你刚才的剧本不完整：缺少三幕弧线上的环节 {_missing}。"
                f"请在不删减现有镜头的前提下，补出对应镜头，使其构成 "
                f"起(setup/inciting)→承(develop)→转(turn)→合(climax)→收(resolution) 的完整闭环，"
                f"并重写完整 JSON。"
            )
            continue
        break  # 完整，跳出重试

    print(f"[*] 成功生成全局设定: {global_setting[:60]}...")
    print(f"[*] 成功拆解出 {len(raw_scenes)} 个连续镜头！")

    safety_check_prompt = (
        SAFETY_PROMPT
        + f"\n\n待检查内容：\n全局视觉设定：{global_setting}\n\n分镜列表：\n"
        + json.dumps([s.get("script", "") for s in raw_scenes], ensure_ascii=False, indent=2)
    )
    safety_result = call_llm(safety_check_prompt, "安全审核员")
    if "FAIL" in safety_result.upper():
        print(f"    ❌ 初始内容安全审核失败: {safety_result}")
        raise Exception(f"内容安全违规: {safety_result}")
    print("    ✅ 初始内容安全审核通过")

    processed_scenes =[]
    for i, s in enumerate(raw_scenes):
        raw_script = s.get("script", s.get("description", ""))
        # ── 兜底清洗：剥离 LLM 可能混入的英文风格/质量后缀（如 "high-quality anime..."），
        #    避免污染导演生图提示词，并与统一 style_suffix 冲突 ──
        raw_script = _sanitize_script(raw_script)
        scene_item = _scene_base(raw_script)
        # ── 情节密度字段：从总导演的结构化输出中提取，强化首尾帧可见内容 ──
        scene_item["scale"] = s.get("scale", "")
        scene_item["camera_note"] = s.get("camera_note", "")
        scene_item["action_beat"] = s.get("action_beat", "")
        ve = s.get("visual_elements", [])
        scene_item["visual_elements"] = ve if isinstance(ve, list) else []
        scene_item["emotion"] = s.get("emotion", "")
        scene_item["beat"] = s.get("beat", "")
        scene_item["transition_in"] = s.get("transition_in", "")
        processed_scenes.append(scene_item)
        print(f"    Shot {i+1} script ready: {scene_item['script'][:60]}...")
        if scene_item["action_beat"]:
            print(f"         ↳ 可见动作: {scene_item['action_beat'][:50]}...")
        if scene_item["visual_elements"]:
            print(f"         ↳ 视觉细节: {len(scene_item['visual_elements'])} 项")

    # ── P2-1: Build narrative arc from scenes (rule-based, no LLM call) ──
    narrative_arc = _build_narrative_arc(processed_scenes)
    print(f"    [*] Narrative arc built: {len(processed_scenes)} scenes, "
          f"emotion curve: {' -> '.join(f'{e:.1f}' for _, e, _ in narrative_arc.get('emotion_trajectory', []))}")

    return {
        "global_setting": global_setting,
        "scenes": processed_scenes,
        "current_scene_index": 0,
        "visual_style": style_key,
        "content_type": content_type,
        "use_first_last_frame": True,
        "character_portrait_url": None,
        "narrative_arc": narrative_arc,
        "reference_images": [],
        "reference_embeddings":[],
        "visual_context": None,
        "shot_plan": None,
        "sequence_graph": None,
        "critic_eval": None,
        "quality_gates": None,
        "rewrite_count": 0,
        "studio_output": None,
        "rag_context": None,
        "aborted": False,
        "abort_reason": None,
        "error_log": None,
    }


def showrunner_review_gate_node(state: MultimediaState):
    payload = {
        "stage": "showrunner_review",
        "title": "总导演人工审片",
        "task": state["task"],
        "global_setting": state["global_setting"],
        "scenes": state["scenes"],
        "message": "请审查总导演输出的全局设定与分镜列表。可直接通过，也可修改 global_setting / scenes 后再继续。",
        "actions":["approve", "rewrite", "edit_prompt"],
    }
    decision = _normalize_decision(interrupt(payload), gate="showrunner_review")

    scenes = state["scenes"]
    global_setting = state["global_setting"]
    task = state["task"]

    if decision.get("task") is not None:
        task = str(decision["task"]).strip() or task
    if decision.get("global_setting") is not None:
        global_setting = str(decision["global_setting"]).strip() or global_setting
    if decision.get("scenes") is not None:
        scenes = _normalize_scenes(decision["scenes"])

    # ── P1-1: Character Reference Portrait (lightweight Character ID) ──
    # After user approves the showrunner output, generate a standard character portrait.
    # This portrait is used as a low-strength auxiliary reference for all subsequent keyframes,
    # anchoring character identity across scenes (skin tone, face shape, clothing, equipment).
    character_portrait_url = state.get("character_portrait_url")
    if not character_portrait_url:
        try:
            portrait_prompt = (
                f"A standard full-body character reference portrait, front-facing, "
                f"neutral studio lighting, clean gradient background. Character: {global_setting}"
            )
            print("    [*] Generating character reference portrait...")
            character_portrait_url = generate_keyframe(
                portrait_prompt,
                size="1024*1024",
            )
            print(f"    [OK] Character portrait generated: {character_portrait_url[-50:]}...")
        except Exception as e:
            print(f"    [WARN] Character portrait generation failed (non-blocking): {e}")
            character_portrait_url = None

    return {
        "task": task,
        "global_setting": global_setting,
        "scenes": scenes,
        "character_portrait_url": character_portrait_url,
        "visual_style": state.get("visual_style"),
        "style_suffix": state.get("style_suffix"),
        "style_description": state.get("style_description"),
    }


def reference_gen_node(state: MultimediaState):
    """
    Reference Sheet Generation Node
    ================================
    在关键帧生成之前，为角色（4视图转面图）、道具（多角度设定图）、
    场景环境（建立镜头）生成一致性参考图。

    架构位置: showrunner_review → 【reference_gen_node】→ visual_context_builder
    """
    print("--- [Reference] Generating consistency reference sheets ---")

    global_setting = state.get("global_setting", "")
    scenes = state.get("scenes", [])
    task = state.get("task", "")
    style = _resolve_style(state)
    style_key = style["style_key"]
    style_suffix = style["style_suffix"]

    try:
        sheets = generate_reference_sheets(
            global_setting=global_setting,
            scenes=scenes,
            task=task,
            style_suffix=style_suffix,
            style_key=style_key,
        )
    except Exception as e:
        print(f"    [WARN] Reference sheet generation failed (non-blocking): {e}")
        sheets = {"characters": {}, "props": {}, "environments": {}}

    # 从角色转面图中提取第一张作为 character_portrait_url（向后兼容）
    character_portrait_url = state.get("character_portrait_url")
    if sheets.get("characters"):
        first_char = next(iter(sheets["characters"].values()))
        if first_char and first_char.get("url"):
            character_portrait_url = first_char["url"]
            print(f"    [OK] character_portrait_url updated from turnaround sheet")

    # 统计
    n_chars = len(sheets.get("characters", {}))
    n_props = len(sheets.get("props", {}))
    n_envs = len(sheets.get("environments", {}))
    print(f"    [Reference] Generated: {n_chars} characters, {n_props} props, {n_envs} environments")

    return {
        "reference_sheets": sheets,
        "character_portrait_url": character_portrait_url,
    }


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


def director_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    global_setting = state.get("global_setting", "")
    print(f"\n--- 🎬 [镜头 {idx+1}] 导演执行拍摄方案 ---")

    # Phase 2: 从 state 消费镜头拍摄方案
    shot_plan = state.get("shot_plan")
    if shot_plan and shot_plan.get("formatted_plan"):
        shot_plan_text = shot_plan["formatted_plan"]
        hero_idx = shot_plan.get("hero_shot_index", 0)
        hero_name = shot_plan["shots"][hero_idx]["shot_name"]
        print(f"    [*] 已注入 Shot Plan (hero=#{hero_idx+1} \"{hero_name}\")")
    else:
        # Phase 1 兜底：如果 Phase 2 未产出，降级使用 visual_context
        visual_ctx = state.get("visual_context")
        if visual_ctx and visual_ctx.get("formatted_prompt_block"):
            shot_plan_text = visual_ctx["formatted_prompt_block"]
            print("    [!] Shot Plan 缺失，降级使用 Visual Context")
        else:
            shot_plan_text = "No shot plan available. Proceed with standard cinematic quality."
            print("    [!] Shot Plan 和 Visual Context 均缺失，使用默认模式")

    # Phase 3: 从 state 消费序列编排图
    seq_graph = state.get("sequence_graph")
    if seq_graph and seq_graph.get("director_context"):
        sequence_context_text = seq_graph["director_context"]
        print(f"    [*] 已注入 Sequence Context (connections={len(seq_graph.get('connections', []))})")
    else:
        sequence_context_text = "No sequence context available."

    # Phase RAG: 从 state 消费 RAG 检索结果
    rag_ctx = state.get("rag_context")
    if rag_ctx and rag_ctx.get("formatted_rag_block"):
        rag_context_text = rag_ctx["formatted_rag_block"]
        print(f"    [*] 已注入 RAG Context ({rag_ctx.get('total_retrieved', 0)} refs)")
    else:
        rag_context_text = "[RAG Reference — No references available (knowledge base empty or retrieval disabled)]"

    # P2-1: Narrative Arc — 本场景在全局叙事弧中的位置
    narrative_arc = state.get("narrative_arc")
    if narrative_arc and narrative_arc.get("formatted_block"):
        narrative_arc_text = narrative_arc["formatted_block"]
        scene_func = narrative_arc.get("scene_functions", [])[idx] if idx < len(narrative_arc.get("scene_functions", [])) else "unknown"
        print(f"    [*] Narrative arc: this scene is [{scene_func}] in the story")
    else:
        narrative_arc_text = "[Narrative Arc — Not available (narrative planning disabled)]"

    critique = scene.get("critique", "无")
    protected_specs_text = _extract_protected_specs(scene["script"], global_setting)

    # ── 角色 Visual DNA 锁：把累积的角色外观特征强制并入受保护规格 ──
    # 借鉴 ViMax/ArcReel 的「角色视觉 DNA」思想：跨镜头单调并锁定外貌，
    # 防止生图模型在相邻镜头里把角色发型/服饰/五官悄悄改写。
    character_identity = (state.get("scene_anchors") or {}).get("character_identity") or {}
    dna_appearance = character_identity.get("appearance") or []
    dna_subject = character_identity.get("subject", "")
    if dna_appearance:
        dna_block = "；".join(dna_appearance)
        if dna_block not in protected_specs_text:
            protected_specs_text = (
                protected_specs_text.rstrip("。") + f"；【角色视觉DNA-必须严格保持一致】{dna_subject}：{dna_block}。"
            )
            print(f"    [*] 角色视觉DNA锁注入：{dna_subject}（{len(dna_appearance)} 项外观）")

    # ── 环境锚点（防止跨场景环境漂移）──
    environment_anchor = _extract_environment_anchors(global_setting, scene["script"])
    if environment_anchor:
        print(f"    [*] Environment anchor injected")

    # ── 情节密度字段（Phase 0 结构化输出，强化首帧可见内容）──
    action_beat = scene.get("action_beat", "")
    visual_elements = scene.get("visual_elements", []) or []
    visual_elements_block = "\n".join(f"- {v}" for v in visual_elements) if visual_elements else "（无额外细节，请从 script 与环境锚点中自行推断具体可见细节）"
    camera_note = scene.get("camera_note", "")
    emotion = scene.get("emotion", "")
    transition_in = scene.get("transition_in", "")

    # 风格上下文（统一从 state 解析）
    style = _resolve_style(state)
    style_key = style["style_key"]
    style_context = _build_style_context(style_key)
    style_suffix = style["style_suffix"]

    prompt = DIRECTOR_SYSTEM_PROMPT.format(
        global_setting=global_setting,
        script=scene["script"],
        critique=critique,
        scene_index=idx + 1,
        shot_plan=shot_plan_text,
        sequence_context=sequence_context_text,
        rag_context=rag_context_text,
        narrative_arc=narrative_arc_text,
        protected_specs=protected_specs_text,
        environment_anchor=environment_anchor,
        style_context=style_context,
        style_suffix=style_suffix,
        action_beat=action_beat or "（无，请从 script 中提取主体可见动作）",
        visual_elements_block=visual_elements_block,
        camera_note=camera_note or "（未指定）",
        emotion=emotion or "（未指定）",
        transition_in=transition_in or "（开场镜头，无需承接）",
    )
    image_prompt = call_llm(prompt, "导演")

    # 安全网：截掉 director 可能附加的多镜头序列列表
    _preset = _get_style_preset(state)
    _style_key = _resolve_style(state)["style_key"]
    image_prompt = _sanitize_single_frame_prompt(
        image_prompt, _preset["suffix"], _get_conflict_keywords(_style_key)
    )
    # 翻译层：生图/视频模型对英文 prompt 出图质量显著更高，将中文叙事翻译为英文再喂模型
    image_prompt = _translate_to_english(image_prompt, purpose="image")

    scenes = state["scenes"].copy()
    scenes[idx]["image_prompt"] = image_prompt
    return {"scenes": scenes}


def prompt_preview_node(state: MultimediaState):
    """
    Prompt Preview Gate — 在生成图片之前让用户审核 director 生成的 image_prompt。
    用户可以：
    - approve: 直接使用该 prompt 生成图片
    - edit_prompt: 使用用户编辑后的 prompt 直接生成图片（跳过重新生成）
    - rewrite: 让 director 重新生成 prompt
    """
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]

    payload = {
        "stage": "prompt_preview",
        "title": f"镜头 {idx + 1} Prompt 预览（生图前审核）",
        "scene_index": idx + 1,
        "script": scene["script"],
        "global_setting": state.get("global_setting", ""),
        "image_prompt": scene.get("image_prompt", ""),
        "message": "请审核 director 生成的 image_prompt。确认无误后点击「生成图片」，或编辑/重写 prompt。",
        "actions": ["approve", "edit_prompt", "rewrite"],
    }

    decision = _normalize_decision(interrupt(payload), gate="prompt_preview")
    action = decision.get("action", "approve")

    scenes = state["scenes"].copy()

    # 如果用户编辑了 prompt，直接替换
    if decision.get("image_prompt") is not None and action == "edit_prompt":
        edited = str(decision["image_prompt"]).strip()
        # 编辑后也走安全网
        _preset = _get_style_preset(state)
        _style_key = _resolve_style(state)["style_key"]
        edited = _sanitize_single_frame_prompt(
            edited, _preset["suffix"], _get_conflict_keywords(_style_key)
        )
        scenes[idx]["image_prompt"] = edited
        scenes[idx]["is_perfect"] = True  # 用户手动确认的，直接通过
        print(f"    [*] 用户编辑了 image_prompt ({len(edited)} chars)")
        return {"scenes": scenes}

    if action == "approve":
        scenes[idx]["is_perfect"] = True
        print(f"    [*] 用户批准了 image_prompt")
        return {"scenes": scenes}

    # rewrite: 回到 director 重新生成
    scenes[idx]["is_perfect"] = False
    scenes[idx]["critique"] = decision.get("reason") or "User requested prompt rewrite"
    print(f"    [*] 用户要求重写 image_prompt")
    return {"scenes": scenes}


def end_frame_prompt_preview_node(state: MultimediaState):
    """
    End Frame Prompt Preview Gate — 在生成尾帧图片之前让用户审核 end_frame_director 生成的 last_image_prompt。
    用户可以：
    - approve: 直接使用该 prompt 生成尾帧图片
    - edit_prompt: 使用用户编辑后的 prompt 直接生成尾帧图片（跳过重新生成）
    - rewrite: 让 end_frame_director 重新生成 prompt
    """
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]

    payload = {
        "stage": "end_frame_prompt_preview",
        "title": f"镜头 {idx + 1} 尾帧 Prompt 预览（生图前审核）",
        "scene_index": idx + 1,
        "script": scene["script"],
        "global_setting": state.get("global_setting", ""),
        "image_prompt": scene.get("last_image_prompt", ""),
        "reference_image_url": scene.get("image_url", ""),  # 首帧作为参考
        "message": "请审核 end_frame_director 生成的尾帧 prompt。确认无误后点击「生成尾帧」，或编辑/重写 prompt。",
        "actions": ["approve", "edit_prompt", "rewrite"],
    }

    decision = _normalize_decision(interrupt(payload), gate="end_frame_prompt_preview")
    action = decision.get("action", "approve")

    scenes = state["scenes"].copy()

    # 如果用户编辑了 prompt，直接替换
    if decision.get("image_prompt") is not None and action == "edit_prompt":
        edited = str(decision["image_prompt"]).strip()
        _preset = _get_style_preset(state)
        _style_key = _resolve_style(state)["style_key"]
        edited = _sanitize_single_frame_prompt(
            edited, _preset["suffix"], _get_conflict_keywords(_style_key)
        )
        scenes[idx]["last_image_prompt"] = edited
        scenes[idx]["last_image_is_perfect"] = True
        print(f"    [*] 用户编辑了 last_image_prompt ({len(edited)} chars)")
        return {"scenes": scenes}

    if action == "approve":
        scenes[idx]["last_image_is_perfect"] = True
        print(f"    [*] 用户批准了 last_image_prompt")
        return {"scenes": scenes}

    # rewrite: 回到 end_frame_director 重新生成
    scenes[idx]["last_image_is_perfect"] = False
    scenes[idx]["last_image_critique"] = decision.get("reason") or "User requested end-frame prompt rewrite"
    print(f"    [*] 用户要求重写 last_image_prompt")
    return {"scenes": scenes}


def _match_reference_elements(reference_sheets: dict, script_text: str) -> dict:
    """匹配 reference sheets 中与当前剧本相关的角色/道具/环境。

    Returns:
        {
            "char_url": str or None,
            "char_desc": str,
            "prop_desc": str,
            "env_url": str or None,
            "matched_char_name": str,
            "matched_prop_name": str,
        }
    """
    result = {"char_url": None, "char_desc": "", "prop_desc": "",
              "env_url": None, "matched_char_name": "", "matched_prop_name": ""}

    def _check_match(name: str, data: dict) -> bool:
        """通用名称匹配：支持 name_cn + name + 单词级 fallback。"""
        name_cn = data.get("name_cn", "") if isinstance(data, dict) else ""
        names = [n for n in [name, name_cn] if n]
        if any(n.lower() in script_text for n in names):
            return True
        return any(
            word in script_text
            for n in names
            for word in n.lower().replace("_", " ").split()
            if len(word) > 1
        )

    # 角色匹配
    for char_name, char_data in reference_sheets.get("characters", {}).items():
        if _check_match(char_name, char_data):
            result["char_url"] = char_data.get("url") if isinstance(char_data, dict) else char_data
            result["char_desc"] = char_data.get("description", "") if isinstance(char_data, dict) else ""
            result["matched_char_name"] = char_name
            break

    # 道具匹配
    for prop_name, prop_data in reference_sheets.get("props", {}).items():
        if _check_match(prop_name, prop_data):
            result["prop_desc"] = prop_data.get("description", "") if isinstance(prop_data, dict) else ""
            result["matched_prop_name"] = prop_name
            break

    # 环境匹配
    for env_name, env_data in reference_sheets.get("environments", {}).items():
        if _check_match(env_name, env_data):
            result["env_url"] = env_data.get("url") if isinstance(env_data, dict) else env_data
            break

    return result


def image_gen_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"--- [Art] [Shot {idx+1}] Rendering keyframe ---")

    scenes = state["scenes"].copy()

    # ── 选择参考图（reference image）用于 img2img 一致性控制 ──
    # 优先级: 上一场景尾帧(FLF连续性) > 匹配角色参考 > 环境参考 > 无参考
    #
    # FLF 模式核心理念（参考 CoAgent GCM + Script-is-All-You-Need 帧锚定方案）:
    #   prev_end_frame 作为 img2img 参考图(ref_strength=0.5) -> 导演 prompt 主导构图
    #   -> end_frame_gen 从 keyframe 生成尾帧 -> kf2v(keyframe, end_frame) 渲染视频
    # 这样既保持场景间像素级连续性，又不放弃导演对每个镜头起始构图的精确控制
    reference_url = None
    ref_strength = 1.0

    use_flf = state.get("use_first_last_frame", False)
    scene_anchors = state.get("scene_anchors") or {}
    prev_end_frame = scene_anchors.get("last_frame_url") if idx > 0 else None

    reference_sheets = state.get("reference_sheets") or {}
    script_text = scene.get("script", "").lower()

    # 源头角色肖像（showrunner 通过后生成的标准角色 ID 基准）。
    # 关键：所有场景对齐"同一个源头基准"，抑制链式(逐场景)传递导致的角色漂移(drift)。
    character_portrait_url = state.get("character_portrait_url")

    # ── 使用 helper 匹配当前场景涉及的角色/道具/环境 ──
    ref_match = _match_reference_elements(reference_sheets, script_text)
    matched_char_url = ref_match["char_url"]
    matched_char_desc = ref_match["char_desc"]
    matched_prop_desc = ref_match["prop_desc"]
    if ref_match["matched_char_name"]:
        print(f"    [*] Matched character reference: {ref_match['matched_char_name']}")
    if ref_match["matched_prop_name"]:
        print(f"    [*] Matched prop reference: {ref_match['matched_prop_name']}")

    if use_flf and prev_end_frame and idx > 0:
        # FLF 模式: 上一场景尾帧 -> img2img 参考 (ref_strength=0.5)
        # 0.5 的平衡点: 保持场景间视觉连续性，同时让导演 prompt 主导构图和景别
        reference_url = prev_end_frame
        ref_strength = 0.5
        print(f"    [*] FLF mode: using prev scene end frame as img2img reference (ref_strength={ref_strength})")
    elif matched_char_url:
        # 匹配到的角色参考图（比固定第一个角色更精准）
        reference_url = matched_char_url
        ref_strength = 0.5
        print(f"    [*] Using matched character reference (ref_strength={ref_strength})")
    elif character_portrait_url:
        # 源头基准兜底：无 FLF 尾帧 / 无匹配角色参考时，用源头角色肖像做 img2img，
        # 保证角色身份对齐源头基准，而非退化为无参考(漂移风险最高)。
        reference_url = character_portrait_url
        ref_strength = 0.4
        print(f"    [*] Using source character portrait as anchor reference (ref_strength={ref_strength})")
    elif ref_match["env_url"]:
        # Fallback: 环境参考图
        reference_url = ref_match["env_url"]
        ref_strength = 0.35
        print(f"    [*] Using environment reference (ref_strength={ref_strength})")

    # ── 文本增强：在 prompt 中注入角色/道具描述，强化一致性 ──
    enhanced_prompt = scene["image_prompt"]
    identity_anchors = []
    if matched_char_desc:
        identity_anchors.append(f"[Character Identity Anchor: {matched_char_desc}]")
    if matched_prop_desc:
        identity_anchors.append(f"[Prop Identity Anchor: {matched_prop_desc}]")
    # 角色肖像文本级恒定锚点：即便本帧参考图用的是"上一场景尾帧"(链式)，
    # 也始终提示模型对齐源头角色外观，进一步抑制跨场景漂移。
    if character_portrait_url and reference_url != character_portrait_url:
        identity_anchors.append(
            "[Character Consistency: keep the SAME character identity as the "
            "established source portrait — same face, hairstyle, outfit, and equipment.]"
        )
    if identity_anchors:
        enhanced_prompt = enhanced_prompt + "\n" + "\n".join(identity_anchors)
        print(f"    [*] Injected {len(identity_anchors)} identity anchor(s) into prompt")

    try:
        if reference_url:
            image_url = generate_keyframe(enhanced_prompt,
                                          reference_image_url=reference_url,
                                          ref_strength=ref_strength)
        else:
            image_url = generate_keyframe(enhanced_prompt)
    except Exception as e:
        # 生图失败（网络瞬断/接口错误）不再直接 FATAL，改为计入失败计数，
        # 交由路由节点按 MAX_VIDEO_GEN_FAILURES 决定重试或跳过该镜头。
        print(f"    ❌ [Art] [Shot {idx+1}] 关键帧生图失败: {e}")
        scenes[idx]["video_gen_failures"] = scenes[idx].get("video_gen_failures", 0) + 1
        return {"scenes": scenes}

    scenes[idx]["image_url"] = image_url
    scenes[idx]["iterations"] += 1
    return {"scenes": scenes}


def reviewer_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"--- 🧐 [镜头 {idx+1}] 艺术总监 + 一致性引擎审核 ---")

    # 【修复】显式获取上一镜头的图片，优先取尾帧
    previous_image_url = None
    if idx > 0:
        prev_scene = state["scenes"][idx - 1]
        previous_image_url = prev_scene.get("last_image_url") or prev_scene.get("image_url")
        if previous_image_url:
            print(f"    [*] 正在对比上一镜头关键帧: {previous_image_url[-30:]}...")

    similarity_score = None
    if previous_image_url and compute_similarity is not None:
        try:
            similarity_score = compute_similarity(scene["image_url"], previous_image_url)
            print(f"    [*] embedding 相似度(相邻镜头): {similarity_score:.4f}")
        except Exception as e:
            print(f"    ⚠️ embedding 计算失败，继续人工审核: {str(e)}")

    # ── 角色一致性闭环度量：与"源头角色肖像"的相似度 ──
    # 源头肖像是全局角色 ID 基准，用它度量角色漂移比"与上一镜头比"更抗累积误差。
    portrait_similarity = None
    portrait_url = state.get("character_portrait_url")
    if portrait_url and compute_similarity is not None:
        try:
            portrait_similarity = compute_similarity(scene["image_url"], portrait_url)
            print(f"    [*] embedding 相似度(源头肖像): {portrait_similarity:.4f}")
        except Exception as e:
            print(f"    ⚠️ 肖像相似度计算失败(不阻断): {str(e)}")

    # 连贯性可观测分数：相邻镜头相似度 + 源头肖像相似度（任一可缺失）
    continuity_score = {
        "neighbor_similarity": similarity_score,
        "portrait_similarity": portrait_similarity,
    }

    # 预注入风格质量标准（evaluate_image 内部会继续 format script）
    style_key = state.get("visual_style") or _DEFAULT_STYLE_KEY
    quality_bar = _STYLE_QUALITY_BAR.get(style_key, _STYLE_QUALITY_BAR[_DEFAULT_STYLE_KEY])
    reviewer_prompt = REVIEWER_SYSTEM_PROMPT.replace("{quality_bar}", quality_bar)

    feedback = evaluate_image(
        scene["image_url"],
        scene["script"],
        reviewer_prompt,
        previous_image_url=previous_image_url
    )

    scenes = state["scenes"].copy()
    scenes[idx]["image_auto_feedback"] = feedback
    scenes[idx]["embedding_similarity"] = similarity_score
    scenes[idx]["portrait_similarity"] = portrait_similarity
    scenes[idx]["continuity_score"] = continuity_score
    scenes[idx]["critique"] = feedback
    scenes[idx]["is_perfect"] = _is_review_pass(feedback)

    return {"scenes": scenes}


def image_review_gate_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]

    # 【修复】显式获取上一镜头的图片
    previous_image_url = None
    if idx > 0:
        prev_scene = state["scenes"][idx - 1]
        previous_image_url = prev_scene.get("last_image_url") or prev_scene.get("image_url")

    payload = {
        "stage": "image_review",
        "title": f"镜头 {idx + 1} 关键帧审片",
        "scene_index": idx + 1,
        "script": scene["script"],
        "global_setting": state.get("global_setting", ""),
        "image_prompt": scene.get("image_prompt", ""),
        "image_url": scene.get("image_url", ""),
        "reference_image_url": previous_image_url,
        "embedding_similarity": scene.get("embedding_similarity"),
        "portrait_similarity": scene.get("portrait_similarity"),
        "continuity_score": scene.get("continuity_score"),
        "auto_feedback": scene.get("image_auto_feedback", scene.get("critique", "")),
        "message": "请审核关键帧：可通过、重写，或修改 image_prompt 后继续。",
        "actions":["approve", "rewrite", "edit_prompt"],
    }

    decision = _normalize_decision(interrupt(payload), gate="image_review")
    action = decision.get("action", "approve")

    scenes = state["scenes"].copy()

    if decision.get("image_prompt") is not None:
        scenes[idx]["image_prompt"] = str(decision["image_prompt"]).strip()

    if action in APPROVE_ACTIONS:
        scenes[idx]["is_perfect"] = True
        scenes[idx]["critique"] = "无"

        ref_images = state.get("reference_images",[]).copy()
        ref_images.append(scenes[idx]["image_url"])

        updates = {
            "scenes": scenes,
            "reference_images": ref_images,
        }

        if get_image_embedding is not None:
            try:
                ref_embs = state.get("reference_embeddings",[]).copy()
                ref_embs.append(get_image_embedding(scenes[idx]["image_url"]))
                updates["reference_embeddings"] = ref_embs
            except Exception as e:
                print(f"    ⚠️ 记录 embedding 失败，但不影响主流程: {str(e)}")

        return updates

    scenes[idx]["is_perfect"] = False
    scenes[idx]["critique"] = decision.get("reason") or scene.get("image_auto_feedback") or "Human requested rewrite"
    return {"scenes": scenes}


# ================= 新增：尾帧相关节点 =================
def end_frame_director_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"\n--- 🎬 [镜头 {idx+1}] 导演执行【尾帧】拍摄方案 ---")

    # Phase 2: 从 state 消费镜头拍摄方案中的尾帧 shot
    shot_plan = state.get("shot_plan")
    if shot_plan and shot_plan.get("shots"):
        end_idx = shot_plan.get("end_shot_index", len(shot_plan["shots"]) - 1)
        end_shot = shot_plan["shots"][end_idx]
        # 构建尾帧专用的 shot spec 块
        cam = end_shot["camera"]
        light = end_shot["lighting"]
        end_shot_text = (
            f"[End Shot: \"{end_shot['shot_name']}\"]\n"
            f"  Camera: {cam['type']}, {cam['movement']}, {cam['angle']}, {cam['lens']}\n"
            f"  Orientation: {cam.get('orientation', 'three_quarter')}\n"
            f"  Tracking: {cam.get('tracking', 'static_observe')}\n"
            f"  Lighting: {light.get('full_description', light['setup'])}\n"
            f"  Emotion target: {end_shot['emotion']}\n"
            f"  Focus object: {end_shot.get('focus_object', 'character')}\n"
            f"  Lens: {end_shot.get('lens_feel', 'standard')}"
        )
        print(f"    [*] 已注入 End Shot (#{end_idx+1} \"{end_shot['shot_name']}\")")
    else:
        end_shot_text = "No end shot plan available. Design end frame based on script progression."
        print("    [!] Shot Plan 缺失，使用默认尾帧模式")

    global_setting = state.get("global_setting", "")
    protected_specs_text = _extract_protected_specs(scene["script"], global_setting)

    # ── 环境锚点（防止跨场景环境漂移）──
    environment_anchor = _extract_environment_anchors(global_setting, scene["script"])

    # ── 情节密度字段（Phase 0 结构化输出，强化尾帧可见内容）──
    action_beat = scene.get("action_beat", "")
    visual_elements = scene.get("visual_elements", []) or []
    visual_elements_block = "\n".join(f"- {v}" for v in visual_elements) if visual_elements else "（无额外细节，请从 script 与环境锚点中自行推断具体可见细节）"
    camera_note = scene.get("camera_note", "")
    emotion = scene.get("emotion", "")
    transition_in = scene.get("transition_in", "")

    # 风格上下文（统一从 state 解析）
    style = _resolve_style(state)
    style_key = style["style_key"]
    style_context = _build_style_context(style_key)
    style_suffix = style["style_suffix"]

    prompt = END_FRAME_DIRECTOR_PROMPT.format(
        global_setting=global_setting,
        script=scene["script"],
        first_frame_prompt=scene["image_prompt"],
        critique=scene.get("last_image_critique", "无"),
        scene_index=idx + 1,
        end_shot_plan=end_shot_text,
        protected_specs=protected_specs_text,
        environment_anchor=environment_anchor,
        style_context=style_context,
        style_suffix=style_suffix,
        action_beat=action_beat or "（无，请从 script 中提取动作结束状态）",
        visual_elements_block=visual_elements_block,
        camera_note=camera_note or "（未指定）",
        emotion=emotion or "（未指定）",
        transition_in=transition_in or "（开场镜头，无需承接）",
    )
    last_image_prompt = call_llm(prompt, "导演")

    # 安全网：清理序列语言、中文、强制 style suffix
    _preset = _get_style_preset(state)
    _style_key = _resolve_style(state)["style_key"]
    last_image_prompt = _sanitize_single_frame_prompt(
        last_image_prompt, _preset["suffix"], _get_conflict_keywords(_style_key)
    )
    # 翻译层：尾帧同样翻译为英文再喂生图模型
    last_image_prompt = _translate_to_english(last_image_prompt, purpose="image")

    scenes = state["scenes"].copy()
    scenes[idx]["last_image_prompt"] = last_image_prompt
    return {"scenes": scenes}


def end_frame_gen_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"--- 🎨[镜头 {idx+1}] 画师渲染【尾帧】 ---")

    # 动态计算 ref_strength：尾帧需要与首帧产生足够大的视觉差异，
    # 这样 kf2v 视频模型才有足够的插值空间来生成动态运动。
    # ref_strength 越低 → 图编模型越自由 → 尾帧与首帧差异越大 → 视频动作幅度越大
    ref_strength = 0.08  # 默认偏低，允许较大差异
    shot_plan = state.get("shot_plan", {})
    script_lower = scene.get("script", "").lower()

    # 从 shot_plan 的 pacing_weight 推断运动幅度
    if shot_plan and shot_plan.get("shots"):
        hero_idx = shot_plan.get("hero_shot_index", 0)
        pacing = shot_plan["shots"][hero_idx].get("pacing_weight", 0.25)
        if pacing >= 0.35:
            ref_strength = 0.02  # 高能量动作场景：极低参考，允许剧烈姿态变化
        elif pacing <= 0.15:
            ref_strength = 0.12  # 静态/情感场景：稍高参考，保持温和过渡
        else:
            ref_strength = 0.08  # 中等

    # 剧本关键词微调
    action_kws = ["战斗", "冲锋", "追逐", "挥剑", "爆炸", "combat", "chase", "attack", "explosion", "slash"]
    calm_kws = ["回忆", "告别", "沉默", "微笑", "祈祷", "tears", "farewell", "silent", "whisper", "hope"]
    if any(kw in script_lower for kw in action_kws):
        ref_strength = min(ref_strength, 0.03)
    elif any(kw in script_lower for kw in calm_kws):
        ref_strength = max(ref_strength, 0.12)

    # 尾帧负面提示词：抑制与首帧相同姿态的静态输出
    end_frame_negative = (
        "same pose as reference, identical composition, static, frozen, "
        "no movement, copy of reference image, unchanged position, "
        "stiff body, no expression change, duplicate"
    )

    # ── 注入 identity anchors 到尾帧 prompt，防止低 ref_strength 下角色外貌漂移 ──
    end_frame_prompt = scene["last_image_prompt"]
    reference_sheets = state.get("reference_sheets") or {}
    script_text = scene.get("script", "").lower()
    ref_match = _match_reference_elements(reference_sheets, script_text)
    end_frame_anchors = []
    if ref_match["char_desc"]:
        end_frame_anchors.append(f"[Character Identity Anchor: {ref_match['char_desc']}]")
    if ref_match["prop_desc"]:
        end_frame_anchors.append(f"[Prop Identity Anchor: {ref_match['prop_desc']}]")
    if end_frame_anchors:
        end_frame_prompt = end_frame_prompt + "\n" + "\n".join(end_frame_anchors)

    try:
        last_image_url = generate_keyframe(
            end_frame_prompt,
            reference_image_url=scene["image_url"],
            ref_strength=ref_strength,
            negative_prompt=end_frame_negative,
        )
    except Exception as e:
        # 尾帧生图失败不 FATAL，计入失败计数交由路由处理（重试/跳过该镜头）
        print(f"    ❌ [尾帧] [Shot {idx+1}] 尾帧生图失败: {e}")
        scenes[idx]["video_gen_failures"] = scenes[idx].get("video_gen_failures", 0) + 1
        return {"scenes": scenes}
    print(f"    [*] 尾帧参考强度: {ref_strength}（{'极低→大差异' if ref_strength <= 0.12 else '偏低→中等差异' if ref_strength <= 0.2 else '中等→保持连续'}）")

    scenes = state["scenes"].copy()
    scenes[idx]["last_image_url"] = last_image_url
    scenes[idx]["last_image_iterations"] += 1
    return {"scenes": scenes}


def end_frame_review_gate_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]

    payload = {
        "stage": "end_frame_review",
        "title": f"镜头 {idx + 1} 【尾帧】审片",
        "scene_index": idx + 1,
        "script": scene["script"],
        "image_prompt": scene.get("last_image_prompt", ""),
        "image_url": scene.get("last_image_url", ""),
        "reference_image_url": scene.get("image_url", ""), # 首帧作为参考图
        "message": "请审核尾帧：需确保与首帧视觉一致。可通过、重写，或修改 prompt 后继续。",
        "actions": ["approve", "rewrite", "edit_prompt"],
    }

    decision = _normalize_decision(interrupt(payload), gate="end_frame_review")
    action = decision.get("action", "approve")
    scenes = state["scenes"].copy()

    if decision.get("image_prompt") is not None:
        scenes[idx]["last_image_prompt"] = str(decision["image_prompt"]).strip()

    if action in APPROVE_ACTIONS:
        scenes[idx]["last_image_is_perfect"] = True
        scenes[idx]["last_image_critique"] = "无"

        # 【修复】将尾帧也追加到 reference_images 中，保持历史完整
        ref_images = state.get("reference_images",[]).copy()
        ref_images.append(scenes[idx]["last_image_url"])

        updates = {
            "scenes": scenes,
            "reference_images": ref_images,
        }

        if get_image_embedding is not None:
            try:
                ref_embs = state.get("reference_embeddings", []).copy()
                ref_embs.append(get_image_embedding(scenes[idx]["last_image_url"]))
                updates["reference_embeddings"] = ref_embs
            except Exception as e:
                print(f"    ⚠️ 记录 embedding 失败，但不影响主流程: {str(e)}")

        return updates

    scenes[idx]["last_image_is_perfect"] = False
    scenes[idx]["last_image_critique"] = decision.get("reason") or "Human requested rewrite"
    return {"scenes": scenes}
# ===================================================


def _build_camera_context(
    scene: dict,
    shot_plan: dict,
    scene_index: int,
    total_scenes: int,
) -> str:
    """
    从 shot_plan 提取结构化摄影参数，配合通用运镜原则传递给运镜 LLM。
    不做硬编码的关键词匹配——让 LLM 根据摄影参数和剧本内容自行推断
    运镜距离、跟随目标和拍摄角度。

    Returns:
        格式化的文本块，注入给 design_camera_movement
    """
    script = scene.get("script", "")
    lines = []

    # ── 1. 从 shot_plan 提取 hero shot 的完整摄影规格 ──
    if shot_plan and shot_plan.get("shots"):
        hero_idx = shot_plan.get("hero_shot_index", 0)
        hero = shot_plan["shots"][hero_idx]
        cam = hero.get("camera", {})

        lines.append("[Hero Shot Camera Specs]")
        for key in ("type", "movement", "angle", "lens", "orientation", "tracking"):
            val = cam.get(key, "")
            if val:
                lines.append(f"  {key.replace('_', ' ').title()}: {val}")

        focus_object = hero.get("focus_object", "")
        if focus_object:
            lines.append(f"  Focus Object: {focus_object}")

        pacing = hero.get("pacing_weight", 0.25)
        lines.append(f"  Pacing Weight: {pacing:.0%} ({'high energy' if pacing >= 0.35 else 'low energy' if pacing <= 0.15 else 'moderate'})")

    # ── 2. 剧本原文（让 LLM 自行识别动作、武器、目光、运动等焦点） ──
    if script:
        lines.append(f"[Scene Script]\n  {script}")

    # ── 3. 通用运镜原则（不依赖硬编码，适用于任何题材） ──
    lines.append("[Camera Motion Principles]")
    lines.append("  - Match camera distance to the shot type: wider types need more distance, tighter types get closer.")
    lines.append("  - Camera should follow the scene's primary action — track whatever moves fastest or carries the most visual energy (weapon arcs, character trajectory, gaze direction, environmental events).")
    lines.append("  - Camera can shift distance within the clip when motivated: push in on impact, pull back on reveal, rise for scope.")
    lines.append("  - Every camera movement should have a visible trigger in the frame. Avoid arbitrary pans or tilts with no on-screen cause.")

    return "\n".join(lines)


def _build_cross_scene_context(state: "MultimediaState") -> str:
    """
    构建跨场景连续性上下文。
    将上一场景的关键信息（剧本、角色描述、视觉锚点）以结构化方式
    传递给运镜 LLM，让其自行理解如何维持场景间的视觉连贯性。
    """
    idx = state["current_scene_index"]
    if idx == 0:
        return ""  # 第一场景无前置上下文

    lines = ["[Previous Scene Context]"]

    # 上一场景剧本
    prev_scene = state["scenes"][idx - 1]
    prev_script = prev_scene.get("script", "")
    if prev_script:
        lines.append(f"  Script: {prev_script}")

    # 角色身份描述（确保跨场景一致）
    global_setting = state.get("global_setting", "")
    if global_setting:
        lines.append(f"  Character: {global_setting}")

    # 上一场景的视觉锚点（运镜惯性、色调、能量等原始数据）
    scene_anchors = state.get("scene_anchors") or {}
    anchor_items = []
    for key in ("camera_momentum", "color_direction", "emotion_energy", "ambient_context"):
        val = scene_anchors.get(key)
        if val is not None and val != "":
            label = key.replace("_", " ").title()
            if key == "emotion_energy":
                anchor_items.append(f"  {label}: {val:.2f}")
            else:
                anchor_items.append(f"  {label}: {val}")
    if anchor_items:
        lines.append("  Visual Anchors:")
        lines.extend(anchor_items)

    return "\n".join(lines) if len(lines) > 1 else ""


def videographer_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    total_scenes = len(state["scenes"])
    print(f"--- 🎥 [镜头 {idx+1}] 摄影师设计运镜 ---")

    # 根据是否开启双帧模式，选择不同的 Prompt
    prompt_template = VIDEOGRAPHER_DUAL_FRAME_PROMPT if state.get("use_first_last_frame") else VIDEOGRAPHER_PROMPT

    # 从 shot_plan 提取当前镜头的摄影规格，注入给运镜设计 LLM
    script_for_camera = scene["script"]
    shot_plan = state.get("shot_plan")
    if shot_plan and shot_plan.get("shots"):
        hero_idx = shot_plan.get("hero_shot_index", 0)
        hero_shot = shot_plan["shots"][hero_idx]
        cam = hero_shot.get("camera", {})
        camera_spec = (
            f"\n[Shot Plan Camera Specs — 请在此基础上设计运镜，不要与之矛盾] "
            f"Shot type: {cam.get('type','')}, Movement: {cam.get('movement','')}, "
            f"Angle: {cam.get('angle','')}, Lens: {cam.get('lens','')}"
        )
        script_for_camera = scene["script"] + camera_spec

    # ── NEW: 构建景别/动作焦点上下文 ──
    camera_context = _build_camera_context(scene, shot_plan, idx, total_scenes)
    if camera_context:
        print(f"    [*] Camera context: shot distance + action focus injected")

    # ── NEW: 构建跨场景连续性上下文 ──
    cross_scene_context = _build_cross_scene_context(state)
    if cross_scene_context:
        print(f"    [*] Cross-scene context: continuity directives injected")

    video_prompt = design_camera_movement(
        scene["image_url"],
        script_for_camera,
        prompt_template,
        critique=scene.get("video_critique", "无"),
        camera_context=camera_context,
        cross_scene_context=cross_scene_context,
    )
    print(f"    [*] 运镜提示词: {video_prompt}")

    scenes = state["scenes"].copy()
    scenes[idx]["video_prompt"] = video_prompt
    return {"scenes": scenes}


def video_gen_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"--- 🎞️ [镜头 {idx+1}] 动画师生成原片 ---")

    scenes = state["scenes"].copy()

    try:
        last_url = scene.get("last_image_url", "") if state.get("use_first_last_frame") else ""
        raw_video_url = generate_video_from_image(scene["image_url"], scene["video_prompt"], last_url)

        # 仅在成功产出后递增质量重试计数（用于审片不通过时的重生循环上限）
        scenes[idx]["video_iterations"] = scenes[idx].get("video_iterations", 0) + 1
        scenes[idx]["raw_video_url"] = raw_video_url
        scenes[idx]["final_video_url"] = raw_video_url
        scenes[idx]["video_is_perfect"] = False

        print(f"    🌟 镜头 {idx+1} 制作完成！URL: {raw_video_url}")
        return {"scenes": scenes, "aborted": False, "abort_reason": None}

    except FreeTierQuotaExhaustedError as e:
        # 配额/余额耗尽：所有模型均不可用，属于全局性中止，不再继续后续镜头
        msg = str(e)
        print(f"    ⛔ [配额中止] {msg}")
        scenes[idx]["raw_video_url"] = ""
        scenes[idx]["final_video_url"] = ""
        scenes[idx]["video_is_perfect"] = False
        scenes[idx]["video_critique"] = "FAIL: " + msg
        return {
            "scenes": scenes,
            "aborted": True,
            "abort_reason": msg,
            "error_log": msg,
        }

    except Exception as e:
        # 单镜头生成失败（网络/超时/接口错误）：不中止整个任务，
        # 由路由决定"有限次重试后跳过该镜头"，保全已生成的其他镜头。
        msg = f"视频生成失败: {str(e)}"
        fails = scenes[idx].get("video_gen_failures", 0) + 1
        scenes[idx]["video_gen_failures"] = fails
        scenes[idx]["video_is_perfect"] = False
        scenes[idx]["video_critique"] = "FAIL: " + msg

        if fails >= MAX_VIDEO_GEN_FAILURES:
            # 达重试上限 -> 占位黑场降级（而非静默丢镜头）：
            # 生成本地黑场写入 final_video_url 槽位，标记 shot_failed，
            # 保证 stitcher/audio_mixer 拿到时间轴完整的序列。
            duration = float(scene.get("duration", 2.0) or 2.0)
            placeholder = _make_placeholder_clip(
                duration=duration,
                idx=idx,  # 按镜头索引命名，避免多镜头覆盖
            )
            if placeholder:
                scenes[idx]["final_video_url"] = placeholder
                scenes[idx]["raw_video_url"] = placeholder
                scenes[idx]["shot_failed"] = True
                print(f"    ⚫ [降级] 镜头 {idx+1} 达失败上限，生成占位黑场: {placeholder}")
            else:
                # moviepy 不可用时退化为原空 URL 跳过
                scenes[idx]["final_video_url"] = ""
                scenes[idx]["raw_video_url"] = ""
                scenes[idx]["shot_failed"] = True
                print(f"    ⚠️ [降级] 镜头 {idx+1} 达失败上限，占位生成失败，跳过该镜头")
        else:
            scenes[idx]["raw_video_url"] = ""
            scenes[idx]["final_video_url"] = ""

        print(f"    ⚠️ [镜头 {idx+1} 生成失败 {fails}/{MAX_VIDEO_GEN_FAILURES}] {msg}")
        return {
            "scenes": scenes,
            "aborted": False,
            "abort_reason": None,
            "error_log": msg,
        }


def video_reviewer_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    print(f"--- 🧐 [镜头 {idx+1}] 视频总监 + 安全官审核 ---")

    # 【修复】显式获取上一镜头的图片，优先取尾帧
    previous_image_url = None
    if idx > 0:
        prev_scene = state["scenes"][idx - 1]
        previous_image_url = prev_scene.get("last_image_url") or prev_scene.get("image_url")
        if previous_image_url:
            print(f"    [*] 正在对比上一镜头关键帧: {previous_image_url[-30:]}...")

    feedback = evaluate_video(
        scene["raw_video_url"],
        scene["script"],
        VIDEO_REVIEWER_SYSTEM_PROMPT,
        previous_image_url=previous_image_url,
        frame_count=4
    )

    scenes = state["scenes"].copy()
    scenes[idx]["video_auto_feedback"] = feedback
    scenes[idx]["video_critique"] = feedback
    scenes[idx]["video_is_perfect"] = _is_review_pass(feedback)

    return {"scenes": scenes}


def video_review_gate_node(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]

    # 【修复】显式获取上一镜头的图片
    previous_image_url = None
    if idx > 0:
        prev_scene = state["scenes"][idx - 1]
        previous_image_url = prev_scene.get("last_image_url") or prev_scene.get("image_url")

    # ── 动作幅度自检：防止"静止空镜 / 看不出在动" ──
    action_beat = scene.get("action_beat", "")
    if len(action_beat.strip()) < 8:
        motion_hint = "⚠️ 动作空洞：本镜头缺少明确的可见动作（action_beat 过短），建议重写时补充主体从起始到结束的具体动作，否则视频模型易生成静止空镜。"
    else:
        motion_hint = f"✓ 已锚定可见动作：{action_beat[:60]}..."

    payload = {
        "stage": "video_review",
        "title": f"镜头 {idx + 1} 视频审片",
        "scene_index": idx + 1,
        "script": scene["script"],
        "global_setting": state.get("global_setting", ""),
        "video_prompt": scene.get("video_prompt", ""),
        "video_url": scene.get("raw_video_url", ""),
        "reference_image_url": previous_image_url,
        "embedding_similarity": scene.get("embedding_similarity"),
        "auto_feedback": scene.get("video_auto_feedback", scene.get("video_critique", "")),
        "action_beat": action_beat,
        "motion_hint": motion_hint,
        "message": "请审核视频成片：可通过、重写，或修改 video_prompt 后再生成。注意检查主体是否有可见动作、画面是否看得出在演什么。",
        "actions": ["approve", "rewrite", "edit_prompt"],
    }

    decision = _normalize_decision(interrupt(payload), gate="video_review")
    action = decision.get("action", "approve")

    scenes = state["scenes"].copy()

    if decision.get("video_prompt") is not None:
        scenes[idx]["video_prompt"] = str(decision["video_prompt"]).strip()

    if action in APPROVE_ACTIONS:
        scenes[idx]["video_is_perfect"] = True
        scenes[idx]["video_critique"] = "无"
        scenes[idx]["final_video_url"] = scene["raw_video_url"]
        return {"scenes": scenes}

    scenes[idx]["video_is_perfect"] = False
    scenes[idx]["video_critique"] = decision.get("reason") or scene.get("video_auto_feedback") or "Human requested rewrite"
    return {"scenes": scenes}


def _extract_character_identity(state: MultimediaState) -> dict:
    """
    从已完成场景中提取角色身份锚点，供下一场景保持角色外观一致性。

    提取来源（优先级从高到低，避免对人类职业名词的硬编码假设）：
    1. reference_sheets.characters — 已生成的角色键（如 ugly_duckling / white_swan），
       由 reference_gen 用 LLM 从故事提取，最权威，直接用作 subject。
    2. 上一轮已累积的 character_identity — 历史 subject 源头锁定（抗漂移）。
    3. shot_plan.focus_object — 镜头焦点对象（过滤纯环境词）。
    4. scene script + global_setting + 角色参考描述 — 抽取通用中性外观词
       （体型 / 颜色 / 羽色 / 自然附肢等），不再局限于人类发型服饰。

    抗漂移设计（源头累积，取代链式覆盖）：
    - appearance 与历史特征取并集，单调递增：某个场景剧本未提及的特征
      不会从锚点链中消失，避免约束被逐级稀释。
    - subject 源头优先：一旦确立便不再被后续场景改写，防止角色身份跳变。

    Returns:
        {
            "subject": str,          # 角色主体标识
            "appearance": List[str], # 外观特征关键词（累积并集）
            "has_character": bool,   # 是否检测到角色存在
        }
    """
    scene = state["scenes"][state["current_scene_index"]]
    script = scene.get("script", "")
    shot_plan = state.get("shot_plan")

    # 源头基准：全局角色设定始终参与匹配，不随场景剧本措辞而丢失
    global_setting = state.get("global_setting", "") or ""
    # 历史累积：上一轮已提取的角色身份锚点
    prev_identity = (state.get("scene_anchors") or {}).get("character_identity") or {}

    # 匹配文本：当前场景剧本（优先），全局设定单独保留作为兜底来源
    script_lower = script.lower()
    global_lower = global_setting.lower()

    identity = {"subject": "", "appearance": [], "has_character": False}

    # 防御跨运行 thread 残留污染：第一个镜头没有任何「前序」可言，
    # 必须忽略任何继承自历史 scene_anchors 的 character_identity（如旧运行残留的 mage），
    # 强制基于本次 reference_sheets + 剧本重新识别。
    is_first_scene = state.get("current_scene_index", 0) == 0
    if is_first_scene:
        prev_identity = {}

    # ── 1.5 从 reference_sheets 已生成的角色键匹配（最高优先级，最权威）──
    # 说明：历史累积的 subject 不再前置强制覆盖，而是作为本步匹配失败时的
    # 兜底（见下方「0.5 历史继承 fallback」），避免旧锚点（如 mage）污染首镜头。
    # reference_gen 已用 LLM 从 global_setting+scenes 提取正确角色名
    # （如 ugly_duckling / white_swan），直接用剧本提及的角色键作 subject，
    # 彻底避免对人类职业名词的硬编码假设（修复「丑小鸭被锁成法师」的 bug）。
    reference_sheets = state.get("reference_sheets") or {}
    char_entries = reference_sheets.get("characters", {}) or {}
    matched_char_key = ""
    matched_char_desc = ""
    if char_entries:
        # 角色名候选（英文键 + 中文名）
        def _names_of(char_key, char_data):
            ns = [char_key]
            if isinstance(char_data, dict):
                cn = char_data.get("name_cn", "")
                if cn:
                    ns.append(cn)
            return ns, (char_data.get("description", "") if isinstance(char_data, dict) else "")

        # 第一轮：仅用「当前场景剧本」匹配 —— 保证按当前镜头区分角色，
        # 不被 global_setting 里罗列的全片角色名抢先（修复「丑小鸭永远抢先」）。
        for char_key, char_data in char_entries.items():
            names, desc = _names_of(char_key, char_data)
            for n in names:
                if n and n.lower() in script_lower:
                    matched_char_key = char_key
                    matched_char_desc = desc
                    break
            if matched_char_key:
                break

        # 第二轮（兜底）：剧本未提及任何角色时，退回 global_setting 全片角色名匹配
        if not matched_char_key:
            for char_key, char_data in char_entries.items():
                names, desc = _names_of(char_key, char_data)
                for n in names:
                    if n and n.lower() in global_lower:
                        matched_char_key = char_key
                        matched_char_desc = desc
                        break
                if matched_char_key:
                    break

        # 若仍无命中，但全片只有一个角色，则默认采用它
        if not matched_char_key and len(char_entries) == 1:
            matched_char_key = next(iter(char_entries.keys()))
            only = char_entries[matched_char_key]
            matched_char_desc = only.get("description", "") if isinstance(only, dict) else ""

    if matched_char_key:
        # 优先采用本轮基于剧本匹配的 reference_sheets 角色键，
        # 覆盖任何历史残留（如 mage），确保角色身份始终由当前 reference 决定。
        identity["subject"] = matched_char_key
        identity["has_character"] = True

    # ── 0.5 历史继承 fallback：仅当本轮 reference_sheets 完全没匹配到角色时，
    # 才继承上一个镜头的 subject（抗漂移），避免旧 mage 锚点在首镜头被强制覆盖。
    if not identity["subject"]:
        prev_subject = prev_identity.get("subject", "")
        if prev_subject:
            identity["subject"] = prev_subject
            identity["has_character"] = True

    # ── 1. 兜底：shot_plan.focus_object 作为 subject（仅在仍为空时，过滤纯环境词）──
    # 优先级低于 reference_sheets 角色键，避免 focus_object 误写覆盖权威角色名。
    if shot_plan and not identity["subject"]:
        focus = (shot_plan.get("focus_object", "") or "").strip()
        _ENV_NOISE = {"environment", "landscape", "scene", "background",
                      "农场", "湖泊", "沼泽", "森林", "天空", "环境", "场景"}
        if focus and focus.lower() not in _ENV_NOISE:
            identity["subject"] = focus
            identity["has_character"] = True

    # ── 2. 从剧本 + 全局设定 + 角色参考描述 提取外观描述关键词（通用、中性）──
    # 不再硬编码「人类发型/服饰/装备」表，而是对动物/精灵/机甲等各类主角通用：
    # 抽取自然界与角色设定中常见的外观维度词（羽毛/体型/颜色/翅膀/角等），
    # 作为补充锚点注入 prompt，而非覆盖式强制锁定。
    _NEUTRAL_APPEARANCE_TOKENS = [
        # 体型 / 体态
        "蓬乱", "蓬松", "纤细", "壮硕", "修长", "粗短", "圆润", "瘦削",
        "slender", "bulky", "lean", "fluffy", "plump",
        # 颜色 / 毛色 / 羽色
        "灰褐", "灰白", "纯白", "雪白", "乌黑", "金黄", "银灰", "棕褐",
        "white", "black", "gray", "grey", "brown", "golden", "silver",
        # 羽毛 / 皮毛 / 鳞片等自然覆盖物
        "羽毛", "绒毛", "鳞甲", "皮毛", "鬃毛", "feathers", "fur", "scales",
        # 自然附肢 / 特征（不再视为畸形）
        "翅膀", "双翅", "长颈", "短颈", "犄角", "尖喙", "利爪",
        "wings", "beak", "claws", "horns", "tail", "fins",
        # 神态 / 气质
        "怯弱", "优雅", "威严", "俏皮", "哀伤", "gentle", "elegant", "fierce",
    ]

    # 匹配文本扩展：纳入已匹配角色的描述（角色设定里的外观词也应被抽取）
    match_text = script_lower
    if matched_char_desc:
        match_text += " " + matched_char_desc.lower()

    for tok in _NEUTRAL_APPEARANCE_TOKENS:
        if tok.lower() in match_text:
            identity["appearance"].append(tok)

    # ── 2.5 与历史累积特征求并集（去重保序，单调递增，抗漂移）──
    # 先放历史特征保持源头顺序，再补充本场景新增特征。
    merged_appearance: List[str] = []
    seen_kw = set()
    for kw in list(prev_identity.get("appearance") or []) + identity["appearance"]:
        kw_key = kw.lower()
        if kw_key not in seen_kw:
            merged_appearance.append(kw)
            seen_kw.add(kw_key)
    identity["appearance"] = merged_appearance

    # 兜底：focus_object 作为 subject（仅在仍为空时，过滤纯环境词）
    if shot_plan and not identity["subject"]:
        focus = (shot_plan.get("focus_object", "") or "").strip()
        _ENV_NOISE = {"environment", "landscape", "scene", "background",
                      "农场", "湖泊", "沼泽", "森林", "天空", "环境", "场景"}
        if focus and focus.lower() not in _ENV_NOISE:
            identity["subject"] = focus
            identity["has_character"] = True

    if identity["appearance"] or identity["subject"]:
        identity["has_character"] = True

    return identity


def _extract_scene_anchors(state: MultimediaState) -> dict:
    """
    从已完成场景的 state 中提取视觉锚点，供下一场景做连续性约束。

    提取维度：
    - color_direction: 色调方向（从风格后缀和光照推断）
    - lighting_anchor: 光照锚点（最后一条光照规则的关键特征）
    - camera_momentum: 运镜惯性（hero shot 的 camera movement）
    - emotion_energy: 结尾能量水平（最后一个 shot 的情绪能量值）
    - ambient_context: 环境上下文（夜晚/雨天/黎明/黄昏）
    - dominant_genres: 检测到的风格列表
    - intent: 场景意图分类
    """
    visual_ctx = state.get("visual_context")
    shot_plan = state.get("shot_plan")

    anchors = {}

    # ── 从 visual_context 提取 ──
    if visual_ctx:
        anchors["intent"] = visual_ctx.get("intent", "")
        anchors["ambient_context"] = visual_ctx.get("ambient_context")
        anchors["dominant_genres"] = visual_ctx.get("genres", [])

        # 色调方向：从风格后缀和 mood 关键词推断
        mood_list = visual_ctx.get("mood_keywords", [])
        warm_moods = {"triumph", "power", "presence", "hope", "anticipation"}
        cool_moods = {"isolation", "mystery", "loss", "vulnerability", "awe"}
        if any(m in warm_moods for m in mood_list):
            anchors["color_direction"] = "warm"
        elif any(m in cool_moods for m in mood_list):
            anchors["color_direction"] = "cool"
        else:
            anchors["color_direction"] = "neutral"

        # 光照锚点：取第一条光照规则的前 40 字符作为特征标识
        lighting = visual_ctx.get("lighting_rules", [])
        anchors["lighting_anchor"] = lighting[0][:40] if lighting else ""

    # ── 从 shot_plan 提取 ──
    if shot_plan and shot_plan.get("shots"):
        shots = shot_plan["shots"]

        # 运镜惯性：hero shot 的 camera movement（代表场景的主要运镜方向）
        hero_idx = shot_plan.get("hero_shot_index", 0)
        if hero_idx < len(shots):
            hero_movement = shots[hero_idx].get("camera", {}).get("movement", "")
            anchors["camera_momentum"] = hero_movement
        else:
            anchors["camera_momentum"] = shots[-1].get("camera", {}).get("movement", "")

        # 结尾能量：最后一个 shot 的情绪能量值
        last_emotion = shots[-1].get("emotion", "")
        anchors["emotion_energy"] = _EMOTION_ENERGY.get(last_emotion, 0.5)
    else:
        anchors["camera_momentum"] = ""
        anchors["emotion_energy"] = 0.5

    # ── 角色身份锚点（跨场景角色外观一致性）──
    anchors["character_identity"] = _extract_character_identity(state)

    # ── 帧级连续性（Frame Continuation）──
    # 提取当前场景的最后一帧图像 URL，供下一场景的首帧生成作为视觉参考
    # 实现行业标准的 frame continuation：前一场景尾帧 → 后一场景首帧的像素级锚定
    scene = state["scenes"][state["current_scene_index"]]
    last_frame_url = scene.get("last_image_url") or scene.get("image_url")
    anchors["last_frame_url"] = last_frame_url
    if last_frame_url:
        print(f"    [*] Frame Continuation: tail frame URL extracted for next scene")

    return anchors


def advance_scene_node(state: MultimediaState):
    idx = state["current_scene_index"]
    next_idx = idx + 1

    # 提取当前场景的视觉锚点，传递给下一场景做连续性约束
    anchors = _extract_scene_anchors(state)
    print(f"--- ✅[镜头 {idx+1}] 当前镜头完成，切换到下一镜头 ---")
    print(f"    [*] 视觉锚点已提取: intent={anchors.get('intent', '?')}, "
          f"energy={anchors.get('emotion_energy', 0):.2f}, "
          f"color={anchors.get('color_direction', '?')}, "
          f"ambient={anchors.get('ambient_context', 'none')}")
    char_id = anchors.get("character_identity", {})
    if char_id.get("has_character"):
        print(f"    [*] 角色锚点: subject=\"{char_id.get('subject', '?')}\", "
              f"appearance={char_id.get('appearance', [])}")

    return {
        "current_scene_index": next_idx,
        "scene_anchors": anchors,
    }


def abort_node(state: MultimediaState):
    reason = state.get("abort_reason") or state.get("error_log") or "未知原因"
    print(f"\n--- ⛔ [任务中止] {reason}")
    # 运行指标埋点：任务中止（可观测性）
    run_metrics.finalize_run("abort", "")
    return {"error_log": reason, "aborted": True, "abort_reason": reason}


def stitcher_node(state: MultimediaState):
    print("\n--- 🎬 [剪辑师] 正在将所有镜头拼接成最终宣传片 ---")
    video_urls = []
    skipped = []
    for i, scene in enumerate(state["scenes"]):
        url = scene.get("final_video_url", "")
        if url:
            video_urls.append(url)
        else:
            skipped.append(i + 1)
            print(f"    ⚠️ 镜头 {i + 1} 无视频（可能生成失败），将跳过")

    if not video_urls:
        print("    ❌ 所有镜头均无可用视频，无法生成最终宣传片")
        # 运行指标埋点：拼接失败（可观测性）
        run_metrics.finalize_run("fail", "")
        return {"final_movie_path": "", "error_log": "所有镜头视频生成失败，无法拼接"}

    if skipped:
        print(f"    [*] 将拼接 {len(video_urls)}/{len(state['scenes'])} 个镜头（跳过: {skipped}）")

    # ── 提取转场数据（基于各场景的叙事功能 beat，做跨场景语义衔接）──
    # 说明：state["sequence_graph"] 是「单个场景内部」的镜头关系图，不能直接当作
    # 跨场景转场使用（否则会错用最后一个场景的内部转场）。正确做法是根据每个
    # 场景的 beat + emotion，用 select_cross_scene_transition 计算相邻场景的转场。
    transitions: List[str] = []
    used_scenes = [s for s in state["scenes"] if s.get("final_video_url")]
    for i in range(1, len(used_scenes)):
        prev = used_scenes[i - 1]
        curr = used_scenes[i]
        t = select_cross_scene_transition(
            prev_beat=prev.get("beat", ""),
            curr_beat=curr.get("beat", ""),
            prev_emotion=prev.get("emotion", ""),
            curr_emotion=curr.get("emotion", ""),
        )
        transitions.append(t)
        print(f"    ↳ 场景 {i}→{i+1} 转场: {t} "
              f"(beat: {prev.get('beat','?')} → {curr.get('beat','?')})")
    non_hardcut = [t for t in transitions if t in ("dissolve", "fade_through_black",
                                                    "whip_pan", "smash_cut", "camera_carry")]
    if non_hardcut:
        print(f"    [*] 跨场景转场: {len(non_hardcut)} 个效果转场（基于叙事节拍语义匹配）")

    # ── 动态文件名 ──
    task = state.get("task", "")
    output_filename = ""  # 空字符串让 stitch_videos 自动生成

    try:
        final_movie_path, valid_indices, segment_durations = stitch_videos(
            video_urls,
            output_filename=output_filename,
            transitions=transitions,
        )
        # P1 字幕精确对齐：把各有效片段真实时长写回对应 scene，
        # 供 audio_mixer_node 的 build_srt 按真实时长逐句对齐（替代原均分策略）。
        scenes_patch = {}
        for orig_idx, dur in zip(valid_indices, segment_durations):
            if 0 <= orig_idx < len(state["scenes"]):
                state["scenes"][orig_idx]["video_duration"] = round(float(dur), 3)
        return {
            "final_movie_path": final_movie_path,
            "segment_durations": segment_durations,
        }
    except Exception as e:
        print(f"    ❌ 拼接失败: {e}")
        return {"final_movie_path": "", "error_log": f"视频拼接失败: {str(e)[:200]}"}


def audio_mixer_node(state: MultimediaState):
    """P0/P1 音频闭环：在无声成片之上叠加配音 + 字幕 + BGM + 混音。

    复用现成组件：edge-tts(TTS) / pysrt(SRT) / 预置曲库 BGM / moviepy(烧录+混音)。
    - enable_audio=False 时整阶段跳过（前端总开关）。
    - 若无声成片缺失则直接跳过，保证图不崩。
    """
    # 总开关：前端可关闭整个音频闭环
    if not state.get("enable_audio", True):
        print("\n--- 🔊 [音频师] 音频闭环已关闭（前端开关），跳过 ---")
        return {}

    final_movie = state.get("final_movie_path") or ""
    if not final_movie or not os.path.exists(final_movie):
        print("\n--- 🔊 [音频师] 无成片，跳过音频合成 ---")
        return {}

    print("\n--- 🔊 [音频师] 正在为成片叠加配音、字幕与配乐 ---")
    voice_role = state.get("voice_role") or "xiaoxiao"

    # 1) 汇总分镜台词作为旁白文本
    narration = "\n".join(
        s.get("script", "").strip() for s in state["scenes"] if s.get("script", "").strip()
    )
    if not narration:
        print("    [*] 无旁白文本，仅保留原声（若有）")
        return {}

    # 2) TTS 配音（edge-tts，失败降级 dashscope CosyVoice）
    voiceover = generate_voiceover(narration, voice=voice_role)
    if not voiceover:
        print("    ⚠️ 配音生成失败，跳过音频合成")
        return {}

    # 3) 生成 SRT 字幕（P1 精确对齐：build_srt 读取 scene["video_duration"] 真实时长）
    srt_path = build_srt(state["scenes"])
    print(f"    ✓ 配音: {os.path.basename(voiceover)} | 字幕: {os.path.basename(srt_path)}")

    # 4) 生成 BGM：P1 按场景情绪映射 mood（复用 scenes 的 emotion 字段），
    #    未提供时回退顶层 bgm_mood 或 ambient。
    scenes_with_emotion = [s for s in state["scenes"] if s.get("emotion")]
    bgm_mood = resolve_bgm_mood(
        scenes_with_emotion[0].get("emotion") if scenes_with_emotion else None,
        fallback=state.get("bgm_mood") or "ambient",
    )
    bgm_path = generate_bgm(bgm_mood, duration=30.0)
    if bgm_path:
        print(f"    ✓ BGM: {os.path.basename(bgm_path)}")
    else:
        print("    [*] 未生成 BGM（无可用引擎），跳过配乐")

    # 5) 烧录字幕 → 混音成片（配音 + 可选 BGM）
    try:
        with_subs = burn_subtitles(final_movie, srt_path)
        final_with_audio = mix_audio(with_subs, voiceover, bgm_path=bgm_path)
        print(f"    ✓ 音画合成完成: {os.path.basename(final_with_audio)}")
        # 运行指标埋点：任务成功出片（可观测性）
        run_metrics.finalize_run("ok", final_with_audio)
        return {
            "audio_track": voiceover,
            "subtitle_path": srt_path,
            "bgm_path": bgm_path,
            "final_movie_with_audio": final_with_audio,
        }
    except Exception as e:
        print(f"    ❌ 音频合成失败: {e}")
        return {"error_log": f"音频合成失败: {str(e)[:200]}"}


# ================= 路由逻辑 =================
def decide_image_quality(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]

    passed = scene["is_perfect"]

    # ── 角色一致性闭环门槛 ──
    # 即便 VLM 判 PASS，若与源头角色肖像的相似度过低（角色漂移）且仍有重试配额，
    # 也强制重生一次。embedding 缺失时该分支自动跳过（优雅降级为纯 VLM 判定）。
    # 阈值随机位自适应：背对镜头/远景时脸部信息本就缺失，
    # 沿用正面基准会系统性误判，重生也无法提分（白耗额度）。
    threshold, relax_reason = get_consistency_threshold(
        CHARACTER_CONSISTENCY_THRESHOLD, scene.get("script", "")
    )
    portrait_sim = scene.get("portrait_similarity")
    if (
        passed
        and threshold > 0
        and portrait_sim is not None
        and portrait_sim < threshold
        and scene["iterations"] < 3
    ):
        note = f"，已按 {relax_reason} 放宽" if relax_reason else ""
        print(
            f"    ⚠️ [镜头 {idx+1}] 角色一致性不足 "
            f"(肖像相似度 {portrait_sim:.3f} < {threshold:.3f}{note})，触发重生"
        )
        run_metrics.record_portrait_regen(portrait_sim, threshold)
        return "director"

    if relax_reason and portrait_sim is not None:
        print(
            f"    ℹ️ [镜头 {idx+1}] 一致性阈值按 {relax_reason} 放宽至 "
            f"{threshold:.3f}（肖像相似度 {portrait_sim:.3f}），通过"
        )

    if passed or scene["iterations"] >= 3:
        if state.get("use_first_last_frame"):
            return "end_frame_director"
        return "videographer"
    return "director"

def decide_end_frame_quality(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    if scene["last_image_is_perfect"] or scene["last_image_iterations"] >= 3:
        return "videographer"
    return "end_frame_director"

def decide_after_video_generation(state: MultimediaState):
    # 配额耗尽属于全局中止
    if state.get("aborted"):
        return "abort"

    idx = state["current_scene_index"]
    scene = state["scenes"][idx]

    # 生成成功 → 进入审片
    if scene.get("raw_video_url"):
        return "video_reviewer"

    # 单镜头生成失败：未达上限则重试生成，达到上限则跳过该镜头继续后续
    if scene.get("video_gen_failures", 0) < MAX_VIDEO_GEN_FAILURES:
        print(f"    🔁 [镜头 {idx+1}] 生成失败，重试...")
        return "retry"

    print(f"    ⏭️ [镜头 {idx+1}] 生成失败达上限，跳过该镜头继续制作其余镜头")
    return "skip"

def decide_after_image_generation(state: MultimediaState):
    """关键帧生图后路由：成功→审核；失败按 video_gen_failures 重试或跳过该镜头（不 FATAL）。"""
    # 配额耗尽属于全局中止
    if state.get("aborted"):
        return "abort"

    idx = state["current_scene_index"]
    scene = state["scenes"][idx]

    if scene.get("image_url"):
        return "reviewer"

    if scene.get("video_gen_failures", 0) < MAX_VIDEO_GEN_FAILURES:
        print(f"    🔁 [镜头 {idx+1}] 关键帧生图失败，重试...")
        return "retry"
    print(f"    ⏭️ [镜头 {idx+1}] 关键帧生图失败达上限，跳过该镜头继续制作其余镜头")
    return "skip"


def decide_after_end_frame_generation(state: MultimediaState):
    """尾帧生图后路由：成功→尾帧审核；失败按 video_gen_failures 重试或跳过。"""
    if state.get("aborted"):
        return "abort"

    idx = state["current_scene_index"]
    scene = state["scenes"][idx]

    if scene.get("last_image_url"):
        return "end_frame_review"

    if scene.get("video_gen_failures", 0) < MAX_VIDEO_GEN_FAILURES:
        print(f"    🔁 [镜头 {idx+1}] 尾帧生图失败，重试...")
        return "retry"
    print(f"    ⏭️ [镜头 {idx+1}] 尾帧生图失败达上限，跳过该镜头继续制作其余镜头")
    return "skip"


def decide_video_quality(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    if scene["video_is_perfect"] or scene["video_iterations"] >= 3:
        return "advance_scene"
    return "videographer"

def decide_next_scene(state: MultimediaState):
    if state["current_scene_index"] < len(state["scenes"]):
        print(f"\n🔄 [路由] 准备开始制作 镜头 {state['current_scene_index'] + 1}...")
        return "continue"
    print("\n✅ [路由] 所有镜头制作完毕，移交剪辑部门！")
    return "stitcher"


def decide_after_critic(state: MultimediaState):
    """Phase 4: 质量门控路由 — 通过则进 director，不通过则回 shot_strategy_builder 重写。"""
    critic_eval = state.get("critic_eval")
    rewrite_count = state.get("rewrite_count", 0)
    max_rewrites = QUALITY_THRESHOLDS.get("max_rewrites", 2)

    if not critic_eval:
        print("\n🔄 [路由] 无评估结果，直接进导演")
        run_metrics.record_critic(0.0, False, rewrite_count, False)
        return "director"

    if critic_eval.get("pass_gates", False):
        print(f"\n✅ [路由] 质量门控通过 (score={critic_eval['overall_score']:.2f})，进入导演")
        run_metrics.record_critic(critic_eval["overall_score"], True, rewrite_count, False)
        return "director"

    if rewrite_count >= max_rewrites:
        print(f"\n⚠️ [路由] 已达最大重写次数 ({max_rewrites})，强制进入导演")
        run_metrics.record_critic(critic_eval["overall_score"], False, rewrite_count, True)
        return "director"

    print(f"\n🔄 [路由] 质量门控未通过 (score={critic_eval['overall_score']:.2f})，触发重写 #{rewrite_count + 1}")
    run_metrics.record_critic(critic_eval["overall_score"], False, rewrite_count, False)
    return "rewrite"


# ================= 组装 Graph =================
workflow = StateGraph(MultimediaState)

workflow.add_node("showrunner", showrunner_node)
workflow.add_node("showrunner_review", showrunner_review_gate_node)
workflow.add_node("reference_gen", reference_gen_node)
workflow.add_node("visual_context_builder", visual_context_builder_node)
workflow.add_node("shot_strategy_builder", shot_strategy_builder_node)
workflow.add_node("sequence_orchestrator", sequence_orchestrator_node)

# Phase 4: Self-Refining Cinematic Director System
workflow.add_node("cinematic_critic", cinematic_critic_node)
workflow.add_node("director_refine", director_refine_node)

# Phase 5: Multi-Agent Film Studio System
workflow.add_node("film_studio", film_studio_node)

workflow.add_node("director", director_node)
workflow.add_node("prompt_preview", prompt_preview_node)
workflow.add_node("image_generator", image_gen_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("image_review", image_review_gate_node)

# 新增尾帧节点
workflow.add_node("end_frame_director", end_frame_director_node)
workflow.add_node("end_frame_prompt_preview", end_frame_prompt_preview_node)
workflow.add_node("end_frame_generator", end_frame_gen_node)
workflow.add_node("end_frame_review", end_frame_review_gate_node)

workflow.add_node("videographer", videographer_node)
workflow.add_node("video_generator", video_gen_node)
workflow.add_node("video_reviewer", video_reviewer_node)
workflow.add_node("video_review", video_review_gate_node)
workflow.add_node("advance_scene", advance_scene_node)
workflow.add_node("abort", abort_node)
workflow.add_node("stitcher", stitcher_node)
workflow.add_node("audio_mixer", audio_mixer_node)

workflow.set_entry_point("showrunner")
workflow.add_edge("showrunner", "showrunner_review")
workflow.add_edge("showrunner_review", "reference_gen")
workflow.add_edge("reference_gen", "visual_context_builder")
workflow.add_edge("visual_context_builder", "shot_strategy_builder")
workflow.add_edge("shot_strategy_builder", "sequence_orchestrator")

# Phase 5: Film Studio → Phase 4: Cinematic Critic + Rewrite Loop
workflow.add_edge("sequence_orchestrator", "film_studio")
workflow.add_edge("film_studio", "cinematic_critic")
workflow.add_conditional_edges(
    "cinematic_critic",
    decide_after_critic,
    {
        "director": "director",
        "rewrite": "shot_strategy_builder",
    }
)

# Phase 4: Director → Refinement → Prompt Preview → Image Generator
workflow.add_edge("director", "director_refine")
workflow.add_edge("director_refine", "prompt_preview")

def decide_after_prompt_preview(state: MultimediaState):
    """prompt_preview 路由：approve/edit_prompt → image_generator, rewrite → director"""
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    if scene.get("is_perfect"):
        return "image_generator"
    return "director"

workflow.add_conditional_edges(
    "prompt_preview",
    decide_after_prompt_preview,
    {
        "image_generator": "image_generator",
        "director": "director",
    }
)

workflow.add_conditional_edges(
    "image_generator",
    decide_after_image_generation,
    {
        "reviewer": "reviewer",
        "retry": "image_generator",
        "skip": "advance_scene",
        "abort": "abort",
    }
)
workflow.add_edge("reviewer", "image_review")

workflow.add_conditional_edges(
    "image_review",
    decide_image_quality,
    {
        "end_frame_director": "end_frame_director",
        "videographer": "videographer",
        "director": "director"
    }
)

# 尾帧路由
workflow.add_edge("end_frame_director", "end_frame_prompt_preview")


def decide_after_end_frame_prompt_preview(state: MultimediaState):
    """end_frame_prompt_preview 路由：approve/edit_prompt → end_frame_generator, rewrite → end_frame_director"""
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    if scene.get("last_image_is_perfect"):
        return "end_frame_generator"
    return "end_frame_director"


workflow.add_conditional_edges(
    "end_frame_prompt_preview",
    decide_after_end_frame_prompt_preview,
    {
        "end_frame_generator": "end_frame_generator",
        "end_frame_director": "end_frame_director",
    }
)

workflow.add_conditional_edges(
    "end_frame_generator",
    decide_after_end_frame_generation,
    {
        "end_frame_review": "end_frame_review",
        "retry": "end_frame_generator",
        "skip": "advance_scene",
        "abort": "abort",
    }
)
workflow.add_conditional_edges(
    "end_frame_review",
    decide_end_frame_quality,
    {
        "videographer": "videographer",
        "end_frame_director": "end_frame_director"
    }
)

workflow.add_edge("videographer", "video_generator")
workflow.add_conditional_edges(
    "video_generator",
    decide_after_video_generation,
    {
        "video_reviewer": "video_reviewer",
        "retry": "video_generator",
        "skip": "advance_scene",
        "abort": "abort"
    }
)

workflow.add_edge("video_reviewer", "video_review")
workflow.add_conditional_edges(
    "video_review",
    decide_video_quality,
    {
        "videographer": "videographer",
        "advance_scene": "advance_scene"
    }
)

workflow.add_conditional_edges(
    "advance_scene",
    decide_next_scene,
    {
        "continue": "visual_context_builder",
        "stitcher": "stitcher"
    }
)

workflow.add_edge("abort", END)
workflow.add_edge("stitcher", "audio_mixer")
workflow.add_edge("audio_mixer", END)

# LangGraph API（langgraph dev / cloud）由平台自动接管持久化，不允许自定义
# checkpointer，否则启动时直接 GraphLoadError。本地脚本/测试则保留自定义
# SqliteSaver 以获得读写并发健壮性。通过 LANGGRAPH_API 环境变量区分。
import os as _os

# LangGraph API（langgraph dev / cloud）由平台自动接管持久化，不允许自定义
# checkpointer，否则启动时直接 GraphLoadError。本地脚本/测试则保留自定义
# SqliteSaver 以获得读写并发健壮性。
# 通过 .env 中的 LANGGRAPH_API_MODE=1 显式声明运行在 LangGraph API 模式下。
_RUNNING_UNDER_LANGGRAPH_API = bool(_os.getenv("LANGGRAPH_API_MODE"))

if _RUNNING_UNDER_LANGGRAPH_API:
    multimedia_agent = workflow.compile()
else:
    multimedia_agent = workflow.compile(checkpointer=checkpointer)


def get_thread_state(thread_id: str):
    return multimedia_agent.get_state(make_thread_config(thread_id))

def get_thread_history(thread_id: str):
    return list(multimedia_agent.get_state_history(make_thread_config(thread_id)))

def get_latest_interrupt_payload(thread_id: str):
    history = get_thread_history(thread_id)
    if not history:
        return None
    latest = history[0]
    tasks = getattr(latest, "tasks", ()) or ()
    for task in tasks:
        interrupts = getattr(task, "interrupts", ()) or ()
        if interrupts:
            first = interrupts[0]
            return getattr(first, "value", first)
    return None
