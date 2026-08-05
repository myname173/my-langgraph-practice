# src/agent/multimedia/graph.py
import json
import re
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt

from .state import MultimediaState
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
from .visual_context import build_visual_context
from .shot_strategy import generate_shot_strategy, rewrite_strategy
from .sequence_orchestrator import build_sequence_graph, _EMOTION_ENERGY
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


def _normalize_decision(decision: Any) -> Dict[str, Any]:
    if isinstance(decision, dict):
        out = dict(decision)
    elif isinstance(decision, str):
        out = {"action": decision}
    elif isinstance(decision, bool):
        out = {"action": "approve" if decision else "rewrite"}
    else:
        out = {"action": "approve"}
    out["action"] = _normalize_action(out.get("action", "approve"))
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


def _scene_base(script: str) -> Dict[str, Any]:
    return {
        "script": script,
        "iterations": 0,
        "critique": "无",
        "is_perfect": False,
        "image_prompt": "",
        "image_url": "",
        "embedding_similarity": None,
        "image_auto_feedback": "",
        
        # 尾帧相关字段
        "last_image_prompt": "",
        "last_image_url": "",
        "last_image_iterations": 0,
        "last_image_critique": "无",
        "last_image_is_perfect": False,
        "last_image_auto_feedback": "",

        "video_iterations": 0,
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
    for field in ("shot_plan", "sequence_graph", "critic_eval", "studio_output"):
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

        # 风格后缀
        style_key = state.get("visual_style") or _detect_visual_style(state.get("task", ""))
        style_suffix = _STYLE_PRESETS.get(style_key, _STYLE_PRESETS[_DEFAULT_STYLE_KEY])["suffix"]

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

    return {
        "studio_output": studio_result,
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
        "formatted_block": str,  # 直接注入 director prompt 的格式化文本
    }
    """
    n = len(scenes)
    if n == 0:
        return {"formatted_block": ""}

    # ── 1. 为每个场景分配叙事功能 ──
    _FUNCTION_MAP = {
        0: "opening",        # 建立世界观、角色登场
    }
    scene_functions = []
    for i in range(n):
        if i == 0:
            scene_functions.append("opening")
        elif i == n - 1:
            scene_functions.append("resolution")
        elif i == n - 2 and n > 3:
            scene_functions.append("climax")
        elif n > 4 and i == (n // 2):
            scene_functions.append("turning_point")
        else:
            scene_functions.append("building")

    # ── 2. 情绪轨迹（线性递增/递减/弧形）──
    # 默认弧形：低→高→peak→缓降→收尾
    _ARC_ENERGY = {
        "opening": 0.4,
        "building": 0.55,
        "turning_point": 0.7,
        "climax": 0.95,
        "resolution": 0.6,
    }
    emotion_trajectory = []
    for i, func in enumerate(scene_functions):
        energy = _ARC_ENERGY.get(func, 0.5)
        emotion_trajectory.append((i + 1, energy, func))

    # ── 3. 因果链（从 script 提取承接/铺垫关系）──
    causal_chain = []
    for i, scene in enumerate(scenes):
        script = scene.get("script", "")[:80]
        func = scene_functions[i]
        causal_chain.append(f"Scene {i+1} [{func}]: {script}")

    # ── 4. 格式化输出块 ──
    func_lines = []
    for i, func in enumerate(scene_functions):
        script_preview = scenes[i].get("script", "")[:50]
        func_lines.append(f"    Scene {i+1} ({func}): \"{script_preview}...\"")

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
        "formatted_block": formatted,
    }


# ================= 节点定义 =================
def showrunner_node(state: MultimediaState):
    print("\n--- 👑[节点1: 总导演] 正在拆解长视频分镜剧本 ---")

    # 清空跨运行累积的风格记忆单例，防止脏数据污染 Global Film Brain 报告
    reset_memory()

    # 检测视觉风格（首次检测，存入 state 供下游复用）
    style_key = _detect_visual_style(state["task"])
    style_context = _build_style_context(style_key)
    print(f"    [*] 检测到视觉风格: {style_key}")

    # 检测内容类型
    content_type = _detect_content_type(state["task"])
    print(f"    [*] 检测到内容类型: {content_type}")

    prompt = SHOWRUNNER_PROMPT.format(
        task=state["task"],
        style_context=style_context,
    )
    response = call_llm(prompt, "总导演")

    parsed_data = safe_parse_json(response)

    global_setting = parsed_data.get("global_setting", "无全局设定")
    raw_scenes = parsed_data.get("scenes", [])

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
        scene_item = _scene_base(s.get("script", s.get("description", "")))
        processed_scenes.append(scene_item)
        print(f"    Shot {i+1} script ready: {scene_item['script'][:60]}...")

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
    decision = _normalize_decision(interrupt(payload))

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
    style_key = state.get("visual_style") or _detect_visual_style(task)
    style_suffix = _STYLE_PRESETS.get(style_key, {}).get("suffix", "")

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
}

# AAA 后缀关键词列表——用于检测 prompt 中是否已有部分风格信息
_AAA_STYLE_KEYWORDS = ["cinematic", "8k", "masterpiece", "unreal engine", "ray tracing"]

# 向后兼容：默认风格（当没有检测到任何 style 关键词时使用）
_DEFAULT_STYLE_KEY = "aaa_cg"

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

    # ── 环境锚点（防止跨场景环境漂移）──
    environment_anchor = _extract_environment_anchors(global_setting, scene["script"])
    if environment_anchor:
        print(f"    [*] Environment anchor injected")

    # 风格上下文
    style_key = state.get("visual_style") or _detect_visual_style(state.get("task", ""))
    style_context = _build_style_context(style_key)
    style_suffix = _STYLE_PRESETS.get(style_key, _STYLE_PRESETS[_DEFAULT_STYLE_KEY])["suffix"]

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
    )
    image_prompt = call_llm(prompt, "导演")

    # 安全网：截掉 director 可能附加的多镜头序列列表
    _preset = _get_style_preset(state)
    image_prompt = _sanitize_single_frame_prompt(
        image_prompt, _preset["suffix"], _preset.get("strip_keywords", [])
    )

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

    decision = _normalize_decision(interrupt(payload))
    action = decision.get("action", "approve")

    scenes = state["scenes"].copy()

    # 如果用户编辑了 prompt，直接替换
    if decision.get("image_prompt") is not None and action == "edit_prompt":
        edited = str(decision["image_prompt"]).strip()
        # 编辑后也走安全网
        _preset = _get_style_preset(state)
        edited = _sanitize_single_frame_prompt(
            edited, _preset["suffix"], _preset.get("strip_keywords", [])
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

    decision = _normalize_decision(interrupt(payload))
    action = decision.get("action", "approve")

    scenes = state["scenes"].copy()

    # 如果用户编辑了 prompt，直接替换
    if decision.get("image_prompt") is not None and action == "edit_prompt":
        edited = str(decision["image_prompt"]).strip()
        _preset = _get_style_preset(state)
        edited = _sanitize_single_frame_prompt(
            edited, _preset["suffix"], _preset.get("strip_keywords", [])
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
    if identity_anchors:
        enhanced_prompt = enhanced_prompt + "\n" + "\n".join(identity_anchors)
        print(f"    [*] Injected {len(identity_anchors)} identity anchor(s) into prompt")

    if reference_url:
        image_url = generate_keyframe(enhanced_prompt,
                                      reference_image_url=reference_url,
                                      ref_strength=ref_strength)
    else:
        image_url = generate_keyframe(enhanced_prompt)

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
            print(f"    [*] embedding 相似度: {similarity_score:.4f}")
        except Exception as e:
            print(f"    ⚠️ embedding 计算失败，继续人工审核: {str(e)}")

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
        "auto_feedback": scene.get("image_auto_feedback", scene.get("critique", "")),
        "message": "请审核关键帧：可通过、重写，或修改 image_prompt 后继续。",
        "actions":["approve", "rewrite", "edit_prompt"],
    }

    decision = _normalize_decision(interrupt(payload))
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

    # 风格上下文
    style_key = state.get("visual_style") or _detect_visual_style(state.get("task", ""))
    style_context = _build_style_context(style_key)
    style_suffix = _STYLE_PRESETS.get(style_key, _STYLE_PRESETS[_DEFAULT_STYLE_KEY])["suffix"]

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
    )
    last_image_prompt = call_llm(prompt, "导演")

    # 安全网：清理序列语言、中文、强制 style suffix
    _preset = _get_style_preset(state)
    last_image_prompt = _sanitize_single_frame_prompt(
        last_image_prompt, _preset["suffix"], _preset.get("strip_keywords", [])
    )

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

    last_image_url = generate_keyframe(
        end_frame_prompt,
        reference_image_url=scene["image_url"],
        ref_strength=ref_strength,
        negative_prompt=end_frame_negative,
    )
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

    decision = _normalize_decision(interrupt(payload))
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
    scenes[idx]["video_iterations"] = scenes[idx].get("video_iterations", 0) + 1

    try:
        last_url = scene.get("last_image_url", "") if state.get("use_first_last_frame") else ""
        raw_video_url = generate_video_from_image(scene["image_url"], scene["video_prompt"], last_url)

        scenes[idx]["raw_video_url"] = raw_video_url
        scenes[idx]["final_video_url"] = raw_video_url
        scenes[idx]["video_is_perfect"] = False

        print(f"    🌟 镜头 {idx+1} 制作完成！URL: {raw_video_url}")
        return {"scenes": scenes, "aborted": False, "abort_reason": None}

    except FreeTierQuotaExhaustedError as e:
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
        msg = f"视频生成失败: {str(e)}"
        print(f"    ⛔ [视频中止] {msg}")
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
        "message": "请审核视频成片：可通过、重写，或修改 video_prompt 后再生成。",
        "actions": ["approve", "rewrite", "edit_prompt"],
    }

    decision = _normalize_decision(interrupt(payload))
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

    提取来源：
    1. shot_plan.focus_object — 镜头焦点对象（如"剑客"、"dragon"）
    2. scene script — 外观描述关键词（发型、服饰、装备等）

    Returns:
        {
            "subject": str,          # 角色主体标识
            "appearance": List[str], # 外观特征关键词
            "has_character": bool,   # 是否检测到角色存在
        }
    """
    scene = state["scenes"][state["current_scene_index"]]
    script = scene.get("script", "")
    shot_plan = state.get("shot_plan")
    script_lower = script.lower()

    identity = {"subject": "", "appearance": [], "has_character": False}

    # ── 1. 从 shot_plan 获取焦点对象 ──
    if shot_plan:
        focus = shot_plan.get("focus_object", "")
        if focus:
            identity["subject"] = focus
            identity["has_character"] = True

    # ── 2. 从剧本提取外观描述关键词 ──
    # 发型/面容
    _APPEARANCE_PATTERNS = {
        "hair": [
            "长发", "短发", "白发", "黑发", "金发", "红发", "银发",
            "long hair", "short hair", "white hair", "black hair",
            "blonde", "red hair", "silver hair", "ponytail", "braid",
        ],
        "clothing": [
            "铠甲", "盔甲", "长袍", "斗篷", "风衣", "战甲", "披风", "盔甲",
            "armor", "cloak", "robe", "coat", "cape", "vest", "suit",
            "leather", "chainmail", "hooded",
        ],
        "equipment": [
            "剑", "刀", "弓", "枪", "盾", "法杖", "权杖", "双刃",
            "sword", "blade", "bow", "staff", "shield", "spear", "wand",
            "katana", "scythe", "halberd",
        ],
        "feature": [
            "伤疤", "纹身", "面具", "眼罩", "角", "翅膀", "机械臂",
            "scar", "tattoo", "mask", "eyepatch", "horns", "wings",
            "cybernetic", "mechanical arm", "glowing eyes",
        ],
    }

    for category, keywords in _APPEARANCE_PATTERNS.items():
        for kw in keywords:
            if kw.lower() in script_lower:
                identity["appearance"].append(kw)

    # 如果检测到外观关键词但没有 subject，从剧本中提取角色名词
    if identity["appearance"] and not identity["subject"]:
        # 常见角色名词模式
        _CHARACTER_NOUNS = [
            "剑客", "战士", "骑士", "英雄", "法师", "刺客", "猎人",
            "女巫", "武士", "忍者", "将军", "女王", "王子", "公主",
            "warrior", "knight", "hero", "mage", "assassin", "hunter",
            "witch", "samurai", "ninja", "general", "queen", "prince",
        ]
        for noun in _CHARACTER_NOUNS:
            if noun.lower() in script_lower:
                identity["subject"] = noun
                identity["has_character"] = True
                break

    if identity["appearance"]:
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
        return {"final_movie_path": "", "error_log": "所有镜头视频生成失败，无法拼接"}

    if skipped:
        print(f"    [*] 将拼接 {len(video_urls)}/{len(state['scenes'])} 个镜头（跳过: {skipped}）")

    # ── 提取转场数据 ──
    transitions: List[str] = []
    seq_graph = state.get("sequence_graph")
    if seq_graph and seq_graph.get("connections"):
        transitions = [
            c.get("transition", "hard_cut")
            for c in seq_graph["connections"]
        ]
        non_hardcut = [t for t in transitions if t in ("dissolve", "fade_through_black",
                                                        "whip_pan", "smash_cut", "camera_carry")]
        if non_hardcut:
            print(f"    [*] 序列转场: {len(non_hardcut)} 个效果转场将从 sequence_graph 应用")

    # ── 动态文件名 ──
    task = state.get("task", "")
    output_filename = ""  # 空字符串让 stitch_videos 自动生成

    try:
        final_movie_path = stitch_videos(
            video_urls,
            output_filename=output_filename,
            transitions=transitions,
        )
        return {"final_movie_path": final_movie_path}
    except Exception as e:
        print(f"    ❌ 拼接失败: {e}")
        return {"final_movie_path": "", "error_log": f"视频拼接失败: {str(e)[:200]}"}


# ================= 路由逻辑 =================
def decide_image_quality(state: MultimediaState):
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    if scene["is_perfect"] or scene["iterations"] >= 3:
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
    if state.get("aborted"):
        return "abort"
    return "video_reviewer"

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
        return "director"

    if critic_eval.get("pass_gates", False):
        print(f"\n✅ [路由] 质量门控通过 (score={critic_eval['overall_score']:.2f})，进入导演")
        return "director"

    if rewrite_count >= max_rewrites:
        print(f"\n⚠️ [路由] 已达最大重写次数 ({max_rewrites})，强制进入导演")
        return "director"

    print(f"\n🔄 [路由] 质量门控未通过 (score={critic_eval['overall_score']:.2f})，触发重写 #{rewrite_count + 1}")
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

workflow.add_edge("image_generator", "reviewer")
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

workflow.add_edge("end_frame_generator", "end_frame_review")
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
workflow.add_edge("stitcher", END)

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
