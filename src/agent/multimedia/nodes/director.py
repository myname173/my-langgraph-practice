# src/agent/multimedia/nodes/director.py
"""导演阶段：电影级评审、导演精修、影棚协作、首尾帧导演、运镜设计。

P2-3：由原 graph.py（单体）按 stage 机械拆分而来，逻辑逐字节保留；
仅新增本文件头注释与模块导入，函数体 / 常量 / 注释均未改动。
"""
import time
from .. import run_metrics
from ..cinematic_critic import QUALITY_THRESHOLDS, evaluate_cinematic_quality, format_critic_report
from ..config_loader import persist_studio_output as _persist_studio_output
from ..film_studio_orchestrator import orchestrate_film_studio
from ..prompts import DIRECTOR_REFINEMENT_PROMPT, DIRECTOR_SYSTEM_PROMPT, END_FRAME_DIRECTOR_PROMPT, VIDEOGRAPHER_DUAL_FRAME_PROMPT, VIDEOGRAPHER_PROMPT
from ..state import MultimediaState
from ..tools.text_llm import call_llm
from ..tools.vision_eval import design_camera_movement
from .common import _build_style_context, _decide, _extract_environment_anchors, _extract_protected_specs, _finalize_single_frame_prompt, _get_conflict_keywords, _get_style_preset, _resolve_style, _sanitize_single_frame_prompt, _visual_script


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
    _t0 = time.time()

    print(f"\n--- 🔍 [电影评审] 镜头 {idx+1}/{total_scenes} 结构质量评估 (rewrite #{rewrite_count}) ---")

    if not shot_plan:
        print("    [!] shot_plan 缺失，跳过评估")
        run_metrics.record_text_llm_eval(
            "cinematic_critic", idx, "skip(no shot_plan)",
            time.time() - _t0, 0, outcome="skip"
        )
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

    run_metrics.record_text_llm_eval(
        "cinematic_critic", idx, "结构质量评估",
        time.time() - _t0, 0,
        outcome="PASS" if eval_result["pass_gates"] else "FAIL",
        score=round(eval_result["overall_score"], 3),
    )

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
    _t0 = time.time()

    image_prompt = scene.get("image_prompt", "")
    if not image_prompt:
        print(f"    [!] 无 image_prompt 可精修，跳过")
        run_metrics.record_text_llm_eval(
            "director_refine", idx, "skip(no image_prompt)",
            time.time() - _t0, 0, outcome="skip"
        )
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

        _llm_t0 = time.time()
        refined_prompt = call_llm(prompt, "导演精修")
        _llm_dt = time.time() - _llm_t0

        scenes = state["scenes"].copy()
        scenes[idx]["image_prompt"] = refined_prompt
        print(f"    [*] 精修完成 ({len(refined_prompt)} chars, 耗时 {_llm_dt:.1f}s)")
        run_metrics.record_text_llm_eval(
            "director_refine", idx, "导演精修",
            time.time() - _t0, len(refined_prompt), outcome="refined"
        )
        return {"scenes": scenes}
    else:
        print(f"    [*] 无精修建议，保持原始 image_prompt")
        run_metrics.record_text_llm_eval(
            "director_refine", idx, "passthrough(no suggestions)",
            time.time() - _t0, 0, outcome="passthrough"
        )
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
    _t0 = time.time()

    print(f"\n--- 🎬 [电影工作室] 镜头 {idx+1}/{total_scenes} 多 Agent 协作优化 ---")

    if not shot_plan:
        print("    [!] shot_plan 缺失，跳过工作室优化")
        run_metrics.record_text_llm_eval(
            "film_studio", idx, "skip(no shot_plan)",
            time.time() - _t0, 0, outcome="skip"
        )
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
        thread_id=state.get("thread_id", ""),
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

    run_metrics.record_text_llm_eval(
        "film_studio", idx, "多Agent协作优化",
        time.time() - _t0, len(report),
        outcome="consensus",
        score=round(studio_result.get("studio_consensus", 0.0), 3),
    )

    return {
        "studio_output": None,             # 剥离大对象，避免随 state 入库
        "studio_output_path": studio_output_path,
        "studio_summary": studio_summary,
        "shot_plan": refined_shot_plan,
        "sequence_graph": refined_seq_graph,
    }


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
    protected_specs_text = _extract_protected_specs(_visual_script(scene["script"]), global_setting)

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
    environment_anchor = _extract_environment_anchors(global_setting, _visual_script(scene["script"]))
    if environment_anchor:
        print(f"    [*] Environment anchor injected")

    # ── 情节密度字段（Phase 0 结构化输出，强化首帧可见内容）──
    action_beat = scene.get("action_beat", "")
    visual_elements = scene.get("visual_elements", []) or []
    visual_elements_block = "\n".join(f"- {v}" for v in visual_elements) if visual_elements else "（无额外细节，请从 script 与环境锚点中自行推断具体可见细节）"
    camera_note = scene.get("camera_note", "")
    emotion = scene.get("emotion", "")
    transition_in = scene.get("transition_in", "")
    # 跨场景连续性指令：把 visual_context_node 存入 scene_anchors 的 continuity_directives
    # 文本化并入 transition_in（导演模板已有占位符），使导演与关键帧都感知"延续上一镜"，
    # 修复"分镜之间完全没有联系"。仅非首镜注入（首镜无承接）。
    continuity_directives = (state.get("scene_anchors") or {}).get("continuity_directives") or []
    if continuity_directives and idx > 0:
        _cd = "；".join(str(d) for d in continuity_directives)
        _sep = "" if "（开场" in (transition_in or "") else "；"
        transition_in = (transition_in or "") + _sep + f"【延续上一镜】{_cd}"

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

    # 安全网 + 翻译层（统一后置处理）
    image_prompt = _finalize_single_frame_prompt(state, image_prompt)

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

    decision = _decide("prompt_preview", payload, state)
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

    decision = _decide("end_frame_prompt_preview", payload, state)
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
    protected_specs_text = _extract_protected_specs(_visual_script(scene["script"]), global_setting)

    # ── 环境锚点（防止跨场景环境漂移）──
    environment_anchor = _extract_environment_anchors(global_setting, _visual_script(scene["script"]))

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

    # 安全网 + 翻译层（统一后置处理）
    last_image_prompt = _finalize_single_frame_prompt(state, last_image_prompt)

    scenes = state["scenes"].copy()
    scenes[idx]["last_image_prompt"] = last_image_prompt
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

    # ── 统一文本 qwen 模型所需的视觉锚点与风格上下文（与首尾帧对齐）──
    # 首帧英文描述（已过 _sanitize_single_frame_prompt + _translate_to_english）
    # 作为「画面锚点」，弥补文本模型看不到首帧图的缺陷。
    first_frame_prompt = scene.get("image_prompt", "")
    last_frame_prompt = scene.get("last_image_prompt", "") if state.get("use_first_last_frame") else ""
    global_setting = state.get("global_setting", "")
    style = _resolve_style(state)
    style_suffix = style.get("style_suffix", "")
    action_beat = scene.get("action_beat", "")
    visual_elements = scene.get("visual_elements", []) or []
    visual_elements_block = (
        "\n".join(f"- {v}" for v in visual_elements)
        if visual_elements
        else "（无额外细节，请从首帧锚点与剧本中自行推断具体可见的动态细节）"
    )

    video_prompt = design_camera_movement(
        scene["image_url"],
        script_for_camera,
        prompt_template,
        critique=scene.get("video_critique", "无"),
        camera_context=camera_context,
        cross_scene_context=cross_scene_context,
        image_prompt=first_frame_prompt,
        global_setting=global_setting,
        style_suffix=style_suffix,
        action_beat=action_beat or "（无，请从 script 与首帧锚点中推断主体可见动作）",
        visual_elements=visual_elements_block,
        first_frame_prompt=first_frame_prompt,
        last_frame_prompt=last_frame_prompt,
    )
    print(f"    [*] 运镜提示词: {video_prompt}")

    scenes = state["scenes"].copy()
    scenes[idx]["video_prompt"] = video_prompt
    return {"scenes": scenes}
