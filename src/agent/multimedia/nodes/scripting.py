# src/agent/multimedia/nodes/scripting.py
"""脚本与规划阶段：视觉上下文、镜头策略、序列编排、showrunner 拆镜。

P2-3：由原 graph.py（单体）按 stage 机械拆分而来，逻辑逐字节保留；
仅新增本文件头注释与模块导入，函数体 / 常量 / 注释均未改动。
"""
import json
from typing import Any, Dict, List
from langchain_core.runnables import RunnableConfig
from .. import run_metrics
from ..cinematic_memory import reset_memory
from ..config_loader import apply_user_settings
from ..prompts import SAFETY_PROMPT, SHOWRUNNER_PROMPT
from ..rag.retriever import retrieve_rag_context
from ..sequence_orchestrator import build_sequence_graph
from ..shot_strategy import generate_shot_strategy, rewrite_strategy
from ..state import MultimediaState
from ..tools.text_llm import call_llm
from ..visual_context import build_visual_context
from .common import _build_style_context, _decide, _detect_content_type, _estimate_scene_duration, _normalize_scenes, _resolve_style, _sanitize_script, _scene_base, safe_parse_json
from ..tools.narrative import (
    episode_beats_hint,
    format_episode_anchors_hint,
    expand_beats,
    resolve_episode,
    resolve_target_duration,
    scene_count_hint,
    template_for_content_type,
)


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
    # 含 checkpoint 残留清理：旧 run 的 scenes 内 per-scene 运行态字段
    # （video_gen_failures / iterations）若被持久化，会在新 run 复用时泄漏，
    # 导致路由误判/重复消耗。进入"处理当前镜"前强制清零这些运行态字段。
    scene_resets = {}
    for field in ("shot_plan", "sequence_graph", "critic_eval",
                  "studio_output", "studio_output_path", "studio_summary",
                  "video_gen_failures", "iterations"):
        _v = state.get(field)
        # iterations 是 scene 级字段，从 scenes[idx] 取；其余是顶层 state 字段
        if field in ("video_gen_failures", "iterations"):
            _v = (state.get("scenes", [])[idx].get(field) if idx < len(state.get("scenes", [])) else None)
        if _v is not None:
            scene_resets[field] = 0 if field in ("video_gen_failures", "iterations") else None
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
            from ..visual_context import build_visual_context as _build_vc
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


# ================= P2-1: Narrative Arc Builder =================
def _build_narrative_arc(scenes: List[Dict], template_key: str | None = None) -> Dict[str, Any]:
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
    # P2-5：回退不再是死板的位置规则，而是「模板驱动」——按模板把 n 个
    # 镜头自适应伸缩成一条叙事功能序列（expand_beats 保证首 setup、末 resolution、
    # climax 靠后），LLM 未给 beat 的镜头按模板落位。这让 15s 快闪与 3min 短剧集
    # 各自拥有合适的节奏形状，而非共用一套固定规则。
    _template_beats = expand_beats(template_key, n)
    scene_functions = []
    used_beats = set()
    for i, scene in enumerate(scenes):
        raw_beat = (scene.get("beat") or "").strip().lower()
        if raw_beat in _VALID_BEATS:
            scene_functions.append(raw_beat)
            used_beats.add(raw_beat)
        else:
            _b = _template_beats[i] if i < len(_template_beats) else "develop"
            scene_functions.append(_b)
            used_beats.add(_b)

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
def showrunner_node(state: MultimediaState, config: RunnableConfig | None = None):
    print("\n--- 👑[节点1: 总导演] 正在拆解长视频分镜剧本 ---")

    # 设置面板跨进程生效：每次 run 开始前把 data/user_settings.json 的非空键
    # 重新合并进本进程 os.environ（mtime 缓存，未变更时零开销）。
    apply_user_settings()

    # 运行指标埋点：开启一条 run record（可观测性）
    run_metrics.start_run(state.get("task", ""), len(state.get("scenes", [])))

    # P0-3：按线程清空风格记忆库（缓存 + data/memory/<thread>.json 落盘），
    # 防止跨运行脏数据污染 Global Film Brain 报告；恢复中断线程时不经过本节点，
    # 记忆库落盘得以保留并由 film_studio 首次访问时自动重建。
    _cfg0 = (config or {}).get("configurable", {}) if config else {}
    reset_memory(str(_cfg0.get("thread_id") or state.get("thread_id") or ""))

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

    # ── P2-5：镜头数不再硬编码，而是由「目标时长 + content_type」反推 ──
    # SMOKE / 显式 max_shots 仍优先（免费额度下的截断保护）。
    cfg = (config or {}).get("configurable", {}) if config else {}
    max_shots = cfg.get("max_shots") or (2 if cfg.get("smoke") else None)
    target_duration = resolve_target_duration(state)
    episode = resolve_episode(state)
    planned_scenes, shot_hint = scene_count_hint(
        target_duration, content_type, max_shots=max_shots
    )
    if episode > 1:
        shot_hint += episode_beats_hint(episode)
        shot_hint += format_episode_anchors_hint(state.get("episode_anchors"))
    print(f"    [*] P2-5 叙事参数：target_duration={target_duration}s "
          f"episode={episode} content_type={content_type} → 规划 {planned_scenes} 镜")
    prompt = SHOWRUNNER_PROMPT.format(
        task=state["task"],
        style_context=style_context,
        scene_count_hint=shot_hint,
    )

    # ── 故事完整性 + 内容安全 自愈循环（借鉴 LangGraph 多智能体 Reviewer 校验→重跑）──
    # 若总导演生成的剧本缺三幕弧线环节（如缺 turn/climax/resolution），
    # 把缺失环节作为补全指令追加进 prompt 重新生成，最多重试 2 次，保证"情节一定完整"。
    # 若内容安全审核 FAIL，则把违规摘要反馈给总导演要求「降尺度改写」（去掉写实人体伤害
    # 描写、改用写意侧面表现对抗），最多重试 2 次；重试耗尽仍 FAIL 才优雅终止，避免裸
    # raise 让整 run 崩溃（此前曾因"刺胸/喷血/骨裂"描写被安全审核判 FAIL 直接中断全流程）。
    raw_scenes = []
    global_setting = "无全局设定"
    _MAX_SHOWRUNNER_RETRIES = 2
    _safety_feedback = ""  # 安全审核失败时的违规摘要，注入下一轮改写
    for _attempt in range(_MAX_SHOWRUNNER_RETRIES + 1):
        # 每轮重建 prompt（基础 prompt + 镜头数约束 + 上一轮反馈），保证重试干净
        _base_prompt = SHOWRUNNER_PROMPT.format(
            task=state["task"],
            style_context=style_context,
            scene_count_hint=shot_hint,
        )
        if _safety_feedback:
            _base_prompt += _safety_feedback
        prompt = _base_prompt
        # 显式放大 max_tokens：已精简 prompt（可选字段可省略），但仍需足够空间写完
        # 6-9 个镜头的 JSON，避免免费额度高峰期因默认 1536 过早截断导致 0 镜头。
        response = call_llm(prompt, "总导演", max_tokens=4096)
        # partial=True：截断/非法 JSON 返回空 dict 而非抛异常，交由下方重试循环兜底
        parsed_data = safe_parse_json(response, partial=True)
        global_setting = parsed_data.get("global_setting", global_setting)
        raw_scenes = parsed_data.get("scenes", [])
        if not raw_scenes:
            print("    [!] 总导演未返回有效 scenes（可能输出被截断），重试补全...")
            _safety_feedback = ""
            continue
        # 用已解析的 scene 字段做三幕弧线校验
        _probe = [{
            "script": s.get("script", s.get("description", "")),
            "beat": s.get("beat", ""),
        } for s in raw_scenes]
        _arc = _build_narrative_arc(_probe, template_for_content_type(content_type))
        _missing = _arc.get("missing_beats", [])
        # P2-5：3 镜以下天然放不下完整三幕弧线（模板已尽力保 setup/climax/resolution），
        # 不对其强制要求 develop/turn，避免无谓重试。
        if _missing and len(_probe) > 3 and not (cfg.get("max_shots") and cfg.get("max_shots") <= 3):
            print(f"    [!] 故事不完整，缺三幕环节 {_missing}，要求总导演补全后重试 "
                  f"({_attempt+1}/{_MAX_SHOWRUNNER_RETRIES})...")
            _safety_feedback = (
                f"\n\n[校验反馈-必须修正] 你刚才的剧本不完整：缺少三幕弧线上的环节 {_missing}。"
                f"请在不删减现有镜头的前提下，补出对应镜头，使其构成 "
                f"起(setup/inciting)→承(develop)→转(turn)→合(climax)→收(resolution) 的完整闭环，"
                f"并重写完整 JSON。"
            )
            continue
        # 三幕完整 → 进入内容安全审核
        safety_check_prompt = (
            SAFETY_PROMPT
            + f"\n\n待检查内容：\n全局视觉设定：{global_setting}\n\n分镜列表：\n"
            + json.dumps([s.get("script", "") for s in raw_scenes], ensure_ascii=False, indent=2)
        )
        safety_result = call_llm(safety_check_prompt, "安全审核员")
        if "FAIL" in safety_result.upper():
            print(f"    ❌ 内容安全审核失败（第 {_attempt+1}/{_MAX_SHOWRUNNER_RETRIES+1} 次）: {safety_result}")
            if _attempt < _MAX_SHOWRUNNER_RETRIES:
                # 把违规摘要反馈给总导演，要求降尺度改写（保留仙侠对决骨架，去掉写实人体伤害）
                _violation = safety_result.replace("FAIL:", "").strip()[:400]
                _safety_feedback = (
                    f"\n\n[内容安全校验反馈-必须修正] 你的剧本被内容安全审核判定为违规，原因：{_violation}。"
                    f"请在不改变故事主线（正邪对决、守护玉佩）与镜头总数的前提下，重写为**非写实、写意、风格化**的"
                    f"对抗描写：禁止刺穿身体、流血喷血、骨折白骨外露、伤口特写等真实人体伤害刻画；改用剑光交错、"
                    f"灵力震荡气浪、衣袂染尘、护盾碎裂、符文明灭、能量对撞等侧面写意表现。重写完整 JSON。"
                )
                continue
            else:
                # 重试耗尽仍违规 → 优雅终止（保全日志、前端可读 abort_reason），不裸 raise
                return _abort_run(f"内容安全审核多次未通过，已终止生成：{safety_result[:200]}")
        print("    ✅ 初始内容安全审核通过")
        break  # 完整且安全，跳出重试

    print(f"[*] 成功生成全局设定: {global_setting[:60]}...")
    print(f"[*] 成功拆解出 {len(raw_scenes)} 个连续镜头！")

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
        scene_item["dialogue"] = (s.get("dialogue") or "").strip()
        # ── 分镜时长：优先采用 LLM 显式给出的 duration_seconds，否则按台词长度估算 ──
        scene_item["duration_seconds"] = _estimate_scene_duration(
            scene_item["dialogue"], s.get("duration_seconds")
        )
        processed_scenes.append(scene_item)
        print(f"    Shot {i+1} script ready: {scene_item['script'][:60]}...")
        if scene_item["action_beat"]:
            print(f"         ↳ 可见动作: {scene_item['action_beat'][:50]}...")
        if scene_item["visual_elements"]:
            print(f"         ↳ 视觉细节: {len(scene_item['visual_elements'])} 项")

    # ── P2-1: Build narrative arc from scenes (rule-based, no LLM call) ──
    narrative_arc = _build_narrative_arc(processed_scenes, template_for_content_type(content_type))
    print(f"    [*] Narrative arc built: {len(processed_scenes)} scenes, "
          f"emotion curve: {' -> '.join(f'{e:.1f}' for _, e, _ in narrative_arc.get('emotion_trajectory', []))}")

    return {
        "global_setting": global_setting,
        "scenes": processed_scenes,
        "current_scene_index": 0,
        "visual_style": style_key,
        "content_type": content_type,
        "target_duration": target_duration,
        "episode": episode,
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


def showrunner_review_gate_node(state: MultimediaState, config: dict | None = None):
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "") if config else ""
    payload = {
        "stage": "showrunner_review",
        "title": "总导演人工审片",
        "task": state["task"],
        "global_setting": state["global_setting"],
        "scenes": state["scenes"],
        "message": "请审查总导演输出的全局设定与分镜列表。可直接通过，也可修改 global_setting / scenes 后再继续。",
        "actions":["approve", "rewrite", "edit_prompt"],
    }
    decision = _decide("showrunner_review", payload, state)

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
        # 不再主动用 jimeng 生成角色肖像（避免 jimeng 产物被当作参考图素材）。
        # 若确实需要，应由用户从素材库注入参考图，或显式调用 reference_gen_node。
        print("    [Skip] 未注入 character_portrait_url，跳过 jimeng 肖像生成（避免污染参考图素材库）")

    return {
        "task": task,
        "global_setting": global_setting,
        "scenes": scenes,
        "character_portrait_url": character_portrait_url,
        "thread_id": thread_id or state.get("thread_id"),
        "visual_style": state.get("visual_style"),
        "style_suffix": state.get("style_suffix"),
        "style_description": state.get("style_description"),
    }
