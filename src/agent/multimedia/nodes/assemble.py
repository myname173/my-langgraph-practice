# src/agent/multimedia/nodes/assemble.py
"""成片阶段：镜头推进 / 中止 / 拼接 / 草稿审片 / 音频混音。

P2-3：由原 graph.py（单体）按 stage 机械拆分而来，逻辑逐字节保留；
仅新增本文件头注释与模块导入，函数体 / 常量 / 注释均未改动。
"""
import os
import os as _os
import re
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from .. import run_metrics
from ..sequence_orchestrator import _EMOTION_ENERGY, select_cross_scene_transition
from ..state import MultimediaState
from ..tools.audio import build_segmented_voiceover, build_sfx_events, generate_bgm, has_audio_track, mix_audio, resolve_bgm_mood
from ..tools.subtitle import build_srt, burn_subtitles, parse_srt_to_entries
from ..tools.tts import generate_voiceover
from ..tools.video_pack import apply_color_preset, attach_title_cards
from ..tools.video_post import enhance_final_cut
from ..tools.video_stitcher import stitch_videos
from ..tools.vision_eval import critic_final_cut
from .common import _normalize_decision


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


def stitcher_node(state: MultimediaState, config: RunnableConfig | None = None):
    print("\n--- 🎬 [剪辑师] 正在将所有镜头拼接成最终宣传片 ---")
    video_urls = []
    skipped = []
    # 【修复】used_scenes 必须与 video_urls 严格一一对应（同一套过滤条件），
    # 否则跨场景转场会算错：此前 used_scenes 只按 final_video_url 过滤（未排除
    # shot_failed），而 video_urls 排除了 shot_failed，且 stitch_videos 内部还会
    # 二次跳过占位片/空 URL，导致 transitions 与实际参与拼接的 clips 数量和
    # 相邻关系双双错位——转场会张冠李戴（本该 A→C 的语义转场被算成 A→B）。
    used_scenes: List[dict] = []
    for i, scene in enumerate(state["scenes"]):
        # shot_failed 镜头（额度耗尽/生成失败写入的占位片）不再拼入成片，
        # 否则成片出现"黑屏"。改为直接跳过，由相邻成功镜头经转场自然衔接。
        if scene.get("shot_failed"):
            skipped.append(i + 1)
            print(f"    ⏭️ 镜头 {i + 1} 标记为 shot_failed（占位片），跳过不拼入成片（避免黑屏）")
            continue
        url = scene.get("final_video_url", "")
        if url:
            video_urls.append(url)
            used_scenes.append(scene)
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

    # 成片按 thread_id 隔离，避免多次运行互相覆盖导致前端“张冠李戴”
    thread_id = ""
    if config:
        thread_id = (config.get("configurable") or {}).get("thread_id", "") or ""
    output_filename = ""
    if thread_id:
        # 仓库根：graph.py 位于 <root>/src/agent/multimedia/，需上溯 4 层
        base_output = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "output",
        )
        thread_clips = os.path.join(base_output, "clips", thread_id)
        os.makedirs(thread_clips, exist_ok=True)
        output_filename = os.path.join(thread_clips, "FINAL_成片.mp4")

    # ── 提取转场数据（基于各场景的叙事功能 beat，做跨场景语义衔接）──
    # 说明：state["sequence_graph"] 是「单个场景内部」的镜头关系图，不能直接当作
    # 跨场景转场使用（否则会错用最后一个场景的内部转场）。正确做法是根据每个
    # 场景的 beat + emotion，用 select_cross_scene_transition 计算相邻场景的转场。
    transitions: List[str] = []
    # used_scenes 已在上方与 video_urls 同步构建，此处不得重新按 final_video_url
    # 过滤（那会把 shot_failed 占位镜头算进来，导致转场与片段错位）。
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

    # ── 成片路径：严格按 thread_id 隔离，避免不同任务互相覆盖导致前端“张冠李戴” ──
    # 注意：上方已基于 thread_id 算出 output_filename（output/clips/<thread_id>/FINAL_成片.mp4），
    # 这里【绝不能】再把它重置为空，否则 stitch_videos 会把成片落到项目根目录/通用位置，
    # 写回 checkpoint 的路径也会错，最终历史页就会串片。
    if not output_filename:
        # 兜底：极端情况下拿不到 thread_id，仍按任务+时间戳隔离，绝不写通用文件名
        task = state.get("task", "")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^\w\u4e00-\u9fff]", "", (task or "video")[:20])
        output_filename = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "output", "clips", f"{slug}_{ts}", "FINAL_成片.mp4",
        )
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # P2-2：后处理板（默认关）——env MULTIMEDIA_POST_HEIGHT=1080 / MULTIMEDIA_POST_FPS=30
    # 清晰度增强（lanczos+unsharp）与补帧（dup/mci），免费后端低配产物的观感补齐。
    _post_height = None
    _post_fps = None
    _ph = (_os.getenv("MULTIMEDIA_POST_HEIGHT") or "").strip()
    _pf = (_os.getenv("MULTIMEDIA_POST_FPS") or "").strip()
    try:
        if _ph:
            _post_height = int(_ph)
        if _pf:
            _post_fps = int(_pf)
    except ValueError:
        print("    [WARN] MULTIMEDIA_POST_HEIGHT/FPS 需为整数，忽略")

        # P1-2：卡点对齐开关 —— env MULTIMEDIA_BEAT_GRID 填 BPM 数值（如 120）时，
    # 每段微调到节拍网格（只剪不补，最多 0.6s）；默认关闭，保持旧行为。
    beat_grid = None
    _bg = (_os.getenv("MULTIMEDIA_BEAT_GRID") or "").strip()
    if _bg:
        try:
            _bpm = float(_bg)
            if _bpm >= 30:
                beat_grid = round(60.0 / _bpm, 3)
                print(f"    [P1-2] 卡点对齐开启：BPM={_bpm:g} → 网格 {beat_grid}s")
        except ValueError:
            print(f"    [WARN] MULTIMEDIA_BEAT_GRID='{_bg}' 不是有效 BPM，忽略")

    try:
        final_movie_path, valid_indices, segment_durations, segment_starts = stitch_videos(
            video_urls,
            output_filename=output_filename,
            transitions=transitions,
            beat_grid=beat_grid,
        )

        # P2-2：后处理板（在包装卡之前处理，避免把卡也插值/放大）
        if _post_height or _post_fps:
            final_movie_path = enhance_final_cut(
                final_movie_path,
                target_height=_post_height,
                target_fps=_post_fps,
                interp=(_os.getenv("MULTIMEDIA_POST_INTERP") or "dup").strip().lower(),
            )

        # P2-1：包装层 —— 片头片尾卡（默认开；RunInput.title_card / env 可关）+ 色彩预设
        _intro = 0.0
        if _os.getenv("MULTIMEDIA_TITLE_CARD", "1") != "0" and state.get("title_card", True):
            _t = (state.get("task") or "").strip()
            if _t:
                _style = (state.get("style_description") or state.get("visual_style") or "").strip()
                final_movie_path, _intro, _outro = attach_title_cards(
                    final_movie_path,
                    title=_t[:24],
                    subtitle=_style[:26],
                )
                # P2-1 色彩预设（可选）：env MULTIMEDIA_COLOR_PRESET=warm|cool|noir
                _cp = (_os.getenv("MULTIMEDIA_COLOR_PRESET") or "").strip().lower()
                if _cp in ("warm", "cool", "noir"):
                    try:
                        from moviepy import VideoFileClip as _VFC
                        _pre = _os.path.splitext(final_movie_path)
                        _cp_out = _pre[0] + f"_color_{_cp}" + _pre[1]
                        _vc = _VFC(final_movie_path)
                        _styled = apply_color_preset(_vc, _cp)
                        _styled.write_videofile(
                            _cp_out, codec="libx264", audio_codec="aac",
                            preset="medium", ffmpeg_params=["-movflags", "+faststart"], logger=None,
                        )
                        _vc.close(); _styled.close()
                        final_movie_path = _cp_out
                        print(f"    [P2-1] 色彩预设 {_cp} 已应用")
                    except Exception as e:  # noqa: BLE001
                        print(f"    [WARN] 色彩预设失败(不阻断): {e}")
        # P1 字幕精确对齐：把各有效片段真实时长写回对应 scene，
        # 供 audio_mixer_node 的 build_srt 按真实时长逐句对齐（替代原均分策略）。
        scenes_patch = {}
        for k, (orig_idx, dur) in enumerate(zip(valid_indices, segment_durations)):
            if 0 <= orig_idx < len(state["scenes"]):
                state["scenes"][orig_idx]["video_duration"] = round(float(dur), 3)
                # 【修复】回写该片段在成片中的真实起始时间（已扣除转场重叠），
                # 字幕不再用「时长累加」，避免累计漂移与末条超出片尾。
                if k < len(segment_starts):
                    # P2-1：片头卡使全片时间轴后移 intro，字幕/配音起点统一偏移
                    state["scenes"][orig_idx]["segment_start"] = round(float(segment_starts[k]) + _intro, 3)
        # P2-4：成片拼好后把整任务「镜头配方卡」落盘（output/recipes/<thread>/），
        # 供前端查看/复制/分享，以及后续「按配方复现」。任何失败都不阻断成片产出。
        try:
            from ..tools import recipe as _recipe_mod  # noqa: PLC0415
            _task_recipe = _recipe_mod.build_task_recipe(state)
            _saved = _recipe_mod.save_task_recipe(_task_recipe)
            print(f"    [P2-4] 镜头配方卡已落盘: {_saved.get('latest')}")
        except Exception as _re:  # noqa: BLE001
            print(f"    [WARN] 镜头配方卡落盘失败(不阻断): {_re}")

        return {
            "final_movie_path": final_movie_path,
            "segment_durations": segment_durations,
        }
    except Exception as e:
        print(f"    ❌ 拼接失败: {e}")
        return {"final_movie_path": "", "error_log": f"视频拼接失败: {str(e)[:200]}"}


def draft_review_gate_node(state: MultimediaState):
    """成片草稿审片闸口：在「无声粗剪」之上、音频混音之前，给用户一次确认机会。

    对标 Runway 的 draft 机制——先看无声粗剪（剪辑节奏 / 镜头顺序 / 转场），
    满意后再叠加配音 + 字幕 + BGM，避免对剪辑不满意却要重跑整条生成链路。
    - auto_mode 为真：跳过人工卡，直接放行（与普通自动放行一致）。
    - enable_audio 为假或无粗剪：跳过（直接进 audio_mixer 或 END）。
    - approve：进入 audio_mixer 叠加音轨。
    - rewrite：回到 stitcher 重拼（支持调整镜头顺序后重新拼接）。
    """
    # 自动模式 / 未开音频闭环：直接放行，无草稿审片
    if state.get("auto_mode") or not state.get("enable_audio", True):
        # auto_mode 也跑成片 Critic（写入 state 供前端展示），但不打断流程
        _rep_auto = None
        if _os.getenv("MULTIMEDIA_FINAL_CRITIC", "1") != "0":
            _fm = state.get("final_movie_path") or ""
            if _fm and os.path.exists(_fm):
                try:
                    _rep_auto = critic_final_cut(_fm)
                except Exception:
                    _rep_auto = None
        return {"draft_decision": "approve", "draft_critic_report": _rep_auto}

    final_movie = state.get("final_movie_path") or ""
    if not final_movie or not os.path.exists(final_movie):
        # 无粗剪则跳过审片，直接进入后续（audio_mixer 内部也会因缺片跳过）
        return {"draft_decision": "approve"}

    # P1-2：成片 Critic —— 对无声粗剪按时间顺序抽帧复审（节奏/连续性/画质）。
    # 报告随草稿卡给用户参考，也写入 state 供前端展示；失败不阻断。
    critic_report = None
    if _os.getenv("MULTIMEDIA_FINAL_CRITIC", "1") != "0":
        try:
            critic_report = critic_final_cut(final_movie)
            if critic_report.get("available"):
                _iss = critic_report.get("issues") or []
                _sc = critic_report.get("score")
                print(
                    f"    [P1-2] 成片 Critic: score={_sc if _sc is not None else '—'} · "
                    f"issues={len(_iss)}"
                )
                for _it in _iss[:3]:
                    print(f"      · [{_it.get('severity')}/{_it.get('type')}] {_it.get('desc')}")
        except Exception as e:  # noqa: BLE001
            print(f"    [WARN] 成片 Critic 失败(不阻断): {e}")

    payload = {
        "stage": "draft_review",
        "title": "🎬 成片草稿（无声粗剪）确认",
        "message": "剪辑师已完成所有镜头拼接。请预览无声粗剪：节奏 / 镜头顺序 / 转场是否满意？"
        "满意则「通过」进入配音与配乐；不满意可「打回重拼」调整后重新拼接（无需重跑各镜头生成）。",
        "video_url": final_movie,
        "actions": ["approve", "rewrite"],
        # P1-2：成片 Critic 报告（前端草稿卡展示问题清单）
        "critic_report": critic_report,
    }
    decision = _normalize_decision(interrupt(payload), gate="draft_review")
    action = decision.get("action", "approve")
    # 运行指标埋点：记录草稿审片卡消费动作
    run_metrics.record_human_gate("draft_review", action)
    if action == "rewrite" and decision.get("reason"):
        print(f"    [草稿审片] 打回重拼，原因：{decision['reason'][:120]}")
    return {"draft_decision": action, "draft_critic_report": critic_report}


def audio_mixer_node(state: MultimediaState):
    """P0/P1 音频闭环：在无声成片之上叠加配音 + 字幕 + BGM + 混音。

    复用现成组件：edge-tts(TTS) / pysrt(SRT) / 预置曲库 BGM / moviepy(烧录+混音)。
    - enable_audio=False 时整阶段跳过（前端总开关）。
    - 若无声成片缺失则直接跳过，保证图不崩。
    - prefer_native_audio=True 且成片自带原生音轨时（如即梦 Seedance 1.5 Pro
      音画一体），优先保留原生音轨，仅烧录字幕，跳过本地 TTS 配音 / BGM 混音；
      若成片无原生音轨则自动回退到本地 TTS + BGM 混音兜底，保证有声音。
    """
    # 总开关：前端可关闭整个音频闭环
    if not state.get("enable_audio", True):
        print("\n--- 🔊 [音频师] 音频闭环已关闭（前端开关），跳过 ---")
        return {}

    final_movie = state.get("final_movie_path") or ""
    if not final_movie or not os.path.exists(final_movie):
        print("\n--- 🔊 [音频师] 无成片，跳过音频合成 ---")
        return {}

    # ── 原生音轨优先分支：保留即梦等模型生成的原生音轨，仅烧录字幕 ──
    prefer_native = bool(state.get("prefer_native_audio", False))
    has_native = has_audio_track(final_movie)
    if prefer_native and has_native:
        print("\n--- 🔊 [音频师] 检测到原生音轨，采用「原生音轨优先」模式 ---")
        srt_path = build_srt(state["scenes"])
        try:
            with_subs = burn_subtitles(
                final_movie, srt_path,
                theme=(_os.getenv("MULTIMEDIA_SUBTITLE_THEME") or "default").strip().lower(),
            )
        except Exception as e:
            print(f"    [WARN] 字幕烧录失败（保留原生音轨，跳过字幕）: {e}")
            with_subs = final_movie
        print(f"    ✓ 已保留生成视频原生音轨（字幕: {os.path.basename(srt_path)}）")
        # 运行指标埋点：任务成功出片（可观测性）
        run_metrics.finalize_run("ok", with_subs)
        return {
            "subtitle_path": srt_path,
            "subtitle_entries": parse_srt_to_entries(srt_path),
            "final_movie_with_audio": with_subs,
            "final_movie_path": with_subs,
            "native_audio_kept": True,
            "audio_status": "native",
        }

    print("\n--- 🔊 [音频师] 正在为成片叠加配音、字幕与配乐 ---")
    if prefer_native and not has_native:
        print("    [*] 已开启原生音轨优先，但成片无原生音轨，自动回退本地 TTS 配音 + BGM")
    voice_role = state.get("voice_role") or os.getenv("TTS_VOICE", "").strip() or "xiaoxiao"  # 设置面板默认音色

    # ── 1) 逐镜头配音（P2 音画同步，替代原「整段一次性 TTS」）──
    # 【修复核心缺陷】原实现把所有台词拼成一整段后一次性 TTS，配音从 t=0 连续播；
    # 而字幕按各镜头视频时长排布，二者时长来源不同（TTS 语速 vs 生视频模型时长），
    # 必然错位 → "台词还没念完画面就结束""字幕与声音对不上"。
    # 现改为逐镜头 TTS：每段配音放到该镜头的真实起点（segment_start），
    # 并把各段真实 [start,end] 交给 build_srt，使字幕与配音同源、严格同步。
    def _line(s):
        return (s.get("dialogue") or "").strip()

    seg_specs = []
    for _i, _s in enumerate(state["scenes"]):
        _txt = _line(_s)
        if not _txt or _s.get("shot_failed"):
            continue  # 无台词 / 已跳过的镜头不配音
        _start = _s.get("segment_start")
        if not isinstance(_start, (int, float)):
            _start = 0.0  # 无真实起点时退化为从片头开始（旧行为）
        seg_specs.append({"index": _i, "text": _txt, "start": float(_start)})

    if not seg_specs:
        print("    [*] 无旁白文本，仅保留原声（若有）")
        return {}

    voice_timings: Dict[int, Tuple[float, float]] = {}
    voiceover = ""
    try:
        voiceover, voice_timings = build_segmented_voiceover(
            seg_specs, voice=voice_role
        )
    except Exception as _e:
        print(f"    [WARN] 逐镜头配音失败，回退整段配音: {_e}")

    if not voiceover:
        # 兜底：整段配音（旧行为），字幕仍按视频时长排布
        narration = "\n".join(spec["text"] for spec in seg_specs)
        try:
            voiceover = generate_voiceover(narration, voice=voice_role)
        except Exception as _e2:
            print(f"    [WARN] 整段配音也失败，跳过音频合成: {_e2}")
            voiceover = ""
        if not voiceover:
            print("    ⚠️ 配音生成失败，跳过音频合成")
            return {}
        print("    [*] 已回退整段配音（字幕按视频时长对齐）")

    # 2) 生成 SRT 字幕（优先 P2 配音真实区间，其次 P1 video_duration 真实时长）
    srt_path = build_srt(state["scenes"], voice_timings=voice_timings or None)
    subtitle_entries = parse_srt_to_entries(srt_path)
    if voice_timings:
        print(f"    ✓ 字幕已按逐镜头配音真实区间对齐（{len(voice_timings)} 段）")
    # 读取配音总时长（秒），供前端展示「配音生成了多长」
    voiceover_duration: Optional[float] = None
    try:
        from moviepy import AudioFileClip as _AFC

        with _AFC(voiceover) as _ac:
            voiceover_duration = round(float(_ac.duration), 2)
    except Exception as _e:
        print(f"    [*] 配音时长读取失败（不影响成片）: {_e}")
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

    # P1-5：SFX 事件 —— 按跨场景转场类型在切点铺音效（assets/sfx/<类别>/，空库自动跳过）
    sfx_events = build_sfx_events(state["scenes"])
    if sfx_events:
        print(f"    [P1-5] SFX: {len(sfx_events)} 个切点音效已排布")

    # 5) 烧录字幕 → 混音成片（配音 + 可选 BGM）
    try:
        with_subs = burn_subtitles(
            final_movie, srt_path,
            theme=(_os.getenv("MULTIMEDIA_SUBTITLE_THEME") or "default").strip().lower(),
        )
        final_with_audio = mix_audio(
            with_subs, voiceover,
            bgm_path=bgm_path,
            voice_timings=voice_timings or None,
            sfx_events=sfx_events,
        )
        print(f"    ✓ 音画合成完成: {os.path.basename(final_with_audio)}")
        # 运行指标埋点：任务成功出片（可观测性）
        run_metrics.finalize_run("ok", final_with_audio)
        return {
            "audio_track": voiceover,
            "subtitle_path": srt_path,
            "subtitle_entries": subtitle_entries,
            "voiceover_duration": voiceover_duration,
            "voice_role": voice_role,
            "bgm_path": bgm_path,
            "bgm_mood": bgm_mood,
            "audio_status": "dubbed",
            "final_movie_with_audio": final_with_audio,
            "final_movie_path": final_with_audio,
        }
    except Exception as e:
        print(f"    ❌ 音频合成失败: {e}")
        return {"error_log": f"音频合成失败: {str(e)[:200]}"}
