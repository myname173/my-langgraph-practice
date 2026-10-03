# src/agent/multimedia/nodes/routing.py
"""路由决策：图内各条件边的分支函数。

P2-3：由原 graph.py（单体）按 stage 机械拆分而来，逻辑逐字节保留；
仅新增本文件头注释与模块导入，函数体 / 常量 / 注释均未改动。
"""
from .. import run_metrics
from ..cinematic_critic import QUALITY_THRESHOLDS
from ..shot_strategy import get_consistency_threshold
from ..state import MultimediaState
from .common import CHARACTER_CONSISTENCY_THRESHOLD, FIX_EXTEND, FIX_KEEP_FIRST_FRAME, FIX_PROMPT_ONLY, FIX_REIMAGE, MAX_VIDEO_GEN_FAILURES, _normalize_fix_type


def decide_after_draft(state: MultimediaState) -> str:
    """草稿审片路由：approve → 音频混音；rewrite → 回到剪辑师重拼。"""
    action = (state.get("draft_decision") or "approve").lower()
    return "stitcher" if action == "rewrite" else "audio_mixer"


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
    # 【优化】审核视觉模型不可用时（如免费额度耗尽 403），embedding 一致性判定不可靠、
    # 且重生也无法提分（VLM 无法复核），强行重生只会空耗图像额度。直接跳过一致性强制重生。
    if state.get("reviewer_unavailable"):
        if portrait_sim is not None and portrait_sim < threshold:
            print(
                f"    ℹ️ [镜头 {idx+1}] 角色一致性偏低 (肖像相似度 {portrait_sim:.3f} < {threshold:.3f})，"
                f"但审核模型不可用，跳过无效重生，直接放行"
            )
    elif (
        passed
        and threshold > 0
        and portrait_sim is not None
        and portrait_sim < threshold
        and scene.get("consistency_attempts", 0) < 3
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

    # ── P1-4：VLM 一致性强制层（embedding 不可用时的兜底闭环）──
    # 与上面的 embedding 闭环互斥：仅当 embedding 相似度缺失(portrait_sim is None)时接管，
    # 避免双通道重复触发重生。needs_realign 由 reviewer_node 写入。
    _vlm = scene.get("vlm_consistency") or {}
    _vlm_score = _vlm.get("score")
    if (
        passed
        and not state.get("reviewer_unavailable")
        and portrait_sim is None
        and _vlm.get("available")
        and _vlm_score is not None
        and bool(scene.get("needs_realign"))
        and scene.get("consistency_attempts", 0) < 3
    ):
        _thr = _vlm.get("threshold")
        _rnote = f"，已按 {_vlm.get('relax_reason')} 放宽" if _vlm.get("relax_reason") else ""
        print(
            f"    [P1-4] 镜头 {idx+1} 角色一致性不足（VLM 分 {_vlm_score:.3f}"
            f"{f' < {_thr:.3f}' if isinstance(_thr, (int, float)) else ''}{_rnote}），"
            f"标记 needs_realign 并触发重生"
        )
        run_metrics.record_portrait_regen(_vlm_score, _thr if isinstance(_thr, (int, float)) else 0.0)
        return "director"

    # P1-3：外科手术 —— 审核未通过但修正类型为「保留关键帧」时，跳过重绘直接进视频侧。
    _fix = _normalize_fix_type(scene.get("fix_type"))
    if (not passed) and _fix in (FIX_KEEP_FIRST_FRAME, FIX_PROMPT_ONLY, FIX_EXTEND) and scene.get("image_url"):
        print(f"    [P1-3] 镜头 {idx+1} 保留关键帧（fix_type={_fix}），跳过重绘进入视频侧")
        if state.get("use_first_last_frame"):
            return "end_frame_director"
        return "videographer"

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
    # 配额耗尽属于全局中止（仅限真实全局错误，单镜头额度耗尽已降级为占位且 aborted=False）
    if state.get("aborted"):
        return "abort"

    idx = state["current_scene_index"]
    scene = state["scenes"][idx]

    # 生成成功 → 进入审片
    if scene.get("raw_video_url"):
        return "video_reviewer"

    # 【修复-B】镜头级降级（额度耗尽/shot_failed 已写入占位黑场）：不再重试，
    # 直接跳过审片推进后续镜头，由 stitcher 把占位镜头一并拼入成片保全。
    if scene.get("shot_failed") and scene.get("final_video_url"):
        print(f"    ⏭️ [镜头 {idx+1}] 已降级（占位黑场），跳过审片推进后续镜头")
        return "skip"

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

    # 【修复】复用素材库模式（方案 A：assets_imported=True 且 use_first_last_frame=False）
    # image_url 为空是预期行为（视频阶段以素材参考图锚定生成，不消耗即梦额度），
    # 不应误判为"生图失败"而无限重试。直接进入视频生成节点。
    if state.get("assets_imported") and not state.get("use_first_last_frame"):
        return "video_generator"

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

    # 【修复】方案 A（复用素材库模式）无尾帧是预期，直接进入视频生成而非重试
    if state.get("assets_imported") and not state.get("use_first_last_frame"):
        return "video_generator"

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
    # 【优化】审核模型不可用时，视频审片"不通过"源于接口异常而非真实质量问题，
    # 重试 videographer 只会空耗视频额度（且仍会因 403 不通过）。直接放行保底。
    if state.get("reviewer_unavailable"):
        print(f"    ℹ️ [镜头 {idx+1}] 视频审片未通过，但审核模型不可用，跳过视频重生直接放行")
        return "advance_scene"
    # P1-3：外科手术 —— 按修正类型决定重试粒度，替代一律回 videographer 的全量重 roll。
    _fix = _normalize_fix_type(scene.get("fix_type"))
    if _fix == FIX_PROMPT_ONLY:
        # 保图改词：直接用（用户编辑后的）video_prompt 重滚视频，跳过 videographer LLM。
        print(f"    [P1-3] 镜头 {idx+1} fix_type=prompt_only：保图改词，直连 video_generator")
        return "video_generator"
    if _fix == FIX_EXTEND:
        # 保首帧 + 延长时长：直连 video_generator（duration 已在门控 bump）。
        print(f"    [P1-3] 镜头 {idx+1} fix_type=extend：保首帧延长时长，直连 video_generator")
        return "video_generator"
    if _fix == FIX_REIMAGE and scene["video_iterations"] < 3:
        # 重绘关键帧：回到 director 走完整图链路（图重生成 → 尾帧 → 视频）。
        print(f"    [P1-3] 镜头 {idx+1} fix_type=reimage：重绘关键帧，回 director")
        return "director"
    # keep_first_frame / 未指定：保首帧重滚视频（现行为）。
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


def decide_after_prompt_preview(state: MultimediaState):
    """prompt_preview 路由：approve/edit_prompt → image_generator, rewrite → director"""
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    if scene.get("is_perfect"):
        return "image_generator"
    return "director"


def decide_after_end_frame_prompt_preview(state: MultimediaState):
    """end_frame_prompt_preview 路由：approve/edit_prompt → end_frame_generator, rewrite → end_frame_director"""
    idx = state["current_scene_index"]
    scene = state["scenes"][idx]
    if scene.get("last_image_is_perfect"):
        return "end_frame_generator"
    return "end_frame_director"
