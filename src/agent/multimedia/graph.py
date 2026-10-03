# src/agent/multimedia/graph.py
"""兼容层（P2-3）：图实现已按 stage 拆分至 nodes/ 包。

原 graph.py（5,779 行单体：节点 + 路由 + compile 混在一起）拆分为：
  nodes/common.py     跨 stage 常量 / 纯工具 / 风格与 prompt 工具箱
  nodes/scripting.py  脚本与规划（视觉上下文 / 镜头策略 / 序列编排 / showrunner）
  nodes/design.py     视觉设计（参考图生成 / 资产匹配）
  nodes/director.py   导演（电影级评审 / 精修 / 影棚 / 首尾帧导演 / 运镜）
  nodes/render.py     渲染（关键帧 / 尾帧 / 视频生成 + 审核闸口）
  nodes/assemble.py   成片（推进 / 中止 / 拼接 / 草稿审片 / 混音）
  nodes/routing.py    路由决策（decide_*）
  nodes/parallel.py   并行 / 链式扇出机制（Send fan-out / 镜头子图 / FLF 链）
  nodes/wiring.py     图组装与编译（build + compile + thread 状态查询）

本文件仅做向后兼容再导出：所有历史调用方（scripts/、streamlit_app.py、
tests/、langgraph.json 的 src.agent.graph:graph）继续按原路径引用即可，
无需改动。实现与逻辑逐字节保留在下游模块中。
"""

# 线程配置构造器（定义于 checkpointer.py，经 wiring 复导出；供 scripts/ 外部调用）
from .checkpointer import make_thread_config

from .nodes.common import (
    APPROVE_ACTIONS,
    CHARACTER_CONSISTENCY_THRESHOLD,
    FIX_EXTEND,
    FIX_KEEP_FIRST_FRAME,
    FIX_PROMPT_ONLY,
    FIX_REIMAGE,
    FIX_TYPES,
    MAX_VIDEO_GEN_FAILURES,
    REWRITE_ACTIONS,
    VLM_CONSISTENCY_THRESHOLD,
    _AAA_STYLE_KEYWORDS,
    _CN_CHAR_RE,
    _CN_EN_TRANSLATIONS,
    _CONSISTENCY_VLM_ENABLED,
    _CONTENT_TYPE_PRESETS,
    _DEFAULT_CONTENT_TYPE,
    _DEFAULT_STYLE_KEY,
    _FIX_ALIASES,
    _GENERIC_CONFLICT_KEYWORDS,
    _GENRE_ATMOSPHERE,
    _PASS_RE,
    _SEQUENCE_PATTERNS,
    _STYLE_PRESETS,
    _STYLE_QUALITY_BAR,
    _STYLE_TAIL_TOKENS,
    _TRANSLATE_PROMPT,
    _VISUAL_STYLE_EXTRACT_PROMPT,
    _build_style_context,
    _decide,
    _detect_content_type,
    _detect_visual_style,
    _estimate_scene_duration,
    _extract_environment_anchors,
    _extract_protected_specs,
    _extract_visual_style,
    _finalize_single_frame_prompt,
    _get_conflict_keywords,
    _get_style_preset,
    _get_style_suffix,
    _is_review_pass,
    _normalize_action,
    _normalize_decision,
    _normalize_fix_type,
    _normalize_scenes,
    _resolve_style,
    _safe_json_loads,
    _sanitize_script,
    _sanitize_single_frame_prompt,
    _scene_base,
    _strip_chinese_from_prompt,
    _translate_to_english,
    _visual_script,
    safe_parse_json,
)

from .nodes.scripting import (
    _build_narrative_arc,
    sequence_orchestrator_node,
    shot_strategy_builder_node,
    showrunner_node,
    showrunner_review_gate_node,
    visual_context_builder_node,
)

from .nodes.design import (
    _REF_TURNAROUND_HINTS,
    _all_asset_names,
    _build_asset_match_baseline,
    _filter_reference_images,
    _first_url,
    _is_turnaround_sheet,
    _match_reference_elements,
    _to_urls,
    reference_gen_node,
)

from .nodes.director import (
    _build_camera_context,
    _build_cross_scene_context,
    cinematic_critic_node,
    director_node,
    director_refine_node,
    end_frame_director_node,
    end_frame_prompt_preview_node,
    film_studio_node,
    prompt_preview_node,
    videographer_node,
)

from .nodes.render import (
    _ANTI_DEFORMATION_BLOCK,
    compute_similarity,
    end_frame_gen_node,
    end_frame_review_gate_node,
    get_image_embedding,
    image_gen_node,
    image_review_gate_node,
    reviewer_node,
    video_gen_node,
    video_review_gate_node,
    video_reviewer_node,
)

from .nodes.assemble import (
    _extract_character_identity,
    _extract_scene_anchors,
    abort_node,
    advance_scene_node,
    audio_mixer_node,
    draft_review_gate_node,
    stitcher_node,
)

from .nodes.routing import (
    decide_after_critic,
    decide_after_draft,
    decide_after_end_frame_generation,
    decide_after_end_frame_prompt_preview,
    decide_after_image_generation,
    decide_after_prompt_preview,
    decide_after_video_generation,
    decide_end_frame_quality,
    decide_image_quality,
    decide_next_scene,
    decide_video_quality,
)

from .nodes.parallel import (
    _FLF_CHAINS_ENV,
    _PARALLEL_SHOTS_ENV,
    _SHOT_BRANCH_SCALARS,
    _SHOT_CTX_KEYS,
    _build_shot_chain_graph,
    _chain_all_locked,
    _compiled_shot_chain,
    _decide_after_end_frame_prompt_preview_shot,
    _decide_after_parallel_join,
    _decide_after_prompt_preview_shot,
    _diff_asset_reports,
    _fan_out_shots,
    _flf_chain_plan,
    _parallel_shots_enabled,
    _run_chain_node,
    _run_shot_chain_node,
    _shot_collect,
    _shot_ctx_snapshot,
    _shot_node,
    _shot_rehydrate,
    _shot_router,
    shots_collected_node,
)

from .nodes.wiring import (
    _RUNNING_UNDER_LANGGRAPH_API,
    get_latest_interrupt_payload,
    get_thread_history,
    get_thread_state,
    multimedia_agent,
    workflow,
)

__all__ = [
    "APPROVE_ACTIONS",
    "CHARACTER_CONSISTENCY_THRESHOLD",
    "FIX_EXTEND",
    "FIX_KEEP_FIRST_FRAME",
    "FIX_PROMPT_ONLY",
    "FIX_REIMAGE",
    "FIX_TYPES",
    "MAX_VIDEO_GEN_FAILURES",
    "REWRITE_ACTIONS",
    "VLM_CONSISTENCY_THRESHOLD",
    "_AAA_STYLE_KEYWORDS",
    "_ANTI_DEFORMATION_BLOCK",
    "_CN_CHAR_RE",
    "_CN_EN_TRANSLATIONS",
    "_CONSISTENCY_VLM_ENABLED",
    "_CONTENT_TYPE_PRESETS",
    "_DEFAULT_CONTENT_TYPE",
    "_DEFAULT_STYLE_KEY",
    "_FIX_ALIASES",
    "_FLF_CHAINS_ENV",
    "_GENERIC_CONFLICT_KEYWORDS",
    "_GENRE_ATMOSPHERE",
    "_PARALLEL_SHOTS_ENV",
    "_PASS_RE",
    "_REF_TURNAROUND_HINTS",
    "_RUNNING_UNDER_LANGGRAPH_API",
    "_SEQUENCE_PATTERNS",
    "_SHOT_BRANCH_SCALARS",
    "_SHOT_CTX_KEYS",
    "_STYLE_PRESETS",
    "_STYLE_QUALITY_BAR",
    "_STYLE_TAIL_TOKENS",
    "_TRANSLATE_PROMPT",
    "_VISUAL_STYLE_EXTRACT_PROMPT",
    "_all_asset_names",
    "_build_asset_match_baseline",
    "_build_camera_context",
    "_build_cross_scene_context",
    "_build_narrative_arc",
    "_build_shot_chain_graph",
    "_build_style_context",
    "_chain_all_locked",
    "_compiled_shot_chain",
    "_decide",
    "_decide_after_end_frame_prompt_preview_shot",
    "_decide_after_parallel_join",
    "_decide_after_prompt_preview_shot",
    "_detect_content_type",
    "_detect_visual_style",
    "_diff_asset_reports",
    "_estimate_scene_duration",
    "_extract_character_identity",
    "_extract_environment_anchors",
    "_extract_protected_specs",
    "_extract_scene_anchors",
    "_extract_visual_style",
    "_fan_out_shots",
    "_filter_reference_images",
    "_finalize_single_frame_prompt",
    "_first_url",
    "_flf_chain_plan",
    "_get_conflict_keywords",
    "_get_style_preset",
    "_get_style_suffix",
    "_is_review_pass",
    "_is_turnaround_sheet",
    "_match_reference_elements",
    "_normalize_action",
    "_normalize_decision",
    "_normalize_fix_type",
    "_normalize_scenes",
    "_parallel_shots_enabled",
    "_resolve_style",
    "_run_chain_node",
    "_run_shot_chain_node",
    "_safe_json_loads",
    "_sanitize_script",
    "_sanitize_single_frame_prompt",
    "_scene_base",
    "_shot_collect",
    "_shot_ctx_snapshot",
    "_shot_node",
    "_shot_rehydrate",
    "_shot_router",
    "_strip_chinese_from_prompt",
    "_to_urls",
    "_translate_to_english",
    "_visual_script",
    "abort_node",
    "advance_scene_node",
    "audio_mixer_node",
    "cinematic_critic_node",
    "compute_similarity",
    "decide_after_critic",
    "decide_after_draft",
    "decide_after_end_frame_generation",
    "decide_after_end_frame_prompt_preview",
    "decide_after_image_generation",
    "decide_after_prompt_preview",
    "decide_after_video_generation",
    "decide_end_frame_quality",
    "decide_image_quality",
    "decide_next_scene",
    "decide_video_quality",
    "director_node",
    "director_refine_node",
    "draft_review_gate_node",
    "end_frame_director_node",
    "end_frame_gen_node",
    "end_frame_prompt_preview_node",
    "end_frame_review_gate_node",
    "film_studio_node",
    "get_image_embedding",
    "get_latest_interrupt_payload",
    "get_thread_history",
    "get_thread_state",
    "image_gen_node",
    "image_review_gate_node",
    "make_thread_config",
    "multimedia_agent",
    "prompt_preview_node",
    "reference_gen_node",
    "reviewer_node",
    "safe_parse_json",
    "sequence_orchestrator_node",
    "shot_strategy_builder_node",
    "shots_collected_node",
    "showrunner_node",
    "showrunner_review_gate_node",
    "stitcher_node",
    "video_gen_node",
    "video_review_gate_node",
    "video_reviewer_node",
    "videographer_node",
    "visual_context_builder_node",
    "workflow",
]


# ─────────────────────────────────────────────────────────────────────────
# monkeypatch 兼容层
#
# 历史上部分脚本 / 测试（scripts/dry_run_full.py、test_style_pipeline_mock.py）用
# `import agent.multimedia.graph as G; G.call_llm = fake` 注入假实现。拆分后真正的
# 绑定在 nodes/ 子模块里，单纯给本模块赋值不再生效。此兼容模块把对 graph.<name>
# 的赋值同步写入所有【实际持有该名字】的 nodes 子模块，保证旧注入方式行为不变。
import sys as _sys
import types as _types

# 名字 -> 持有该绑定的 nodes 子模块（短名）；运行时按本模块实际包路径展开
_REF_SUBS = {

    "APPROVE_ACTIONS": ("common", "render"),

    "Any": ("common", "parallel", "scripting"),

    "CHARACTER_CONSISTENCY_THRESHOLD": ("common", "render", "routing"),

    "DIRECTOR_REFINEMENT_PROMPT": ("director",),

    "DIRECTOR_SYSTEM_PROMPT": ("director",),

    "Dict": ("common", "parallel", "scripting"),

    "END": ("parallel", "wiring"),

    "END_FRAME_DIRECTOR_PROMPT": ("director",),

    "FIX_EXTEND": ("common", "render", "routing"),

    "FIX_KEEP_FIRST_FRAME": ("common", "render", "routing"),

    "FIX_PROMPT_ONLY": ("common", "render", "routing"),

    "FIX_REIMAGE": ("common", "render", "routing"),

    "FIX_TYPES": ("common",),

    "FreeTierQuotaExhaustedError": ("render",),

    "List": ("common", "parallel", "scripting"),

    "MAX_VIDEO_GEN_FAILURES": ("common", "render", "routing"),

    "MultimediaState": ("assemble", "common", "design", "director", "parallel", "render", "routing", "scripting", "wiring"),

    "Optional": ("common", "parallel"),

    "QUALITY_THRESHOLDS": ("director", "routing"),

    "REVIEWER_SYSTEM_PROMPT": ("render",),

    "REWRITE_ACTIONS": ("common",),

    "RunnableConfig": ("assemble", "parallel", "render", "scripting"),

    "SAFETY_PROMPT": ("scripting",),

    "SHOWRUNNER_PROMPT": ("scripting",),

    "Send": ("parallel",),

    "ShotState": ("parallel",),

    "StateGraph": ("parallel", "wiring"),

    "VIDEOGRAPHER_DUAL_FRAME_PROMPT": ("director",),

    "VIDEOGRAPHER_PROMPT": ("director",),

    "VIDEO_REVIEWER_SYSTEM_PROMPT": ("render",),

    "VLM_CONSISTENCY_THRESHOLD": ("common", "render"),

    "_AAA_STYLE_KEYWORDS": ("common",),

    "_ANTI_DEFORMATION_BLOCK": ("render",),

    "_CN_CHAR_RE": ("common",),

    "_CN_EN_TRANSLATIONS": ("common",),

    "_CONSISTENCY_VLM_ENABLED": ("common", "render"),

    "_CONTENT_TYPE_PRESETS": ("common",),

    "_DEFAULT_CONTENT_TYPE": ("common",),

    "_DEFAULT_STYLE_KEY": ("common", "render"),

    "_EMOTION_ENERGY": ("assemble",),

    "_FIX_ALIASES": ("common",),

    "_FLF_CHAINS_ENV": ("parallel",),

    "_GENERIC_CONFLICT_KEYWORDS": ("common",),

    "_GENRE_ATMOSPHERE": ("common",),

    "_MOTION_NEGATIVE_PROMPT": ("render",),

    "_PARALLEL_SHOTS_ENV": ("parallel",),

    "_PASS_RE": ("common",),

    "_REF_TURNAROUND_HINTS": ("design",),

    "_RUNNING_UNDER_LANGGRAPH_API": ("wiring",),

    "_SEQUENCE_PATTERNS": ("common",),

    "_SHOT_BRANCH_SCALARS": ("parallel",),

    "_SHOT_CTX_KEYS": ("parallel",),

    "_STYLE_PRESETS": ("common",),

    "_STYLE_QUALITY_BAR": ("common", "render"),

    "_STYLE_TAIL_TOKENS": ("common",),

    "_TRANSLATE_PROMPT": ("common",),

    "_VISUAL_STYLE_EXTRACT_PROMPT": ("common",),

    "_all_asset_names": ("design",),

    "_build_asset_match_baseline": ("design",),

    "_build_camera_context": ("director",),

    "_build_cross_scene_context": ("director",),

    "_build_narrative_arc": ("scripting",),

    "_build_shot_chain_graph": ("parallel",),

    "_build_style_context": ("common", "director", "scripting"),

    "_chain_all_locked": ("parallel",),

    "_compiled_shot_chain": ("parallel",),

    "_decide": ("common", "director", "render", "scripting"),

    "_decide_after_end_frame_prompt_preview_shot": ("parallel",),

    "_decide_after_parallel_join": ("parallel", "wiring"),

    "_decide_after_prompt_preview_shot": ("parallel",),

    "_detect_content_type": ("common", "scripting"),

    "_detect_visual_style": ("common",),

    "_diff_asset_reports": ("parallel",),

    "_estimate_scene_duration": ("common", "scripting"),

    "_extract_character_identity": ("assemble",),

    "_extract_environment_anchors": ("common", "director"),

    "_extract_protected_specs": ("common", "director"),

    "_extract_scene_anchors": ("assemble",),

    "_extract_visual_style": ("common",),

    "_fan_out_shots": ("parallel", "wiring"),

    "_filter_reference_images": ("design", "render"),

    "_finalize_single_frame_prompt": ("common", "director"),

    "_first_url": ("design", "render"),

    "_flf_chain_plan": ("parallel",),

    "_get_conflict_keywords": ("common", "director"),

    "_get_style_preset": ("common", "director"),

    "_get_style_suffix": ("common",),

    "_is_review_pass": ("common", "render"),

    "_is_turnaround_sheet": ("design", "render"),

    "_make_placeholder_clip": ("render",),

    "_match_reference_elements": ("design", "render"),

    "_normalize_action": ("common",),

    "_normalize_decision": ("assemble", "common"),

    "_normalize_fix_type": ("common", "render", "routing"),

    "_normalize_scenes": ("common", "scripting"),

    "_parallel_shots_enabled": ("parallel",),

    "_persist_studio_output": ("director",),

    "_resolve_style": ("common", "design", "director", "scripting"),

    "_run_chain_node": ("parallel", "wiring"),

    "_run_shot_chain_node": ("parallel", "wiring"),

    "_safe_json_loads": ("common",),

    "_sanitize_script": ("common", "scripting"),

    "_sanitize_single_frame_prompt": ("common", "director"),

    "_scene_base": ("common", "scripting"),

    "_shot_collect": ("parallel",),

    "_shot_ctx_snapshot": ("parallel",),

    "_shot_node": ("parallel",),

    "_shot_rehydrate": ("parallel",),

    "_shot_router": ("parallel",),

    "_strip_chinese_from_prompt": ("common",),

    "_to_urls": ("design",),

    "_translate_to_english": ("common",),

    "_visual_script": ("common", "director"),

    "abort_node": ("assemble", "wiring"),

    "advance_scene_node": ("assemble", "wiring"),

    "apply_color_preset": ("assemble",),

    "apply_user_settings": ("scripting",),

    "assess_character_consistency": ("render",),

    "attach_title_cards": ("assemble",),

    "audio_mixer_node": ("assemble", "wiring"),

    "build_segmented_voiceover": ("assemble",),

    "build_sequence_graph": ("scripting",),

    "build_sfx_events": ("assemble",),

    "build_srt": ("assemble",),

    "build_visual_context": ("scripting",),

    "burn_subtitles": ("assemble",),

    "call_llm": ("common", "director", "scripting"),

    "character_consistency_threshold": ("common",),

    "checkpointer": ("wiring",),

    "cinematic_critic_node": ("director", "wiring"),

    "compute_similarity": ("render",),

    "critic_final_cut": ("assemble",),

    "datetime": ("assemble",),

    "decide_after_critic": ("parallel", "routing"),

    "decide_after_draft": ("routing", "wiring"),

    "decide_after_end_frame_generation": ("parallel", "routing", "wiring"),

    "decide_after_end_frame_prompt_preview": ("routing", "wiring"),

    "decide_after_image_generation": ("parallel", "routing", "wiring"),

    "decide_after_prompt_preview": ("routing", "wiring"),

    "decide_after_video_generation": ("parallel", "routing", "wiring"),

    "decide_end_frame_quality": ("parallel", "routing", "wiring"),

    "decide_image_quality": ("parallel", "routing", "wiring"),

    "decide_next_scene": ("routing", "wiring"),

    "decide_video_quality": ("parallel", "routing", "wiring"),

    "design_camera_movement": ("director",),

    "director_node": ("director", "parallel", "wiring"),

    "director_refine_node": ("director", "parallel", "wiring"),

    "download_video": ("render",),

    "draft_review_gate_node": ("assemble", "wiring"),

    "end_frame_director_node": ("director", "parallel", "wiring"),

    "end_frame_gen_node": ("parallel", "render", "wiring"),

    "end_frame_prompt_preview_node": ("director", "parallel", "wiring"),

    "end_frame_review_gate_node": ("parallel", "render", "wiring"),

    "enhance_final_cut": ("assemble",),

    "evaluate_cinematic_quality": ("director",),

    "evaluate_image": ("render",),

    "evaluate_video": ("render",),

    "film_studio_node": ("director", "wiring"),

    "format_critic_report": ("director",),

    "generate_bgm": ("assemble",),

    "generate_keyframe": ("render",),

    "generate_reference_sheets": ("design",),

    "generate_shot_strategy": ("scripting",),

    "generate_video_from_image": ("render",),

    "generate_voiceover": ("assemble",),

    "get_consistency_threshold": ("render", "routing"),

    "get_image_embedding": ("render",),

    "get_latest_interrupt_payload": ("wiring",),

    "get_thread_history": ("wiring",),

    "get_thread_state": ("wiring",),

    "has_audio_track": ("assemble",),

    "image_gen_node": ("parallel", "render", "wiring"),

    "image_review_gate_node": ("parallel", "render", "wiring"),

    "interrupt": ("assemble", "common"),

    "make_thread_config": ("wiring",),

    "max_video_gen_failures": ("common",),

    "mix_audio": ("assemble",),

    "multimedia_agent": ("wiring",),

    "orchestrate_film_studio": ("director",),

    "parse_srt_to_entries": ("assemble",),

    "prompt_preview_node": ("director", "parallel", "wiring"),

    "reference_gen_node": ("design", "wiring"),

    "reset_memory": ("scripting",),

    "resolve_bgm_mood": ("assemble",),

    "retrieve_rag_context": ("scripting",),

    "reviewer_node": ("parallel", "render", "wiring"),

    "rewrite_strategy": ("scripting",),

    "run_metrics": ("assemble", "common", "director", "render", "routing", "scripting"),

    "safe_parse_json": ("common", "scripting"),

    "select_cross_scene_transition": ("assemble",),

    "sequence_orchestrator_node": ("scripting", "wiring"),

    "shot_strategy_builder_node": ("scripting", "wiring"),

    "shots_collected_node": ("parallel", "wiring"),

    "showrunner_node": ("scripting", "wiring"),

    "showrunner_review_gate_node": ("scripting", "wiring"),

    "stitch_videos": ("assemble",),

    "stitcher_node": ("assemble", "wiring"),

    "video_gen_node": ("parallel", "render", "wiring"),

    "video_review_gate_node": ("parallel", "render", "wiring"),

    "video_reviewer_node": ("parallel", "render", "wiring"),

    "videographer_node": ("director", "parallel", "wiring"),

    "visual_context_builder_node": ("scripting", "wiring"),

    "workflow": ("wiring",),

}

_PKG = __name__.rsplit(".", 1)[0]  # 兼容 'agent.multimedia' 与 'src.agent.multimedia' 两种导入路径
_REF_OWNER = {
    name: tuple(f"{_PKG}.nodes.{m}" for m in subs) for name, subs in _REF_SUBS.items()
}


class _CompatModule(_types.ModuleType):
    """对 graph.<name> 的赋值转发到持有该绑定的 nodes 子模块（保留 monkeypatch 语义）。"""

    def __setattr__(self, name, value):
        for sub in _REF_OWNER.get(name, ()):
            mod = _sys.modules.get(sub)
            if mod is not None and hasattr(mod, name):
                try:
                    setattr(mod, name, value)
                except Exception:
                    pass
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _CompatModule
