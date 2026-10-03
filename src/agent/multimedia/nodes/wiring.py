# src/agent/multimedia/nodes/wiring.py
"""图组装与编译：节点注册、边连接、compile、thread 状态查询。

P2-3：由原 graph.py（单体）按 stage 机械拆分而来，逻辑逐字节保留；
仅新增本文件头注释与模块导入，函数体 / 常量 / 注释均未改动。
"""
import os as _os
from langgraph.graph import END, StateGraph
from ..checkpointer import checkpointer, make_thread_config
from ..state import MultimediaState
from .assemble import abort_node, advance_scene_node, audio_mixer_node, draft_review_gate_node, stitcher_node
from .design import reference_gen_node
from .director import cinematic_critic_node, director_node, director_refine_node, end_frame_director_node, end_frame_prompt_preview_node, film_studio_node, prompt_preview_node, videographer_node
from .parallel import _decide_after_parallel_join, _fan_out_shots, _run_chain_node, _run_shot_chain_node, shots_collected_node
from .render import end_frame_gen_node, end_frame_review_gate_node, image_gen_node, image_review_gate_node, reviewer_node, video_gen_node, video_review_gate_node, video_reviewer_node
from .routing import decide_after_draft, decide_after_end_frame_generation, decide_after_end_frame_prompt_preview, decide_after_image_generation, decide_after_prompt_preview, decide_after_video_generation, decide_end_frame_quality, decide_image_quality, decide_next_scene, decide_video_quality
from .scripting import sequence_orchestrator_node, shot_strategy_builder_node, showrunner_node, showrunner_review_gate_node, visual_context_builder_node


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
workflow.add_node("draft_review", draft_review_gate_node)
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
# P0-2：并行扇出（auto_mode + MULTIMEDIA_PARALLEL_SHOTS）或回退串行
workflow.add_node("shot_chain", _run_shot_chain_node)
# P1-1：链式扇出节点（一条链内逐镜串行承接尾帧，链间并行）
workflow.add_node("shot_chain_seq", _run_chain_node)
workflow.add_node("shots_collected", shots_collected_node)
workflow.add_conditional_edges(
    "cinematic_critic",
    _fan_out_shots,
    {
        "director": "director",
        "rewrite": "shot_strategy_builder",
    }
)
workflow.add_edge("shot_chain", "shots_collected")
workflow.add_edge("shot_chain_seq", "shots_collected")
workflow.add_conditional_edges(
    "shots_collected",
    _decide_after_parallel_join,
    {
        "stitcher": "stitcher",
        "abort": "abort",
    }
)

# Phase 4: Director → Refinement → Prompt Preview → Image Generator
workflow.add_edge("director", "director_refine")
workflow.add_edge("director_refine", "prompt_preview")


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
        "video_generator": "video_generator",
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
        "video_generator": "video_generator",
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
        "video_generator": "video_generator",
        "director": "director",
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
# 剪辑完成后先经「成片草稿审片」闸口（无声粗剪确认），再进入音频混音；
# auto_mode / 关闭音频闭环时该闸口内部直接放行，行为与改动前一致。
workflow.add_edge("stitcher", "draft_review")
workflow.add_conditional_edges(
    "draft_review",
    decide_after_draft,
    {
        "audio_mixer": "audio_mixer",
        "stitcher": "stitcher",
    },
)
workflow.add_edge("audio_mixer", END)

# 无论本地脚本还是 LangGraph API（langgraph dev / cloud）模式，都使用同一份 SqliteSaver
# 持久化层：本地模式直接在 compile 传入；平台模式则通过 langgraph.json 的
# checkpointer.path 字段声明 create_checkpointer 工厂注入（平台不允许在 compile()
# 中直接传自定义 checkpointer，否则 GraphLoadError）。两者复用 ./data/
# multimedia_checkpoints.sqlite，因此前端能列出并恢复本地跑一半的历史会话。

# .env 中的 LANGGRAPH_API_MODE=1 用于区分运行模式：平台模式下不在此处传 checkpointer，
# 而交由平台按 langgraph.json 的 checkpointer 配置注入。
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
