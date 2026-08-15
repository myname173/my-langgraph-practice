# src/agent/multimedia/state.py
from typing import TypedDict, List, Optional, Dict, Any


class MultimediaState(TypedDict):
    """
    顶层状态：管理整个长视频生成任务的全局流转
    """
    task: str
    global_setting: str
    scenes: List[Dict[str, Any]]
    current_scene_index: int
    final_movie_path: Optional[str]
    error_log: Optional[str]

    # ── P0 音频闭环（配音 + 字幕 + 混音）──
    audio_track: Optional[str]               # 配音音频本地路径 (.mp3)
    subtitle_path: Optional[str]             # 生成的 SRT 字幕路径
    bgm_path: Optional[str]                  # 背景音乐路径（P1 接入）
    final_movie_with_audio: Optional[str]    # 音画合成后的最终成片
    voice_role: Optional[str]                # 配音音色 key（见 tts.EDGE_TTS_VOICES_ZH）
    enable_audio: Optional[bool]             # 音频闭环总开关（前端控制，默认开启）
    bgm_mood: Optional[str]                  # BGM 情绪标签（如 ambient/tense/upbeat）
    prefer_native_audio: Optional[bool]      # 原生音轨优先（即梦等模型自带声音时，跳过本地TTS/BGM混音，仅烧字幕）
    
    # 新增：是否启用首尾帧双控模式
    use_first_last_frame: bool

    # 视觉风格（Style Registry）
    # 由 _extract_visual_style（LLM 抽取）从用户输入中识别，存储 _STYLE_PRESETS 的 key，
    # 或 "custom:<label>" 表示未知但由 LLM 给出英文风格描述的自定义风格。
    # 供 director / end_frame_director / sanitize 等节点消费。
    visual_style: Optional[str]

    # 风格英文描述（由 _extract_visual_style 产出），供导演/审核 prompt 注入
    style_description: Optional[str]

    # 风格英文后缀（由 _extract_visual_style 产出，命中预设时取预设后缀，
    # 自定义时取 LLM 给出的 style_prompt），全链路生图/视频统一追加
    style_suffix: Optional[str]

    # 内容类型（Content Type Registry）
    # 由 _detect_content_type 从用户 task 中检测，存储 _CONTENT_TYPE_PRESETS 的 key
    # 决定视频的结构形态和叙事节奏模式
    content_type: Optional[str]

    # Phase 1: Visual Context Retrieval Layer
    # 由 visual_context_builder 节点生成，供 director 节点消费
    visual_context: Optional[Dict[str, Any]]

    # Phase 2: Shot Strategy Layer
    # 由 shot_strategy_builder 节点生成，供 director / end_frame_director 消费
    shot_plan: Optional[Dict[str, Any]]

    # Phase 3: Sequence Orchestration Layer
    # 由 sequence_orchestrator 节点生成，提供镜头间关系和序列上下文
    sequence_graph: Optional[Dict[str, Any]]

    # Phase 4: Self-Refining Cinematic Director System
    # 由 cinematic_critic 节点生成，包含质量评估、问题检测和优化建议
    critic_eval: Optional[Dict[str, Any]]
    quality_gates: Optional[Dict[str, Any]]
    rewrite_count: int

    # Phase 5: Multi-Agent Film Studio System
    # 由 film_studio 节点生成，包含多 Agent 协作优化的结果
    studio_output: Optional[Dict[str, Any]]
    # studio_output 落盘指针：为缓解大 state 死锁，完整 studio 结果落本地文件，
    # state 仅存路径（轻量指针）。film_studio_node 返回时剥离大对象。
    studio_output_path: Optional[str]
    # studio 结果摘要（共识度/修改数/冲突数），供下游观测，避免携带大对象。
    studio_summary: Optional[Dict[str, Any]]

    # Phase RAG: Knowledge Retrieval Layer
    # 由 visual_context_builder 节点中的 RAG 检索生成，供 director 消费
    rag_context: Optional[Dict[str, Any]]

    # 全局叙事弧（Narrative Arc）
    # 由 showrunner 完成后自动从 scenes 推导生成，供 director 消费
    # 包含每个场景的叙事功能、情绪轨迹、因果链
    narrative_arc: Optional[Dict[str, Any]]

    # 跨场景视觉锚点（Temporal State Model）
    # 由 advance_scene 节点从已完成场景提取，供下一场景的 visual_context_builder 消费
    # 实现场景间的视觉连续性（色调渐变、光照衔接、运镜惯性、能量过渡）
    scene_anchors: Optional[Dict[str, Any]]

    # 角色标准肖像（Character Reference Portrait）
    # 由 showrunner_review 通过后自动生成，作为所有关键帧的辅助参考图
    # 轻量版 Character ID 系统：正面全身标准光照，低 ref_strength 叠加使用
    character_portrait_url: Optional[str]

    # 一致性参考图（Reference Sheets）
    # 由 reference_gen_node 生成，包含角色转面图、道具设定图、场景环境参考图
    # 结构: {"characters": {"name": url}, "props": {"name": url}, "environments": {"name": url}}
    reference_sheets: Optional[Dict[str, Any]]

    # 一致性跟踪
    reference_images: List[str]
    reference_embeddings: List[List[float]]

    # 当前运行所属 thread_id（由入口注入，用于把参考图等中间产物
    # 按 thread 隔离落盘到 output/reference_sheets/<thread_id>/，
    # 即使即梦等远程临时 URL 过期，前端也能通过 /media 稳定访问）
    thread_id: Optional[str]

    # 用户已注入参考图资产（即梦历史图经前端 AssetLibraryPanel 标注并保存为
    # reference_sheets）。为真时 reference_gen_node 跳过重新生成，直接复用现有
    # reference_sheets，避免重复消耗即梦/百炼额度且保持角色/武器/场景一致。
    assets_imported: Optional[bool]

    # 任务控制
    aborted: bool
    abort_reason: Optional[str]

    # 视频免费额度耗尽降级的镜头索引列表（供最终摘要如实报告，避免全局中止浪费已生成产物）
    quota_exhausted_scenes: Optional[List[int]]

    # 审核视觉模型是否不可用（如免费额度耗尽 403）：不可用时应跳过"一致性强制重生"等依赖 VLM 的无效重试
    reviewer_unavailable: Optional[bool]
