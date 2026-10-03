# src/agent/multimedia/state.py
from typing import Annotated, TypedDict, List, Optional, Dict, Any


# ── P0-2（架构评审）：Send 扇出并行归并 reducer ─────────────────────────────
# 串行模式下单写者语义不变；并行模式下多个 shot 分支各自只写自己的槽位，
# 归并规则保证「谁的槽位谁做主」，其余槽位不被过期副本回滚。

def merge_scenes_by_index(
    left: Optional[List[Dict[str, Any]]],
    right: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """scenes 按 index 合并：right 中「发生变化的槽位」覆盖 left，其余保留 left。"""
    if not left:
        return list(right or [])
    if not right:
        return list(left)
    out = list(left)
    while len(out) < len(right):
        out.append({})
    for i, d in enumerate(right):
        if i < len(out) and d != out[i]:
            # right 的该槽位与 left 不同 → 视为写入，覆盖（分支只写自己的镜头）。
            # 并行时若两分支同写一个 index（不应发生），后完成者胜。
            out[i] = d
    return out


def merge_dicts_shallow(left, right):
    """dict 浅合并（right 覆盖同 key）：用于按 shot_index 键控的并行结果归并。"""
    out = dict(left or {})
    out.update(right or {})
    return out


def any_true(left, right):
    """布尔取或：并行分支任一 True 即 True（aborted / shot_failed 语义）。"""
    return bool(left) or bool(right)


def keep_non_empty(left, right):
    """字符串保留非空：None/空 不覆盖已有值（abort_reason 语义）。"""
    if right is None or right == "":
        return left
    return right


def union_list(left, right):
    """列表按序去重合并（quota_exhausted_scenes / locked_scenes 语义）。"""
    out = list(left or [])
    for x in (right or []):
        if x not in out:
            out.append(x)
    return out


def max_value(left, right):
    """数值取最大（rewrite_count 并行分支计数归并；该计数只增不复位）。"""
    return max(int(left or 0), int(right or 0))



class MultimediaState(TypedDict):
    """
    顶层状态：管理整个长视频生成任务的全局流转
    """
    task: str
    global_setting: str
    scenes: Annotated[List[Dict[str, Any]], merge_scenes_by_index]
    current_scene_index: int
    final_movie_path: Optional[str]
    error_log: Annotated[Optional[str], keep_non_empty]

    # ── P0 音频闭环（配音 + 字幕 + 混音）──
    audio_track: Optional[str]               # 配音音频本地路径 (.mp3)
    subtitle_path: Optional[str]             # 生成的 SRT 字幕路径
    subtitle_entries: Optional[List[Dict[str, Any]]]  # 解析后的字幕条目 JSON（前端逐句时间轴）
    voiceover_duration: Optional[float]      # 配音总时长（秒），前端展示「配音生成了多长」
    audio_status: Optional[str]              # 音频闭环状态：native(原生音轨)/dubbed(本地配音)/None(未跑)
    bgm_path: Optional[str]                  # 背景音乐路径（P1 接入）
    final_movie_with_audio: Optional[str]    # 音画合成后的最终成片
    voice_role: Optional[str]                # 配音音色 key（见 tts.EDGE_TTS_VOICES_ZH）
    enable_audio: Optional[bool]             # 音频闭环总开关（前端控制，默认开启）
    bgm_mood: Optional[str]                  # BGM 情绪标签（如 ambient/tense/upbeat）
    # P2-1：成片包装——片头片尾卡开关（前端 TaskLauncher / env MULTIMEDIA_TITLE_CARD）
    title_card: Optional[bool]
    prefer_native_audio: Optional[bool]      # 原生音轨优先（即梦等模型自带声音时，跳过本地TTS/BGM混音，仅烧字幕）
    
    # 新增：是否启用首尾帧双控模式
    use_first_last_frame: Annotated[bool, any_true]

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

    # ── P2-5：叙事结构参数化（目标时长 / 连续剧集）──
    # target_duration：目标成片总时长（秒）。showrunner 据此反推镜头数
    #   （见 tools/narrative.plan_scene_count），取代此前硬编码的固定镜头数。
    target_duration: Optional[float]
    # episode：连续剧集数（>=1，缺省 1）。>1 时注入「承接上集 / 留下集钩子」提示。
    episode: Optional[int]
    # episode_anchors：上一集的 end_state 锚点（scene_anchors 快照），
    #   本集首镜据此承接；跨集连续性由此打通。
    episode_anchors: Optional[Dict[str, Any]]
    # series_assets：跨集复用的资产库（reference_sheets 集合，存于 run 目录之外的
    #   资产库）；本集若有则直接复用，省额度且保持角色/场景一致。
    series_assets: Optional[Dict[str, Any]]

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
    rewrite_count: Annotated[int, max_value]

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

    # 素材匹配可见化报告：记录用户注入的参考图「是否被剧本用到、被多少个镜头命中」。
    # 消除当前"名称对不上就静默失效"的隐性坑——前端据此给每张参考图标
    # 「已用于 N 个镜头 / 未命中剧本」。结构:
    #   {"matched": {"资产名": 命中镜头数}, "unmatched": ["资产名", ...]}
    # 由 reference_gen_node(基线) + image_gen_node(逐镜头累加) 共同维护。
    asset_match_report: Optional[Dict[str, Any]]

    # 任务控制
    aborted: Annotated[bool, any_true]
    abort_reason: Annotated[Optional[str], keep_non_empty]

    # 低干预 / 自动模式：为真时跳过 6 道人工审核闸口（直接以 approve 放行），
    # 普通用户免逐镜点击；专业用户默认 False 走人审闭环。
    auto_mode: Optional[bool]

    # 成片草稿审片（stitcher 之后、audio_mixer 之前的「无声粗剪」确认闸口）决策：
    # approve → 进入音频混音；rewrite → 回到 stitcher 重拼。仅 enable_audio 且非 auto_mode 时触发。
    # P1-2：成片 Critic 报告（抽帧复审节奏/连续性/画质），随草稿审片卡展示
    draft_critic_report: Optional[Dict[str, Any]]
    draft_decision: Optional[str]

    # 视频免费额度耗尽降级的镜头索引列表（供最终摘要如实报告，避免全局中止浪费已生成产物）
    quota_exhausted_scenes: Annotated[Optional[List[int]], union_list]

    # 审核视觉模型是否不可用（如免费额度耗尽 403）：不可用时应跳过"一致性强制重生"等依赖 VLM 的无效重试
    reviewer_unavailable: Annotated[Optional[bool], any_true]

    # ── S1（架构评审 P0-1）：质量档位 fast / standard / cinema ──
    # 由前端任务创建面板或 QUALITY_TIER 环境变量注入；决定视频后端候选顺序与时长策略。
    quality_tier: Optional[str]
    # ── S1（架构评审 P0-1）：已锁定镜头索引列表 ──
    # video_review 门控通过并勾选「锁定」时写入；locked 镜头在 video_gen_node 跳过再生成，
    # 重跑/续跑直接复用既有视频，不再消耗视频额度。
    locked_scenes: Annotated[Optional[List[int]], union_list]


    # ── P0-2（架构评审）：镜头级并行（Send fan-out）归并通道 ──
    # shot_chain 子图每分支输出 {str(shot_index): ...}，由 shots_collected 汇合。
    # 分支私有 state 不携带全量 scenes，规避过期副本回滚（详见 graph._shot_rehydrate）。
    scene_results: Annotated[Optional[Dict[str, Dict[str, Any]]], merge_dicts_shallow]
    branch_results: Annotated[Optional[Dict[str, Dict[str, Any]]], merge_dicts_shallow]


class ShotState(TypedDict, total=False):
    """单镜头分支（shot_chain subgraph）的私有状态。

    - scene:      当前镜头的可变工作副本（分支内各节点累积更新）
    - ctx:        扇出时注入的只读共享上下文快照（plan/风格/参考图/全片 scenes 等）
    - 其余:       分支私有控制流字段；子图结束后经 _shot_collect 折叠回主图。
    分支内复用现有节点函数：节点收到的是 _shot_rehydrate 还原的
    MultimediaState 视图，写回时只取回自己的镜头槽位。
    """
    shot_index: int
    scene: Dict[str, Any]
    ctx: Dict[str, Any]
    aborted: bool
    abort_reason: Optional[str]
    shot_failed: bool
    error_log: Optional[str]
    reviewer_unavailable: Optional[bool]
    rewrite_count: int
    quota_exhausted: List[int]
    use_first_last_frame: bool
    asset_match_report: Dict[str, Any]
    scene_results: Dict[str, Dict[str, Any]]
    branch_results: Dict[str, Dict[str, Any]]
