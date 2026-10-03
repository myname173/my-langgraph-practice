## AI 多模态游戏宣传片生成 Agent 系统

基于 LangGraph 构建的 22 节点 StateGraph 管线，输入一段创意描述（如"赛博朋克风格的未来都市枪战宣传片"），自动完成从剧本策划、视觉设计、镜头编排、质量评审、图像/视频生成到最终拼接的全流程。全系统仅 Director 节点调用 LLM 做创意决策，其余模块均为纯规则驱动——零额外 LLM 成本、确定性输出、零延迟。

---

### 系统架构

```
用户创意输入
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: 剧本策划                                            │
│  showrunner ──→ showrunner_review (HITL interrupt)           │
│  LLM 拆分 4 个 scene，人工审核修改                              │
└──────────────────────────┬───────────────────────────────────┘
                           │ 逐 scene 处理
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2: 视觉设计（Phase 1-5，纯规则驱动）                      │
│                                                              │
│  visual_context_builder  [Phase 1: 意图分类 + 视觉规则匹配]     │
│        │          ↑ 同时检索 RAG 知识库                        │
│        ▼                                                     │
│  shot_strategy_builder   [Phase 2: 4 镜头设计 + hero 选择]     │
│        │                                                     │
│        ▼                                                     │
│  sequence_orchestrator   [Phase 3: 转场/运动/情绪弧建模]        │
│        │                                                     │
│        ▼                                                     │
│  film_studio             [Phase 5: 三专家协作优化]              │
│        │                                                     │
│        ▼                                                     │
│  cinematic_critic        [Phase 4: 质量评估 + 重写循环]         │
│        │                                                     │
│        ├── FAIL → shot_strategy (rewrite, max 2次)           │
│        └── PASS ──────────────┐                              │
└───────────────────────────────┼──────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 3: 导演创作（LLM 创意决策）                              │
│  director ──→ director_refine                                │
│  综合 Phase 1-5 规则输出 + RAG 知识 → 生成 image_prompt        │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 4: 图像生成 + 审核循环                                   │
│  image_generator ──→ reviewer ──→ image_review (HITL)        │
│        │                                                     │
│        ├── FAIL → director (重生成, max 3次)                   │
│        └── PASS ──┐                                          │
│                   ▼                                          │
│  [可选] end_frame_director → end_frame_generator              │
│         → end_frame_review (HITL)                            │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 5: 视频生成 + 审核循环                                   │
│  videographer ──→ video_generator ──→ video_reviewer         │
│                   → video_review (HITL)                      │
│        │                                                     │
│        ├── FAIL → videographer (重生成, max 3次)              │
│        └── PASS ──┐                                          │
└───────────────────┼──────────────────────────────────────────┘
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 6: 场景推进 + 拼接                                      │
│  advance_scene                                               │
│        │                                                     │
│        ├── 还有 scene → visual_context_builder (下一轮)       │
│        └── 全部完成 → stitcher → 最终宣传片 .mp4 → END         │
└──────────────────────────────────────────────────────────────┘
```

---

### 技术栈

| 组件 | 技术 | 用途 |
|---|---|---|
| 工作流引擎 | LangGraph (StateGraph) | 22 节点有向图 + 条件路由 + 中断恢复 |
| 前端控制台 | Streamlit | 7-tab HITL 交互界面 |
| 状态持久化 | SQLite (langgraph.checkpoint.sqlite) | 图状态快照，支持暂停/恢复/时间旅行 |
| 任务注册表 | SQLite (自建) | 应用层任务元数据，跨会话恢复 |
| 向量数据库 | ChromaDB (PersistentClient) | RAG 知识检索层 |
| 文本 LLM | DashScope `tongyi-xiaomi-analysis-pro` | 剧本拆分 + 导演创作 + 知识生成 |
| 图像生成 | DashScope `wan2.7-image` | 关键帧图片生成 |
| 图生视频 | DashScope `wan2.7-i2v` | 首帧/首尾双帧视频生成 |
| 视频精修 | DashScope `wan2.7-videoedit` | 视频后期风格增强 |
| 图像审核 | DashScope `qwen-image-2.0` | 关键帧质量 + 内容安全审核 |
| 视频审核 | DashScope `qwen3.5-omni-plus-2026-03-15` | 视频抽帧审核 + 运镜设计 |
| 文本向量化 | DashScope `text-embedding-v3` | RAG 知识向量化 (1024 维) |
| 图像向量化 | DashScope `multimodal-embedding-v1` | 跨镜头视觉一致性计算 |
| 视频处理 | MoviePy | 视频标准化 (1920x1080, 24fps) + 拼接 |

所有 DashScope 模型共用同一个 API Key (`DASHSCOPE_API_KEY`)，通过 OpenAI 兼容接口或原生 REST API 两种模式调用。

---

### 30 个节点详解

#### 编排层模块化（P2-3）

原 `graph.py` 是 5,779 行单体（节点 + 路由 + compile 全挤在一个文件里，已超出「能靠记忆维护」的规模）。P2-3 按 **stage** 把它机械拆成 `nodes/` 包，**逻辑逐字节保留**（132 个顶层定义经 AST 级核验：零重复、零缺失、零源码差异）：

| 模块 | 行数 | 职责 |
|---|---:|---|
| `nodes/common.py` | ~1,470 | 跨 stage 常量、纯工具、风格 / prompt 工具箱（`_decide` / `safe_parse_json` / `_resolve_style` / `_finalize_single_frame_prompt` …） |
| `nodes/scripting.py` | ~660 | 脚本与规划：`visual_context_builder` / `shot_strategy_builder` / `sequence_orchestrator` / `showrunner` |
| `nodes/design.py` | ~440 | 视觉设计：`reference_gen` + 资产匹配（角色 / 场景锚点基线） |
| `nodes/director.py` | ~1,000 | 导演：`cinematic_critic` / `director_refine` / `film_studio` / `director` / 提示词预览 / `end_frame_director` / `videographer` |
| `nodes/render.py` | ~1,760 | 渲染：`image_gen` / `reviewer` / `image_review` / `end_frame_gen` / `end_frame_review` / `video_gen` / `video_reviewer` / `video_review` |
| `nodes/assemble.py` | ~1,000 | 成片：`advance_scene` / `abort` / `stitcher` / `draft_review` / `audio_mixer` |
| `nodes/routing.py` | ~330 | 路由决策：全部 `decide_*` 条件边函数 |
| `nodes/parallel.py` | ~620 | 并行 / 链式扇出：Send fan-out、镜头子图、FLF 链式衔接 |
| `nodes/wiring.py` | ~250 | 图组装与编译：节点注册、边连接、`compile`、thread 状态查询 |
| `graph.py` | ~90 | **兼容层**：仅把上述实现再导出，历史调用方零改动 |

依赖是单向无环的：`common → design → render → parallel → wiring`，`common → {scripting, director, assemble, routing} → parallel → wiring`。`graph.py` 保留为向后兼容再导出层，`langgraph.json` 的 `src.agent.graph:graph` 入口与 `tests/` / `streamlit_app.py` / `scripts/` 全部无需改动。

#### 节点注册表

| # | 节点名 | 实现函数 | 所属模块 | 所属阶段 |
|---|---|---|---|---|
| 1 | `showrunner` | `showrunner_node` | scripting | 剧本策划 |
| 2 | `showrunner_review` | `showrunner_review_gate_node` | scripting | 剧本审核 (HITL) |
| 3 | `reference_gen` | `reference_gen_node` | design | 参考图生成 |
| 4 | `visual_context_builder` | `visual_context_builder_node` | scripting | Phase 1 视觉规则 |
| 5 | `shot_strategy_builder` | `shot_strategy_builder_node` | scripting | Phase 2 镜头设计 |
| 6 | `sequence_orchestrator` | `sequence_orchestrator_node` | scripting | Phase 3 序列编排 |
| 7 | `film_studio` | `film_studio_node` | director | Phase 5 专家协作 |
| 8 | `cinematic_critic` | `cinematic_critic_node` | director | Phase 4 质量评估 |
| 9 | `shot_chain` | `_run_shot_chain_node` | parallel | P0-2 逐镜并行分支 |
| 10 | `shot_chain_seq` | `_run_chain_node` | parallel | P1-1 FLF 链式分支 |
| 11 | `shots_collected` | `shots_collected_node` | parallel | 并行归并 |
| 12 | `director` | `director_node` | director | 导演创作 (LLM) |
| 13 | `director_refine` | `director_refine_node` | director | Prompt 精炼 |
| 14 | `prompt_preview` | `prompt_preview_node` | director | 提示词预览 (HITL) |
| 15 | `image_generator` | `image_gen_node` | render | 图像生成 |
| 16 | `reviewer` | `reviewer_node` | render | 图像审核 (LLM) |
| 17 | `image_review` | `image_review_gate_node` | render | 图像审核门控 (HITL) |
| 18 | `end_frame_director` | `end_frame_director_node` | director | 尾帧设计 (LLM) |
| 19 | `end_frame_prompt_preview` | `end_frame_prompt_preview_node` | director | 尾帧提示词预览 (HITL) |
| 20 | `end_frame_generator` | `end_frame_gen_node` | render | 尾帧生成 |
| 21 | `end_frame_review` | `end_frame_review_gate_node` | render | 尾帧审核 (HITL) |
| 22 | `videographer` | `videographer_node` | director | 运镜设计 (LLM) |
| 23 | `video_generator` | `video_gen_node` | render | 视频生成 |
| 24 | `video_reviewer` | `video_reviewer_node` | render | 视频审核 (LLM) |
| 25 | `video_review` | `video_review_gate_node` | render | 视频审核门控 (HITL) |
| 26 | `advance_scene` | `advance_scene_node` | assemble | 场景推进 |
| 27 | `abort` | `abort_node` | assemble | 异常终止 |
| 28 | `stitcher` | `stitcher_node` | assemble | 最终拼接 |
| 29 | `draft_review` | `draft_review_gate_node` | assemble | 成片草稿审片 (HITL) |
| 30 | `audio_mixer` | `audio_mixer_node` | assemble | 配音 / 字幕 / 混音 |

#### 条件路由（11 个分支函数）

| 路由函数 | 触发节点 | 路径 A | 路径 B |
|---|---|---|---|
| `_fan_out_shots` | cinematic_critic | Send 扇出（并行 / 链式） | `director` / `shot_strategy_builder` |
| `decide_after_critic` | cinematic_critic（串行回退） | `director` (PASS) | `shot_strategy_builder` (rewrite) |
| `decide_after_prompt_preview` | prompt_preview | `image_generator` | `director` |
| `decide_after_image_generation` | image_generator | `reviewer` | `retry` / `skip` / `abort` |
| `decide_image_quality` | image_review | `end_frame_director` / `videographer` | `director` (重生成) |
| `decide_after_end_frame_prompt_preview` | end_frame_prompt_preview | `end_frame_generator` | `end_frame_director` |
| `decide_after_end_frame_generation` | end_frame_generator | `end_frame_review` | `retry` / `skip` / `abort` |
| `decide_end_frame_quality` | end_frame_review | `videographer` | `end_frame_director` (重生成) |
| `decide_after_video_generation` | video_generator | `video_reviewer` | `retry` / `skip` / `abort` |
| `decide_video_quality` | video_review | `advance_scene` | `videographer` / `video_generator` / `director` |
| `decide_next_scene` | advance_scene | `visual_context_builder` | `stitcher` |
| `_decide_after_parallel_join` | shots_collected | `stitcher` | `abort` |
| `decide_after_draft` | draft_review | `audio_mixer` | `stitcher`（重剪） |

#### Human-in-the-Loop 中断点（6 个）

管线在以下节点会暂停执行，等待用户审核并决策（approve / rewrite / edit）：

| 中断节点 | stage 标识 | 审核内容 |
|---|---|---|
| `showrunner_review` | `showrunner_review` | 全局设定 + scene 剧本 |
| `prompt_preview` | `prompt_preview` | 关键帧 prompt |
| `image_review` | `image_review` | 关键帧图片 + 前帧对比 + 一致性 |
| `end_frame_prompt_preview` | `end_frame_prompt_preview` | 尾帧 prompt |
| `end_frame_review` | `end_frame_review` | 尾帧图片 + 首帧对比 |
| `video_review` | `video_review` | 视频片段 + 运镜 prompt |
| `draft_review` | `draft_review` | 无声粗剪成片 |

---

### Phase 详解

#### Phase 0 — 视觉知识库

`vision_knowledge/` 目录下存放手写 + LLM 生成的摄影知识 markdown 文件，作为 RAG 层的数据源：

| 文件 | 内容 |
|---|---|
| `camera_grammar.md` | 镜头语言词汇表（景别/机位/运镜/Prompt 关键词） |
| `lighting_grammar.md` | 光影词汇表（主光/氛围光/风格化/时段预设） |
| `trailer_patterns.md` | 4 种预告片叙事模板 + 节奏指南 + 转场技法 |
| `generated/*.md` | LLM 生成的 10 个专业知识文件（光照/色彩/构图/运镜等） |

这些文件不参与管线的规则匹配逻辑，仅在 RAG 灌入时被切块、向量化、存入 ChromaDB。

#### Phase 1 — 视觉上下文构建 (`visual_context.py`, ~450 行)

纯规则驱动。对 scene script 做加权关键词分类，匹配 4 种场景意图之一，输出结构化视觉规则：

- **意图分类器**: `classify_scene_intent(script)` — 4 种意图（world_building / character_introduction / action / emotional），基于 `_INTENT_KEYWORDS` 加权匹配
- **风格检测器**: `detect_global_genre(setting)` — 5 种风格（cyberpunk / fantasy / gothic / post_apocalyptic / sci_fi），基于 `_GENRE_KEYWORDS`
- **节奏控制器**: `determine_pacing_phase(index, total)` — 4 个阶段（opening / build_up / climax / resolution），基于 scene 位置比例
- **输出**: `visual_context` dict 包含 camera_rules、lighting_rules、motion_rules、composition_rules、mood_keywords、genre_modifiers、pacing，以及格式化文本 `formatted_prompt_block`

核心字典：`INTENT_VISUAL_RULES`（4 意图 x 5 维度规则）、`GENRE_MODIFIERS`（5 风格 x 光照/调色修正）、`TRAILER_PACING`（4 阶段 x 节奏/剪辑韵律）。

#### Phase 2 — 镜头策略 (`shot_strategy.py`, ~864 行)

纯规则驱动。将 Phase 1 的视觉规则转化为 4 个具体镜头的拍摄计划：

- **镜头模板**: `SHOT_TEMPLATES` — 4 种意图 x 4 个镜头 = 16 个预定义镜头模板，每个包含完整的 camera 规格（type/movement/angle/lens）、lighting setup、emotion、pacing_weight
- **焦点提取**: `_extract_focus(script)` — 从剧本关键词识别视觉焦点（character/vehicle/weapon/creature/environment）
- **Hero Shot 选择**: `_select_hero_shot()` — 综合节奏匹配(+30)、权重分数(+35)、叙事位置(+10) 选出高潮镜头
- **输出**: `shot_plan` dict 包含 shots 列表、hero_shot_index、end_shot_index、formatted_plan

重写引擎 `rewrite_strategy()` 支持 9 种定向修复策略：complete_camera、diversify_camera、enhance_lighting、diversify_emotion、reshape_arc、reshape_pacing、diversify_focus、strengthen_hero、upgrade_transition。

#### Phase 3 — 序列编排 (`sequence_orchestrator.py`, ~562 行)

纯规则驱动。将独立的 4 个镜头组织为有电影感的连贯序列：

- **转场选择**: 3 级优先级——情绪转换矩阵 (`_EMOTION_TRANSITION_MATRIX`, 16 种情绪对) → 镜头类型矩阵 (`_SHOT_TYPE_TRANSITIONS`) → 能量级启发式 → 默认 hard_cut。共 7 种转场类型
- **运动连续性**: `_MOTION_CARRY_RULES` — 15 种运动对过渡指令（如 "slow dolly in" 衔接 "slow crane ascending" 时的桥接方式）
- **空间连续性**: 检测焦点一致性、景别跳跃、角度变化
- **情绪弧建模**: 计算全局曲线形状（crescendo/anti_crescendo/decay/ascent/arc/single）、peak/valley 位置、能量曲线、整体方向（escalating/de-escalating/balanced）
- **输出**: `sequence_graph` dict 包含 connections（镜头间关系列表）、emotional_arc、formatted_graph、director_context

#### Phase 4 — 电影级质量评估 (`cinematic_critic.py`, ~619 行)

纯规则驱动。4 维度量化评分 + 3 级质量门控，驱动自优化重写循环：

- **评分维度**（加权综合）：

| 维度 | 权重 | 子检查 |
|---|---|---|
| visual_quality | 0.25 | camera 完整度 (50%) + camera 多样性 (50%) |
| coherence | 0.20 | 光照连贯性 (60%) + 焦点多样性 (40%) |
| cinematic_flow | 0.30 | 节奏变化 (35%) + 转场质量 (35%) + hero 强度 (30%) |
| emotion_consistency | 0.25 | 情绪推进 (50%) + 弧形状 (50%) |

- **质量门控**（3 级全部通过 + 综合分 >= 0.65 才放行）：

| 门控 | 阈值 | 检查内容 |
|---|---|---|
| shot_quality | >= 0.60 | camera 完整度 + 多样性 + 节奏变化 |
| sequence_consistency | >= 0.55 | 连贯性 + 弧形状 |
| trailer_flow | >= 0.55 且零 issue | 连接数 + 弧平坦度 + 转场多样性 |

- **问题检测**: 12 种问题类型（camera_incomplete、lighting_flat、redundant_camera、weak_hero 等），每种映射到一个重写策略名
- **重写循环**: FAIL 时调 `shot_strategy.rewrite_strategy()` → 重建 sequence_graph → 重新评估，最多 2 次

#### Phase 5 — 电影工作室协作 (`film_studio_orchestrator.py`)

纯规则驱动。3 个专家 Agent 协作优化镜头计划，通过辩论机制解决跨领域冲突：

| 专家 | 文件 | 职责 |
|---|---|---|
| 摄影指导 | `cinematography_agent.py` | 镜头系统验证（lens 校验、运动叙事匹配、角度叙事功能、跳跃剪辑检测、跨 scene 一致性） |
| 灯光指导 | `lighting_agent.py` | 氛围光照叠加、对比度进程控制、大气层系统、光照连续性、跨 scene 一致性 |
| 剪辑指导 | `editor_agent.py` | 转场优化（能量差→转场映射）、节奏权重校准、情绪弧平滑、视觉疲劳防护、跨 scene 一致性 |

协作流程：
1. 从 `cinematic_memory` 获取前序 scene 的风格快照
2. 依次运行 3 个专家优化
3. 运行辩论机制 (`_run_debate`)，检测 4 类跨领域冲突（运动-氛围冲突、节奏-镜头不匹配、节奏-光照不匹配、镜头跳跃警告）
4. 计算 `studio_consensus` 综合分（3 专家平均分 - 冲突惩罚）
5. 将精炼后的 transitions 注入 sequence_graph
6. 将当前 scene 风格快照写入 `cinematic_memory`，供后续 scene 参考

跨 scene 一致性由 `cinematic_memory.py` 维护——**按 thread 作用域的记忆库**，存储每个 scene 的摄影/灯光/剪辑风格摘要，提供 `get_previous_scene_style()` 和 `generate_film_brain_report()` 接口。记忆库**写入即原子落盘**到 `data/memory/<thread_id>.json`（进程内 dict 仅作缓存层）；长任务中断恢复后，新进程首次访问会自动从落盘重建，避免风格记忆清空导致跨镜头一致性静默退化（P0-3 记忆持久化）。`reset_memory(thread_id)` 在新 run 开始时清空该线程的缓存与落盘文件。

#### Phase RAG — 语义知识检索

叠加层设计，不替换现有规则，检索失败时静默降级。

**数据灌入**（一次性操作，`rag/ingest.py` CLI）：

```
python -m src.agent.multimedia.rag.ingest --source gen
```

灌入流程：`knowledge_generator.py`（LLM 生成 10 个摄影专业 md 文件）→ `markdown_ingestor.py`（按 `##` 标题切块）→ `vector_store.py`（DashScope text-embedding-v3 向量化 → ChromaDB 持久化存储）。

3 个 ChromaDB Collection：

| Collection | 用途 | 查询 top_k |
|---|---|---|
| `cinematography_knowledge` | 摄影/布光/剪辑知识 | 3 |
| `prompt_examples` | 高质量 prompt 样本 | 2 |
| `film_references` | 影视参考描述 | 2 |

**运行时检索**（`rag/retriever.py`，每个 scene 调用一次）：
1. 构建查询：`"{script} | scene intent: {intent} | visual style: {global_setting}"`
2. 对 3 个 Collection 分别查询，按 L2 距离过滤（> 0.6 丢弃）
3. 格式化为 `formatted_rag_block`（摄影知识 + prompt 样本 + 影视参考的编号文本块）
4. 写入 `state["rag_context"]`，由 director 节点消费

---

### 全局状态 (`MultimediaState`)

所有节点共享一个 TypedDict，19 个字段贯穿管线全程：

| 字段 | 类型 | 说明 |
|---|---|---|
| `task` | str | 用户原始创意描述 |
| `global_setting` | str | 全局视觉设定（showrunner 生成） |
| `scenes` | List[Dict] | 4 个 scene 列表 |
| `current_scene_index` | int | 当前处理的 scene 索引 |
| `use_first_last_frame` | bool | 是否启用首尾帧双控模式 |
| `visual_context` | Dict | Phase 1 输出 |
| `shot_plan` | Dict | Phase 2 输出 |
| `sequence_graph` | Dict | Phase 3 输出 |
| `studio_output` | Dict | Phase 5 输出 |
| `critic_eval` | Dict | Phase 4 评估报告 |
| `quality_gates` | Dict | Phase 4 门控结果 |
| `rewrite_count` | int | Phase 4 重写计数 |
| `rag_context` | Dict | RAG 检索结果 |
| `reference_images` | List[str] | 跨镜头一致性：已生成图片 URL 列表 |
| `reference_embeddings` | List[List[float]] | 跨镜头一致性：图片 embedding 列表 |
| `final_movie_path` | str | 最终宣传片路径 |
| `error_log` | str | 错误日志 |
| `aborted` | bool | 是否异常终止 |
| `abort_reason` | str | 终止原因 |

---

### P0/P1 架构增强（并行加速 + FLF 链式衔接）

| 项 | 机制 | 环境变量 |
|---|---|---|
| P0-2 镜头并行 | `cinematic_critic` 通过后按镜头 `Send` 扇出 `shot_chain` 子图；槽位经 `scene_results` / `branch_results` 无锁归并 | `MULTIMEDIA_PARALLEL_SHOTS=0` 关闭 |
| P1-1 FLF 链式衔接 × 并行共存 | 镜头按【连续片段】切链：**链内逐镜串行**（前镜尾帧 → `scene_anchors.last_frame_url` → 后镜 img2img 承接 + 视频首尾帧），**链间并行**（`Send` 扇出 `shot_chain_seq`） | `MULTIMEDIA_FLF_CHAINS`：`0`=关闭（回退逐镜并行）；正整数=链数；**未设置=连续性优先（单链，全部镜头串行承接）** |

- 串行模式（非 auto_mode / 关闭并行）下，`advance_scene` 写 `scene_anchors.last_frame_url`，实现同样的「上一镜尾帧 → 下一镜首帧」像素级承接。
- **默认连续性优先**：不设 `MULTIMEDIA_FLF_CHAINS` 时为单链，全部镜头串行承接尾帧、无链缝（画面像素级连续）。
- **链缝取舍**：设 `N>1` 后，相邻链的首/末镜之间不承接 —— 这是「跨链并行」的固有代价。需最大连续性时设 `MULTIMEDIA_FLF_CHAINS=1`（单链、全承接、无并行）；需最大并行度时设 `0`（逐镜并行、无链内承接）。

---

### P1-3 镜头级外科手术（修正粒度）

审核 FAIL 的重试不再一律「回 director 全量重 roll」，而是由审核决策携带**修正类型** `fix_type`，按粒度选择入口节点：

| fix_type | 语义 | 路由落点 | 相对成本 |
|---|---|---|---|
| `prompt_only` | 保关键帧，仅改运镜词后重滚视频 | 直连 `video_generator`（跳过 videographer LLM） | 最低 |
| `keep_first_frame` | 保首帧，重滚视频（重新设计运镜） | `videographer`（默认行为） | 中 |
| `reimage` | 重生成关键帧 | `director`（完整图链路 → 尾帧 → 视频） | 最高 |
| `extend` | 保首帧，尾帧续接「延长时长」 | 直连 `video_generator`（duration ×1.5，封顶 12s） | 中 |

- 入口：`image_review` / `video_review` 两个审核闸口的决策字典可带 `fix_type`（含中文别名「改词/保首帧/重绘/延长」）；闸口把它写入 `scene["fix_type"]`，`decide_image_quality` / `decide_video_quality` 读取并分流。
- 未提供 `fix_type` 时行为与改动前一致（image 侧默认 `reimage`、video 侧默认 `keep_first_frame`）。
- 通过 `render_params` 记录每次渲染的后端/时长/prompt 等，供 `prompt_only`/`extend` 冻结「其它条件、仅改一处」。
- auto_mode 下人审闸口自动放行，不产生 `fix_type`，即并行/自动链路行为不变。

---

### S3 成片观感层（P1-2 / P1-5 / F-2 / F-3 / F-5）

**P1-2 卡点剪辑 + 成片 Critic**
- **成片 Critic**：`draft_review_gate_node` 对无声粗剪按时间顺序抽 6 帧，交视觉模型
  （`vision_eval.critic_final_cut`）从「节奏 / 连续性 / 画质」三维度复审，产出结构化
  问题清单 `draft_critic_report`（severity/type/desc），随草稿审片卡展示给用户，供
  「通过 / 打回重拼」决策。开关 `MULTIMEDIA_FINAL_CRITIC`（默认开）；失败不阻断；
  auto_mode 也跑（写 state 供前端展示）。
- **卡点对齐**：`stitch_videos(beat_grid=...)` 把每段微调到节拍网格（**只剪不补**，
  每段最多剪 0.6s，需 ≥2 个网格），切点落拍；`segment_durations/starts` 基于修剪后
  时长计算，字幕/配音对齐自动跟随。开关 env `MULTIMEDIA_BEAT_GRID` 填 BPM 数值
  （如 `120` → 0.5s 网格），**默认关闭**（剪帧是创作取舍，交给用户决定）。

**P1-5 音频层升级**
- **BGM 自动闪避**：`mix_audio` 接收逐镜头配音真实区间（`voice_timings`），用 50fps
  音量包络把配音段压低 duck_factor=0.35（≈ -9dB）、间隙恢复，滑动平均平滑防爆音；
  失败回退恒定音量旧行为。整段配音兜底路径（无 timings）同样回退。
- **SFX 转场音效**：`build_sfx_events` 按跨场景转场类型（与 stitcher 完全同源的
  `select_cross_scene_transition` 重算）在切点前 0.15s 排布音效，类别映射
  smash_cut→impact / whip_pan→whoosh / dissolve→soft / fade_through_black→deep，
  曲库 `assets/sfx/<类别>/`（env `SFX_LIBRARY_DIR` 可覆盖），**空库静默跳过**。
  详见 `assets/sfx/README.md`。

**P2-1 成片包装层**
- **片头片尾卡**：`tools/video_pack.py` —— `build_title_card` 深色底 + 居中标题/副标题
  （复用 `fonts.make_text_clip`，moviepy 2.x + 中文字体），带淡入淡出；
  `attach_title_cards` 把首尾各 ~2.2s 的卡接到成片并返回 `intro` 时长，
  `stitcher_node` 据此把所有 `scene.segment_start` 统一后移，字幕/配音对齐不漂移。
  默认开启（RunInput `title_card` / env `MULTIMEDIA_TITLE_CARD=0` 可关）。
- **色彩预设**：`apply_color_preset` 提供 warm / cool / noir 三档，256 级通道 LUT
  查表（noir 为灰度+对比映射）；env `MULTIMEDIA_COLOR_PRESET`（默认空=不应用）。
  轻量逐帧 LUT，不引入 3D LUT 解析与重型依赖。
- **字幕主题**：`fonts.burn_subtitles(theme=)` 三套——default（白字黑边）/
  minimal（小号浅灰）/ cinema（大号金白描边），env `MULTIMEDIA_SUBTITLE_THEME`。

**P2-2 后处理板**
- `tools/video_post.py` —— `enhance_final_cut`，基于项目已有 imageio_ffmpeg 自带
  ffmpeg 二进制（零新增依赖）：
  清晰度增强 `MULTIMEDIA_POST_HEIGHT=1080`（lanczos 上采样 + unsharp 锐化，**非模型
  超分**）；补帧 `MULTIMEDIA_POST_FPS=30`（`MULTIMEDIA_POST_INTERP=dup|mci`，mci 为
  运动补偿插值，观感最好但极慢）。两者默认关闭；目标低于当前时自动跳过；失败
  一律返回原成片不阻断。接入点在 stitcher_node：拼接 → 后处理 → 包装卡。

**F-2 成本与额度仪表**：顶栏常驻 `RunCostBar`——运行耗时（1s 刷新）、镜头进度
（含失败数）、视频生成尝试次数（成功+重试，额度消耗代理）、重写次数、各生成后端
熔断健康灯（`/settings/health`，60s 轮询）。数据全部 state 派生 + 现有端点，零后端改动。

**F-3 质量档位选择器**：`TaskLauncher` 新增三档卡片（fast 省额度 / standard 均衡 /
cinema 电影级），选择随 `RunInput.quality_tier` 进 state——后端 P0-1 已全链路消费
该字段，本次纯前端接入。

**F-5 产物流式生命周期**：活动镜头卡显示当前阶段徽标（分镜设计/图像生成/关键帧
审核/尾帧/运镜/视频生成/视频审核），由时间轴最新节点推导（`NODE_PHASE` 映射）。

> 涉及：后端 `tools/audio.py`（duck + SFX）、`tools/video_stitcher.py`（beat_grid）、
> `tools/vision_eval.py`（critic_final_cut）、`state.py`（draft_critic_report）、
> `graph.py`（三处接线）；前端 `types.ts`、`components/RunCostBar.tsx`（新）、
> `TaskLauncher.tsx`、`App.tsx`、`SceneGallery.tsx`、`InterruptApprovalCard.tsx`、
> `styles.css`。S3 剩余 P2-1 包装层 / P2-2 超分补帧未实现（后续 Sprint）。

---

### 多生图后端自助接入（连通即用）

jimeng（每日免费积分）仍是首选生图后端；其余后端按「用户自助配置 → 连通测试 → 通了
才启用」接入，不要求预先手动修 key：

- **fallback 链**（`tools/image_gen.py`）：jimeng 因**积分/额度/限流类错误**失败时，
  按 `MULTIMEDIA_IMAGE_FALLBACK`（默认 `"dashscope,siliconflow"`）顺序尝试**已配置
  key** 的备选后端，现场连通验证、失败继续下一个、全部失败抛原错误（保持旧行为）。
  非额度类配置错误不触发 fallback，避免掩盖真问题。备选：
  - `dashscope`：qwen-image 系列同步 multimodal messages（复用遗留
    `_IMG2IMG_CHAIN` / `_build_payload` / `_extract_image_url`，有参考图走多图融合链）；
  - `siliconflow`：Qwen-Image `images/generations`（参考图注入 `image` 数组，
    支持负向词与 image_size 映射）。
- **零额度连通测试**（设置面板「连通测试」按钮）：`_probe_dashscope_image`（缺参请求
  手法：401=Key 无效 / 模型不存在=模型无效 / InvalidParameter=通过）与
  `_probe_siliconflow_image`（GET /v1/models 验证 Key，不消耗额度）。
  探测 kind：`dashscope-image` / `siliconflow-image`。
- **设置面板**：`DASHSCOPE_IMAGE_MODEL` / `SILICONFLOW_IMAGE_MODEL` 字段旁均有
  「连通测试」按钮（`PROBE_BY_FIELD` 泛化），填 Key → 测试 → 显示通过/失败详情。
- **后端健康**：备选后端调用同样走熔断器 + 成本台账（`/settings/health` 可见）。

### F-1/F-4 审核台与镜头卡（前端）

把 P1-3 的修正粒度、P0-1 的锁定语义、以及批量审核能力接到前端（React + TS）：

**F-4 外科手术入口与镜头卡四态**
- **修正方式选择器**：`InterruptApprovalCard` 在 `image_review` / `video_review`
  闸口渲染后端下发的 `fix_types`（仅改词 / 保首帧 / 重绘关键帧 / 延长），点选后随
  `rewrite` 决策回传 `fix_type`；不选则按后端默认（image=reimage、video=keep_first_frame）。
- **通过后锁定**：`video_review` 卡新增「通过后锁定」勾选，决策携带 `lock=true`，
  后端写 `locked_scenes`；重跑/续跑复用该镜头成片。
- **镜头卡四态**：`SceneGallery` 每张卡推导 `failed / locked / needs_fix / draft`
  （含 P1-4 `needs_realign` 的「角色漂移」徽标）。failed 卡一键「从此镜重跑」，
  locked 卡「解锁并重跑」（`rerun_from` 请求携带 `unlock=true`，后端顺带把它移出
  `locked_scenes`）。
- **闸口快捷决策**：当审核中断恰好停在某镜时，该镜头卡上直接出现「通过 + 四类修正」
  按钮，无需滚回审核卡。注意：修正决策只能在闸口中断时刻消费——非中断状态的修正
  走「重跑此镜头」，到闸口时再选修正方式（这是 LangGraph 决策模型的固有语义）。

**F-1 批量审核与关键帧候选**
- **批量预通过**：画廊卡片带多选框；选中后「预通过选中 / 通过并锁定」把镜头加入
  **自动通过队列**（前端 `useAgentStream` 维护）：命中这些镜头的审核闸口时自动
  `resume(approve)` 放行，其余镜头仍正常停下等待人工审。video_review 通过后该镜
  出队；可随时取消；新建会话/切换会话时清空。
- **关键帧候选 A/B 挑选**：`image_gen_node` 每次生成把关键帧登记进
  `scene.image_candidates`（最多 6 版）；`image_review` payload 携带最近 4 版
  `candidates`；审核卡渲染候选条，点选某一版后「通过」即回传
  `pick_candidate=<下标>`，闸口先落位该候选再走 approve 基准登记。配合
  「重绘关键帧」修正可积累多版候选逐版对比（hero 镜头的 2 候选挑选即此流程）。

> 涉及文件：后端 `graph.py`（候选登记 / payload / pick_candidate）、
> `static_server.py`（rerun unlock + 新增可再生字段清理）；前端 `types.ts`、
> `lib/fixTypes.ts`（新）、`components/InterruptApprovalCard.tsx`、
> `components/SceneGallery.tsx`、`hooks/useAgentStream.ts`、`lib/historyClient.ts`、
> `App.tsx`、`styles.css`。

---

### P1-4 前置：人脸对齐（mediapipe）

VLM 成对判断本身对「头部倾斜/朝向差异」较敏感。为此在比较前插入一步**几何归一化**：

- **实现**：`tools/face_align.py`，用 mediapipe `FaceLandmarker`（478 关键点）检测人脸，
  以双眼（优先虹膜中心）连线计算 roll 角，绕人脸中心旋转摆正，再按人脸包围盒外扩
  `margin=0.6` 裁剪成正方形输出。
- **模型缓存**：`.task` 模型（3.6MB）首次使用自动下载并缓存到
  `MULTIMEDIA_FACE_MODEL_DIR`（默认 `~/.cache/multimedia_face`）。
- **接入点**：`vision_eval._prepare_for_compare()` 在 `assess_character_consistency`
  比较前对两张图分别对齐；返回结果新增 `aligned` 字段标识本次是否走了对齐。
- **开关 / 降级**：env `MULTIMEDIA_FACE_ALIGN`（默认 `"1"` 开启）；mediapipe 缺失、
  未检出人脸或任何异常 → 回退原图，`aligned=False`，绝不阻断主流程（实测海鸥关键帧
  等无人脸图即走此路径）。
- **实测**：真人脸对齐后人脸占画面比例 0.102 → 0.355（3.5×）；将输入旋转 25° 后对齐
  可摆正（再次对齐残差角 0.06°）；单人脸检测+关键点约 27ms。

---

### P1-4 角色一致性强制层（VLM 成对判断）

原一致性闭环（`decide_image_quality` 的「角色一致性低于阈值 → 强制重生」）依赖 DashScope
`multimodal-embedding-v1` 计算余弦相似度。该通道 Key 失效（401）后相似度恒为 `None`，闭环
**静默降级为失效**。P1-4 给它补上一条 VLM 兜底数据源：

- **数据源切换**：`reviewer_node` 在 embedding 相似度缺失时，调用
  `assess_character_consistency(当前帧, 参照帧)`，用视觉模型（`glm-4v-flash`）对两张图做
  「是否同一角色」判断。参照帧优先取源头角色肖像（抗累积误差），其次上一镜尾帧。
- **结构化输出**：强制模型只回 JSON `{"same_character": bool, "score": 0-1, "reason": str}`，
  解析失败即视为不可用。**不注入剧本上下文**——实测弱模型会把文字描述当判据，去比对
  「画面 vs 描述」而非「图1 vs 图2」，导致同角色被误判为漂移。
- **漂移判定**：`score < 阈值` **且** `same_character == false` 两个信号同时成立才算漂移，
  降低误杀（避免空耗生图额度）。阈值默认 0.6，env `MULTIMEDIA_VLM_CONSISTENCY_THRESHOLD`
  可覆盖，并复用 `get_consistency_threshold` 按镜头朝向/景别自适应放宽。
- **闭环动作**：判定漂移 → `reviewer_node` 写 `scene["needs_realign"]=True` →
  `decide_image_quality` 触发重生（回 `director`），受 `iterations < 3` 上限保护；
  与 embedding 闭环**互斥**（仅当 embedding 缺失时接管，避免双通道重复触发）。
- **开关 / 降级**：env `MULTIMEDIA_CONSISTENCY_VLM`（默认 `"1"` 开启）；VLM 接口异常或
  `reviewer_unavailable` 时自动跳过，绝不阻断主流程。

> 说明：真正的「本地人脸对齐」重生成未实现（依赖 insightface 等重型依赖），当前闭环动作止于
> 「标记 needs_realign → 重生成 → 交由人工审核」。

---

### P2-4 镜头配方卡（Shot Recipe）

**问题**：一个镜头「长什么样、怎么被造出来的」此前散落在 scene 的十几处字段 +
`render_params` 里，既不可复现（换个 run 无法照着重来），也不可分享（没法交给别人）。

**做法**：新增 `tools/recipe.py`，把「一个 run 的全部镜头」提炼成一份结构化配方：

```json
{
  "schema": "multimedia.shot_recipe.v1",
  "thread_id": "...", "created_at": "2026-10-03T02:30:00+08:00",
  "task": "...", "global_setting": "...",
  "style": {"visual_style": "...", "description": "...", "suffix": "..."},
  "post":  {"color_preset": "...", "subtitle_theme": "default", "title_card": true},
  "shots": [
    {"index": 0, "script": "...", "action_beat": "...",
     "image":     {"backend": "jimeng", "prompt": "...", "size": "2K",
                    "ref_images": ["..."], "ref_strength": 0.6},
     "end_frame": {"prompt": "...", "ref_strength": 0.45, "ref_images": ["..."]},
     "video":     {"backend": "agnes", "quality_tier": "standard", "duration": 5.0,
                    "prompt": "...", "use_first_last_frame": true, "reference_images": ["..."]},
     "fix":       {"fix_type": "keep_first_frame", "extend_applied": false},
     "artifacts": {"image_url": "...", "last_image_url": "...", "final_video_url": "..."}}
  ]
}
```

**三个能力**：

| 能力 | 实现 |
|---|---|
| **可复现** | graph 在各生图/生视频节点把参数记入 `render_params`（image / end_frame / video 三段），成片拼好后由 `stitcher_node` 自动落盘到 `output/recipes/<thread>/latest.json`（+ 时间戳快照） |
| **可分享** | 前端「🧾 任务配方 / 查看配方」按钮拉取配方，一键复制 JSON / Markdown，或「另存文件」写盘 |
| **可微调** | 配方卡里直接改首帧/尾帧/视频 prompt、ref_strength、时长 → 「应用微调并重跑此镜」：后端 `apply_shot_patch` 白名单深合并 → `shot_patch_to_scene_fields` 写回 scene → 从该镜续跑（只清产物、**保留 render_params**） |

**接口（static_server.py）**：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/recipe/{thread_id}` | 即时从 checkpoint 构建整任务配方 + 可读 Markdown |
| POST | `/recipe/{thread_id}/save` | 落盘为 `output/recipes/<thread>/*.json` |
| POST | `/recipe/{thread_id}/apply` | 微调：写回某镜配方并标记从该镜重跑 |

**前端**：`components/RecipeCard.tsx`（配方面板：镜头列表 + 可编辑字段 + 只读参数 +
复制/另存/应用）+ `lib/recipeClient.ts`；`SceneGallery` 每张镜头卡加「🧾 查看配方」，
标题区加「🧾 任务配方」。

**注意**：`render_params` 由「替换」改为「合并写入」（`video_gen_node` 不再冲掉
`image_gen_node` / `end_frame_gen_node` 已写的首/尾帧段），旧键
（backend / quality_tier / duration / video_prompt / use_first_last_frame）原样保留，
对 P1-3 外科手术向后兼容。整个 P2-4 路径**零额度**（纯序列化 + 本地 JSON）。

---

### P1-6 审核飞轮（Review Flywheel）

把每次 HITL 人审决策结构化成可复用的 **(rejected, chosen) 偏好对**，攒够后离线蒸馏一个「自动审核员」，把多道人审逐步收敛为 1 道 final review。

#### 三层结构

| 层 | 位置 | 作用 |
|---|---|---|
| 收集端 | `review_flywheel.record_decision()`（挂载于 `nodes/common.py::_decide`，6 处闸口统一收口） | 闸口消费决策时自动配对 chosen/rejected |
| 存储端 | `workspace/_training_data/multimedia_review_pairs.jsonl` | 每条偏好对一行 JSON；可导出 DPO / SFT |
| 收敛端 | `scripts/distill_reviewer.py` → `review_policy.json`；`review_flywheel.suggest()` | 离线蒸馏轻量策略；影子模式给建议（不拦主流程） |

#### 偏好对的构成规则

| 场景 | 是否构对 | chosen / rejected |
|---|---|---|
| `edit_prompt` / 编辑后提交 | ✅ 立即成对 | rejected = 原产物，chosen = 人改后的产物 |
| `rewrite` → 之后再 approve | ✅ 配对成对 | rejected = 重写前产物，chosen = 重写后最终通过版（保留首次理由） |
| 直接 `approve`（无 pending） | ❌ | 无偏好信号 |
| `auto_mode` 自动放行 | ❌（计入 auto 计数） | 无人工介入 |

偏好对结构（`multimedia.review_pair.v1`）：

```json
{
  "schema": "multimedia.review_pair.v1",
  "pair_id": "a1b2c3d4e5f6",
  "gate": "image_review",
  "scene_index": 1,
  "source": "rewrite",
  "decision": {"action": "approve", "fix_type": null, "edited_by_human": false, "reason": "人物比例失真"},
  "rejected": {"kind": "image", "text": "", "url": "http://.../bad.png", "meta": {"portrait_similarity": 0.1}},
  "chosen":   {"kind": "image", "text": "", "url": "http://.../good.png", "meta": {}},
  "context": {"script": "对峙", "style_key": "xianxia", "quality_tier": "standard"},
  "thread_id": "t2", "created_at": "2026-10-03T11:05:00"
}
```

#### 离线蒸馏与影子模式

```bash
# 1) 查看飞轮积累进度
python scripts/distill_reviewer.py --stats

# 2) 蒸馏（默认单闸口 ≥5 对文本可比偏好对才产出策略）
python scripts/distill_reviewer.py --min-pairs 8 --export-dpo --export-sft

# 3) 影子模式：只记录「自动审核员 vs 人审」的建议，不改变主流程
MULTIMEDIA_AUTO_REVIEW=shadow        # 全闸口影子
MULTIMEDIA_AUTO_REVIEW=image_review  # 仅指定闸口
```

蒸馏策略形态为 **词面对比**（log-odds 判别词表 + 最优切点阈值）——零依赖、可解释；`MULTIMEDIA_REVIEW_POLICY_PATH` 可指定策略路径。embedding 通道恢复后可在此处平滑替换为向量质心，接口不变。

#### 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `MULTIMEDIA_FLYWHEEL_ENABLED` | `1` | 收集端总开关 |
| `MULTIMEDIA_FLYWHEEL_DIR` | `workspace/_training_data` | 偏好对落盘目录 |
| `MULTIMEDIA_REVIEW_POLICY_PATH` | `<dir>/review_policy.json` | 蒸馏策略路径 |
| `MULTIMEDIA_AUTO_REVIEW` | `off` | 影子模式：`off` / `shadow` / `all` / `<gate1,gate2>` |

设计约束：零外部依赖、零侵入（`_decide` 收口处 +1 次调用）、绝不阻断（全部 try/except 兜底）。

---

### P2-5 叙事结构参数化（Narrative Parameterization）

把「拆几个镜头、每个镜头承担什么叙事功能」从 prompt 里的**硬编码常量**，提升为
**由目标时长驱动、可自适应伸缩**的模板层。

#### 改造前的问题

| 问题 | 证据 |
|---|---|
| 镜头数硬编码 | `prompts.SHOWRUNNER_PROMPT` 原文写死「包含2个连续镜头」 |
| `scene_count_range` 是死字段 | `_CONTENT_TYPE_PRESETS` 定义了 6 种范围，**从未被任何代码消费** |
| 结构一刀切 | 15s 快闪与 3min 短剧集走同一结构，无「集」概念 |
| 无目标时长输入 | `state` 无 `target_duration` / `episode` 字段 |

#### 三层参数化

<div class="rich-grid">
<div class="rich-card"><div class="rich-card-title">时长 → 镜头数</div>target_duration ÷ 8s/镜 → ceil → 按 content_type 的 scene_count_range 钳制</div>
<div class="rich-card"><div class="rich-card-title">类型 → beat 模板</div>6 个 content_type 各映射一个叙事弧模板（crescendo/arc/flat_wave/rhythmic/single_peak/ascending_steps）</div>
<div class="rich-card"><div class="rich-card-title">模板 → 自适应伸缩</div>expand_beats(template, n) 保证恰好 n 个 beat，首 setup / 尾 resolution / climax 靠后</div>
</div>

#### 核心模块：`tools/narrative.py`

| API | 职责 |
|---|---|
| `plan_scene_count(target_duration, content_type, shot_seconds=8.0)` | 时长 → 镜头数（护栏 `[2, 20]` + content_type 范围） |
| `NARRATIVE_TEMPLATES` | 6 个模板：刚性 anchors（含 develop，保证三幕完整）+ 填充池 fillers |
| `expand_beats(template_key, n)` | 保形伸缩为恰好 n 个 beat（不变量：首 setup、末 resolution、保 climax） |
| `scene_count_hint(...)` | 生成注入 showrunner 的中文镜头数约束（`max_shots`/smoke 优先） |
| `resolve_target_duration / resolve_episode` | 从 state 读取（含非法值兜底） |
| `format_episode_anchors_hint(anchors)` | 上集 end_state 锚点 → 本集承接提示 |
| `export_episode_assets(sheets, anchors)` | 本集资产 + 锚点 → 下一集可复用快照 |
| `build_series_inputs(task, episode_tasks, n, dur)` | 构造连续剧每集初始输入 |

#### 连续剧化（跨集接力）

```
第 N 集 end_state ──export_episode_assets──▶ {series_assets, episode_anchors}
                                                    │
第 N+1 集 initial state ◀── 注入 ────────────────────┘
    · series_assets   → design.reference_gen 直接复用（省额度 + 跨集角色一致）
    · episode_anchors → showrunner 注入「承接上集」提示（尾帧/色调/角色/能量）
```

新增 CLI：`scripts/run_series.py`（逐集驱动 + 自动 approve + 集间接力），
亦提供 `--dry-run` 零额度查看每集将注入的输入。

#### 改动文件

| 文件 | 改动 |
|---|---|
| `tools/narrative.py` | **新增**：全部模板与编排函数 |
| `state.py` | +4 字段：`target_duration` / `episode` / `episode_anchors` / `series_assets` |
| `prompts.py` | `SHOWRUNNER_PROMPT`：硬编码「2 个镜头」→ `{scene_count_hint}` 占位 |
| `nodes/scripting.py` | showrunner 接时长驱动；`_build_narrative_arc` 回退改模板驱动；episode 承接 |
| `nodes/design.py` | 跨集 `series_assets` 复用（跳过参考图生成） |
| `scripts/run_series.py` | **新增**：跨集编排 CLI |
| `graph.py` | 修复兼容层遗漏：补导出 `make_thread_config`（run_stories.py 依赖） |

#### 环境/行为

- `target_duration` 缺省时不硬性约束镜头数，只注入「叙事结构偏好」提示（向后兼容）。
- 3 镜以下豁免三幕完整性硬校验（篇幅天然不足，避免无谓重试）。
- `max_shots` / SMOKE 模式仍最高优先（免费额度截断保护）。

---

### F-6 移动端与窄屏审计（Mobile & Narrow-Screen Audit）

把「桌面优先」的前端补上移动端能力：三处关键区在窄屏可用，并顺带把 85KB 单文件
`styles.css` 重构为**令牌 / 布局 / 组件 / 响应式**四层。

#### 改造前的问题

| 问题 | 证据（改造前磁盘状态） |
|---|---|
| 布局硬绑双列 | `.app { grid-template-columns: 280px 1fr }`，1100px 以下仅简单塌陷为单列 |
| 侧栏在手机上霸屏 | 单列后 `.sidebar` 变成一整块堆在内容**上方**，须滚过整屏才见到主体 |
| 嵌套滚动 | `.main` / `.product-zone` 用 `max-height: 100vh` + `overflow-y:auto`；手机地址栏伸缩时 100vh 失准、双滚动层相互抢手势 |
| 无手机断点 | 仅 1100 / 900 / 720px 三处零散 media query，无 768 / 430 层 |
| 触控目标偏小 | 审核按钮约 39px 高（低于 44px 触控下限） |
| iOS 聚焦缩放 | 表单控件字号 12–13px，iOS 聚焦时自动放大整页 |
| 单文件难维护 | `styles.css` 85KB / 3,883 行，无分层 |

#### 四层样式架构

```
src/styles.css          入口：仅 4 行 @import（Vite 内联为单一 CSS 产物）
  └─ styles/tokens.css     设计令牌（:root 变量）+ reset + body
  └─ styles/layout.css     App shell / 侧栏 / 主区 / 顶栏 / 内容栅格
  └─ styles/components.css 面板 / 按钮 / 表单 / 媒体 / 时间轴 / 审核卡 / 画廊
  └─ styles/responsive.css 移动端适配层（最后加载，用于覆盖）
```

分层是**按原行范围机械切分**（1–57 / 58–406 / 407–末），切分脚本内置
「拼接后逐字节等于原文」断言，因此拆分**零逻辑变更**。

#### 断点体系（responsive.css）

| 断点 | 覆盖场景 | 主要动作 |
|---|---|---|
| `≤1024px` | 平板 / 小笔记本 | 解除 `100vh` 固定高度与嵌套滚动 |
| `≤900px` | 手机横屏 / 平板竖屏 | 侧栏改**左侧抽屉**；三处关键区适配 |
| `≤768px` | 手机 | 排版降级、输入控件字号 ≥16px、Tab 条横向滚动 |
| `≤430px` | 小屏手机 | 进一步收紧字号与内边距 |
| `pointer: coarse` | 触控设备 | 统一放大点击目标至 ≥44px |

#### 三处关键区（路线图硬指标）

<div class="rich-grid">
<div class="rich-card"><div class="rich-card-title">① 审核卡片</div>操作按钮由横排改**纵向全宽**（46px 高）；修正方式单列；候选关键帧横向滚动；通过后锁定项加大点击区；textarea 16px 防 iOS 缩放</div>
<div class="rich-card"><div class="rich-card-title">② 成片播放</div>视频撑满内容宽、限高 <code>56dvh</code>；下载 / 无音轨链接改**纵向卡片式**（≥44px）</div>
<div class="rich-card"><div class="rich-card-title">③ 进度时间轴</div>解除 <code>max-height</code> 锁；标题行可换行；音频子步骤自动折行</div>
</div>

#### 移动端导航抽屉

`≤900px` 时侧栏由文档流改为 `position: fixed` 抽屉：

```
app-header [☰ 会话] ──点击──▶ .app.nav-open + .sidebar.open
                                   │  translateX(-100% → 0)
                                   ├─ .nav-scrim 遮罩（点击关闭）
                                   └─ .sidebar-close 关闭按钮
选中任一历史会话 ──▶ 自动收起抽屉
```

`.nav-scrim` 默认 `display: none`（桌面即便误渲染也不占位，不破坏 grid 轨道）。

#### 改动文件

| 文件 | 改动 |
|---|---|
| `src/styles.css` | 85KB 单文件 → 4 行 `@import` 入口 |
| `src/styles/{tokens,layout,components}.css` | **新增**：按原行范围机械切分 |
| `src/styles/responsive.css` | **新增**：移动端适配层 |
| `src/App.tsx` | 抽屉状态 + 汉堡按钮 + 遮罩 + 侧栏 props |
| `src/components/ThreadSidebar.tsx` | `mobileOpen` / `onClose` props + 关闭按钮 + 选中自动收起 |
| `index.html` | `viewport-fit=cover`、`theme-color`、`color-scheme`、`mobile-web-app-capable` |
| `scripts/audit-mobile.mjs` | **新增**：CDP 窄屏审计工具（多断点断言，退出码可入 CI） |
| `scripts/mobile-audit-fixture.html` | **新增**：审计夹具（三处关键区 DOM） |
| `scripts/check-css-layers.mjs` | **新增**：CSS 分层与内联产物校验 |

#### 验证方式（客观测量，非肉眼）

用 Chrome DevTools Protocol 在真机视口（390 / 430 / 768 / 1280）下渲染，读取
`getBoundingClientRect` / `getComputedStyle` 做断言，并验证真实应用的抽屉交互：

```bash
cd frontend
npm run build
node scripts/audit-mobile.mjs      # 48/48 断言（4 断点 × 三处关键区 + 导航抽屉）
node scripts/check-css-layers.mjs  # 19/19 分层与内联校验
```

真实应用（vite dev）抽屉链路实测：汉堡可见 → 点击后 `.app.nav-open` +
`sidebar translateX(0)` → 遮罩 `display:block` → 点遮罩后回到 `translateX(-330px)`。

---

### 持久化机制

| 层 | 技术 | 存储位置 | 用途 |
|---|---|---|---|
| 图状态检查点 | langgraph SqliteSaver | `data/multimedia_checkpoints.sqlite` | 每次节点转换自动快照，支持暂停/恢复/时间旅行 |
| 任务注册表 | 自建 SQLite | `data/multimedia_tasks.sqlite` | 应用层任务元数据（状态/进度/结果），跨会话恢复 |
| 向量知识库 | ChromaDB PersistentClient | `data/chroma_db/` | RAG 语义检索数据源 |
| Streamlit Session | st.session_state | 内存 | UI 状态，含 35+ 个 session key |

---

### Streamlit HITL 控制台 (`streamlit_app.py`, ~1644 行)

7 个 Tab 覆盖完整的交互流程：

| Tab | 名称 | 功能 |
|---|---|---|
| 1 | 新手模式 | 模板选择器 + 结构化字段（标题/标签/风格/时长） + 任务描述编辑器 |
| 2 | 专业模式 | 直接全文本任务编辑器 |
| 3 | 恢复中心 | 恢复/创建/列出所有任务，支持 resume/fork/timeline 操作 |
| 4 | 分叉恢复 | 从任意 checkpoint 分叉出新线程，可编辑状态后继续执行 |
| 5 | 时间轴 | 按时间线查看所有 checkpoint，每个可展开查看状态摘要和分叉 |
| 6 | 增量设计 | Phase 1-5 + RAG 可视化面板，实时展示每个阶段的输出结果 |
| 7 | 当前状态 | 4 个顶层指标 + 原始 JSON 状态 + 逐 scene 摘要卡片 |

侧边栏包含：任务设置（首尾帧双控开关）、线程控制（thread_id 输入/新建/恢复）、帮助说明、当前线程摘要（scene 进度/critic 分数/RAG 引用数等）。

中断面板在 Tab 下方渲染，根据 stage 类型展示不同的审核表单（approve/rewrite/edit prompt），用户提交后通过 `Command(resume=decision)` 恢复管线。

---

### 项目文件结构

```
my-langgraph-practice-main/
├── streamlit_app.py                  # Streamlit HITL 控制台入口
├── pyproject.toml                    # 项目配置 + 依赖声明
├── langgraph.json                    # LangGraph 服务配置
├── start.sh                          # LangGraph API 服务启动脚本
├── Makefile                          # 开发工具链 (test/lint/format)
│
├── src/agent/multimedia/
│   ├── __init__.py
│   ├── graph.py                      # 兼容层：再导出 nodes/ 全部实现（P2-3 后 ~90 行）
│   ├── nodes/                        # P2-3：原 graph.py 单体按 stage 拆分
│   │   ├── __init__.py               #   包入口（stdout/stderr UTF-8 容错）
│   │   ├── common.py                 #   跨 stage 常量 / 纯工具 / 风格与 prompt 工具箱
│   │   ├── scripting.py              #   脚本与规划（视觉上下文 / 镜头策略 / 序列编排 / showrunner）
│   │   ├── design.py                 #   视觉设计（参考图生成 / 资产匹配）
│   │   ├── director.py               #   导演（电影级评审 / 精修 / 影棚 / 首尾帧导演 / 运镜）
│   │   ├── render.py                 #   渲染（关键帧 / 尾帧 / 视频生成 + 审核闸口）
│   │   ├── assemble.py               #   成片（推进 / 中止 / 拼接 / 草稿审片 / 混音）
│   │   ├── routing.py                #   路由决策（decide_*）
│   │   ├── parallel.py               #   并行 / 链式扇出（Send fan-out / 镜头子图 / FLF 链）
│   │   └── wiring.py                 #   图组装与编译（build + compile + thread 状态查询）
│   ├── state.py                      # MultimediaState TypedDict (~48 行)
│   ├── prompts.py                    # 9 个 Prompt 模板 (~184 行)
│   ├── checkpointer.py               # SQLite checkpointer 单例 (~47 行)
│   ├── task_store.py                 # 任务注册表 SQLite 持久化
│   │
│   ├── visual_context.py             # Phase 1: 视觉规则层 (~450 行)
│   ├── shot_strategy.py              # Phase 2: 镜头策略层 (~864 行)
│   ├── sequence_orchestrator.py      # Phase 3: 序列编排层 (~562 行)
│   ├── cinematic_critic.py           # Phase 4: 质量评估层 (~619 行)
│   ├── film_studio_orchestrator.py   # Phase 5: 专家协作编排
│   ├── cinematography_agent.py       # Phase 5 子Agent: 摄影指导
│   ├── lighting_agent.py             # Phase 5 子Agent: 灯光指导
│   ├── editor_agent.py               # Phase 5 子Agent: 剪辑指导
│   ├── cinematic_memory.py           # 跨 scene 风格一致性 memory bank（thread 作用域 + data/memory 落盘）
│   │
│   ├── tools/
│   │   ├── text_llm.py               # DashScope 文本 LLM (tongyi-xiaomi-analysis-pro)
│   │   ├── image_gen.py              # DashScope 图像生成 (wan2.7-image)
│   │   ├── video_gen.py              # DashScope 图生视频 (wan2.7-i2v)
│   │   ├── video_edit.py             # DashScope 视频精修 (wan2.7-videoedit)
│   │   ├── video_stitcher.py         # MoviePy 视频标准化 + 拼接
│   │   ├── vision_eval.py            # DashScope 视觉审核 (qwen-image-2.0 / qwen3.5-omni)
│   │   ├── text_embedding.py         # DashScope 文本向量化 (text-embedding-v3)
│   │   └── embedding.py              # DashScope 图像向量化 (multimodal-embedding-v1)
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── vector_store.py           # ChromaDB 初始化 + 3 Collection 管理
│   │   ├── retriever.py              # 语义检索 + 格式化 + 静默降级
│   │   ├── ingest.py                 # 灌入 CLI 入口 (--source md/gen/prompts/films)
│   │   ├── README.md                 # RAG 模块使用文档
│   │   └── ingestors/
│   │       ├── __init__.py
│   │       ├── markdown_ingestor.py  # md 文件切块 + 向量化灌入
│   │       ├── knowledge_generator.py # LLM 生成 10 个专业知识文件
│   │       ├── prompt_scraper.py     # Civitai prompt 爬虫
│   │       └── film_knowledge_scraper.py # 影视知识爬虫
│   │
│   └── vision_knowledge/             # Phase 0 知识库（RAG 数据源）
│       ├── camera_grammar.md
│       ├── lighting_grammar.md
│       ├── trailer_patterns.md
│       └── generated/                # LLM 生成的知识文件
│
└── data/
    ├── multimedia_checkpoints.sqlite # LangGraph 图状态检查点
    ├── multimedia_tasks.sqlite       # 任务注册表
    └── chroma_db/                    # ChromaDB 持久化存储
```

---

### 环境配置与运行

#### 1. 环境准备

创建 `.env` 文件（项目根目录）：

```env
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
LANGSMITH_PROJECT=new-agent
```

安装依赖（推荐 conda 环境）：

```bash
pip install langgraph>=1.0.0 python-dotenv chromadb>=0.5.0
pip install openai streamlit requests httpx moviepy Pillow numpy
```

#### 2. 初始化 RAG 知识库（首次运行）

```bash
python -m src.agent.multimedia.rag.ingest --source gen
```

该命令会调用 LLM 生成 10 个摄影专业主题的知识文件，然后连同手写的 3 个 md 文件一起切块、向量化、存入 ChromaDB。后续运行管线时自动检索，无需重复灌入。

其他灌入选项：`--source md`（仅手写文件）、`--source prompts`（Civitai 爬虫）、`--force`（清空重建）、`--stats`（查看统计）。

#### 3. 启动 Streamlit 控制台

```bash
streamlit run streamlit_app.py
```

在"新手模式"或"专业模式"中输入创意描述，管线自动开始执行。遇到 interrupt 节点时 UI 会暂停，显示审核面板。

#### 4. 启动 LangGraph API 服务（可选）

```bash
bash start.sh
```

激活 conda 环境 `myenv`，设置 `PYTHONPATH=src`，在端口 8123 启动 LangGraph API 服务。

---

### 数据流一句话总结

用户创意 → showrunner 拆 4 个 scene → 每个 scene 经规则驱动视觉设计（5 个 Phase 节点）→ RAG 知识叠加 → director LLM 生成 prompt → 图像生成+审核循环 → 视频生成+审核循环 → 全 scene 拼接为最终宣传片。
