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

### 22 个节点详解

#### 节点注册表

| # | 节点名 | 实现函数 | 所属阶段 |
|---|---|---|---|
| 1 | `showrunner` | `showrunner_node` | 剧本策划 |
| 2 | `showrunner_review` | `showrunner_review_gate_node` | 剧本审核 (HITL) |
| 3 | `visual_context_builder` | `visual_context_builder_node` | Phase 1 视觉规则 |
| 4 | `shot_strategy_builder` | `shot_strategy_builder_node` | Phase 2 镜头设计 |
| 5 | `sequence_orchestrator` | `sequence_orchestrator_node` | Phase 3 序列编排 |
| 6 | `film_studio` | `film_studio_node` | Phase 5 专家协作 |
| 7 | `cinematic_critic` | `cinematic_critic_node` | Phase 4 质量评估 |
| 8 | `director` | `director_node` | 导演创作 (LLM) |
| 9 | `director_refine` | `director_refine_node` | Prompt 精炼 |
| 10 | `image_generator` | `image_gen_node` | 图像生成 |
| 11 | `reviewer` | `reviewer_node` | 图像审核 (LLM) |
| 12 | `image_review` | `image_review_gate_node` | 图像审核门控 (HITL) |
| 13 | `end_frame_director` | `end_frame_director_node` | 尾帧设计 (LLM) |
| 14 | `end_frame_generator` | `end_frame_gen_node` | 尾帧生成 |
| 15 | `end_frame_review` | `end_frame_review_gate_node` | 尾帧审核 (HITL) |
| 16 | `videographer` | `videographer_node` | 运镜设计 (LLM) |
| 17 | `video_generator` | `video_gen_node` | 视频生成 |
| 18 | `video_reviewer` | `video_reviewer_node` | 视频审核 (LLM) |
| 19 | `video_review` | `video_review_gate_node` | 视频审核门控 (HITL) |
| 20 | `advance_scene` | `advance_scene_node` | 场景推进 |
| 21 | `abort` | `abort_node` | 异常终止 |
| 22 | `stitcher` | `stitcher_node` | 最终拼接 |

#### 条件路由（6 个分支点）

| 路由函数 | 触发节点 | 路径 A | 路径 B |
|---|---|---|---|
| `decide_after_critic` | cinematic_critic | `director` (PASS) | `shot_strategy_builder` (rewrite) |
| `decide_image_quality` | image_review | `end_frame_director` / `videographer` (PASS) | `director` (重生成) |
| `decide_end_frame_quality` | end_frame_review | `videographer` (PASS) | `end_frame_director` (重生成) |
| `decide_after_video_generation` | video_generator | `video_reviewer` (正常) | `abort` (异常) |
| `decide_video_quality` | video_review | `advance_scene` (PASS) | `videographer` (重生成) |
| `decide_next_scene` | advance_scene | `visual_context_builder` (下一 scene) | `stitcher` (全部完成) |

#### Human-in-the-Loop 中断点（4 个）

管线在以下节点会暂停执行，等待用户通过 Streamlit UI 审核并决策（approve / rewrite / edit）：

| 中断节点 | stage 标识 | 审核内容 |
|---|---|---|
| `showrunner_review` | `showrunner_review` | 全局设定 + 4 个 scene 剧本 |
| `image_review` | `image_review` | 关键帧图片 + 前帧对比 + embedding 相似度 |
| `end_frame_review` | `end_frame_review` | 尾帧图片 + 首帧对比 |
| `video_review` | `video_review` | 视频片段 + 运镜 prompt |

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

跨 scene 一致性由 `cinematic_memory.py` 维护——一个模块级单例 memory bank，存储每个 scene 的摄影/灯光/剪辑风格摘要，提供 `get_previous_scene_style()` 和 `generate_film_brain_report()` 接口。

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
│   ├── graph.py                      # 22 节点 StateGraph 定义 + 编译 (~1151 行)
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
│   ├── cinematic_memory.py           # 跨 scene 风格一致性 memory bank
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
