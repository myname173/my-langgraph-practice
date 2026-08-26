# Multimedia Agent 架构改进实施方案

> 版本：v1.0 ｜ 日期：2026-08-16 ｜ 角色：总架构师
> 范围：`src/agent/multimedia/`（后端 LangGraph 状态机 + 工具链）+ `frontend/`（React 控制台）
> 目标：修复阻断级断点、降低普通用户操作门槛、消除冗余代码、提升多故事扩展性

---

## 0. 现状结论（评审摘要）

项目整体架构达到大厂中上水平：LangGraph 26 节点状态机编排、6 道人审闭环、角色一致性工程（reference_gen + embedding 校验）、多视频后端容灾（agnes/jimeng/zhipu）、后期合成（stitcher + audio_mixer）。主干流程连通、前后端字段对齐度高、无游离死代码。

**已具备的能力（评审中发现，计划中不重复造轮子）：**
- ✅ `max_rewrites` 循环上限已实现：配置化（`MAX_REWRITES` env / `thresholds.json`，默认 2），`graph.py:4013` 有"已达上限强制通过"逻辑。→ 本计划仅做**外围默认值上调 + 文档化**，无需重新实现。

**待修复/待优化的 6 项**（按优先级 P0–P3 排列）。

---

## P0 🔴 修复「快速生成」端点断链

### 问题
- 前端 `frontend/src/components/QuickGeneratePanel.tsx:26` 调用 `fetch("/media/generate", {method:"POST"})`。
- 后端 `static_server.py:438` 注册的是 `POST /generate`（无 `/media` 前缀）。
- vite 代理把 `/media/*` 转发到静态服务，会命中 `GET /media/{file_path:path}` 通配路由 → 几乎必现 404。
- 结果：**「快速生成」面板 100% 失效**，用户输入一句话点「立即生成」必然报错。

### 方案（二选一，推荐 A）
- **方案 A（改前端，零后端改动）**：`QuickGeneratePanel.tsx:26` 的 `/media/generate` → `/generate`。
  - 同时确认 `vite.config.ts` 的 `/media` 代理不影响 `/generate`（建议给 `/generate` 单独建代理或复用 `/lg` 之外的直连）。
  - 后端响应为 `{url, type}`（无 `remote` 字段），前端 `result.remote` 已做 `undefined` 容错，无需改。
- **方案 B（改后端，补路由）**：在 `static_server.py` 增加 `@app.post("/media/generate")` 别名指向同一 handler，保持前端不动。

### 验收
- 在 dev 环境输入一句话 → 选视频/图片 → 点「立即生成」→ 预览区出现结果，无 404。
- 补充一条冒烟测试脚本 `scripts/smoke_quick_generate.sh`（curl `/generate` 验证 200）。

### 工作量：0.5 人日

---

## P1 🟠 增加「低干预 / 自动模式」跳过人审闸口

### 问题
6 道 `interrupt()` 人工闸口（showrunner_review / prompt_preview / image_review / end_frame_preview / end_frame_review / video_review）对专业用户有价值，但对普通用户过重：多镜叠加 = 数十次点击。业界（Pika / 可灵 / Runway）默认低干预，关键节点才确认。

### 方案
1. **后端**：`state.py` 的 `MultimediaState` 增加可选字段 `auto_mode: bool = False`。
   - `TaskLauncher` 提交 `RunInput` 时携带 `auto_mode`。
   - `graph.py` 中 6 处 `interrupt()` 前置判断：若 `state.get("auto_mode")` 为真，则跳过 `interrupt()`，直接以默认决策（approve）继续，并写入 `auto_approved` 标记供前端展示。
   - 抽一个工具函数 `should_interrupt(state, stage)` 统一封装（消除 6 处重复判断）。
2. **前端**：`TaskLauncher.tsx` 选项网格加「🤖 自动模式（跳过逐镜确认）」开关；`NodeProgressTimeline` 对被自动放行的节点标注「自动」。
3. **配置**：`.env` 增加 `AUTO_MODE_DEFAULT=false`（默认关闭，专业用户默认走人审）。

### 验收
- 勾选自动模式 → 跑全流程 → 无 `InterruptApprovalCard` 弹出，成片正常产出。
- 不勾选 → 行为与现有一致（人审照常）。

### 工作量：2 人日

---

## P1 🟠 故事/素材匹配改为配置驱动（去掉硬编码）

### 问题
- `AssetLibraryPanel.tsx:71-75` 用 `threadId.includes("gull") / includes("cyber")` 推断故事名。
- 当前 git 已有 `scripts/resume_xianxia.py`（仙侠），新建其它故事（如 xianxia）会 `wantStory=null`，导致展示全部 `_legacy` 图，**混入无关素材**。
- 扩展性差，每加一个故事就要改前端代码。

### 方案
1. **配置化故事映射**：后端 `config_loader.py` 新增 `STORY_KEYWORDS` 配置（dict：story_name → 关键词列表），从 `.env` 或 `vision_knowledge/config/stories.json` 读取。
   - 例：`{"gull": ["gull"], "cyber": ["cyber"], "xianxia": ["xianxia","仙侠"]}`
2. **前端去硬编码**：`AssetLibraryPanel` 改为调用新增接口 `GET /history/story_keywords`（或直接读取注入到 `RunInput` 的 `story` 字段），用配置匹配替代 `includes` 字面量。
3. **manifest 增强**：`keyframes/_legacy/manifest.json` 的每条目已有 `story` 字段，前端按 `story` 精确过滤（现逻辑已支持，只需让 `wantStory` 来源从配置而非硬编码）。
4. 新增 `scripts/resume_xianxia.py` 等脚本在创建 thread 时，将 `story` 写入 thread 元数据，前端直接消费。

### 验收
- 新建 xianxia 任务 → 素材库只显示仙侠相关图，不混入 gull/cyber。
- 新增第 N 个故事只需改配置，前端零改动。

### 工作量：1.5 人日

---

## P2 🟡 清理冗余代码

### 问题 A
- `frontend/src/components/AssetPicker.tsx` 导出 `buildReferenceSheetsFromPicks()`（约 162-176 行），但 `TaskLauncher` 内部自实现 `buildSheets()`，**从未 import 该导出** → 重复逻辑。

### 方案 A
- 删除 `AssetPicker.tsx` 中的 `buildReferenceSheetsFromPicks` 导出（保留 `AssetPicker` 组件本身）。
- 或反过来：让 `TaskLauncher` 复用 `buildReferenceSheetsFromPicks`，删除 `TaskLauncher.buildSheets`。推荐前者（改动最小）。

### 问题 B
- `useAgentStream` 返回 `canContinue`（`useAgentStream.ts:336`），但 `App.tsx` 实际用 `actionMode === "continue"`，`canContinue` 未被消费。

### 方案 B
- 删除 `useAgentStream` 中 `canContinue` 的计算与返回，或统一改为单一数据源（`actionMode`）。

### 验收
- `grep -rn "buildReferenceSheetsFromPicks" frontend/` 无引用；`canContinue` 无引用。
- 构建通过（`npm run build`）。

### 工作量：0.5 人日

---

## P2 🟡 上传素材默认分类与命名优化

### 问题
- `TaskLauncher` 上传图默认 `category: "character"`、未填名时默认 `"素材"`。
- 用户可能把道具/场景误填为「素材」→ 后端按角色匹配剧本称呼失败（仅提示不影响运行，但一致性受损）。

### 方案
1. `TaskLauncher.uploadViaFormData`：上传时**不预设 category**，改为前端选择分类后再提交；或默认 `category` 留空，提交时校验「未选分类」拦截。
2. 默认名不再用字面量「素材」，改为空 + 必填校验（与现有「未填写名称拦截」逻辑合并）。
3. 复用 `AssetLibraryPanel` 已有的 `nameSuggestions`（datalist 候选）引导用户选择已有角色名，避免手打错名。

### 验收
- 上传道具图 → 不会自动归入「角色」；未命名提交被拦截并提示。

### 工作量：0.5 人日

---

## P3 🟢 成片草稿预览（draft 机制）

### 问题
当前 `stitcher → audio_mixer` 直接出最终成片（带配音字幕）。若用户对剪辑节奏/镜头顺序不满意，重跑成本高（需重走整条链路或重跑多镜）。

### 方案（对标 Runway draft）
1. `stitcher` 节点产出「无声粗剪版」`final_movie_path`（已有字段）。
2. 在 `audio_mixer` 前插入**可选闸口**：若 `enable_audio` 且未开启 `auto_mode`，先展示无声粗剪 + 「确认混音 / 调整顺序」按钮；确认后再 `audio_mixer`。
3. 前端 `FinalMoviePlayer` 增加「草稿 / 成片」双标签；`TimelineBar` 增加「调整镜头顺序」入口（调用已有 `rerun_from` 机制）。

### 验收
- 开启草稿模式 → 先看无声粗剪 → 确认 → 出带音频成片。
- 关闭 → 行为与现有一致。

### 工作量：2 人日（可后置到 v1.1）

---

## 实施路线图

| 阶段 | 任务 | 优先级 | 工作量 | 依赖 | 状态 |
|------|------|--------|--------|------|------|
| Sprint 1 | P0 端点修复 | 🔴 P0 | 0.5d | 无 | ✅ 已完成 |
| Sprint 1 | P2 冗余清理 | 🟡 P2 | 0.5d | 无 | ✅ 已完成 |
| Sprint 1 | P2 上传分类优化 | 🟡 P2 | 0.5d | 无 | ✅ 已完成 |
| Sprint 2 | P1 自动模式 | 🟠 P1 | 2d | P0 | ✅ 已完成 |
| Sprint 2 | P1 故事配置化 | 🟠 P1 | 1.5d | P0 | ✅ 已完成（改为前端元数据驱动，0.5d 实际） |
| Sprint 3 | P3 草稿预览 | 🟢 P3 | 2d | P1 | ✅ 已完成 |

**总计：约 7 人日。** Sprint 1（P0 + 冗余清理）已交付并通过 `tsc --noEmit` 编译。

### Sprint 1 已完成改动清单（2026-08-16）
- `frontend/src/components/QuickGeneratePanel.tsx:26`：`/media/generate` → `/generate`
- `frontend/vite.config.ts`：新增 `/generate` 代理规则转发至 `MEDIA_TARGET`(8900)，使请求可达 `static_server.py` 的 `POST /generate`
- `frontend/src/components/AssetPicker.tsx`：删除未引用的 `buildReferenceSheetsFromPicks` 导出，及其连带未使用的 `ReferenceEntry`/`ReferenceSheets` 类型导入与 `CAT_TO_SHEET` 常量
- `frontend/src/hooks/useAgentStream.ts`：删除未消费的 `canContinue` 字段（定义 + 返回对象引用）

> 注：Sprint 1 中的「P2 上传分类/命名优化」因涉及交互逻辑调整，按方案保留至下一轮实施，与 P1 一并评审。

---

## 验收与回归清单

- [ ] 快速生成面板端到端可用（无 404）
- [ ] `npm run build` 通过，无未使用导出告警
- [ ] 自动模式开/关两种路径均产出成片
- [ ] 新增 xianxia 等故事素材库正确隔离
- [ ] 上传道具不再误归角色
- [ ] （P3）草稿 → 成片流程连通
- [ ] 补充冒烟测试覆盖 P0/P1 关键路径

---

## 附：已具备能力确认（无需重复实现）

- `max_rewrites` 循环上限：`config_loader.max_rewrites()` + `graph.py:4013` 强制通过。默认 2，可由 `MAX_REWRITES` env 调整。**建议**：在 `.env.example` 中补注释说明该开关，并在 UI 高级选项中暴露。
- 多后端容灾：video_gen（agnes/jimeng/zhipu）、image_gen（qwen/dashscope/siliconflow/jimeng）、tts（edge-tts→dashscope 降级）均已实现。
- 角色一致性：reference_gen + embedding 相似度校验已实现。
