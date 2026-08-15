# 交互对接核查清单

> 生成时间：2026-08-10
> 覆盖范围：`frontend/src` 下所有 `onClick` / `onSubmit` / `onKeyDown` / `onChange` 交互点
> 后端端点：LangGraph Platform (`/lg/*`) + 自建 history 接口 (`/history/*`) + 产物托管 (`/media/*`)

## 核查结论总览

| 分类 | 数量 | 状态 |
|---|---|---|
| 有后端调用的交互 | 8 | ✅ 全部对接正确 |
| 纯前端交互 | 17 | ✅ 行为与标签一致 |
| **总计** | **25** | **0 缺陷** |

---

## 一、App.tsx —— 核心流程控制

| # | 元素 / 文案 | 事件 | 处理链路 | 后端接口 | 实测结论 |
|---|---|---|---|---|---|
| 1 | `<button>`「中断」(仅 busy 时) | onClick → `stop()` | `abortRef.current.abort()` | **纯前端**：中止 AbortController，SSE 断开。不通知后端，run 在服务端继续。 | ✅ 立即停止前端流消费 |
| 2 | `<ThreadSidebar onSelect>` | onSelect → `loadThread(tid)` | 通道1: `GET /history/threads/{id}` → 通道2: `client.threads.getState(tid)` | `/history/threads/{id}` + 兜底 `/lg/threads/{id}/state` | ✅ 双通道容错，两条都败才报错 |
| 3 | `<TaskLauncher onStart>` | onStart → `handleStart` → `startRun` | `client.threads.create()` → `client.runs.stream(tid, "agent", {input})` → `client.threads.getState` | `POST /lg/threads` + `POST /lg/threads/{id}/runs/stream` | ✅ 新建线程 + 启动任务 |
| 4 | `<InterruptApprovalCard onSubmit>` | onSubmit → `handleDecision` → `resumeRun` | `client.runs.stream(tid, ASSISTANT_ID, {command:{resume:decision}})` | `POST /lg/threads/{id}/runs/stream` | ✅ 提交审核恢复图执行 |
| 5 | `<button>`「继续执行」(actionMode="continue" 时) | onClick → `handleContinue` → `continueRun` | `client.runs.stream(tid, ASSISTANT_ID, {input:null})` | `POST /lg/threads/{id}/runs/stream` | ✅ 无中断会话续跑 |

**本次修复**：
- 审核卡片与续跑面板改用 `actionMode`（review/continue/none）互斥渲染，从结构上杜绝双面板。
- `handleDecision` 补 `setRefreshKey` 刷新侧栏，与 `handleStart`/`handleContinue` 一致。
- `useAgentStream` 所有异步入口补齐 try/catch → setError(可读中文)，消除静默失败。

---

## 二、components/ThreadSidebar.tsx —— 历史会话列表

| # | 元素 / 文案 | 事件 | 后端接口 | 实测结论 |
|---|---|---|---|---|
| 6 | `<button>`「⟳」刷新 | onClick → `load()` | `GET /history/threads?limit=50` → 兜底 `POST /lg/threads/search` | ✅ loading 时旋转并禁用 |
| 7 | `<button class="thread-item">` 会话条目 | onClick → `onSelect(thread_id)` → App `loadThread` | 同 #2 | ✅ 损坏条目禁用点击 |
| 8 | `<button>`「重试」(错误时) | onClick → `load()` | 同 #6 | ✅ 本次新增，侧栏内联重试 |
| — | 挂载 / refreshKey 变化 | useEffect → `load()` | 同 #6 | ✅ 自动首屏 + 任务完成后刷新 |

---

## 三、components/TaskLauncher.tsx —— 新建任务

| # | 元素 / 文案 | 事件 | 后端接口 | 实测结论 |
|---|---|---|---|---|
| 9 | `<textarea>` 任务描述 | onChange → `setTask` | **纯前端** | ✅ 受控输入 |
| 10 | 同上 `<textarea>` | onKeyDown: Ctrl+Enter → `submit()` | 经 onStart → `startRun` → `client.threads.create()` + `runs.stream` | ✅ 快捷键提交 |
| 11 | `<input type="checkbox">`「首尾帧双控」 | onChange → `setUseFirstLastFrame` | **纯前端** | ✅ |
| 12 | `<input type="checkbox">`「启用配音」 | onChange → `setEnableAudio` | **纯前端**（联动禁用下方 select） | ✅ |
| 13 | `<select>` 配音音色 | onChange → `setVoiceRole` | **纯前端** | ✅ |
| 14 | `<select>` BGM 情绪 | onChange → `setBgmMood` | **纯前端** | ✅ |
| 15 | `<button>`「开始生成 (Ctrl+Enter)」 | onClick → `submit()` | `client.threads.create()` + `runs.stream` | ✅ busy 或空输入时禁用 |

---

## 四、components/InterruptApprovalCard.tsx —— 人工审核

| # | 元素 / 文案 | 事件 | 后端接口 | 实测结论 |
|---|---|---|---|---|
| 16 | `<textarea>` 任务描述 (showrunner_review) | onChange → `setTask` | **纯前端** | ✅ |
| 17 | `<textarea>` 全局设定 | onChange → `setGlobalSetting` | **纯前端** | ✅ |
| 18 | `<textarea class="mono">` 分镜 JSON | onChange → `setScenesText` | **纯前端**；提交时校验 JSON 解析 | ✅ 解析失败阻断提交 |
| 19 | `<textarea class="mono">` Prompt (非 showrunner) | onChange → `setPrompt` | **纯前端** | ✅ 改动显示「已修改」标记 |
| 20 | `<textarea>` 打回理由 | onChange → `setReason` | **纯前端** | ✅ |
| 21 | `<button>`「通过」/「打回重写」/「改写后继续」 | onClick → `handle(action)` → `buildDecision` → `onSubmit` | `runs.stream({command:{resume:decision}})` | ✅ busy 时按钮禁用 + 显示"提交中…" |
| — | payload 变化 | useEffect 重置表单 | **纯前端** | ✅ |

---

## 五、components/MediaPreview.tsx + FinalMoviePlayer.tsx —— 产物展示

| # | 元素 | 事件 | 后端接口 | 实测结论 |
|---|---|---|---|---|
| 22 | `<video controls>` 产物播放 | 原生控件 | `GET /media/{path}` (隐式) | ✅ 走 static_server |
| 23 | `<a target="_blank">` `<img>` 缩略图 | 原生导航 | `GET /media/{path}` (隐式) | ✅ 新标签页打开原图 |
| 24 | `<video controls>` 最终成片 | 原生控件 | `GET /media/...` | ✅ 优先 final_movie_with_audio |
| 25 | `<a download>`「下载成片」 | 原生导航 | `GET /media/...` | ✅ |

---

## 六、无交互的展示组件（零事件、零后端调用）

- **components/NodeProgressTimeline.tsx** — 纯展示 `<ol>/<li>` 时间线
- **components/MetricBadges.tsx** — 纯展示 `<span class="badge">` 指标徽标
- **components/SceneGallery.tsx** — `<article class="scene-card">` 无 onClick；内嵌 MediaPreview 产生隐式 `/media` 请求

---

## 七、关键设计说明

### 7.1 assistant 标识
所有 `client.runs.*` 调用统一使用 `ASSISTANT_ID = import.meta.env.VITE_ASSISTANT_ID ?? "agent"`（来自 `langgraphClient.ts`），由 `useAgentStream` 内部注入，不依赖调用方传入。

### 7.2 会话动作模式（actionMode）
- `review`：有中断，只允许 `resumeRun(Command.resume)`
- `continue`：无中断但有待执行节点，只允许 `continueRun({input:null})`
- `none`：已完成 / 已中止 / 无可执行动作

App.tsx 严格按此互斥渲染操作入口。

### 7.3 续跑语义
- `continueRun({input: null})` — LangGraph 标准续跑，从最新 checkpoint 推进
- `resumeRun({command:{resume:decision}})` — 中断恢复，`decision` 经 `_normalize_decision` 兼容 dict/str/bool/None

### 7.4 错误可见化
所有异步入口均补齐 try/catch → setError(可读中文) + setStatus("error")，确保点击失败时有明确 UI 反馈。

### 7.5 双通道容错
列表（ThreadSidebar `load`）和详情（`loadThread`）都是 `/history` 优先 → 平台 `client.threads.*` 兜底，两条都败才报错。

---

## 八、未覆盖项（已知限制）

| 项 | 说明 |
|---|---|
| 「中断」按钮不真中断后端 | `stop()` 仅 abort 前端 AbortController，服务端 run 继续执行。若需真中断需调用 `client.runs.cancel(threadId, runId)`，当前未实现。 |
| 无"加载更多"分页 | ThreadSidebar 固定拉 50 条，无 offset 递增。45 条会话在阈值内，暂不需要。 |
