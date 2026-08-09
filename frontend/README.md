# 多模态 Agent 前端控制台

基于 React + TypeScript + Vite 的控制台，用于驱动 `src/agent/graph:graph` 多模态生成工作流，
专为「人在回路（Human-in-the-loop）」设计：可视化每个节点的执行进度、预览生成产物、
并在 6 个审批关卡上进行人工决策（通过 / 打回 / 改 Prompt 后续跑）。

## 架构

```
浏览器 (Vite dev :5173)
  ├─ /lg/*   ──proxy──> LangGraph Server (:2024)   # 图执行 + 流式 + interrupt/resume
  └─ /media/*──proxy──> Media Server (:8900)        # 静态托管 output/ 下的产物
```

- `src/lib/langgraphClient.ts`：封装 `@langchain/langgraph-sdk` 的 `Client`，及把后端
  绝对路径归一化为 `/media/...` 的 `resolveMediaUrl()`。
- `src/hooks/useAgentStream.ts`：核心流式 hook，负责启动 run、监听 `events`、
  捕获 `interrupt` 并回传 `Command(resume=...)`。
- `src/components/InterruptApprovalCard.tsx`：覆盖全部 6 类中断关卡的审批卡。
- `src/components/*`：进度时间线、媒体预览、成片播放器、分镜画廊、任务启动器、会话侧栏。

## 环境变量

`frontend/.env`（可参照 `.env.example`）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `VITE_LANGGRAPH_API_URL` | `http://127.0.0.1:2024` | LangGraph Server 地址（若与前端同源部署到 `/lg` 则填相对路径） |
| `VITE_LANGGRAPH_API_KEY` | 空 | 远程部署时需要 |
| `VITE_MEDIA_BASE_URL` | `/media` | 产物 URL 前缀（需与 vite proxy 的 `/media` 一致） |
| `VITE_LANGGRAPH_PROXY_TARGET` | `http://127.0.0.1:2024` | dev proxy `/lg` 目标 |
| `VITE_MEDIA_PROXY_TARGET` | `http://127.0.0.1:8900` | dev proxy `/media` 目标 |

## 启动步骤

### 1. 后端：LangGraph Server

```bash
# 项目根目录
langgraph dev            # 默认监听 :2024，注册 graphs.agent
```

### 2. 后端：产物静态服务（独立进程）

```bash
# 项目根目录，端口默认 8900
python -m src.agent.multimedia.static_server
# 自定义端口：MEDIA_SERVER_PORT=8900 python -m src.agent.multimedia.static_server
```

> 该服务把项目根 `output/` 目录只读暴露为 `/media/<相对路径>`，仅支持 GET/HEAD，
> 并对 `..` 目录逃逸做了防护。

### 3. 前端：Vite dev

```bash
cd frontend
npm install
npm run dev             # 默认 :5173
```

打开 `http://localhost:5173`。

## 构建产物

```bash
cd frontend
npm run build           # tsc 类型检查 + vite 打包到 dist/
npm run preview         # 本地预览构建结果
```

生产部署时，请将 `dist/` 静态资源与 Media Server（:8900）一并托管，
并将前端请求的后端地址通过 `VITE_LANGGRAPH_API_URL` / `VITE_MEDIA_BASE_URL` 指向实际域名。
