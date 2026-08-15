import { useCallback, useState } from "react";
import { useAgentStream } from "./hooks/useAgentStream";
import { TaskLauncher } from "./components/TaskLauncher";
import { QuickGeneratePanel } from "./components/QuickGeneratePanel";
import { ThreadSidebar } from "./components/ThreadSidebar";
import { NodeProgressTimeline } from "./components/NodeProgressTimeline";
import { InterruptApprovalCard } from "./components/InterruptApprovalCard";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { SceneGallery } from "./components/SceneGallery";
import { TimelineBar } from "./components/TimelineBar";
import { ReferenceSheetsPanel } from "./components/ReferenceSheetsPanel";
import { AssetLibraryPanel } from "./components/AssetLibraryPanel";
import { FinalMoviePlayer } from "./components/FinalMoviePlayer";
import type { ReviewDecision, RunInput } from "./types";

const STATUS_TEXT: Record<string, string> = {
  idle: "待机",
  running: "运行中",
  interrupted: "等待人工审核",
  done: "已完成",
  error: "出错",
};

export default function App() {
  const {
    threadId,
    status,
    state,
    interrupt,
    timeline,
    error,
    pendingNodes,
    actionMode,
    startRun,
    resumeRun,
    continueRun,
    rerunFrom,
    loadThread,
    stop,
  } = useAgentStream();

  const [refreshKey, setRefreshKey] = useState(0);
  const busy = status === "running";

  const handleStart = useCallback(
    async (input: RunInput) => {
      await startRun(input);
      setRefreshKey((k) => k + 1);
    },
    [startRun],
  );

  const handleDecision = useCallback(
    async (decision: ReviewDecision) => {
      await resumeRun(decision);
      setRefreshKey((k) => k + 1);
    },
    [resumeRun],
  );

  const handleContinue = useCallback(async () => {
    await continueRun();
    setRefreshKey((k) => k + 1);
  }, [continueRun]);

  return (
    <div className="app">
      <ThreadSidebar
        activeThreadId={threadId}
        onSelect={(tid) => void loadThread(tid)}
        refreshKey={refreshKey}
      />

      <main className="main">
        <header className="app-header">
          <h1>Multimedia Agent 控制台</h1>
          <div className="status-area">
            <span className={`status status-${status}`}>
              {STATUS_TEXT[status] ?? status}
            </span>
            {threadId && (
              <code className="thread-code">{threadId.slice(0, 8)}</code>
            )}
            {busy && (
              <button className="btn btn-ghost" onClick={stop}>
                中断
              </button>
            )}
          </div>
        </header>

        {error && <div className="error-banner">运行错误：{error}</div>}
        {state.error_log && (
          <div className="error-banner">后端日志：{state.error_log}</div>
        )}
        {state.aborted && (
          <div className="error-banner">
            任务已中止：{state.abort_reason ?? "未知原因"}
          </div>
        )}

        <div className="content-grid">
          <div className="left-col">
            <QuickGeneratePanel />
            <TaskLauncher busy={busy} onStart={(i) => void handleStart(i)} />

            {actionMode === "review" && (
              <ErrorBoundary label="审核卡片">
                <InterruptApprovalCard
                  payload={interrupt!}
                  busy={busy}
                  onSubmit={(d) => void handleDecision(d)}
                />
              </ErrorBoundary>
            )}

            {actionMode === "continue" && (
              <div className="resume-panel">
                <div className="resume-info">
                  <strong>该会话尚未跑完</strong>
                  <span>
                    下一步：{pendingNodes.join(" → ")}
                    {typeof state.current_scene_index === "number" &&
                    state.scenes?.length
                      ? ` · 进度 ${Math.min(
                          state.current_scene_index + 1,
                          state.scenes.length,
                        )}/${state.scenes.length} 镜头`
                      : ""}
                  </span>
                </div>
                <button
                  className="btn btn-primary"
                  disabled={busy}
                  onClick={() => void handleContinue()}
                >
                  继续执行
                </button>
              </div>
            )}

            <AssetLibraryPanel threadId={threadId} />

            <div className="product-zone">
              <ErrorBoundary label="产物区">
                <FinalMoviePlayer state={state} />
                <TimelineBar state={state} busy={busy} onRerun={(i) => void rerunFrom(i)} />
                <SceneGallery state={state} />
                <ReferenceSheetsPanel state={state} />
              </ErrorBoundary>
            </div>
          </div>

          <div className="right-col">
            <NodeProgressTimeline
              timeline={timeline}
              status={status}
              currentSceneIndex={state.current_scene_index}
              totalScenes={state.scenes?.length}
            />

            {(state.visual_style || state.content_type) && (
              <div className="meta-panel">
                <h3>
                  <span className="section-icon">📋</span>任务元信息
                </h3>
                {state.visual_style && (
                  <p>
                    <span>视觉风格</span>
                    {state.visual_style}
                  </p>
                )}
                {state.content_type && (
                  <p>
                    <span>内容类型</span>
                    {state.content_type}
                  </p>
                )}
                {typeof state.rewrite_count === "number" && (
                  <p>
                    <span>重写次数</span>
                    {state.rewrite_count}
                  </p>
                )}
                {state.global_setting && (
                  <p className="meta-long">
                    <span>全局设定</span>
                    {state.global_setting}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
