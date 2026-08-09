import { useCallback, useState } from "react";
import { useAgentStream } from "./hooks/useAgentStream";
import { TaskLauncher } from "./components/TaskLauncher";
import { ThreadSidebar } from "./components/ThreadSidebar";
import { NodeProgressTimeline } from "./components/NodeProgressTimeline";
import { InterruptApprovalCard } from "./components/InterruptApprovalCard";
import { SceneGallery } from "./components/SceneGallery";
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
    startRun,
    resumeRun,
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
    },
    [resumeRun],
  );

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
            <TaskLauncher busy={busy} onStart={(i) => void handleStart(i)} />

            {interrupt && (
              <InterruptApprovalCard
                payload={interrupt}
                busy={busy}
                onSubmit={(d) => void handleDecision(d)}
              />
            )}

            <FinalMoviePlayer state={state} />
            <SceneGallery state={state} />
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
