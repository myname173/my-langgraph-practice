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
import {
  AudioSubtitlePanel,
  mapSubtitlesToScenes,
} from "./components/AudioSubtitlePanel";
import { ReferenceSheetsPanel } from "./components/ReferenceSheetsPanel";
import { AssetLibraryPanel } from "./components/AssetLibraryPanel";
import { AssetManagerView } from "./components/AssetManagerView";
import { FinalMoviePlayer } from "./components/FinalMoviePlayer";
import { SplashScreen, shouldPlaySplash } from "./components/SplashScreen";
import { SettingsPanel } from "./components/SettingsPanel";
import { RunCostBar } from "./components/RunCostBar";
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
    autoMode,
    actionMode,
    startRun,
    resumeRun,
    continueRun,
    rerunFrom,
    autoQueue,
    planAutoApprove,
    cancelAutoApprove,
    loadThread,
    stop,
    reset,
  } = useAgentStream();

  const [refreshKey, setRefreshKey] = useState(0);
  // 当前任务的故事标识（来自 RunInput.story），用于素材库按故事隔离
  const [activeStory, setActiveStory] = useState<string | undefined>(undefined);
  // 第4项：参考图面板「去修正」回传的待高亮素材名，路由到创建面板高亮对应卡片
  const [highlightName, setHighlightName] = useState<string | undefined>(undefined);
  // 布局优化：左列「创建任务 / 成果」Tab。启动/加载历史后自动跳成果，去修正时跳回创建
  const [activeTab, setActiveTab] = useState<"create" | "result" | "assets">("create");
  // 生成引擎设置面板（API Key / 模型名 / 音色），挂在头部齿轮按钮
  const [showSettings, setShowSettings] = useState(false);
  // 开场动画：仅首次访问播放（localStorage 记忆），之后不再弹
  const [showSplash, setShowSplash] = useState<boolean>(() => shouldPlaySplash());
  // F-6：窄屏导航抽屉（≤900px 出现汉堡按钮，侧栏改为左侧抽屉）
  const [navOpen, setNavOpen] = useState(false);
  const busy = status === "running";

  // F-5：当前执行阶段（从时间轴最新节点推导），供镜头卡展示「生成中」细态
  const lastNode = timeline.length ? timeline[timeline.length - 1].node : "";
  const NODE_PHASE: Record<string, string> = {
    director: "分镜设计中", director_refine: "分镜设计中", prompt_preview: "分镜设计中",
    image_generator: "图像生成中", reviewer: "关键帧审核中", image_review: "关键帧审核中",
    end_frame_director: "尾帧设计中", end_frame_prompt_preview: "尾帧设计中",
    end_frame_generator: "尾帧生成中", end_frame_review: "尾帧审核中",
    videographer: "运镜设计中", video_generator: "视频生成中",
    video_reviewer: "视频审核中", video_review: "视频审核中",
  };
  const scenePhase = busy ? NODE_PHASE[lastNode] ?? "生成中" : "";

  // 字幕逐句 → 按 scenes[].video_duration 累积时间轴映射到所属镜头，供卡片/面板展示
  const subtitleByScene = mapSubtitlesToScenes(
    state.scenes ?? [],
    state.subtitle_entries,
  );
  // 是否已生成配音：音画合成成片存在，或 audio_status 标记为已配音/原生音轨
  const dubbed = !!(
    state.final_movie_with_audio ||
    state.audio_status === "dubbed" ||
    state.audio_status === "native"
  );

  const handleStart = useCallback(
    async (input: RunInput) => {
      setActiveStory(input.story);
      setActiveTab("result");
      await startRun(input);
      setRefreshKey((k) => k + 1);
    },
    [startRun],
  );

  /** 新建会话：重置状态并回到干净的创建 Tab。运行中有会话时二次确认，避免误触丢失进度 */
  const handleNewSession = useCallback(() => {
    if (busy || threadId) {
      const tip = busy
        ? "当前任务正在生成中，新建会话会中断并清空当前进度。确定继续？"
        : "新建会话会清空当前会话的进度与素材状态。确定继续？";
      if (!window.confirm(tip)) return;
    }
    reset();
    setActiveStory(undefined);
    setHighlightName(undefined);
    setActiveTab("create");
    setRefreshKey((k) => k + 1);
  }, [reset, busy, threadId]);

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
    <div className={`app${navOpen ? " nav-open" : ""}`}>
      {showSplash && (
        <SplashScreen onDone={() => setShowSplash(false)} />
      )}

      {/* F-6：抽屉打开时的背景遮罩（点击关闭） */}
      {navOpen && (
        <button
          className="nav-scrim"
          aria-label="关闭导航"
          onClick={() => setNavOpen(false)}
        />
      )}

      {showSettings && <SettingsPanel onClose={() => setShowSettings(false)} />}

      <ThreadSidebar
        activeThreadId={threadId}
        onSelect={(tid) => {
          void loadThread(tid);
          setActiveTab("result");
          setNavOpen(false);
        }}
        refreshKey={refreshKey}
        mobileOpen={navOpen}
        onClose={() => setNavOpen(false)}
      />

      <main className="main">
        <header className="app-header">
          {/* F-6：窄屏汉堡按钮，展开历史会话抽屉 */}
          <button
            className="nav-toggle"
            aria-label="打开导航"
            aria-expanded={navOpen}
            onClick={() => setNavOpen(true)}
          >
            ☰ 会话
          </button>
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
            <button
              className="btn btn-ghost"
              onClick={() => setShowSettings(true)}
              title="配置 API Key / 模型名 / 配音音色"
            >
              ⚙️ 设置
            </button>
          </div>
        </header>

        {/* F-2：顶栏常驻成本与额度仪表 */}
        <RunCostBar state={state} busy={busy} threadId={threadId} />

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
            {/* 快捷生成：与主流程解耦的独立小工具，常驻顶部，不占用创建/成果任一阶段 */}
            <QuickGeneratePanel />

            <div className="work-tabs">
              <button
                className={`work-tab ${activeTab === "create" ? "active" : ""}`}
                onClick={() => setActiveTab("create")}
              >
                <span className="wt-num">①</span> 创建任务
              </button>
              <button
                className={`work-tab ${activeTab === "result" ? "active" : ""}`}
                onClick={() => setActiveTab("result")}
              >
                <span className="wt-num">②</span> 成果
                {busy && <span className="wt-dot" title="生成中" />}
              </button>
              <button
                className={`work-tab ${activeTab === "assets" ? "active" : ""}`}
                onClick={() => setActiveTab("assets")}
                title="上传 / 管理一致性参考素材"
              >
                <span className="wt-num">🎨</span> 素材库
              </button>
              <button
                className="work-tab work-tab-new"
                onClick={handleNewSession}
                title="中断当前运行并清空，回到干净的创建态"
              >
                ＋ 新建会话
              </button>
            </div>

            {activeTab === "create" && (
              <>
                <TaskLauncher
                  busy={busy}
                  onStart={(i) => void handleStart(i)}
                  highlightName={highlightName}
                />
                <AssetLibraryPanel threadId={threadId} story={activeStory} />
              </>
            )}

            {activeTab === "assets" && (
              <AssetManagerView story={activeStory} />
            )}

            {activeTab === "result" && (
              <>
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

                <div className="product-zone">
                  <ErrorBoundary label="产物区">
                    <FinalMoviePlayer state={state} />
                    <AudioSubtitlePanel state={state} />
                    <TimelineBar state={state} busy={busy} onRerun={(i) => void rerunFrom(i)} dubbed={dubbed} />
                    <SceneGallery
                      state={state}
                      subtitleByScene={subtitleByScene}
                      interrupt={interrupt}
                      busy={busy}
                      runActive={busy || status === "interrupted"}
                      scenePhase={scenePhase}
                      onDecide={(d) => void handleDecision(d)}
                      onRerun={(i, opts) => void rerunFrom(i, opts)}
                      onPlanApprovals={(idxs, lock) => planAutoApprove(idxs, lock)}
                      autoQueue={autoQueue}
                      onCancelQueue={cancelAutoApprove}
                      threadId={threadId}
                      onRecipeApplied={() => void handleContinue()}
                    />
                    <ReferenceSheetsPanel
                      state={state}
                      onRequestFix={(name) => {
                        // 跳回创建 Tab 并高亮对应素材卡片，形成闭环
                        setActiveTab("create");
                        setHighlightName(name);
                        window.setTimeout(() => setHighlightName(undefined), 2500);
                      }}
                    />
                  </ErrorBoundary>
                </div>
              </>
            )}
          </div>

          <div className="right-col">
            <NodeProgressTimeline
              timeline={timeline}
              status={status}
              autoMode={autoMode}
              currentSceneIndex={state.current_scene_index}
              totalScenes={state.scenes?.length}
              audioState={{
                audio_track: state.audio_track,
                subtitle_path: state.subtitle_path,
                bgm_path: state.bgm_path,
                final_movie_with_audio: state.final_movie_with_audio,
                voiceover_duration: state.voiceover_duration,
                audio_status: state.audio_status,
              }}
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
