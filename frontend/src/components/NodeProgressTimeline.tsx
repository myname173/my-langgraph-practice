import type { RunStatus, TimelineEntry } from "../types";

/**
 * 节点名 → 中文可读名。
 * 严格对齐 graph.py 中 workflow.add_node 注册的 27 个节点名：
 * showrunner / showrunner_review / reference_gen / visual_context_builder /
 * shot_strategy_builder / sequence_orchestrator / film_studio / cinematic_critic /
 * director / director_refine / prompt_preview / image_generator / reviewer /
 * image_review / end_frame_director / end_frame_prompt_preview /
 * end_frame_generator / end_frame_review / videographer / video_generator /
 * video_reviewer / video_review / advance_scene / abort / stitcher /
 * draft_review / audio_mixer
 *
 * 注：旧表里的 generate_image / generate_end_frame / video_director / generate_video /
 * finalize 均非真实节点名，会导致界面回落显示英文节点 ID。
 */
const NODE_LABELS: Record<string, string> = {
  showrunner: "总导演拆解剧本",
  showrunner_review: "总导演人工审片",
  reference_gen: "生成一致性参考图",
  visual_context_builder: "构建视觉上下文 / RAG",
  shot_strategy_builder: "镜头拍摄策略",
  sequence_orchestrator: "序列编排",
  film_studio: "多智能体协作优化",
  cinematic_critic: "影视化自我批评",
  director: "导演生成图片 Prompt",
  director_refine: "导演 Prompt 精修",
  prompt_preview: "生图前 Prompt 审核",
  image_generator: "生成关键帧",
  reviewer: "关键帧一致性审核",
  image_review: "关键帧审片",
  end_frame_director: "尾帧导演",
  end_frame_prompt_preview: "尾帧 Prompt 审核",
  end_frame_generator: "生成尾帧",
  end_frame_review: "尾帧审片",
  videographer: "视频 Prompt 生成",
  video_generator: "生成视频片段",
  video_reviewer: "视频质量审核",
  video_review: "视频审片",
  advance_scene: "推进到下一镜头",
  abort: "中止任务",
  stitcher: "拼接成片",
  draft_review: "粗剪草稿审片",
  audio_mixer: "音频混流 / 字幕",
};

function label(node: string): string {
  return NODE_LABELS[node] ?? node;
}

/** 人工审核闸口节点：在自动模式下会被自动放行 */
const REVIEW_NODES = new Set([
  "showrunner_review",
  "prompt_preview",
  "image_review",
  "end_frame_prompt_preview",
  "end_frame_review",
  "video_review",
  "draft_review",
]);

interface Props {
  timeline: TimelineEntry[];
  status: RunStatus;
  currentSceneIndex?: number;
  totalScenes?: number;
  /** 低干预 / 自动模式：为真时审核类节点追加「自动」徽标 */
  autoMode?: boolean;
  /** 音频闭环相关状态：用于在 audio_mixer 节点下展示子步骤进度 */
  audioState?: {
    audio_track?: string | null;
    subtitle_path?: string | null;
    bgm_path?: string | null;
    final_movie_with_audio?: string | null;
    voiceover_duration?: number | null;
    audio_status?: string | null;
  };
}

/** audio_mixer 节点的子步骤（按完成度展示） */
function AudioSubSteps({ audio }: { audio: NonNullable<Props["audioState"]> }) {
  const steps: { key: string; label: string; done: boolean }[] = [
    {
      key: "tts",
      label: audio.voiceover_duration != null ? `配音 ${audio.voiceover_duration}s` : "配音生成",
      done: !!audio.audio_track || !!audio.voiceover_duration,
    },
    { key: "sub", label: "字幕生成", done: !!audio.subtitle_path },
    { key: "bgm", label: "BGM 配乐", done: !!audio.bgm_path },
    { key: "mix", label: "音画混音", done: !!audio.final_movie_with_audio },
  ];
  return (
    <div className="audio-substeps">
      {steps.map((s) => (
        <span
          key={s.key}
          className={`substep ${s.done ? "done" : "pending"}`}
          title={s.done ? "已完成" : "进行中"}
        >
          {s.done ? "✓" : "…"} {s.label}
        </span>
      ))}
    </div>
  );
}

export function NodeProgressTimeline({
  timeline,
  status,
  currentSceneIndex,
  totalScenes,
  autoMode,
  audioState,
}: Props) {
  return (
    <div className="timeline">
      <div className="timeline-head">
        <h3>
          <span className="section-icon">📈</span>执行进度
        </h3>
        {typeof currentSceneIndex === "number" && !!totalScenes && (
          <span className="scene-counter">
            镜头 {Math.min(currentSceneIndex + 1, totalScenes)} / {totalScenes}
          </span>
        )}
      </div>

      {!timeline.length && (
        <p className="empty-hint">尚未开始执行</p>
      )}

      <ol className="timeline-list">
        {timeline.map((entry, idx) => {
          const isLast = idx === timeline.length - 1;
          const active = isLast && status === "running";
          const showAudio =
            entry.node === "audio_mixer" &&
            !!audioState &&
            (active ||
              audioState.audio_track ||
              audioState.subtitle_path ||
              audioState.bgm_path ||
              audioState.final_movie_with_audio);
          return (
            <li
              key={entry.id}
              className={`timeline-item ${active ? "active" : "finished"}`}
            >
              <span className="dot" />
              <div className="timeline-body">
                <span className="node-name">{label(entry.node)}</span>
                {autoMode && REVIEW_NODES.has(entry.node) && (
                  <span className="auto-badge" title="自动模式：已自动放行">
                    自动
                  </span>
                )}
                <time>
                  {new Date(entry.at).toLocaleTimeString("zh-CN", {
                    hour12: false,
                  })}
                </time>
                {showAudio && <AudioSubSteps audio={audioState!} />}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
