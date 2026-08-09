import type { RunStatus, TimelineEntry } from "../types";

/** 节点名 → 中文可读名（对齐 graph.py 节点注册） */
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
  prompt_preview: "生图前 Prompt 审核",
  generate_image: "生成关键帧",
  image_review: "关键帧审片",
  end_frame_director: "尾帧导演",
  end_frame_prompt_preview: "尾帧 Prompt 审核",
  generate_end_frame: "生成尾帧",
  end_frame_review: "尾帧审片",
  video_director: "视频 Prompt 生成",
  generate_video: "生成视频片段",
  video_review: "视频审片",
  advance_scene: "推进到下一镜头",
  stitcher: "拼接成片",
  audio_mixer: "音频混流",
  finalize: "收尾输出",
};

function label(node: string): string {
  return NODE_LABELS[node] ?? node;
}

interface Props {
  timeline: TimelineEntry[];
  status: RunStatus;
  currentSceneIndex?: number;
  totalScenes?: number;
}

export function NodeProgressTimeline({
  timeline,
  status,
  currentSceneIndex,
  totalScenes,
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
          return (
            <li
              key={entry.id}
              className={`timeline-item ${active ? "active" : "finished"}`}
            >
              <span className="dot" />
              <div className="timeline-body">
                <span className="node-name">{label(entry.node)}</span>
                <time>
                  {new Date(entry.at).toLocaleTimeString("zh-CN", {
                    hour12: false,
                  })}
                </time>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
