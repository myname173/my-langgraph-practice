import { useState } from "react";
import type { MultimediaState, SceneLike } from "../types";
import { resolveMediaUrl, isVideoUrl } from "../lib/langgraphClient";

interface TimelineBarProps {
  state: MultimediaState;
  busy: boolean;
  onRerun: (sceneIndex: number) => void;
  /** 是否已生成配音（音画合成成片存在即视为已配音） */
  dubbed?: boolean;
}

type SceneStatus = "failed" | "dubbed" | "done" | "keyframe" | "pending";

function sceneStatus(
  scene: SceneLike | undefined,
  dubbed: boolean,
): SceneStatus {
  if (!scene) return "pending";
  // 【修复】生成失败的镜头（后端写占位黑场并置 shot_failed）必须显式标识，
  // 否则用户会误以为该镜头已成功入片——实际 stitcher 已跳过它。
  if (scene.shot_failed) return "failed";
  if (dubbed) return "dubbed";
  if (scene.final_video_url || scene.raw_video_url) return "done";
  if (scene.image_url || scene.last_image_url) return "keyframe";
  return "pending";
}

const STATUS_LABEL: Record<SceneStatus, string> = {
  failed: "生成失败",
  dubbed: "已配音",
  done: "已生片",
  keyframe: "已出图",
  pending: "未生成",
};

/**
 * 镜头时间轴：把所有镜头横向铺开，用户可点任意镜头上的“重跑此镜头”
 * 按钮，让图从该镜头重头制作（后端会重置 checkpoint 并续跑）。
 */
export function TimelineBar({ state, busy, onRerun, dubbed }: TimelineBarProps) {
  const scenes = state.scenes ?? [];
  const [rerunIdx, setRerunIdx] = useState<number | null>(null);

  if (scenes.length === 0) return null;

  const currentIdx = state.current_scene_index;

  const handleRerun = async (idx: number) => {
    if (busy) return;
    setRerunIdx(idx);
    try {
      await onRerun(idx);
    } finally {
      setRerunIdx(null);
    }
  };

  return (
    <div className="timeline-bar">
      <div className="timeline-head">
        <h3>
          <span className="section-icon">🎞️</span>镜头时间轴
        </h3>
        <span className="timeline-hint">
          点任意镜头可“从该镜头重跑”（清空其后产物并续跑）
        </span>
      </div>
      <div className="timeline-track">
        {scenes.map((scene, idx) => {
          const st = sceneStatus(scene, !!dubbed);
          const isCurrent =
            typeof currentIdx === "number" && currentIdx === idx;
          // 【修复】缩略图必须优先用静态图：视频 URL 用 <img> 解码会 100% 裂图。
          // 仅当该镜头没有任何关键帧图时，才退而用视频做静音预览。
          const still =
            scene.last_image_url || scene.image_url || "";
          const clip =
            scene.final_video_url || scene.raw_video_url || "";
          const thumb = still || clip;
          const thumbIsVideo = !still && isVideoUrl(clip);
          return (
            <div
              key={idx}
              className={[
                "timeline-node",
                `node-${st}`,
                isCurrent ? "node-current" : "",
              ].join(" ")}
            >
              <div className="tn-index">{idx + 1}</div>
              <div className="tn-thumb">
                {thumb ? (
                  thumbIsVideo ? (
                    // 只有视频（无关键帧图）时用静音 video 预载首帧，避免裂图
                    <video
                      src={resolveMediaUrl(thumb)}
                      muted
                      playsInline
                      preload="metadata"
                    />
                  ) : (
                    // 后端产物常为本地绝对路径，必须归一为 /media/ URL 否则必然裂图
                    <img src={resolveMediaUrl(thumb)} alt={`镜头 ${idx + 1}`} loading="lazy" />
                  )
                ) : (
                  <div className="tn-thumb-empty">—</div>
                )}
                {st === "failed" && (
                  <div className="tn-thumb-failed" title="该镜头生成失败，未拼入成片">
                    ⚠
                  </div>
                )}
              </div>
              <div className="tn-status">{STATUS_LABEL[st]}</div>
              <button
                className="tn-rerun"
                disabled={busy || rerunIdx === idx}
                onClick={() => void handleRerun(idx)}
                title={`从镜头 ${idx + 1} 重新跑`}
              >
                {rerunIdx === idx ? "重跑中…" : "↺ 重跑此镜头"}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
