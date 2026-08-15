import { useEffect, useMemo, useState } from "react";
import type {
  InterruptPayload,
  ReviewAction,
  ReviewDecision,
  SceneLike,
} from "../types";
import { MediaPreview } from "./MediaPreview";
import { MetricBadges } from "./MetricBadges";

interface Props {
  payload: InterruptPayload;
  busy: boolean;
  onSubmit: (decision: ReviewDecision) => void;
}

const ACTION_LABEL: Record<ReviewAction, string> = {
  approve: "通过",
  rewrite: "打回重写",
  edit_prompt: "改写后继续",
};

/** 该 stage 下用户可编辑的主 prompt 字段名 */
function editableField(
  stage: InterruptPayload["stage"],
): "image_prompt" | "video_prompt" | null {
  switch (stage) {
    case "prompt_preview":
    case "end_frame_prompt_preview":
    case "image_review":
    case "end_frame_review":
      return "image_prompt";
    case "video_review":
      return "video_prompt";
    default:
      return null;
  }
}

export function InterruptApprovalCard({ payload, busy, onSubmit }: Props) {
  const field = editableField(payload.stage);

  const initialPrompt = useMemo(() => {
    if (field === "image_prompt" && "image_prompt" in payload) {
      return payload.image_prompt ?? "";
    }
    if (field === "video_prompt" && "video_prompt" in payload) {
      return payload.video_prompt ?? "";
    }
    return "";
  }, [payload, field]);

  const [prompt, setPrompt] = useState(initialPrompt);
  const [reason, setReason] = useState("");
  const [task, setTask] = useState(
    "task" in payload ? payload.task : "",
  );
  const [globalSetting, setGlobalSetting] = useState(
    "global_setting" in payload ? (payload.global_setting ?? "") : "",
  );
  const [scenesText, setScenesText] = useState(() =>
    "scenes" in payload ? JSON.stringify(payload.scenes ?? [], null, 2) : "",
  );
  const [scenesError, setScenesError] = useState<string | null>(null);

  // 切换到新的 interrupt 时重置表单
  useEffect(() => {
    setPrompt(initialPrompt);
    setReason("");
    setScenesError(null);
    if ("task" in payload) setTask(payload.task);
    if ("global_setting" in payload)
      setGlobalSetting(payload.global_setting ?? "");
    if ("scenes" in payload)
      setScenesText(JSON.stringify(payload.scenes ?? [], null, 2));
  }, [payload, initialPrompt]);

  const promptDirty = prompt.trim() !== initialPrompt.trim();

  function buildDecision(action: ReviewAction): ReviewDecision | null {
    const decision: ReviewDecision = { action };

    if (action === "rewrite" && reason.trim()) {
      decision.reason = reason.trim();
    }

    if (payload.stage === "showrunner_review") {
      if (task.trim() && task !== payload.task) decision.task = task.trim();
      if (globalSetting.trim() && globalSetting !== payload.global_setting) {
        decision.global_setting = globalSetting.trim();
      }
      const original = JSON.stringify(payload.scenes ?? [], null, 2);
      if (scenesText.trim() && scenesText !== original) {
        try {
          const parsed = JSON.parse(scenesText) as SceneLike[];
          if (!Array.isArray(parsed)) throw new Error("scenes 必须是数组");
          decision.scenes = parsed;
        } catch (e) {
          setScenesError(e instanceof Error ? e.message : String(e));
          return null;
        }
      }
      setScenesError(null);
      return decision;
    }

    // edit_prompt：无论是否改过文本框，都回传当前 prompt 以严格表达"改写后继续"
    if (field && action === "edit_prompt") {
      const trimmed = prompt.trim();
      if (trimmed) decision[field] = trimmed;
    }

    return decision;
  }

  function handle(action: ReviewAction) {
    const decision = buildDecision(action);
    if (decision) onSubmit(decision);
  }

  const availableActions = payload.actions?.length
    ? payload.actions
    : (["approve", "rewrite"] as ReviewAction[]);

  return (
    <section className="approval-card">
      <header className="approval-header">
        <span className="stage-tag">{payload.stage}</span>
        <h2>{payload.title}</h2>
      </header>

      <p className="approval-message">{payload.message}</p>

      {"script" in payload && payload.script && (
        <div className="field-block">
          <label>剧本</label>
          <p className="readonly-text">{payload.script}</p>
        </div>
      )}

      {"motion_hint" in payload && payload.motion_hint && (
        <p
          className={
            payload.motion_hint.startsWith("⚠️") ? "hint hint-warn" : "hint"
          }
        >
          {payload.motion_hint}
        </p>
      )}

      {"auto_feedback" in payload && payload.auto_feedback && (
        <div className="field-block">
          <label>AI 自动评审意见</label>
          <p className="readonly-text">{payload.auto_feedback}</p>
        </div>
      )}

      <MetricBadges
        metrics={[
          {
            label: "整体相似度",
            value: "embedding_similarity" in payload
              ? payload.embedding_similarity
              : undefined,
            warnBelow: 0.7,
          },
          {
            label: "人物相似度",
            value: "portrait_similarity" in payload
              ? payload.portrait_similarity
              : undefined,
            warnBelow: 0.7,
          },
          {
            label: "连贯性",
            value: "continuity_score" in payload
              ? payload.continuity_score
              : undefined,
            warnBelow: 0.6,
          },
        ]}
      />

      <div className="media-row">
        {"image_url" in payload && (
          <MediaPreview url={payload.image_url} label="当前产物" />
        )}
        {"video_url" in payload && (
          <MediaPreview url={payload.video_url} label="生成视频" forceVideo />
        )}
        {"reference_image_url" in payload && (
          <MediaPreview url={payload.reference_image_url} label="参考图" />
        )}
      </div>

      {payload.stage === "showrunner_review" ? (
        <>
          <div className="field-block">
            <label>任务描述 (task)</label>
            <textarea
              value={task}
              rows={2}
              onChange={(e) => setTask(e.target.value)}
            />
          </div>
          <div className="field-block">
            <label>全局设定 (global_setting)</label>
            <textarea
              value={globalSetting}
              rows={4}
              onChange={(e) => setGlobalSetting(e.target.value)}
            />
          </div>
          <div className="field-block">
            <label>分镜列表 (scenes, JSON)</label>
            <textarea
              className="mono"
              value={scenesText}
              rows={12}
              onChange={(e) => setScenesText(e.target.value)}
            />
            {scenesError && <p className="error-text">JSON 解析失败: {scenesError}</p>}
          </div>
        </>
      ) : (
        field && (
          <div className="field-block">
            <label>
              {field === "video_prompt" ? "视频 Prompt" : "图片 Prompt"}
              {promptDirty && <span className="dirty-tag">已修改</span>}
            </label>
            <textarea
              className="mono"
              value={prompt}
              rows={8}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </div>
        )
      )}

      <div className="field-block">
        <label>打回理由（选填，将作为重写依据）</label>
        <textarea
          value={reason}
          rows={2}
          placeholder="例如：主体动作不明显，需要更强的镜头运动"
          onChange={(e) => setReason(e.target.value)}
        />
      </div>

      <div className="action-row">
        {availableActions.map((action) => (
          <button
            key={action}
            disabled={busy}
            className={`btn btn-${action}`}
            onClick={() => handle(action)}
          >
            {busy ? "提交中…" : ACTION_LABEL[action] ?? action}
          </button>
        ))}
      </div>
    </section>
  );
}
