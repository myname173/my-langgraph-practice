import { useEffect, useMemo, useState } from "react";
import type {
  InterruptPayload,
  ReviewAction,
  ReviewDecision,
  SceneLike,
} from "../types";
import { FIX_TYPE_META, type FixType } from "../lib/fixTypes";
import { MediaPreview } from "./MediaPreview";
import { MetricBadges } from "./MetricBadges";
import { resolveMediaUrl } from "../lib/langgraphClient";

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

  // F-4：外科手术修正方式（rewrite 时随决策回传 fix_type）
  const [selectedFix, setSelectedFix] = useState<FixType | null>(null);
  // F-1：候选关键帧挑选（approve 时随决策回传 pick_candidate）
  const [picked, setPicked] = useState<number | null>(null);
  // F-4：通过后锁定（video_review，随决策回传 lock）
  const [lockAfter, setLockAfter] = useState(false);

  // 切换到新的 interrupt 时重置表单
  useEffect(() => {
    setPrompt(initialPrompt);
    setReason("");
    setScenesError(null);
    setSelectedFix(null);
    setPicked(null);
    setLockAfter(false);
    if ("task" in payload) setTask(payload.task);
    if ("global_setting" in payload)
      setGlobalSetting(payload.global_setting ?? "");
    if ("scenes" in payload)
      setScenesText(JSON.stringify(payload.scenes ?? [], null, 2));
  }, [payload, initialPrompt]);

  const promptDirty = prompt.trim() !== initialPrompt.trim();

  /** F-1：候选关键帧（仅 image_review 且历史候选 ≥2 版时展示 A/B 挑选） */
  const candidates =
    payload.stage === "image_review" && "candidates" in payload
      ? (payload.candidates ?? []).filter(Boolean)
      : [];
  const showCandidates = candidates.length >= 2;
  const currentImage =
    payload.stage === "image_review" && "image_url" in payload
      ? (payload.image_url ?? "")
      : "";

  /** F-4：外科手术修正类型列表（image_review / video_review 由后端下发） */
  const fixTypes =
    (payload.stage === "image_review" || payload.stage === "video_review") &&
    "fix_types" in payload
      ? (payload.fix_types ?? [])
      : [];

  function buildDecision(action: ReviewAction): ReviewDecision | null {
    const decision: ReviewDecision = { action };

    if (action === "rewrite" && reason.trim()) {
      decision.reason = reason.trim();
    }
    // F-4：打回重写时携带修正粒度（后端 _normalize_decision 透传给审核闸口）
    if (action === "rewrite" && selectedFix) {
      decision.fix_type = selectedFix;
    }
    // F-1：通过时携带选定的候选关键帧（后端会先落位再走 approve 基准登记）
    if (action === "approve" && payload.stage === "image_review" && picked != null) {
      decision.pick_candidate = picked;
    }
    // F-4：通过后锁定（后端 video_review 闸口读取 decision.lock 写 locked_scenes）
    if (action === "approve" && payload.stage === "video_review" && lockAfter) {
      decision.lock = true;
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

      {/* P1-2：成片 Critic 报告（draft_review 阶段） */}
      {payload.stage === "draft_review" && "critic_report" in payload && payload.critic_report?.available && (
        <div className="field-block critic-report">
          <label>
            成片 Critic（机器审片参考）
            {typeof payload.critic_report.score === "number" && (
              <span className="dirty-tag">评分 {payload.critic_report.score.toFixed(2)}</span>
            )}
          </label>
          {(payload.critic_report.issues?.length ?? 0) > 0 ? (
            <ul className="critic-issues">
              {payload.critic_report.issues.map((it, k) => (
                <li key={k} className={`critic-issue sev-${it.severity}`}>
                  <span className="critic-sev">{it.severity}</span>
                  <span className="critic-type">{it.type}</span>
                  <span className="critic-desc">{it.desc}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="readonly-text">未发现明显问题（机器抽帧复审）</p>
          )}
          <p className="hint">报告来自视觉模型对粗剪抽帧的复审，供「通过 / 打回重拼」决策参考。</p>
        </div>
      )}

      {/* F-1：关键帧候选 A/B 挑选（重绘过至少一次后出现） */}
      {showCandidates && (
        <div className="field-block">
          <label>
            关键帧候选（{candidates.length} 版）
            {picked != null && (
              <span className="dirty-tag">已选候选 #{picked + 1}</span>
            )}
          </label>
          <div className="candidate-strip">
            {candidates.map((u, i) => (
              <button
                key={`${i}-${u.slice(-12)}`}
                type="button"
                disabled={busy}
                className={[
                  "candidate-item",
                  u === currentImage ? "current" : "",
                  picked === i ? "selected" : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                onClick={() => setPicked(picked === i ? null : i)}
                title={`候选 #${i + 1}${u === currentImage ? "（当前版本）" : ""}`}
              >
                <img src={resolveMediaUrl(u)} alt={`候选 ${i + 1}`} loading="lazy" />
                <span>
                  #{i + 1}
                  {u === currentImage ? " 当前" : ""}
                </span>
              </button>
            ))}
          </div>
          <p className="hint">
            点选候选后按「通过」即采用该版关键帧继续；也可以选「重绘关键帧」再生成新一版候选来对比。
          </p>
        </div>
      )}

      {/* F-4：外科手术修正方式（打回重写时生效） */}
      {fixTypes.length > 0 && (
        <div className="field-block">
          <label>
            修正方式（「打回重写」时生效
            {payload.stage === "image_review" ? "，默认重绘关键帧" : "，默认保首帧重滚"}）
          </label>
          <div className="fix-row">
            {fixTypes.map((ft) => (
              <button
                key={ft}
                type="button"
                disabled={busy}
                className={`fix-btn ${selectedFix === ft ? "selected" : ""}`}
                title={FIX_TYPE_META[ft]?.desc ?? ft}
                onClick={() => setSelectedFix(selectedFix === ft ? null : ft)}
              >
                <span className="fix-btn-label">
                  {FIX_TYPE_META[ft]?.label ?? ft}
                </span>
                <span className="fix-btn-desc">{FIX_TYPE_META[ft]?.desc}</span>
              </button>
            ))}
          </div>
          {selectedFix && (
            <p className="hint">
              将以「{FIX_TYPE_META[selectedFix].label}」打回重写：
              {FIX_TYPE_META[selectedFix].desc}
            </p>
          )}
        </div>
      )}

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

      {/* F-4：通过后锁定（仅 video_review，后端写 locked_scenes） */}
      {payload.stage === "video_review" && (
        <label className="lock-check">
          <input
            type="checkbox"
            checked={lockAfter}
            onChange={(e) => setLockAfter(e.target.checked)}
          />
          通过后锁定此镜头（后续重跑/续跑将复用该视频，不再重新生成）
        </label>
      )}

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
