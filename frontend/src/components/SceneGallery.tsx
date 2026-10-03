import { useEffect, useRef, useState } from "react";
import type {
  MultimediaState,
  SceneLike,
  SubtitleEntry,
  InterruptPayload,
  ReviewDecision,
} from "../types";
import { FIX_TYPE_META, type FixType, type SceneCardState } from "../lib/fixTypes";
import { RecipeCard } from "./RecipeCard";
import { fetchRecipe } from "../lib/recipeClient";
import type { TaskRecipe } from "../types";
import { MediaPreview } from "./MediaPreview";
import { useImageDim } from "../hooks/useImageDims";
import { ArtifactLightbox, type LightboxItem } from "./ArtifactLightbox";

interface Props {
  state: MultimediaState;
  /** 每镜的字幕条目（由 App 层映射好传入），用于卡片内展示本镜字幕时间轴 */
  subtitleByScene?: Map<number, SubtitleEntry[]>;
  /** F-4：当前人审中断（若有）——命中本镜时卡片上直接出「通过 / 四类修正」快捷按钮 */
  interrupt?: InterruptPayload | null;
  busy?: boolean;
  /** 任务是否处于「可批量预通过」的活跃期（运行中 / 等待审核） */
  runActive?: boolean;
  /** F-5：当前执行阶段文案（如图像生成中/视频生成中），展示在活动镜头卡上 */
  scenePhase?: string;
  /** 快捷审核决策（本镜被中断时，卡片按钮直接提交 approve / rewrite+fix_type） */
  onDecide?: (decision: ReviewDecision) => void;
  /** 从此镜重跑（failed / needs_fix 卡）；opts.unlock 为真时顺带移出 locked_scenes */
  onRerun?: (sceneIndex: number, opts?: { unlock?: boolean }) => void;
  /** F-1：批量预通过——把选中镜头加入自动审核队列 */
  onPlanApprovals?: (sceneIndices: number[], lockAfterApprove: boolean) => void;
  /** F-1：当前自动通过队列（0-based 场景下标，剩余未消费部分） */
  autoQueue?: number[];
  onCancelQueue?: () => void;
  /** P2-4：当前 thread（用于拉取镜头配方） */
  threadId?: string | null;
  /** P2-4：微调应用后，父组件据此发起续跑 */
  onRecipeApplied?: (sceneIndex: number) => void;
}

type LightboxState =
  | { kind: "single"; item: LightboxItem }
  | { kind: "pair"; left: LightboxItem; right: LightboxItem; title: string }
  | null;

function fmtScore(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}

function fmtTime(sec: number | null | undefined): string {
  if (sec == null || Number.isNaN(sec as number)) return "—";
  const m = Math.floor(sec / 60);
  const s = (sec as number) - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}

/** F-4：镜头卡四态推导（failed > locked > needs_fix > draft） */
function deriveCardState(
  scene: SceneLike,
  lockedFlag: boolean,
): SceneCardState {
  if (scene.shot_failed) return "failed";
  const hasVideo = !!(scene.final_video_url || scene.raw_video_url);
  if (lockedFlag && hasVideo) return "locked";
  const imageBad =
    !!scene.critique && scene.critique !== "无" && !scene.is_perfect;
  const videoBad =
    !!scene.video_critique &&
    scene.video_critique !== "无" &&
    !scene.video_is_perfect;
  if (scene.needs_realign === true || imageBad || videoBad) return "needs_fix";
  return "draft";
}

const CARD_STATE_META: Record<
  SceneCardState,
  { label: string; cls: string; title: string } | null
> = {
  failed: {
    label: "生成失败",
    cls: "badge-failed",
    title: "该镜头生成失败，未拼入成片；可一键从此镜重跑",
  },
  locked: {
    label: "已锁定",
    cls: "badge-locked",
    title: "已锁定：重跑/续跑将复用该镜头视频，不再重新生成",
  },
  needs_fix: {
    label: "待修正",
    cls: "badge-fix",
    title: "审核提出了修改意见，可通过重跑或等待审核闸口修正",
  },
  draft: null,
};

interface QuickFix {
  fixTypes: FixType[];
  busy: boolean;
  onApprove: () => void;
  onFix: (ft: FixType) => void;
}

function SceneCard({
  scene,
  idx,
  active,
  onZoom,
  onCompare,
  subtitles,
  cardState,
  phaseLabel,
  quickFix,
  onRerunCard,
  onUnlockRerun,
  rerunning,
  selectMode,
  selected,
  onToggleSelect,
  queued,
  onShowRecipe,
}: {
  scene: SceneLike;
  idx: number;
  active: boolean;
  onZoom: (item: LightboxItem) => void;
  onCompare: (left: LightboxItem, right: LightboxItem, title: string) => void;
  subtitles?: SubtitleEntry[];
  phaseLabel?: string;
  cardState: SceneCardState;
  quickFix: QuickFix | null;
  onRerunCard: () => void;
  onUnlockRerun: () => void;
  rerunning: boolean;
  selectMode: boolean;
  selected: boolean;
  onToggleSelect: () => void;
  queued: boolean;
  onShowRecipe: () => void;
}) {
  const [expanded, setExpanded] = useState(active);
  const cardRef = useRef<HTMLElement | null>(null);

  // 镜头变为当前执行镜头时自动展开 + 滚入视口
  useEffect(() => {
    if (active) {
      setExpanded(true);
      cardRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [active]);

  const finished = !!(scene.video_is_perfect && scene.final_video_url);
  const stateMeta = CARD_STATE_META[cardState];

  // 识图：读取素材真实尺寸，决定每块的比例类别
  const headDim = useImageDim(scene.image_url);
  const tailDim = useImageDim(scene.last_image_url);
  const emb = scene.embedding_similarity;
  const portrait = scene.portrait_similarity;
  const cont = scene.continuity_score;
  const hasScores = emb != null || portrait != null || cont != null;

  return (
    <article
      ref={cardRef as unknown as React.RefObject<HTMLElement>}
      className={[
        "scene-card",
        active ? "scene-active" : "",
        finished ? "scene-finished" : "",
        `scene-state-${cardState}`,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <header className="scene-card-head">
        {selectMode && (
          <input
            type="checkbox"
            className="scene-select"
            checked={selected}
            onChange={onToggleSelect}
            title={`选择镜头 ${idx + 1} 加入批量审核`}
            aria-label={`选择镜头 ${idx + 1}`}
          />
        )}
        <div className="scene-title">
          <strong>镜头 {idx + 1}</strong>
          {active && <span className="badge badge-active">执行中</span>}
          {active && phaseLabel && <span className="badge badge-phase">{phaseLabel}</span>}
          {stateMeta && (
            <span
              className={`badge ${stateMeta.cls}`}
              title={stateMeta.title}
            >
              {stateMeta.label}
            </span>
          )}
          {scene.needs_realign === true && (
            <span className="badge badge-drift" title="P1-4 视觉模型判断角色身份出现漂移">
              角色漂移
            </span>
          )}
          {queued && (
            <span className="badge badge-queued" title="该镜头在自动通过队列中，命中审核闸口时将自动放行">
              队列中
            </span>
          )}
          {scene.video_is_perfect && <span className="badge badge-perfect">视频已定稿</span>}
          {scene.is_perfect && !scene.video_is_perfect && <span className="badge badge-perfect">关键帧已定稿</span>}
          {typeof scene.video_duration === "number" && (
            <span className="badge badge-duration" title="该镜头视频真实时长">
              ⏱ {fmtTime(scene.video_duration)}
            </span>
          )}
        </div>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
        >
          {expanded ? "收起详情 ▴" : "展开详情 ▾"}
        </button>
      </header>

      {scene.script && <p className="scene-script">{scene.script}</p>}

      <div
        className={`media-row ratio-grid-${headDim?.kind ?? "landscape"}-${tailDim?.kind ?? "square"}`}
      >
        <MediaPreview
          url={scene.image_url}
          label="首帧"
          ratioKind={headDim?.kind}
          badge={fmtScore(emb)}
          onZoom={() =>
            onZoom({
              url: String(scene.image_url ?? ""),
              label: `镜头 ${idx + 1} · 首帧`,
              caption: `与上一镜头相似度: ${fmtScore(emb)}`,
            })
          }
        />
        <MediaPreview
          url={scene.last_image_url}
          label="尾帧"
          ratioKind={tailDim?.kind}
          onZoom={() =>
            onZoom({
              url: String(scene.last_image_url ?? ""),
              label: `镜头 ${idx + 1} · 尾帧`,
            })
          }
        />
        <MediaPreview
          url={scene.final_video_url || scene.raw_video_url}
          label="片段"
          forceVideo
        />
      </div>

      {/* F-4：审核闸口正停在本镜 —— 卡片上直接给「通过 / 四类修正」快捷决策 */}
      {quickFix && (
        <div className="quick-fix-block">
          <div className="quick-fix-label">该镜头正在等待你的审核：</div>
          <div className="quick-fix-row">
            <button
              type="button"
              className="btn btn-approve btn-sm"
              disabled={quickFix.busy}
              onClick={quickFix.onApprove}
            >
              ✓ 通过
            </button>
            {quickFix.fixTypes.map((ft) => (
              <button
                key={ft}
                type="button"
                className="btn btn-rewrite btn-sm"
                disabled={quickFix.busy}
                title={FIX_TYPE_META[ft]?.desc ?? ft}
                onClick={() => quickFix.onFix(ft)}
              >
                {FIX_TYPE_META[ft]?.label ?? ft}
              </button>
            ))}
          </div>
          <p className="hint">
            「打回重写」默认会在此镜的审核闸口按所选修正类型执行（改词最省额度，重绘最贵）。
          </p>
        </div>
      )}

      {/* F-4：镜头卡状态操作行 */}
      {(cardState === "failed" || cardState === "locked" || cardState === "needs_fix") && !quickFix && (
        <div className="scene-row-actions">
          {cardState === "failed" && (
            <button
              type="button"
              className="btn btn-rewrite btn-sm"
              disabled={rerunning}
              onClick={onRerunCard}
              title="清空该镜头及其后的产物，并从此镜重头制作"
            >
              {rerunning ? "重跑中…" : "↺ 从此镜重跑"}
            </button>
          )}
          {cardState === "locked" && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              disabled={rerunning}
              onClick={onUnlockRerun}
              title="先移出锁定，再清空该镜头及其后的产物并重头制作"
            >
              {rerunning ? "重跑中…" : "🔓 解锁并重跑"}
            </button>
          )}
          {cardState === "needs_fix" && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              disabled={rerunning}
              onClick={onRerunCard}
              title="重跑此镜头（到审核闸口时可再选修正方式）"
            >
              {rerunning ? "重跑中…" : "↺ 重跑此镜头"}
            </button>
          )}
        </div>
      )}

      {(scene.image_url || scene.last_image_url || scene.final_video_url || scene.raw_video_url) && (
        <div className="scene-row-actions">
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={onShowRecipe}
            title="P2-4：查看该镜头的完整生成配方（可复制 / 另存 / 微调后重跑）"
          >
            🧾 查看配方
          </button>
          {scene.image_url && scene.last_image_url && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() =>
                onCompare(
                  { url: String(scene.image_url ?? ""), label: "首帧" },
                  { url: String(scene.last_image_url ?? ""), label: "尾帧" },
                  `镜头 ${idx + 1} · 首帧 ↔ 尾帧 对比`,
                )
              }
            >
              🔍 首帧 vs 尾帧 对比
            </button>
          )}
        </div>
      )}

      {expanded && (
        <div className="scene-detail">
          {scene.image_prompt && (
            <DetailBlock title="生图 Prompt（image_prompt）" body={scene.image_prompt} />
          )}
          {scene.last_image_prompt && scene.last_image_prompt !== scene.image_prompt && (
            <DetailBlock title="尾帧 Prompt（last_image_prompt）" body={scene.last_image_prompt} />
          )}
          {scene.video_prompt && (
            <DetailBlock title="生视频 Prompt（video_prompt）" body={scene.video_prompt} />
          )}
          {scene.action_beat && (
            <DetailBlock title="动作节拍（action_beat）" body={scene.action_beat} />
          )}

          {subtitles && subtitles.length > 0 && (
            <div className="scene-subtitle-block">
              <DetailBlock
                title="本镜字幕时间轴"
                body={subtitles
                  .map((e) => `${fmtTime(e.start)} – ${fmtTime(e.end)}  ${e.text}`)
                  .join("\n")}
              />
            </div>
          )}

          {hasScores && (
            <div className="score-grid">
              <ScoreChip label="与上一镜头相似度" value={fmtScore(emb)} />
              <ScoreChip label="角色肖像相似度" value={fmtScore(portrait)} />
              <ScoreChip label="连贯性分数" value={fmtScore(cont)} />
            </div>
          )}

          {(scene.critique || scene.video_critique) && (
            <div className="critique-block">
              {scene.critique && (
                <DetailBlock title="关键帧评审（critique）" body={scene.critique} />
              )}
              {scene.video_critique && scene.video_critique !== scene.critique && (
                <DetailBlock title="视频评审（video_critique）" body={scene.video_critique} />
              )}
            </div>
          )}

          {scene.auto_feedback != null && scene.auto_feedback !== scene.critique && (
            <DetailBlock title="自动反馈（auto_feedback）" body={String(scene.auto_feedback)} />
          )}
        </div>
      )}
    </article>
  );
}

function DetailBlock({ title, body }: { title: string; body: string }) {
  return (
    <div className="detail-block">
      <div className="detail-title">{title}</div>
      <pre className="detail-body">{body}</pre>
    </div>
  );
}

function ScoreChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="score-chip">
      <span className="score-label">{label}</span>
      <span className="score-value">{value}</span>
    </div>
  );
}

const DEFAULT_FIX_TYPES: FixType[] = [
  "prompt_only",
  "keep_first_frame",
  "reimage",
  "extend",
];

export function SceneGallery({
  state,
  subtitleByScene,
  interrupt,
  busy,
  runActive,
  scenePhase,
  onDecide,
  onRerun,
  onPlanApprovals,
  autoQueue,
  onCancelQueue,
  threadId,
  onRecipeApplied,
}: Props) {
  const scenes = state.scenes ?? [];
  const activeIdx = state.current_scene_index;
  const [lightbox, setLightbox] = useState<LightboxState>(null);
  const [selection, setSelection] = useState<Set<number>>(new Set());
  const [rerunIdx, setRerunIdx] = useState<number | null>(null);

  // P2-4：镜头配方卡面板状态
  const [recipe, setRecipe] = useState<TaskRecipe | null>(null);
  const [recipeMd, setRecipeMd] = useState("");
  const [recipeShotIdx, setRecipeShotIdx] = useState(0);
  const [recipeBusy, setRecipeBusy] = useState(false);
  const [recipeErr, setRecipeErr] = useState<string | null>(null);

  const openRecipe = async (idx: number) => {
    if (!threadId) return;
    setRecipeBusy(true);
    setRecipeErr(null);
    try {
      const r = await fetchRecipe(threadId);
      setRecipe(r.recipe);
      setRecipeMd(r.markdown);
      setRecipeShotIdx(idx);
    } catch (e) {
      setRecipeErr(e instanceof Error ? e.message : String(e));
    } finally {
      setRecipeBusy(false);
    }
  };

  // F-4：仅当当前中断是关键帧/视频审核闸口时，卡片才出现快捷决策按钮
  const gate =
    interrupt && (interrupt.stage === "image_review" || interrupt.stage === "video_review")
      ? interrupt
      : null;
  const gateIdx =
    gate && typeof gate.scene_index === "number" ? gate.scene_index - 1 : -1;
  const gateFixTypes: FixType[] =
    gate && "fix_types" in gate && Array.isArray(gate.fix_types) && gate.fix_types.length
      ? gate.fix_types
      : DEFAULT_FIX_TYPES;

  const selectMode = !!runActive && scenes.length > 0;
  const lockedSet = new Set(state.locked_scenes ?? []);
  const queueSet = new Set(autoQueue ?? []);

  if (!scenes.length) return null;

  const toggleSelect = (idx: number) => {
    setSelection((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const planBatch = (lock: boolean) => {
    if (!onPlanApprovals || selection.size === 0) return;
    onPlanApprovals([...selection].sort((a, b) => a - b), lock);
    setSelection(new Set());
  };

  const handleRerun = async (idx: number, opts?: { unlock?: boolean }) => {
    if (!onRerun || busy || rerunIdx !== null) return;
    setRerunIdx(idx);
    try {
      await onRerun(idx, opts);
    } finally {
      setRerunIdx(null);
    }
  };

  return (
    <section className="scene-gallery">
      <h3>
        <span className="section-icon">🎞️</span>分镜产物
        <span className="scene-count">
          {scenes.length} 个镜头{typeof activeIdx === "number" ? ` · 当前第 ${activeIdx + 1} 个` : ""}
        </span>
        {threadId && (
          <button
            type="button"
            className="btn btn-ghost btn-sm recipe-open-task"
            disabled={recipeBusy}
            onClick={() => void openRecipe(typeof activeIdx === "number" ? activeIdx : 0)}
            title="P2-4：查看整个任务的镜头配方卡（可复制 / 另存 / 微调）"
          >
            {recipeBusy ? "读取配方中…" : "🧾 任务配方"}
          </button>
        )}
        {!!autoQueue?.length && (
          <span className="queue-chip" title="命中这些镜头的审核闸口时会自动通过，无需逐个点确认">
            自动通过队列：{autoQueue.map((i) => i + 1).join("、")}
            {onCancelQueue && (
              <button type="button" onClick={onCancelQueue}>
                取消
              </button>
            )}
          </span>
        )}
      </h3>

      {/* F-1：批量审核条（选中镜头后出现） */}
      {selection.size > 0 && (
        <div className="batch-bar">
          <strong>已选 {selection.size} 镜</strong>
          {runActive ? (
            <>
              <button
                type="button"
                className="btn btn-approve btn-sm"
                disabled={busy}
                onClick={() => planBatch(false)}
                title="把这些镜头加入自动通过队列：命中各自审核闸口时自动放行，其余镜头仍会正常停下等你审"
              >
                预通过选中（自动放行）
              </button>
              <button
                type="button"
                className="btn btn-primary btn-sm"
                disabled={busy}
                onClick={() => planBatch(true)}
                title="同上，且每个镜头视频审核通过后自动锁定，重跑/续跑时复用"
              >
                通过并锁定
              </button>
            </>
          ) : (
            <span className="hint">任务未在运行，批量预通过仅在「运行中 / 等待审核」时可用</span>
          )}
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() => setSelection(new Set())}
          >
            清除选择
          </button>
        </div>
      )}

      <div className="scene-list">
        {scenes.map((scene, idx) => {
          const lockedFlag = lockedSet.has(idx);
          const cardState = deriveCardState(scene, lockedFlag);
          const quickFix: QuickFix | null =
            gate && idx === gateIdx && onDecide
              ? {
                  fixTypes: gateFixTypes,
                  busy: !!busy,
                  onApprove: () => onDecide({ action: "approve" }),
                  onFix: (ft) => onDecide({ action: "rewrite", fix_type: ft }),
                }
              : null;
          return (
            <SceneCard
              key={idx}
              scene={scene}
              idx={idx}
              active={activeIdx === idx}
              subtitles={subtitleByScene?.get(idx)}
              cardState={cardState}
              phaseLabel={scenePhase}
              quickFix={quickFix}
              rerunning={rerunIdx === idx}
              selectMode={selectMode}
              selected={selection.has(idx)}
              onToggleSelect={() => toggleSelect(idx)}
              queued={queueSet.has(idx)}
              onRerunCard={() => void handleRerun(idx)}
              onUnlockRerun={() => void handleRerun(idx, { unlock: true })}
              onShowRecipe={() => void openRecipe(idx)}
              onZoom={(item) => setLightbox({ kind: "single", item })}
              onCompare={(left, right, title) =>
                setLightbox({ kind: "pair", left, right, title })
              }
            />
          );
        })}
      </div>

      {lightbox?.kind === "single" && (
        <ArtifactLightbox item={lightbox.item} onClose={() => setLightbox(null)} />
      )}
      {lightbox?.kind === "pair" && (
        <ArtifactLightbox
          pair={{ left: lightbox.left, right: lightbox.right, title: lightbox.title }}
          onClose={() => setLightbox(null)}
        />
      )}

      {recipeErr && (
        <div className="recipe-error" onClick={() => setRecipeErr(null)}>
          读取配方失败：{recipeErr}（点击关闭）
        </div>
      )}

      {recipe && (
        <RecipeCard
          threadId={threadId ?? ""}
          recipe={recipe}
          markdown={recipeMd}
          initialShot={recipeShotIdx}
          canApply={!!onRecipeApplied}
          onClose={() => setRecipe(null)}
          onApplied={(idx) => {
            setRecipe(null);
            onRecipeApplied?.(idx);
          }}
        />
      )}
    </section>
  );
}
