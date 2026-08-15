import { useEffect, useRef, useState } from "react";
import type { MultimediaState, SceneLike } from "../types";
import { MediaPreview } from "./MediaPreview";
import { useImageDim } from "../hooks/useImageDims";
import { ArtifactLightbox, type LightboxItem } from "./ArtifactLightbox";

interface Props {
  state: MultimediaState;
}

type LightboxState =
  | { kind: "single"; item: LightboxItem }
  | { kind: "pair"; left: LightboxItem; right: LightboxItem; title: string }
  | null;

function fmtScore(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}

function SceneCard({
  scene,
  idx,
  active,
  onZoom,
  onCompare,
}: {
  scene: SceneLike;
  idx: number;
  active: boolean;
  onZoom: (item: LightboxItem) => void;
  onCompare: (left: LightboxItem, right: LightboxItem, title: string) => void;
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
      className={`scene-card ${active ? "scene-active" : ""} ${finished ? "scene-finished" : ""}`}
    >
      <header className="scene-card-head">
        <div className="scene-title">
          <strong>镜头 {idx + 1}</strong>
          {active && <span className="badge badge-active">执行中</span>}
          {scene.video_is_perfect && <span className="badge badge-perfect">视频已定稿</span>}
          {scene.is_perfect && !scene.video_is_perfect && <span className="badge badge-perfect">关键帧已定稿</span>}
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

      {scene.image_url && scene.last_image_url && (
        <div className="scene-row-actions">
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

export function SceneGallery({ state }: Props) {
  const scenes = state.scenes ?? [];
  const activeIdx = state.current_scene_index;
  const [lightbox, setLightbox] = useState<LightboxState>(null);

  if (!scenes.length) return null;

  return (
    <section className="scene-gallery">
      <h3>
        <span className="section-icon">🎞️</span>分镜产物
        <span className="scene-count">
          {scenes.length} 个镜头{typeof activeIdx === "number" ? ` · 当前第 ${activeIdx + 1} 个` : ""}
        </span>
      </h3>

      <div className="scene-list">
        {scenes.map((scene, idx) => (
          <SceneCard
            key={idx}
            scene={scene}
            idx={idx}
            active={activeIdx === idx}
            onZoom={(item) => setLightbox({ kind: "single", item })}
            onCompare={(left, right, title) =>
              setLightbox({ kind: "pair", left, right, title })
            }
          />
        ))}
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
    </section>
  );
}
