import type { MultimediaState } from "../types";
import { MediaPreview } from "./MediaPreview";

export function SceneGallery({ state }: { state: MultimediaState }) {
  const scenes = state.scenes ?? [];
  if (!scenes.length) return null;

  return (
    <section className="scene-gallery">
      <h3>
        <span className="section-icon">🎞️</span>分镜产物
      </h3>
      <div className="scene-list">
        {scenes.map((scene, idx) => {
          const active = state.current_scene_index === idx;
          return (
            <article
              key={idx}
              className={`scene-card ${active ? "scene-active" : ""}`}
            >
              <header>
                <strong>镜头 {idx + 1}</strong>
                {scene.video_is_perfect && <span className="tick">已定稿</span>}
              </header>
              <p className="scene-script">{scene.script}</p>
              <div className="media-row">
                <MediaPreview url={scene.image_url} label="首帧" />
                <MediaPreview url={scene.last_image_url} label="尾帧" />
                <MediaPreview
                  url={scene.final_video_url || scene.raw_video_url}
                  label="片段"
                  forceVideo
                />
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
