import { resolveMediaUrl } from "../lib/langgraphClient";
import type { MultimediaState } from "../types";

export function FinalMoviePlayer({ state }: { state: MultimediaState }) {
  // 优先播放带音频的最终成片
  const withAudio = resolveMediaUrl(state.final_movie_with_audio);
  const silent = resolveMediaUrl(state.final_movie_path);
  const src = withAudio || silent;

  if (!src) return null;

  return (
    <section className="final-movie">
      <h3>
        <span className="section-icon">🎥</span>
        最终成片 {withAudio ? "（含音轨）" : "（无音轨）"}
      </h3>
      <video src={src} controls playsInline />
      <div className="final-links">
        <a href={src} download target="_blank" rel="noreferrer">
          下载成片
        </a>
        {withAudio && silent && (
          <a href={silent} target="_blank" rel="noreferrer">
            查看无音轨版本
          </a>
        )}
      </div>
    </section>
  );
}
