import type { MultimediaState, SceneLike, SubtitleEntry } from "../types";

interface Props {
  state: MultimediaState;
}

/** 秒 → mm:ss.t 格式（与视频播放器时间码一致） */
function fmtTime(sec: number | null | undefined): string {
  if (sec == null || Number.isNaN(sec)) return "—";
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}

/**
 * 由 scenes[].video_duration 累积时间轴，把全局字幕条目映射到所属镜头序号。
 * build_srt 按 scenes 顺序排布、按各镜真实时长切分，所以逐条 start 落在哪个镜
 * 的累积区间 [cumStart, cumStart+dur) 即归属该镜。
 */
export function mapSubtitlesToScenes(
  scenes: SceneLike[],
  entries: SubtitleEntry[] | null | undefined,
): Map<number, SubtitleEntry[]> {
  const out = new Map<number, SubtitleEntry[]>();
  if (!entries || !scenes.length) return out;

  // 优先用后端 stitcher 回写的真实起始时间（已扣除转场重叠，见 graph.stitcher_node）。
  // 只要任一镜头带 segment_start，就整体采用真实起点，与后端 build_srt 保持同一时间轴，
  // 避免前端累加与转场重叠不一致造成的错位。
  const hasRealStart = scenes.some((sc) => typeof sc.segment_start === "number");
  const sceneStarts = hasRealStart
    ? scenes.map((sc) => sc.segment_start ?? 0)
    : (() => {
        let cum = 0;
        return scenes.map((sc) => {
          const start = cum;
          const dur = typeof sc.video_duration === "number" ? sc.video_duration : 0;
          cum += dur;
          return start;
        });
      })();

  for (const e of entries) {
    // 找到 e.start 落入的镜头区间（末尾兜底到最后一镜）。
    // 时长缺失时用 duration_seconds 兜底，绝不用 Infinity —— 否则缺失镜头会
    // 形成 [st, ∞) 区间，把后面所有字幕全部吸走，其余镜头字幕为空。
    let idx = sceneStarts.findIndex((st, i) => {
      const vd = scenes[i].video_duration;
      const est = scenes[i].duration_seconds;
      const dur =
        typeof vd === "number" && vd > 0
          ? vd
          : typeof est === "number" && est > 0
            ? est
            : 0;
      if (dur <= 0) return false; // 无有效时长：不参与区间匹配
      return e.start >= st && e.start < st + dur;
    });
    if (idx === -1) {
      // 落在所有区间之外（舍入误差）时，归到最接近的镜
      idx = sceneStarts.reduce(
        (best, st, i) => (Math.abs(st - e.start) < Math.abs(sceneStarts[best] - e.start) ? i : best),
        0,
      );
    }
    const list = out.get(idx) ?? [];
    list.push(e);
    out.set(idx, list);
  }
  return out;
}

const AUDIO_STATUS_LABEL: Record<string, string> = {
  native: "原生音轨（保留生成视频自带声音）",
  dubbed: "本地配音 + BGM 混音",
};

export function AudioSubtitlePanel({ state }: Props) {
  const entries = state.subtitle_entries;
  const hasAudio = !!state.final_movie_with_audio || !!state.audio_track;
  const voiceRole = state.voice_role;
  const voiceDur = state.voiceover_duration;
  const bgm = state.bgm_mood;
  const audioStatus = state.audio_status;

  if (!hasAudio && !entries?.length) return null;

  return (
    <section className="audio-panel">
      <h3>
        <span className="section-icon">🔊</span>配音 &amp; 字幕
        {audioStatus && (
          <span className="audio-status-badge">{AUDIO_STATUS_LABEL[audioStatus] ?? audioStatus}</span>
        )}
      </h3>

      <div className="audio-meta">
        {voiceRole && (
          <span className="meta-chip">
            🎙️ 配音音色：<code>{voiceRole}</code>
          </span>
        )}
        {typeof voiceDur === "number" && (
          <span className="meta-chip">
            ⏱️ 配音总时长：<strong>{fmtTime(voiceDur)}</strong>
          </span>
        )}
        {bgm && (
          <span className="meta-chip">
            🎵 BGM 情绪：<code>{bgm}</code>
          </span>
        )}
        {state.subtitle_path && (
          <span className="meta-chip">
            📝 字幕条数：<strong>{entries?.length ?? 0}</strong>
          </span>
        )}
      </div>

      {entries && entries.length > 0 && (
        <div className="subtitle-timeline">
          <div className="subtitle-head">字幕逐句时间轴（生成在第几秒）</div>
          <ul className="subtitle-list">
            {entries.map((e) => (
              <li key={e.index} className="subtitle-row">
                <span className="subtitle-time">
                  {fmtTime(e.start)} – {fmtTime(e.end)}
                </span>
                <span className="subtitle-text">{e.text}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {state.final_movie_with_audio && (
        <div className="audio-final">
          <span className="meta-chip">✅ 音画合成成片已生成</span>
        </div>
      )}
    </section>
  );
}
