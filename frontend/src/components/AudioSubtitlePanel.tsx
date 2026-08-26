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
  let cum = 0;
  // 每镜的全局起始时间
  const sceneStarts = scenes.map((sc) => {
    const start = cum;
    const dur = typeof sc.video_duration === "number" ? sc.video_duration : 0;
    cum += dur;
    return start;
  });
  for (const e of entries) {
    // 找到 e.start 落入的镜头区间（末尾兜底到最后一镜）
    let idx = sceneStarts.findIndex(
      (st, i) =>
        e.start >= st &&
        e.start < st + (scenes[i].video_duration || Infinity),
    );
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
