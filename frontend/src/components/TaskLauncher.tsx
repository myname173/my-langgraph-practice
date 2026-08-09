import { useState } from "react";
import type { RunInput } from "../types";

interface Props {
  busy: boolean;
  onStart: (input: RunInput) => void;
}

/** 与后端 tts.EDGE_TTS_VOICES_ZH 对齐的常用音色 */
const VOICE_OPTIONS = [
  { value: "xiaoxiao", label: "晓晓（女声·温柔）" },
  { value: "yunxi", label: "云希（男声·青年）" },
  { value: "yunjian", label: "云健（男声·浑厚）" },
  { value: "xiaoyi", label: "晓伊（女声·活泼）" },
];

const BGM_OPTIONS = [
  { value: "ambient", label: "氛围 / 舒缓" },
  { value: "tense", label: "紧张 / 悬疑" },
  { value: "upbeat", label: "明快 / 激昂" },
];

export function TaskLauncher({ busy, onStart }: Props) {
  const [task, setTask] = useState("");
  const [useFirstLastFrame, setUseFirstLastFrame] = useState(true);
  const [enableAudio, setEnableAudio] = useState(true);
  const [voiceRole, setVoiceRole] = useState("xiaoxiao");
  const [bgmMood, setBgmMood] = useState("ambient");

  function submit() {
    const trimmed = task.trim();
    if (!trimmed || busy) return;

    // 构造完整初始 state，避免后端节点读取缺失键
    onStart({
      task: trimmed,
      global_setting: "",
      scenes: [],
      current_scene_index: 0,
      use_first_last_frame: useFirstLastFrame,
      enable_audio: enableAudio,
      voice_role: voiceRole,
      bgm_mood: bgmMood,
      reference_images: [],
      reference_embeddings: [],
      aborted: false,
      final_movie_path: null,
      error_log: null,
      rewrite_count: 0,
    });
  }

  return (
    <section className="task-launcher">
      <h3>
        <span className="section-icon">✨</span>新建视频任务
      </h3>
      <textarea
        rows={4}
        value={task}
        disabled={busy}
        placeholder="描述你想生成的宣传片，例如：为一款国风茶饮品牌做一支 30 秒水彩风格宣传片，突出手工与自然"
        onChange={(e) => setTask(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
        }}
      />

      <div className="options-grid">
        <label className="check">
          <input
            type="checkbox"
            checked={useFirstLastFrame}
            disabled={busy}
            onChange={(e) => setUseFirstLastFrame(e.target.checked)}
          />
          首尾帧双控（画面更连贯，耗时更长）
        </label>

        <label className="check">
          <input
            type="checkbox"
            checked={enableAudio}
            disabled={busy}
            onChange={(e) => setEnableAudio(e.target.checked)}
          />
          启用配音 / 字幕 / BGM
        </label>

        <label className="select">
          配音音色
          <select
            value={voiceRole}
            disabled={busy || !enableAudio}
            onChange={(e) => setVoiceRole(e.target.value)}
          >
            {VOICE_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>

        <label className="select">
          BGM 情绪
          <select
            value={bgmMood}
            disabled={busy || !enableAudio}
            onChange={(e) => setBgmMood(e.target.value)}
          >
            {BGM_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <button className="btn btn-primary" disabled={busy || !task.trim()} onClick={submit}>
        {busy ? "运行中…" : "开始生成 (Ctrl+Enter)"}
      </button>
    </section>
  );
}
