import { useRef, useState } from "react";
import { AssetPicker } from "./AssetPicker";
import type {
  AssetCategory,
  ReferenceEntry,
  ReferenceSheets,
  RunInput,
  SelectedAsset,
} from "../types";

interface Props {
  busy: boolean;
  onStart: (input: RunInput) => void;
}

type Cat = Exclude<AssetCategory, "ignore">;

/** 资产分类 → reference_sheets 字段名 */
const CAT_TO_SHEET: Record<Cat, "characters" | "props" | "environments"> = {
  character: "characters",
  prop: "props",
  environment: "environments",
};

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
  const [preferNativeAudio, setPreferNativeAudio] = useState(false);
  const [voiceRole, setVoiceRole] = useState("xiaoxiao");
  const [bgmMood, setBgmMood] = useState("ambient");

  // ── 素材选择区状态 ─────────────────────────────────────────────
  // 本地上传图（SelectedAsset，source=upload）
  const [uploaded, setUploaded] = useState<SelectedAsset[]>([]);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);
  // 勾选的已有素材库图：资产名 → {分类, urls, 描述}
  const [libPicks, setLibPicks] = useState<Record<string, { category: Cat; urls: string[]; description?: string }>>({});
  // 用户对已有素材手动改过的分类
  const [libCatOverride, setLibCatOverride] = useState<Record<string, Cat>>({});

  const hasAssets = uploaded.length > 0 || Object.keys(libPicks).length > 0;
  const counts = {
    character: uploaded.filter((u) => u.category === "character").length + Object.values(libPicks).filter((p) => p.category === "character").length,
    prop: uploaded.filter((u) => u.category === "prop").length + Object.values(libPicks).filter((p) => p.category === "prop").length,
    environment: uploaded.filter((u) => u.category === "environment").length + Object.values(libPicks).filter((p) => p.category === "environment").length,
  };

  // 选中素材库面板展开态（受控，贴合框架折叠风格）
  const [libOpen, setLibOpen] = useState(false);
  // 拖拽高亮态
  const [dragging, setDragging] = useState(false);

  async function onFilesPicked(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (fileRef.current) fileRef.current.value = ""; // 允许重复选同一文件
    if (!files || files.length === 0) return;
    const fd = new FormData();
    for (const f of Array.from(files)) fd.append("files", f);
    await uploadViaFormData(fd);
  }

  async function uploadViaFormData(fd: FormData) {
    if (busy) return;
    setUploading(true);
    try {
      const r = await fetch("/assets/upload", { method: "POST", body: fd });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${r.status}`);
      }
      const data = await r.json();
      const added: SelectedAsset[] = (data.files as { filename: string; url: string }[]).map((f) => {
        // 用文件名（去扩展名/去 uuid 前缀）猜测默认角色名，方便后端按名匹配脚本
        const base = f.filename.replace(/\.[^.]+$/, "");
        const guessed = base.replace(/^[0-9a-f]{8,}-/, "").replace(/[_-]+/g, " ").trim() || "素材";
        return {
          url: f.url,
          name: guessed,
          category: "character" as const,
          description: "",
          source: "upload" as const,
        };
      });
      setUploaded((prev) => [...prev, ...added]);
    } catch (err) {
      console.error("[TaskLauncher] 上传失败", err);
      alert(`素材上传失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setUploading(false);
    }
  }

  function setUploadedCat(idx: number, cat: Cat) {
    setUploaded((prev) => prev.map((u, i) => (i === idx ? { ...u, category: cat } : u)));
  }
  function removeUploaded(idx: number) {
    setUploaded((prev) => prev.filter((_, i) => i !== idx));
  }

  function toggleLib(name: string, urls: string[], checked: boolean) {
    setLibPicks((prev) => {
      const next = { ...prev };
      if (checked) {
        next[name] = { category: libCatOverride[name] ?? "character", urls };
      } else {
        delete next[name];
      }
      return next;
    });
  }
  function changeLibCat(name: string, cat: Cat) {
    setLibCatOverride((prev) => ({ ...prev, [name]: cat }));
    setLibPicks((prev) => (prev[name] ? { ...prev, [name]: { ...prev[name], category: cat } } : prev));
  }

  /** 合并上传图 + 勾选已有图，构建 reference_sheets；图只作一致性参考，不生成视频。 */
  function buildSheets(): ReferenceSheets {
    const sheets: ReferenceSheets = { characters: {}, props: {}, environments: {} };
    const put = (name: string, cat: Cat, urls: string[], desc?: string) => {
      const key = CAT_TO_SHEET[cat];
      const entry: ReferenceEntry = { urls, description: desc || name, name };
      (sheets[key] as Record<string, ReferenceEntry>)[name] = entry;
    };
    // 上传图用用户填写的名称作 key（需与剧本称呼一致，后端靠名称匹配），description 写清语义
    uploaded.forEach((u) => {
      const nm = u.name.trim();
      if (nm) put(nm, u.category, [u.url], u.description?.trim() || undefined);
    });
    for (const [name, p] of Object.entries(libPicks)) put(name, p.category, p.urls, p.description);
    return sheets;
  }

  function submit() {
    const trimmed = task.trim();
    if (!trimmed || busy) return;

    // 上传图必须填写名称（用于后端按名匹配剧本中的角色/道具/场景）
    const unnamed = uploaded.filter((u) => !u.name.trim());
    if (unnamed.length > 0) {
      alert(
        `有 ${unnamed.length} 张上传素材未填写名称。\n请为每张上传图填写「角色/道具/场景名」，且名称需与剧本中的称呼一致，素材才会参与生成。`,
      );
      return;
    }

    const sheets = hasAssets ? buildSheets() : undefined;

    // 构造完整初始 state，避免后端节点读取缺失键
    onStart({
      task: trimmed,
      global_setting: "",
      scenes: [],
      current_scene_index: 0,
      use_first_last_frame: useFirstLastFrame,
      enable_audio: enableAudio,
      prefer_native_audio: preferNativeAudio,
      voice_role: voiceRole,
      bgm_mood: bgmMood,
      reference_images: [],
      reference_embeddings: [],
      reference_sheets: sheets,
      assets_imported: hasAssets,
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

      {/* ── 素材选择区：上传本地图 / 勾选已有素材，作为一致性参考 ── */}
      <div className="asset-upload-card">
        <div className="au-head">
          <span className="au-title">素材参考图（可选）</span>
          <span className="panel-hint">上传或勾选的图用于<b>指导生成内容的角色/道具/场景一致</b>，不会直接生成视频。</span>
        </div>

        {/* 本地上传：框架风格按钮 + 拖拽区 */}
        <label
          className={`au-drop ${dragging ? "is-dragging" : ""} ${busy ? "is-disabled" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            if (!busy) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            if (busy) return;
            const dt = e.dataTransfer;
            if (dt?.files?.length) {
              const fd = new FormData();
              for (const f of Array.from(dt.files)) fd.append("files", f);
              void uploadViaFormData(fd);
            }
          }}
        >
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange={onFilesPicked}
          />
          <span className="au-drop-icon">⬆</span>
          <span className="au-drop-text">
            {uploading ? "上传中…" : dragging ? "松开以上传" : "点击或拖拽上传本地素材图"}
          </span>
          <span className="au-drop-sub">支持多选 · jpg / png / webp / gif</span>
        </label>

        {uploaded.length > 0 && (
          <div className="au-thumbs">
            {uploaded.map((u, i) => (
              <div key={i} className={`au-thumb ${u.name.trim() ? "" : "is-unnamed"}`}>
                <img src={u.url} alt={u.name} />
                <input
                  className="au-thumb-name"
                  list="au-name-suggestions"
                  placeholder="角色/道具/场景名（必填，需与剧本称呼一致）"
                  value={u.name}
                  onChange={(e) =>
                    setUploaded((prev) =>
                      prev.map((p, j) => (j === i ? { ...p, name: e.target.value } : p)),
                    )
                  }
                />
                <textarea
                  className="au-thumb-desc"
                  rows={2}
                  placeholder="语义描述（如：白身灰翅尖黄喙的海鸥）"
                  value={u.description ?? ""}
                  onChange={(e) =>
                    setUploaded((prev) =>
                      prev.map((p, j) => (j === i ? { ...p, description: e.target.value } : p)),
                    )
                  }
                />
                <select
                  value={u.category}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => setUploadedCat(i, e.target.value as Cat)}
                >
                  <option value="character">角色</option>
                  <option value="prop">道具</option>
                  <option value="environment">场景</option>
                </select>
                <button type="button" className="au-remove" onClick={() => removeUploaded(i)}>
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}
        <datalist id="au-name-suggestions">
          <option value="女主" />
          <option value="海鸥" />
          <option value="能量手枪" />
          <option value="无人机" />
          <option value="海边日落" />
        </datalist>

        {/* 已有素材库勾选：受控折叠面板，贴合框架风格 */}
        <div className={`au-library ${libOpen ? "is-open" : ""}`}>
          <button
            type="button"
            className="au-library-toggle"
            onClick={() => setLibOpen((v) => !v)}
            aria-expanded={libOpen}
          >
            <span className="au-library-title">选择已有素材图库</span>
            <span className="au-library-count">{Object.keys(libPicks).length} 已选</span>
            <span className="au-caret">{libOpen ? "▾" : "▸"}</span>
          </button>
          {libOpen && (
            <div className="au-library-body">
              <AssetPicker
                selected={new Set(Object.keys(libPicks))}
                onToggle={toggleLib}
                categoryOverride={libCatOverride}
                onCategoryChange={changeLibCat}
              />
            </div>
          )}
        </div>

        {hasAssets && (
          <div className="au-summary">
            <span className="ov-pill ov-character">角色 {counts.character}</span>
            <span className="ov-pill ov-prop">道具 {counts.prop}</span>
            <span className="ov-pill ov-environment">场景 {counts.environment}</span>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => {
                setUploaded([]);
                setLibPicks({});
                setLibCatOverride({});
              }}
            >
              清空已选
            </button>
          </div>
        )}
      </div>

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

        <label
          className="check"
          title="若生成视频自带音轨（如即梦 Seedance 1.5 Pro 音画一体），优先保留原生声音，仅叠加字幕；无原生音轨时自动回退本地配音+BGM。"
        >
          <input
            type="checkbox"
            checked={preferNativeAudio}
            disabled={busy || !enableAudio}
            onChange={(e) => setPreferNativeAudio(e.target.checked)}
          />
          优先保留生成视频原生音轨
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
