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
  /** 第4项：参考图面板「去修正」传入的待高亮素材名，命中则高亮并滚动进视图 */
  highlightName?: string;
  /** 第4项：点击参考图「去修正」时回传未命中名称，由 App 路由到本面板 */
  onRequestFix?: (name: string) => void;
}

/** 名称实时预检状态（第2项）：让用户在上传时就知道这名字能否用于匹配剧本 */
export type NameStatus = { tone: "ok" | "bad" | "warn"; text: string };

/** 单个上传素材缩略卡（支持独立卡 / 组内卡两种形态） */
function UploadThumb({
  u,
  grouped,
  status,
  highlight,
  onName,
  onDesc,
  onCat,
  onRemove,
}: {
  u: SelectedAsset;
  grouped: boolean;
  status?: NameStatus;
  highlight?: boolean;
  onName: (v: string) => void;
  onDesc: (v: string) => void;
  onCat: (v: AssetCategory | "") => void;
  onRemove: () => void;
}) {
  return (
    <div
      className={`au-thumb ${u.name.trim() ? "" : "is-unnamed"} ${highlight ? "is-highlight" : ""}`}
      ref={highlight ? (el) => el?.scrollIntoView({ block: "center", behavior: "smooth" }) : undefined}
    >
      <img src={u.url} alt={u.name} />
      {!grouped && (
        <input
          className="au-thumb-name"
          list="au-name-suggestions"
          placeholder="角色/道具/场景名（必填，需与剧本称呼一致）"
          value={u.name}
          onChange={(e) => onName(e.target.value)}
        />
      )}
      {!grouped && status && (
        <span className={`au-name-status au-ns-${status.tone}`}>
          <span className="au-ns-icon">{status.tone === "ok" ? "✓" : status.tone === "bad" ? "✗" : "⚠"}</span>
          {status.text}
        </span>
      )}
      <textarea
        className="au-thumb-desc"
        rows={2}
        placeholder="语义描述（如：白身灰翅尖黄喙的海鸥）"
        value={u.description ?? ""}
        onChange={(e) => onDesc(e.target.value)}
      />
      <select
        value={u.category}
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => onCat(e.target.value as AssetCategory | "")}
      >
        <option value="">请选择分类</option>
        <option value="character">角色</option>
        <option value="prop">道具</option>
        <option value="environment">场景</option>
      </select>
      {u.story && <span className="au-thumb-story" title="故事标签：素材归类与素材库隔离依据">#{u.story}</span>}
      <button type="button" className="au-remove" onClick={onRemove}>
        ✕
      </button>
    </div>
  );
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



export function TaskLauncher({ busy, onStart, highlightName }: Props) {
  const [task, setTask] = useState("");
  const [useFirstLastFrame, setUseFirstLastFrame] = useState(true);
  const [enableAudio, setEnableAudio] = useState(true);
  const [preferNativeAudio, setPreferNativeAudio] = useState(false);
  const [voiceRole, setVoiceRole] = useState("xiaoxiao");
  const [bgmMood, setBgmMood] = useState("ambient");
  // 低干预 / 自动模式：跳过 6 道人工审核闸口，普通用户免逐镜点击
  const [autoMode, setAutoMode] = useState(false);
  // 故事标识：用于素材库按故事隔离（如 cyber/gull/xianxia），空则由前端回退 threadId 推断
  const [story, setStory] = useState("");

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
    // 把当前任务故事标签（如 xianxia）一并上传，素材即被打上该故事标记
    const tag = story.trim();
    if (tag) fd.append("story", tag);
    setUploading(true);
    try {
      const r = await fetch("/assets/upload", { method: "POST", body: fd });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${r.status}`);
      }
      const data = await r.json();
      const added: SelectedAsset[] = (data.files as { filename: string; url: string; story?: string }[]).map((f) => {
        // 用文件名（去扩展名/去 uuid 前缀）猜测默认名，方便用户在此基础上修正；
        // 猜测为空则留空（不再默认「素材」），由提交校验强制用户填写，避免误匹配。
        // 分类不预设：默认空串，由用户在下拉中明确选择，避免道具/场景误归为「角色」。
        const base = f.filename.replace(/\.[^.]+$/, "");
        const guessed = base.replace(/^[0-9a-f]{8,}-/, "").replace(/[_-]+/g, " ").trim();
        return {
          url: f.url,
          name: guessed,
          category: "",
          description: "",
          source: "upload" as const,
          story: f.story || tag || undefined,
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

  function setUploadedCat(idx: number, cat: Cat | "") {
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
    const put = (name: string, cat: Cat, urls: string[], desc?: string): ReferenceEntry | null => {
      const key = CAT_TO_SHEET[cat];
      const bucket = sheets[key] as Record<string, ReferenceEntry>;
      const existing = bucket[name];
      if (existing) {
        // 同名（多视图）素材合并为一个参考条目，URL 累加
        const merged = Array.from(new Set([...existing.urls, ...urls]));
        bucket[name] = { ...existing, urls: merged };
        return bucket[name];
      }
      const entry: ReferenceEntry = { urls, description: desc || name, name };
      bucket[name] = entry;
      return entry;
    };
    // 上传图用用户填写的名称作 key（需与剧本称呼一致，后端靠名称匹配），description 写清语义
    // 同一角色传多张图时填写相同名称即可自动归为一组（多视图设定图）
    uploaded.forEach((u) => {
      const nm = u.name.trim();
      if (nm && u.category) {
        const entry = put(nm, u.category, [u.url], u.description?.trim() || undefined);
        if (u.story && entry) entry.story = u.story;
      }
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

    // 上传图必须明确选择分类（角色/道具/场景），避免道具或场景被默认误归为「角色」
    const uncategorized = uploaded.filter((u) => !u.category);
    if (uncategorized.length > 0) {
      alert(
        `有 ${uncategorized.length} 张上传素材未选择分类。\n请为每张上传图在下拉中选择「角色 / 道具 / 场景」，分类错误会导致一致性参考图被错误匹配。`,
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
      auto_mode: autoMode,
      story: story.trim() || undefined,
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
          <span className="au-title">一致性参考图（可选）</span>
          <span className="panel-hint">
            上传或勾选的图用于<b>指导生成内容的角色/道具/场景一致</b>，不会直接生成视频。
            给每张图标好<b>分类与角色名</b>，生成后到产物区「一致性参考图」面板可查看<b>匹配结果</b>。
          </span>
        </div>

        {/* 多视图引导：单视图是身份漂移的根因，引导用户传正面+侧面等多视图设定图 */}
        <div className="au-multiview-tip">
          <span className="au-mv-icon">💡</span>
          <span className="au-mv-text">
            想让角色在多镜头里长相一致？建议为每个角色传 <b>多视图设定图</b>
            （如 <b>正面 + 侧面 + 全身</b>），比只传一张正面图能显著减少「换镜就变脸」。
            同一角色的多张图<b>用相同名称</b>即可自动归入一组。
            <br />
            <b>注意：</b>这里的「故事标签 / 分类 / 角色名」只是<b>素材管理标记</b>，模型读不到——
            仙侠感来自你上传图的<b>画风/服饰</b>，以及剧本里的<b>仙侠描述词</b>。
            标签只是帮你把不同故事的素材分开，不会让模型自动「变仙侠」。
          </span>
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
          <span className="au-drop-sub">
            支持多选 · jpg / png / webp / gif
            <br />
            建议：<b>正面 · 中性姿态 · 背景干净 · ≥1024px</b>；同一角色传多视图（正面+侧面+全身）更稳。
          </span>
        </label>

        {uploaded.length > 0 && (() => {
          const groups = buildUploadGroups(uploaded);
          const isHi = (nm: string) => Boolean(highlightName && highlightName.trim() && nm.trim() === highlightName.trim());
          return (
          <div className="au-thumbs">
            {groups.unnamed.map(({ u, i }) => (
              <UploadThumb
                key={`u-${i}`}
                u={u}
                grouped={false}
                status={nameStatus(uploaded, u.name, i)}
                highlight={isHi(u.name)}
                onName={(v) => setUploaded((prev) => prev.map((p, j) => (j === i ? { ...p, name: v } : p)))}
                onDesc={(v) => setUploaded((prev) => prev.map((p, j) => (j === i ? { ...p, description: v } : p)))}
                onCat={(v) => setUploadedCat(i, v as Cat)}
                onRemove={() => removeUploaded(i)}
              />
            ))}
            {groups.named.map((g) => {
              const st = nameStatus(uploaded, g.name, g.indexes[0]);
              const hi = isHi(g.name);
              return (
                <div
                  key={`g-${g.name}`}
                  className={`au-thumb-group ${hi ? "is-highlight" : ""}`}
                  ref={hi ? (el) => el?.scrollIntoView({ block: "center", behavior: "smooth" }) : undefined}
                >
                  <div className="au-thumb-group-head">
                    <input
                      className="au-thumb-name au-thumb-group-name"
                      list="au-name-suggestions"
                      placeholder="角色/道具/场景名（必填，需与剧本称呼一致）"
                      value={g.name}
                      onChange={(e) => {
                        const v = e.target.value;
                        setUploaded((prev) => prev.map((p, j) => (g.indexes.includes(j) ? { ...p, name: v } : p)));
                      }}
                    />
                    <span className={`au-name-status au-ns-${st.tone}`}>
                      <span className="au-ns-icon">{st.tone === "ok" ? "✓" : st.tone === "bad" ? "✗" : "⚠"}</span>
                      {st.text}
                    </span>
                    <span className="au-thumb-group-count">{g.items.length} 视图</span>
                  </div>
                  <div className="au-thumb-group-imgs">
                    {g.items.map(({ u, i }) => (
                      <UploadThumb
                        key={i}
                        u={u}
                        grouped
                        highlight={false}
                        onName={() => {}}
                        onDesc={(v) => setUploaded((prev) => prev.map((p, j) => (j === i ? { ...p, description: v } : p)))}
                        onCat={(v) => setUploadedCat(i, v as Cat)}
                        onRemove={() => removeUploaded(i)}
                      />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
          );
        })()}
        <datalist id="au-name-suggestions">
          <option value="女主" />
          <option value="男主" />
          <option value="剑仙" />
          <option value="灵宠" />
          <option value="飞剑" />
          <option value="仙山" />
          <option value="秘境" />
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

        {/* 低干预 / 自动模式：跳过逐镜人工审核，普通用户免逐次点击 */}
        <label className="check auto-mode">
          <input
            type="checkbox"
            checked={autoMode}
            disabled={busy}
            onChange={(e) => setAutoMode(e.target.checked)}
          />
          🤖 自动模式（跳过逐镜确认，一键直出）
        </label>

        {/* 故事标识：素材库按故事隔离，支撑多故事（cyber/gull/xianxia 等） */}
        <label className="select story-input">
          故事标识（可选）
          <input
            type="text"
            value={story}
            disabled={busy}
            placeholder="如 cyber / gull / xianxia，留空则自动推断"
            onChange={(e) => setStory(e.target.value)}
          />
        </label>
      </div>

      <button className="btn btn-primary" disabled={busy || !task.trim()} onClick={submit}>
        {busy ? "运行中…" : "开始生成 (Ctrl+Enter)"}
      </button>
    </section>
  );
}

/** 第1项：把上传素材按名称分组（同名=同一角色多视图），未命名单独列出 */
interface UploadGroup {
  name: string;
  indexes: number[];
  items: { u: SelectedAsset; i: number }[];
}
function buildUploadGroups(list: SelectedAsset[]): {
  named: UploadGroup[];
  unnamed: { u: SelectedAsset; i: number }[];
} {
  const map = new Map<string, UploadGroup>();
  const unnamed: { u: SelectedAsset; i: number }[] = [];
  list.forEach((u, i) => {
    const nm = u.name.trim();
    if (!nm) {
      unnamed.push({ u, i });
      return;
    }
    if (!map.has(nm)) map.set(nm, { name: nm, indexes: [], items: [] });
    const g = map.get(nm)!;
    g.indexes.push(i);
    g.items.push({ u, i });
  });
  return { named: [...map.values()], unnamed };
}

/** 第2项：名称实时预检——上传阶段即告知用户这名字能否用于匹配剧本（剧本此时尚未生成，故只做结构性校验） */
function nameStatus(list: SelectedAsset[], name: string, selfIndex: number): NameStatus {
  const t = name.trim();
  if (!t) return { tone: "bad", text: "需填角色名" };
  const dup = list.some((p, j) => j !== selfIndex && p.name.trim() === t);
  if (dup) return { tone: "warn", text: "与其他重名" };
  return { tone: "ok", text: "可用于匹配剧本" };
}
