import { useEffect, useMemo, useState } from "react";
import { AssetGroupCard } from "./AssetGroupCard";
import type { AssetCategory, AssetItem, AssetsApplyPayload, ReferenceEntry, ReferenceSheets } from "../types";

interface Props {
  threadId?: string | null;
}

interface ManifestEntry {
  filename: string;
  story?: string;
  category: AssetCategory;
  label: string;
  role_name?: string;
  description?: string;
}

/** 资产分类 → reference_sheets 字段名（复数 key） */
const CAT_TO_SHEET: Record<Exclude<AssetCategory, "ignore">, "characters" | "props" | "environments"> = {
  character: "characters",
  prop: "props",
  environment: "environments",
};

/** 读取清单：优先该 thread 目录，空则回退 _legacy（历史即梦图） */
async function listKeyframes(threadId?: string | null): Promise<{ dir: string; items: { name: string }[] }> {
  const dirs = threadId ? [`keyframes/${threadId}`, "keyframes/_legacy"] : ["keyframes/_legacy"];
  for (const dir of dirs) {
    const r = await fetch(`/media/list?dir=${encodeURIComponent(dir)}`);
    if (!r.ok) continue;
    const data = await r.json();
    if (data.ok && data.count > 0) return { dir, items: data.items };
  }
  return { dir: "keyframes/_legacy", items: [] };
}

async function loadManifest(): Promise<Record<string, ManifestEntry>> {
  try {
    const r = await fetch("/media/keyframes/_legacy/manifest.json");
    if (!r.ok) return {};
    const arr: ManifestEntry[] = await r.json();
    return Object.fromEntries(arr.map((m) => [m.filename, m]));
  } catch {
    return {};
  }
}

export function AssetLibraryPanel({ threadId }: Props) {
  const [items, setItems] = useState<AssetItem[]>([]);
  const [manifest, setManifest] = useState<Record<string, ManifestEntry>>({});
  const [wantStory, setWantStory] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [zoom, setZoom] = useState<AssetItem | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      if (!threadId) {
        setItems([]);
        return;
      }
      setLoading(true);
      setSaved(false);
      const [list, manifestData] = await Promise.all([listKeyframes(threadId), loadManifest()]);
      if (alive) setManifest(manifestData);
      // 由 thread_id 推断所属故事（cyber / gull），只展示该故事的已标注即梦图，
      // 避免两个故事的历史图混在同一素材库；其余渠道产物
      // 不混入，避免误标并注入 reference_sheets。
      const wantStory = threadId.includes("gull")
        ? "gull"
        : threadId.includes("cyber")
        ? "cyber"
        : null;
      if (alive) setWantStory(wantStory);
      const mapped: AssetItem[] = list.items
        .map((it): AssetItem | null => {
          const m = manifest[it.name];
          if (!m) return null;
          if (wantStory && m.story && m.story !== wantStory) return null;
          return {
            filename: it.name,
            url: `/media/${list.dir}/${it.name}`,
            category: m.category,
            label: m.label ?? it.name,
            description: m.description,
            role_name: m.role_name,
            story: m.story,
          };
        })
        .filter((x): x is AssetItem => x !== null);
      if (alive) {
        setItems(mapped);
        setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [threadId]);

  // 按 role_name 聚合为资产卡（AssetGroup）。资产名 = role_name；
  // 同一 role_name 下多张图（如“女主”正面+三视图）聚合为一卡，多图融合注入续跑生图。
  const assets = useMemo(() => {
    const map = new Map<
      string,
      { category: Exclude<AssetCategory, "ignore">; description: string; items: AssetItem[] }
    >();
    for (const it of items) {
      if (it.category === "ignore") continue;
      const name = it.role_name?.trim() || it.label;
      if (!name) continue;
      const cur = map.get(name) ?? {
        category: it.category as Exclude<AssetCategory, "ignore">,
        description: it.description ?? "",
        items: [],
      };
      cur.items.push(it);
      if (!cur.description && it.description) cur.description = it.description;
      map.set(name, cur);
    }
    // 顺序：角色 → 道具 → 场景
    const order: Record<Exclude<AssetCategory, "ignore">, number> = {
      character: 0,
      prop: 1,
      environment: 2,
    };
    return Array.from(map.entries())
      .map(([name, v]) => ({ name, ...v }))
      .sort((a, b) => order[a.category] - order[b.category] || a.name.localeCompare(b.name));
  }, [items]);

  const update = (filename: string, patch: Partial<AssetItem>) =>
    setItems((prev) => prev.map((it) => (it.filename === filename ? { ...it, ...patch } : it)));

  // 候选素材名：取该故事 manifest 中“同类别”已有的 role_name / label，供用户下拉选择，
  // 同时保留自由输入（datalist 兼得“选择 + 自定义”）。避免用户手打错名导致重复资产。
  const nameSuggestions = useMemo(() => {
    const byCat: Record<AssetCategory, string[]> = {
      character: [],
      prop: [],
      environment: [],
      ignore: [],
    };
    for (const m of Object.values(manifest)) {
      if (wantStory && m.story && m.story !== wantStory) continue;
      const src = m.role_name?.trim() || m.label?.trim();
      if (!src) continue;
      const set = byCat[m.category] ?? (byCat[m.category] = []);
      if (!set.includes(src)) set.push(src);
    }
    return byCat;
  }, [manifest, wantStory]);

  // 资产级编辑：改分类 / Persona / 重命名会同步到该资产下所有图（保持聚合一致）。
  const updateAssetCategory = (name: string, cat: Exclude<AssetCategory, "ignore">) =>
    setItems((prev) =>
      prev.map((it) =>
        (it.role_name?.trim() || it.label) === name ? { ...it, category: cat } : it
      )
    );
  const updateAssetDescription = (name: string, desc: string) =>
    setItems((prev) =>
      prev.map((it) =>
        (it.role_name?.trim() || it.label) === name ? { ...it, description: desc } : it
      )
    );
  const renameItem = (filename: string, roleName: string) =>
    update(filename, { role_name: roleName });

  const onSave = async () => {
    if (!threadId) return;
    setSaving(true);
    setSaved(false);
    const sheets: ReferenceSheets = { characters: {}, props: {}, environments: {} };
    for (const a of assets) {
      const sheetKey = CAT_TO_SHEET[a.category];
      (sheets[sheetKey] as Record<string, ReferenceEntry>)[a.name] = {
        urls: a.items.map((it) => it.url),
        description: a.description || a.items[0]?.label || a.name,
        name: a.name,
      };
    }
    const payload: AssetsApplyPayload = { thread_id: threadId, reference_sheets: sheets };
    try {
      const r = await fetch("/assets/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setSaved(true);
    } catch (e) {
      console.error("[AssetLibraryPanel] 保存失败", e);
      alert("保存失败，请重试");
    } finally {
      setSaving(false);
    }
  };

  if (!threadId) return null;
  if (loading) return <div className="panel-hint">素材库加载中…</div>;
  if (items.length === 0)
    return (
      <div className="panel-hint">
        当前会话暂无即梦图素材。生成关键帧 / 参考图后会自动归入本会话目录。
      </div>
    );

  const ignoredCount = items.filter((i) => i.category === "ignore").length;
  const counts = {
    character: assets.filter((a) => a.category === "character").length,
    prop: assets.filter((a) => a.category === "prop").length,
    environment: assets.filter((a) => a.category === "environment").length,
  };

  return (
    <section className="asset-library">
      <div className="panel-head">
        <h3>素材资产库</h3>
        <button
          type="button"
          className={`btn ${saved ? "btn-success" : "btn-primary"}`}
          disabled={saving}
          onClick={() => void onSave()}
        >
          {saving ? "保存中…" : saved ? "✓ 已保存为素材" : "保存为素材（续跑参考图）"}
        </button>
      </div>

      <div className="asset-overview">
        <span className="ov-pill ov-character">角色 {counts.character}</span>
        <span className="ov-pill ov-prop">道具 {counts.prop}</span>
        <span className="ov-pill ov-environment">场景 {counts.environment}</span>
        {ignoredCount > 0 && <span className="ov-ignored">已忽略 {ignoredCount} 张</span>}
      </div>

      <p className="panel-hint">
        素材按<b>角色 / 道具 / 场景</b>聚合成资产卡。每张卡可挂多张参考图（如正面 + 三视图），
        保存后写入本会话参考图资产（reference_sheets），续跑生图时<b>多图融合注入</b>保持一致性，省额度。
      </p>

      <div className="asset-group-list">
        {assets.map((a) => (
          <AssetGroupCard
            key={a.name}
            name={a.name}
            category={a.category}
            description={a.description}
            items={a.items}
            nameSuggestions={nameSuggestions[a.category]}
            onCategoryChange={(cat) => updateAssetCategory(a.name, cat)}
            onDescriptionChange={(desc) => updateAssetDescription(a.name, desc)}
            onRenameItem={renameItem}
            onZoom={setZoom}
          />
        ))}
      </div>

      {zoom && (
        <div className="ref-lightbox" onClick={() => setZoom(null)}>
          <div className="ref-lightbox-inner" onClick={(e) => e.stopPropagation()}>
            <img src={zoom.url} alt={zoom.label} />
            <p>{zoom.label}</p>
            <button type="button" className="btn" onClick={() => setZoom(null)}>
              关闭
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
