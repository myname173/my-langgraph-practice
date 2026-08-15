import { useEffect, useMemo, useState } from "react";
import type { AssetCategory, AssetItem, ReferenceEntry, ReferenceSheets } from "../types";

type Cat = Exclude<AssetCategory, "ignore">;

interface ManifestEntry {
  filename: string;
  story?: string;
  category: AssetCategory;
  label: string;
  role_name?: string;
  description?: string;
}

/** 资产分类 → reference_sheets 字段名 */
const CAT_TO_SHEET: Record<Cat, "characters" | "props" | "environments"> = {
  character: "characters",
  prop: "props",
  environment: "environments",
};

/** 读取历史素材库（_legacy 即梦图 + manifest 标注）。返回归一化的 AssetItem 列表。 */
async function loadLibraryItems(): Promise<{ items: AssetItem[]; manifest: Record<string, ManifestEntry> }> {
  // 历史素材库目录优先 _legacy；若后续扩展多故事可按需追加
  const r = await fetch(`/media/list?dir=${encodeURIComponent("keyframes/_legacy")}`);
  if (!r.ok) return { items: [], manifest: {} };
  const data = await r.json();
  const files: { name: string }[] = data.ok ? data.items : [];

  let manifest: Record<string, ManifestEntry> = {};
  try {
    const mr = await fetch("/media/keyframes/_legacy/manifest.json");
    if (mr.ok) {
      const arr: ManifestEntry[] = await mr.json();
      manifest = Object.fromEntries(arr.map((m) => [m.filename, m]));
    }
  } catch {
    manifest = {};
  }

  const items: AssetItem[] = files
    .map((it): AssetItem | null => {
      // 只保留即梦生成的素材（三视图/设定图），过滤其他渠道的历史素材
      if (!it.name.startsWith("jimeng_")) return null;
      const m = manifest[it.name];
      if (m && m.category === "ignore") return null;
      // 无 manifest 的图也允许选择，默认归为角色，用户可改分类
      const cat: AssetCategory = m?.category ?? "character";
      if (cat === "ignore") return null;
      return {
        filename: it.name,
        url: `/media/keyframes/_legacy/${it.name}`,
        category: cat,
        label: m?.label ?? it.name,
        description: m?.description,
        role_name: m?.role_name,
      };
    })
    .filter((x): x is AssetItem => x !== null);
  return { items, manifest };
}

interface Props {
  /** 当前已勾选的资产名（role_name）集合，受控于父组件 */
  selected: Set<string>;
  /** 勾选切换：回传资产名、该资产全部图 url、勾选态 */
  onToggle: (name: string, urls: string[], checked: boolean) => void;
  /** 用户手动改过的分类（资产名 → 分类） */
  categoryOverride: Record<string, Cat>;
  onCategoryChange: (name: string, cat: Cat) => void;
}

export function AssetPicker({ selected, onToggle, categoryOverride, onCategoryChange }: Props) {
  const [items, setItems] = useState<AssetItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    (async () => {
      const { items: lib } = await loadLibraryItems();
      if (alive) {
        setItems(lib);
        setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // 按 role_name 聚合为资产卡
  const groups = useMemo(() => {
    const map = new Map<string, { category: Cat; description: string; items: AssetItem[] }>();
    for (const it of items) {
      const effCat = (categoryOverride[it.role_name?.trim() || it.label] ??
        (it.category as Cat)) as Cat;
      const name = it.role_name?.trim() || it.label;
      if (!name) continue;
      const cur = map.get(name) ?? { category: effCat, description: it.description ?? "", items: [] };
      cur.items.push(it);
      if (!cur.description && it.description) cur.description = it.description;
      // 若分类被覆盖则同步
      cur.category = effCat;
      map.set(name, cur);
    }
    const order: Record<Cat, number> = { character: 0, prop: 1, environment: 2 };
    return Array.from(map.entries())
      .map(([name, v]) => ({ name, ...v }))
      .sort((a, b) => order[a.category] - order[b.category] || a.name.localeCompare(b.name));
  }, [items, categoryOverride]);

  if (loading) return <div className="panel-hint">已有素材库加载中…</div>;
  if (groups.length === 0)
    return <div className="panel-hint">暂无历史素材图。可改用上方「上传本地素材」。</div>;

  return (
    <div className="asset-picker">
      <p className="panel-hint">
        勾选已有素材图作为<b>一致性参考</b>（角色 / 道具 / 场景）。这些图只用于指导生成内容的
        角色/道具/场景一致，<b>不会直接生成视频</b>。
      </p>
      <div className="asset-picker-grid">
        {groups.map((g) => {
          const checked = selected.has(g.name);
          return (
            <label key={g.name} className={`asset-pick-card ${checked ? "is-checked" : ""}`}>
              <input
                type="checkbox"
                checked={checked}
                onChange={(e) => onToggle(g.name, g.items.map((it) => it.url), e.target.checked)}
              />
              <div className="apc-thumbs">
                {g.items.slice(0, 3).map((it) => (
                  <img key={it.filename} src={it.url} alt={it.label} />
                ))}
              </div>
              <div className="apc-meta">
                <span className="apc-name">{g.name}</span>
                <span className="apc-count">{g.items.length} 张</span>
              </div>
              <select
                value={g.category}
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => onCategoryChange(g.name, e.target.value as Cat)}
              >
                <option value="character">角色</option>
                <option value="prop">道具</option>
                <option value="environment">场景</option>
              </select>
            </label>
          );
        })}
      </div>
    </div>
  );
}

/**
 * 由勾选结果构建 reference_sheets（供 TaskLauncher 随 RunInput 透传）。
 * selectedItems 为父组件维护的「资产名 → 该资产首图」映射（含分类）。
 */
export function buildReferenceSheetsFromPicks(
  picks: { name: string; category: Cat; urls: string[]; description?: string }[]
): ReferenceSheets {
  const sheets: ReferenceSheets = { characters: {}, props: {}, environments: {} };
  for (const p of picks) {
    const key = CAT_TO_SHEET[p.category];
    const entry: ReferenceEntry = {
      urls: p.urls,
      description: p.description || p.name,
      name: p.name,
    };
    (sheets[key] as Record<string, ReferenceEntry>)[p.name] = entry;
  }
  return sheets;
}
