import { useEffect, useMemo, useState } from "react";
import type { AssetCategory, AssetItem } from "../types";

type Cat = Exclude<AssetCategory, "ignore">;

/** 分类色板：与 AssetGroupCard / 参考图面板徽标体系保持一致（角色紫 / 道具橙 / 场景绿） */
const CATEGORY_COLOR: Record<Cat, string> = {
  character: "#8B5CF6",
  prop: "#F59E0B",
  environment: "#22C55E",
};

interface ManifestEntry {
  filename: string;
  story?: string;
  category: AssetCategory;
  label: string;
  role_name?: string;
  description?: string;
}

/** 读取参考图素材库（仅 reference_sheets/library 里 source=user 的图）。返回归一化的 AssetItem 列表。 */
async function loadLibraryItems(): Promise<{ items: AssetItem[]; manifest: Record<string, ManifestEntry> }> {
  // 数据源改为按 manifest 驱动，只展示用户登记的参考图；jimeng 关键帧（source=keyframe）不进入素材库
  let manifest: Record<string, ManifestEntry> = {};
  try {
    const mr = await fetch("/media/reference_sheets/library/manifest.json");
    if (mr.ok) {
      const arr: ManifestEntry[] = await mr.json();
      manifest = Object.fromEntries(arr.map((m) => [m.filename, m]));
    }
  } catch {
    manifest = {};
  }

  const items: AssetItem[] = Object.values(manifest)
    .filter((m) => m.source === "user" && m.category !== "ignore")
    .map((m): AssetItem => {
      const cat: AssetCategory = (m.category as AssetCategory) ?? "character";
      const name = (m.filename as string) ?? "";
      return {
        filename: name,
        url: `/media/reference_sheets/library/${m.category}/${name}`,
        category: cat,
        label: (m.label as string) ?? name,
        description: (m.description as string) ?? "",
        role_name: (m.role_name as string) ?? (m.label as string),
      };
    });
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
            <label
              key={g.name}
              className={`asset-pick-card ${checked ? "is-checked" : ""}`}
              style={{ borderLeftColor: CATEGORY_COLOR[g.category] }}
            >
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
