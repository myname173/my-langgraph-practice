import { useState } from "react";
import { MediaPreview } from "./MediaPreview";
import { useImageDims, type MediaRatio } from "../hooks/useImageDims";
import type { AssetCategory, AssetItem } from "../types";

const CATEGORY_COLOR: Record<AssetCategory, string> = {
  character: "#8B5CF6",
  prop: "#F59E0B",
  environment: "#22C55E",
  ignore: "#6B7280",
};

const CATEGORY_LABEL: Record<Exclude<AssetCategory, "ignore">, string> = {
  character: "角色",
  prop: "道具",
  environment: "场景",
};

interface Props {
  name: string;
  category: Exclude<AssetCategory, "ignore">;
  description: string;
  items: AssetItem[];
  nameSuggestions: string[];
  onCategoryChange: (cat: Exclude<AssetCategory, "ignore">) => void;
  onDescriptionChange: (desc: string) => void;
  onRenameItem: (filename: string, roleName: string) => void;
  onZoom: (item: AssetItem) => void;
}

/** 单个资产卡（Cast Card）：分类色描边 + 名称 + 参考图缩略图网格 + Persona 描述框 */
export function AssetGroupCard({
  name,
  category,
  description,
  items,
  nameSuggestions,
  onCategoryChange,
  onDescriptionChange,
  onRenameItem,
  onZoom,
}: Props) {
  const [expanded, setExpanded] = useState(true);
  const color = CATEGORY_COLOR[category];
  const dims = useImageDims(items.map((it) => it.url));

  return (
    <div className="asset-group-card" style={{ borderLeftColor: color }}>
      <div className="agc-head">
        <span className="agc-badge" style={{ background: color }}>
          {CATEGORY_LABEL[category]}
        </span>
        <h4 className="agc-name">{name}</h4>
        <span className="agc-count">{items.length} 张参考图</span>
        <button
          type="button"
          className="agc-toggle"
          onClick={() => setExpanded((v) => !v)}
          aria-label={expanded ? "收起" : "展开"}
        >
          {expanded ? "▾" : "▸"}
        </button>
      </div>

      {expanded && (
        <>
          <div className="agc-thumb-grid">
            {items.map((it) => {
              const kind: MediaRatio = dims.get(it.url)?.kind ?? "landscape";
              return (
                <div className={`agc-thumb ratio-${kind}`} key={it.filename}>
                  <MediaPreview
                    url={it.url}
                    label={it.label}
                    ratioKind={kind}
                    onZoom={() => onZoom(it)}
                  />
                  <input
                    className="agc-thumb-name"
                    list="agc-rename-suggestions"
                    placeholder="重命名素材"
                    defaultValue={it.role_name ?? ""}
                    onBlur={(e) => {
                      const v = e.target.value.trim();
                      if (v && v !== it.role_name) onRenameItem(it.filename, v);
                    }}
                  />
                </div>
              );
            })}
            <datalist id="agc-rename-suggestions">
              {nameSuggestions.map((n) => (
                <option key={n} value={n} />
              ))}
            </datalist>
          </div>

          <div className="agc-meta">
            <label className="agc-meta-label">分类</label>
            <select
              className="asset-select"
              value={category}
              onChange={(e) =>
                onCategoryChange(e.target.value as Exclude<AssetCategory, "ignore">)
              }
            >
              <option value="character">角色（人物）</option>
              <option value="prop">武器 / 道具</option>
              <option value="environment">场景 / 环境</option>
            </select>
          </div>

          <div className="agc-persona">
            <label className="agc-meta-label">Persona 设定（角色外观 / 道具描述 / 场景氛围）</label>
            <textarea
              className="agc-textarea"
              rows={2}
              placeholder="描述该素材的外观、风格、关键特征，续跑生图时按此保持一致性…"
              value={description}
              onChange={(e) => onDescriptionChange(e.target.value)}
            />
          </div>
        </>
      )}
    </div>
  );
}
