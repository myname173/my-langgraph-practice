import { useState, useMemo, type ReactNode } from "react";
import type { MultimediaState, AssetMatchReport } from "../types";
import { MediaPreview } from "./MediaPreview";
import { useImageDims, useImageDim, type MediaRatio } from "../hooks/useImageDims";

/** 单个参考图条目：url + 名称 + 描述 */
interface RefItem {
  key: string;
  /** 资产名（report 匹配键）；array 分支无名字时退化为 label */
  name: string;
  url: string;
  label: string;
  desc?: string;
}

/** 从 reference_sheets 的某个分类（dict 或 list）中提取条目 */
function collectItems(
  group: unknown,
  defaultLabel: string,
): RefItem[] {
  if (!group) return [];
  if (Array.isArray(group)) {
    return group
      .map((it, i) => {
        if (typeof it === "string") return { key: String(i), name: `${defaultLabel} ${i + 1}`, url: it, label: `${defaultLabel} ${i + 1}` };
        if (it && typeof it === "object") {
          const o = it as Record<string, unknown>;
          const label = String(o.name || o.name_cn || `${defaultLabel} ${i + 1}`);
          return {
            key: String(i),
            name: label,
            url: String(o.url || o.image_url || ""),
            label,
            desc: String(o.description || ""),
          };
        }
        return null;
      })
      .filter((x): x is RefItem => Boolean(x && x.url));
  }
  // dict: { name: {url, description, name_cn} }
  return Object.entries(group as Record<string, unknown>)
    .map(([k, v]) => {
      if (typeof v === "string") return { key: k, name: k, url: v, label: k };
      if (v && typeof v === "object") {
        const o = v as Record<string, unknown>;
        return {
          key: k,
          name: k,
          url: String(o.url || o.image_url || ""),
          label: String(o.name || o.name_cn || k),
          desc: String(o.description || ""),
        };
      }
      return null;
    })
    .filter((x): x is RefItem => Boolean(x && x.url));
}

/** 根据匹配报告，给单张参考图标上「已用于 N 个镜头 / 未命中剧本」状态 */
function matchBadge(name: string, report?: AssetMatchReport): { text: string; tone: "used" | "unused" | "none" } {
  if (!report) return { text: "", tone: "none" };
  const hit = report.matched?.[name];
  if (typeof hit === "number" && hit > 0) {
    return { text: `已用于 ${hit} 个镜头`, tone: "used" };
  }
  if (report.unmatched?.includes(name)) {
    return { text: "未命中剧本", tone: "unused" };
  }
  return { text: "", tone: "none" };
}

/** 分组渲染一个分类的参考图网格 */
function RefGroup({
  title,
  icon,
  items,
  onZoom,
  report,
  onRequestFix,
}: {
  title: string;
  icon: string;
  items: RefItem[];
  onZoom: (url: string, label: string) => void;
  report?: AssetMatchReport;
  onRequestFix?: (name: string) => void;
}) {
  const dims = useImageDims(items.map((it) => it.url));
  if (!items.length) return null;
  return (
    <section className="ref-group">
      <h4>
        <span className="section-icon">{icon}</span>
        {title}
        <span className="ref-count">{items.length}</span>
      </h4>
      <div className="ref-grid">
        {items.map((it) => {
          const kind: MediaRatio = dims.get(it.url)?.kind ?? "landscape";
          const badge = matchBadge(it.name, report);
          return (
            <figure className={`ref-card ratio-${kind}`} key={it.key}>
              <MediaPreview
                url={it.url}
                label={it.label}
                ratioKind={kind}
                onZoom={() => onZoom(it.url, it.label)}
              />
              {badge.tone !== "none" && (
                <span className={`ref-match-badge ref-match-${badge.tone}`}>
                  {badge.text}
                  {badge.tone === "unused" && onRequestFix && (
                    <button
                      type="button"
                      className="ref-fix-btn"
                      onClick={() => onRequestFix(it.name)}
                      title="去创建面板修正该素材的名称 / 分类"
                    >
                      去修正
                    </button>
                  )}
                </span>
              )}
              {it.desc && <figcaption className="ref-desc">{it.desc}</figcaption>}
            </figure>
          );
        })}
      </div>
    </section>
  );
}

/**
 * 参考图面板：展示角色肖像 + 角色转写 / 道具 / 环境参考图等中间产物，
 * 让用户能看到后端生成这些一致性基准的进度与结果。
 */
export function ReferenceSheetsPanel({
  state,
  onRequestFix,
}: {
  state: MultimediaState;
  /** 第4项：点击「去修正」回传未命中素材名，由 App 路由到创建面板高亮对应卡片 */
  onRequestFix?: (name: string) => void;
}) {
  const [lightbox, setLightbox] = useState<{ url: string; label: string } | null>(null);

  const characterUrl = state.character_portrait_url as string | undefined;
  const sheets = (state.reference_sheets as Record<string, unknown> | undefined) || {};
  const report = state.asset_match_report ?? undefined;

  // hook 必须在顶层无条件调用，不能放进条件分支（否则在 characterUrl 有无之间
  // 切换时会触发 "Rendered more hooks than during the previous render"）。
  const portraitDim = useImageDim(characterUrl);

  const characters = useMemo(
    () => collectItems(sheets.characters, "角色"),
    [sheets],
  );
  const props = useMemo(() => collectItems(sheets.props, "道具"), [sheets]);
  const environments = useMemo(
    () => collectItems(sheets.environments, "环境"),
    [sheets],
  );

  const hasAny = Boolean(characterUrl) || characters.length > 0 || props.length > 0 || environments.length > 0;
  if (!hasAny) return null;

  const groups: ReactNode[] = [];
  if (characterUrl) {
    const portraitKind = portraitDim?.kind ?? "landscape";
    groups.push(
      <section className="ref-group" key="portrait">
        <h4>
          <span className="section-icon">🧑</span>角色肖像
          <span className="ref-count">1</span>
        </h4>
        <div className="ref-grid ref-grid-single">
          <figure className={`ref-card ratio-${portraitKind}`}>
            <MediaPreview
              url={characterUrl}
              label="角色肖像"
              ratioKind={portraitKind}
              onZoom={() => setLightbox({ url: characterUrl, label: "角色肖像" })}
            />
            <span className="ref-match-badge ref-match-base">身份基准</span>
          </figure>
        </div>
      </section>,
    );
  }
  if (characters.length) {
    groups.push(
      <RefGroup key="char" title="角色转写图" icon="👤" items={characters} report={report} onRequestFix={onRequestFix} onZoom={(u, l) => setLightbox({ url: u, label: l })} />,
    );
  }
  if (props.length) {
    groups.push(
      <RefGroup key="props" title="道具参考图" icon="🔧" items={props} report={report} onRequestFix={onRequestFix} onZoom={(u, l) => setLightbox({ url: u, label: l })} />,
    );
  }
  if (environments.length) {
    groups.push(
      <RefGroup key="env" title="环境参考图" icon="🏞️" items={environments} report={report} onRequestFix={onRequestFix} onZoom={(u, l) => setLightbox({ url: u, label: l })} />,
    );
  }

  return (
    <div className="panel reference-panel">
      <h3>
        <span className="section-icon">🖼️</span>一致性参考图
      </h3>
      <p className="panel-hint">
        这些<b>一致性参考图</b>来自你在创建任务时上传/勾选的素材（经上方「素材资产库」保存），
        或由后端自动生成，用于保持跨镜头一致性。
        {report && <span className="panel-hint-sub"> 徽标显示每张图「是否被剧本用到」——红色「未命中剧本」表示素材名与剧本称呼不一致，需到素材库改角色名。</span>}
      </p>
      {groups}

      {lightbox && (
        <div
          className="ref-lightbox"
          role="dialog"
          aria-modal="true"
          onClick={() => setLightbox(null)}
        >
          <div className="ref-lightbox-inner" onClick={(e) => e.stopPropagation()}>
            <button
              type="button"
              className="ref-lightbox-close"
              onClick={() => setLightbox(null)}
              aria-label="关闭"
            >
              ×
            </button>
            <img src={lightbox.url} alt={lightbox.label} />
            <p className="ref-lightbox-label">{lightbox.label}</p>
            <a href={lightbox.url} target="_blank" rel="noreferrer">
              新窗口打开 / 下载
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
