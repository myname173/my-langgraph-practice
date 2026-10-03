import { useEffect, useState } from "react";
import { resolveMediaUrl } from "../lib/langgraphClient";

export type MediaRatio = "landscape" | "square" | "portrait" | "unknown";

export interface ImageDim {
  width: number;
  height: number;
  ratio: number; // width / height
  kind: MediaRatio;
}

const cache = new Map<string, ImageDim>();

function classify(w: number, h: number): MediaRatio {
  if (!w || !h) return "unknown";
  const r = w / h;
  if (r >= 1.5) return "landscape"; // 16:9 / 更宽
  if (r <= 0.75) return "portrait"; // 9:16 / 更高
  return "square";
}

/**
 * 轻量"识图"：预加载素材图，读取真实像素尺寸，
 * 据此推断宽高比类别，供前端约束展示卡片尺寸（横/方/竖）。
 */
export function useImageDim(url?: string | null): ImageDim | null {
  // 后端产物常为本地绝对路径（C:\...\output\...），必须先归一为 /media/ 可访问 URL，
  // 否则 img.src 指向本地路径必然 404，kind 永远回落 landscape，横竖版式判断失效。
  const resolved = resolveMediaUrl(url);
  const [dim, setDim] = useState<ImageDim | null>(() =>
    resolved ? cache.get(resolved) ?? null : null,
  );

  useEffect(() => {
    if (!resolved) {
      setDim(null);
      return;
    }
    const cached = cache.get(resolved);
    if (cached) {
      setDim(cached);
      return;
    }
    let alive = true;
    const img = new Image();
    img.onload = () => {
      if (!alive) return;
      const d: ImageDim = {
        width: img.naturalWidth,
        height: img.naturalHeight,
        ratio: img.naturalWidth / img.naturalHeight,
        kind: classify(img.naturalWidth, img.naturalHeight),
      };
      cache.set(resolved, d);
      setDim(d);
    };
    img.onerror = () => {
      if (alive) setDim(null);
    };
    img.src = resolved;
    return () => {
      alive = false;
    };
  }, [resolved]);

  return dim;
}

/** 批量识图：用于一整组素材统一网格列数 */
export function useImageDims(urls: (string | null | undefined)[]): Map<string, ImageDim> {
  const [dims, setDims] = useState<Map<string, ImageDim>>(new Map());
  useEffect(() => {
    // 统一归一为 /media/ 可访问 URL（后端产物可能是本地绝对路径），
    // 同时保留「原始值 → 解析值」映射，使返回的 Map 仍以调用方传入的 key 索引。
    const valid = urls
      .filter((u): u is string => !!u)
      .map((u) => [u, resolveMediaUrl(u)] as [string, string])
      .filter(([, r]) => !!r);
    if (!valid.length) return;
    let alive = true;
    Promise.all(
      valid.map(
        ([orig, u]) =>
          new Promise<[string, ImageDim]>((resolve) => {
            const c = cache.get(u);
            if (c) return resolve([orig, c]);
            const img = new Image();
            img.onload = () =>
              resolve([
                orig,
                {
                  width: img.naturalWidth,
                  height: img.naturalHeight,
                  ratio: img.naturalWidth / img.naturalHeight,
                  kind: classify(img.naturalWidth, img.naturalHeight),
                },
              ]);
            img.onerror = () => resolve([orig, { width: 0, height: 0, ratio: 0, kind: "unknown" }]);
            img.src = u;
          }),
      ),
    ).then((entries) => {
      if (!alive) return;
      const m = new Map<string, ImageDim>();
      entries.forEach(([u, d]) => {
        cache.set(u, d);
        m.set(u, d);
      });
      setDims(m);
    });
    return () => {
      alive = false;
    };
  }, [urls.join("|")]);

  return dims;
}
