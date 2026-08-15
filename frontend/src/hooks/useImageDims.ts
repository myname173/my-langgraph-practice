import { useEffect, useState } from "react";

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
  const [dim, setDim] = useState<ImageDim | null>(() =>
    url ? cache.get(url) ?? null : null,
  );

  useEffect(() => {
    if (!url) {
      setDim(null);
      return;
    }
    const cached = cache.get(url);
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
      cache.set(url, d);
      setDim(d);
    };
    img.onerror = () => {
      if (alive) setDim(null);
    };
    img.src = url;
    return () => {
      alive = false;
    };
  }, [url]);

  return dim;
}

/** 批量识图：用于一整组素材统一网格列数 */
export function useImageDims(urls: (string | null | undefined)[]): Map<string, ImageDim> {
  const [dims, setDims] = useState<Map<string, ImageDim>>(new Map());
  useEffect(() => {
    const valid = urls.filter((u): u is string => !!u);
    if (!valid.length) return;
    let alive = true;
    Promise.all(
      valid.map(
        (u) =>
          new Promise<[string, ImageDim]>((resolve) => {
            const c = cache.get(u);
            if (c) return resolve([u, c]);
            const img = new Image();
            img.onload = () =>
              resolve([
                u,
                {
                  width: img.naturalWidth,
                  height: img.naturalHeight,
                  ratio: img.naturalWidth / img.naturalHeight,
                  kind: classify(img.naturalWidth, img.naturalHeight),
                },
              ]);
            img.onerror = () => resolve([u, { width: 0, height: 0, ratio: 0, kind: "unknown" }]);
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
