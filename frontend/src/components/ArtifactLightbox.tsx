import { useEffect } from "react";
import { resolveMediaUrl } from "../lib/langgraphClient";

export interface LightboxItem {
  url: string;
  label: string;
  /** 可选附加说明（用于对比模式下显示在图下） */
  caption?: string;
}

interface Props {
  /** 单图模式 */
  item?: LightboxItem | null;
  /** 对比模式：两张图并排（首帧 vs 尾帧） */
  pair?: { left: LightboxItem; right: LightboxItem; title?: string } | null;
  onClose: () => void;
}

/**
 * 全屏产物灯箱：
 * - 单图模式：item 单独展示
 * - 对比模式：pair 展示左右两张图 + 标题
 * - ESC / 点击遮罩关闭
 * - 受控组件，单例挂载（由父组件持有状态）
 */
export function ArtifactLightbox({ item, pair, onClose }: Props) {
  const open = !!(item || pair);

  // ESC 关闭
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    // 锁定背景滚动
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [open, onClose]);

  if (!open) return null;

  const renderImage = (it: LightboxItem) => {
    const src = resolveMediaUrl(it.url);
    return (
      <figure className="lightbox-figure">
        <div className="lightbox-img-wrap">
          <img src={src} alt={it.label} />
        </div>
        <figcaption>
          <strong>{it.label}</strong>
          {it.caption && <span className="lightbox-caption">{it.caption}</span>}
        </figcaption>
      </figure>
    );
  };

  return (
    <div
      className="lightbox-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="产物预览"
    >
      <button
        className="lightbox-close"
        onClick={(e) => {
          e.stopPropagation();
          onClose();
        }}
        aria-label="关闭"
      >
        ✕
      </button>

      <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
        {pair ? (
          <>
            {pair.title && <h3 className="lightbox-title">{pair.title}</h3>}
            <div className="lightbox-pair">
              {renderImage(pair.left)}
              <div className="lightbox-divider" aria-hidden="true" />
              {renderImage(pair.right)}
            </div>
            <p className="lightbox-hint">点击空白或按 ESC 关闭</p>
          </>
        ) : item ? (
          <>
            {renderImage(item)}
            <p className="lightbox-hint">点击空白或按 ESC 关闭</p>
          </>
        ) : null}
      </div>
    </div>
  );
}
