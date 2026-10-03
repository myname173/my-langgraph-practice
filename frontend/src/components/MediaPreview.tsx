import { useEffect, useState } from "react";
import { resolveMediaUrl, isVideoUrl } from "../lib/langgraphClient";
import { useImageDim, type MediaRatio } from "../hooks/useImageDims";

interface Props {
  url?: string | null;
  label: string;
  /** 强制按视频渲染（后端 video_url 有时无扩展名） */
  forceVideo?: boolean;
  /** 视频播放失败时的回退源（如本地真实文件 / 另一远程链接） */
  fallbackUrl?: string | null;
  /** 图片点击放大回调（由父组件挂载灯箱） */
  onZoom?: () => void;
  /** 可选副标题（显示在 label 下方，如 "0.92"） */
  badge?: string;
  /** 已知宽高比类别（由父组件识图后传入，避免重复加载） */
  ratioKind?: MediaRatio;
}

/**
 * 单个产物缩略图：
 * - 图片：懒加载 + 点击触发 onZoom（由父组件打开灯箱）
 * - 视频：内联 controls 预览，悬停时显示操作栏（新窗口/下载）
 * - 主源加载失败自动切到 fallback
 */
export function MediaPreview({
  url,
  label,
  forceVideo,
  fallbackUrl,
  onZoom,
  badge,
  ratioKind,
}: Props) {
  const primary = resolveMediaUrl(url);
  const fallback = resolveMediaUrl(fallbackUrl);
  const src = primary || fallback;

  // ⚠️ Hooks 必须无条件调用：早退只能放在所有 hooks 之后。
  // 若在此处 `if (!src) return null`，则 src 由空变非空时（关键帧刚生成）
  // 同一次挂载的 hook 数量会从 0 变成 5，React 会抛
  // "Rendered more hooks than during the previous render" 并让整个成果区白屏。
  const asVideo = forceVideo || (src ? isVideoUrl(src) : false);
  // 识图：自身未识别时回退到父组件传入的 ratioKind
  const dim = useImageDim(asVideo ? null : primary);
  const kind: MediaRatio = ratioKind ?? dim?.kind ?? "landscape";
  const [current, setCurrent] = useState(src);
  const [errored, setErrored] = useState(false);
  useEffect(() => {
    setCurrent(primary || fallback);
    setErrored(false);
  }, [primary, fallback]);

  if (!src) return null;

  return (
    <figure className={`media-preview ${asVideo ? "is-video" : "is-image"} ratio-${kind}`}>
      <figcaption>
        <span className="media-label">{label}</span>
        {badge && <span className="media-badge">{badge}</span>}
      </figcaption>

      <div className="media-body">
        {asVideo ? (
          <video
            key={current}
            src={current}
            controls
            playsInline
            preload="metadata"
            onError={() => {
              if (fallback && current !== fallback) setCurrent(fallback);
              else setErrored(true);
            }}
          />
        ) : (
          <button
            type="button"
            className="media-zoom-btn"
            onClick={onZoom}
            disabled={!onZoom || errored}
            aria-label={`放大查看 ${label}`}
            title={onZoom ? "点击放大" : undefined}
          >
            <img
              src={current}
              alt={label}
              loading="lazy"
              onError={() => {
                if (fallback && current !== fallback) setCurrent(fallback);
                else setErrored(true);
              }}
            />
            {onZoom && !errored && (
              <span className="media-zoom-icon" aria-hidden="true">🔍</span>
            )}
          </button>
        )}

        {errored && (
          <div className="media-error">加载失败：{label}</div>
        )}
      </div>

      {src && (
        <div className="media-actions">
          <a href={src} target="_blank" rel="noreferrer" title="新窗口打开">↗ 打开</a>
          <a href={src} download target="_blank" rel="noreferrer" title="下载到本地">⬇ 下载</a>
        </div>
      )}
    </figure>
  );
}
