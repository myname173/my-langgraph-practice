import { resolveMediaUrl, isVideoUrl } from "../lib/langgraphClient";

interface Props {
  url?: string | null;
  label: string;
  /** 强制按视频渲染（后端 video_url 有时无扩展名） */
  forceVideo?: boolean;
}

export function MediaPreview({ url, label, forceVideo }: Props) {
  const src = resolveMediaUrl(url);
  if (!src) return null;

  const asVideo = forceVideo || isVideoUrl(src);

  return (
    <figure className="media-preview">
      <figcaption>{label}</figcaption>
      {asVideo ? (
        <video src={src} controls playsInline preload="metadata" />
      ) : (
        <a href={src} target="_blank" rel="noreferrer">
          <img src={src} alt={label} loading="lazy" />
        </a>
      )}
    </figure>
  );
}
