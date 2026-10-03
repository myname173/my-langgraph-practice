import { useCallback, useEffect, useState } from "react";
import { client } from "../lib/langgraphClient";
import {
  fetchHistoryThreads,
  HISTORY_STATUS_META,
  type HistoryStatus,
  type HistoryThreadSummary,
} from "../lib/historyClient";

/** 侧栏列表项（归一化后的展示模型） */
interface ThreadItem {
  thread_id: string;
  task: string | null;
  updated_at: string | null;
  total_scenes: number;
  current_scene_index: number;
  status: HistoryStatus;
  has_interrupt: boolean;
  /** 封面缩略图（首镜首帧 /media URL），可选 */
  cover?: string;
}

function formatTime(iso?: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** 平台 threads.search 的返回结构（兜底通道用） */
interface PlatformThread {
  thread_id: string;
  created_at?: string;
  updated_at?: string;
  values?: { task?: string; scenes?: unknown[]; current_scene_index?: number } | null;
  status?: string;
}

/** 把 /history 摘要归一化为展示模型 */
function fromHistory(t: HistoryThreadSummary): ThreadItem {
  return {
    thread_id: t.thread_id,
    task: t.task,
    updated_at: t.updated_at,
    total_scenes: t.total_scenes,
    current_scene_index: t.current_scene_index,
    status: t.status,
    has_interrupt: t.has_interrupt,
    cover: t.cover || undefined,
  };
}

/** 把平台 thread 归一化为展示模型（字段更少，尽力而为） */
function fromPlatform(t: PlatformThread): ThreadItem {
  const scenes = t.values?.scenes;
  return {
    thread_id: t.thread_id,
    task: t.values?.task?.trim() || null,
    updated_at: t.updated_at ?? t.created_at ?? null,
    total_scenes: Array.isArray(scenes) ? scenes.length : 0,
    current_scene_index: t.values?.current_scene_index ?? 0,
    status: t.status === "interrupted" ? "interrupted" : "idle",
    has_interrupt: t.status === "interrupted",
  };
}

interface Props {
  activeThreadId: string | null;
  onSelect: (threadId: string) => void;
  /** 变化时触发刷新 */
  refreshKey: unknown;
  /** F-6：窄屏抽屉是否展开（≤900px 生效） */
  mobileOpen?: boolean;
  /** F-6：关闭抽屉（点关闭按钮 / 选中会话后调用） */
  onClose?: () => void;
}

export function ThreadSidebar({
  activeThreadId,
  onSelect,
  refreshKey,
  mobileOpen,
  onClose,
}: Props) {
  const [threads, setThreads] = useState<ThreadItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * 双通道加载：优先自建 /history 接口（能拿到真实标题、进度与状态），
   * 失败时降级到平台 threads.search，保证列表始终可用。
   */
  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchHistoryThreads(50);
      setThreads(res.threads.map(fromHistory));
    } catch (historyErr) {
      try {
        const res = await client.threads.search({ limit: 30 });
        setThreads((res as unknown as PlatformThread[]).map(fromPlatform));
        setError(
          historyErr instanceof Error
            ? `${historyErr.message}（已降级为平台列表）`
            : "历史接口不可用，已降级为平台列表",
        );
      } catch (platformErr) {
        const platformMsg =
          platformErr instanceof Error ? platformErr.message : String(platformErr);
        const historyMsg =
          historyErr instanceof Error ? historyErr.message : String(historyErr);
        // 双通道都失败：区分「服务未启动」与「接口异常」，给出可操作提示
        const serviceDown = /连接产物服务|无法连接|fetch/i.test(
          historyMsg + platformMsg,
        );
        setError(
          serviceDown
            ? "无法连接历史/产物服务（static_server.py 端口 8900）。请确认已用一键启动脚本启动全部服务后刷新页面。"
            : `${historyMsg}（备用通道亦失败：${platformMsg}）`,
        );
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, refreshKey]);

  return (
    <aside className={`sidebar${mobileOpen ? " open" : ""}`}>
      {/* F-6：窄屏抽屉关闭按钮 */}
      <button
        className="sidebar-close"
        aria-label="收起会话列表"
        onClick={() => onClose?.()}
      >
        ✕
      </button>
      <div className="brand">
        <div className="brand-mark">🎬</div>
        <div className="brand-text">
          <span className="brand-title">Multimedia Agent</span>
          <span className="brand-sub">视频生成控制台</span>
        </div>
      </div>

      <div className="sidebar-head">
        <h3>历史会话</h3>
        <button
          className={`refresh-btn ${loading ? "spinning" : ""}`}
          title="刷新会话列表"
          disabled={loading}
          onClick={() => void load()}
        >
          ⟳
        </button>
      </div>

      {loading && <p className="empty-hint">加载中…</p>}
      {error && (
        <div className="sidebar-error">
          <span className="error-text">{error}</span>
          <button
            className="btn btn-ghost btn-small"
            disabled={loading}
            onClick={() => void load()}
          >
            重试
          </button>
        </div>
      )}
      {!loading && !error && !threads.length && (
        <p className="empty-hint">还没有历史会话，新建一个任务开始创作吧</p>
      )}

      <ul className="thread-list">
        {threads.map((t) => {
          const shortId = t.thread_id.slice(0, 8);
          const time = formatTime(t.updated_at);
          const meta = HISTORY_STATUS_META[t.status] ?? HISTORY_STATUS_META.idle;
          const corrupted = t.status === "corrupted";
          const title = t.task || `未命名任务 · ${shortId}`;
          const progress =
            t.total_scenes > 0
              ? t.status === "done" || t.status === "interrupted" || t.status === "aborted"
                ? `全部完成 · ${t.total_scenes} 镜头`
                : `${Math.min(t.current_scene_index + 1, t.total_scenes)}/${t.total_scenes} 镜头`
              : "";

          return (
            <li key={t.thread_id}>
              <button
                className={`thread-item ${
                  t.thread_id === activeThreadId ? "thread-active" : ""
                } ${corrupted ? "thread-corrupted" : ""}`}
                onClick={() => {
                  if (corrupted) return;
                  onSelect(t.thread_id);
                  onClose?.(); // F-6：移动端选中后自动收起抽屉
                }}
                disabled={corrupted}
                title={
                  corrupted
                    ? `${shortId} 的存档已损坏，无法恢复`
                    : `${title} (${shortId})`
                }
              >
                <span className="thread-title">{title}</span>
                <span className="thread-meta-row">
                  <span
                    className="thread-badge"
                    style={{ color: meta.color, borderColor: meta.color }}
                  >
                    {meta.label}
                  </span>
                  {progress && (
                    <span className="thread-progress">{progress}</span>
                  )}
                </span>
                {t.cover && (
                  <img
                    className="thread-cover"
                    src={t.cover}
                    alt=""
                    loading="lazy"
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).style.display = "none";
                    }}
                  />
                )}
                <span className="thread-id">
                  {shortId}
                  {time ? ` · ${time}` : ""}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
