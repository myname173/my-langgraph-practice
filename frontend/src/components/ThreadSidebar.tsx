import { useCallback, useEffect, useState } from "react";
import { client } from "../lib/langgraphClient";

interface ThreadItem {
  thread_id: string;
  created_at?: string;
  values?: { task?: string } | null;
}

interface Props {
  activeThreadId: string | null;
  onSelect: (threadId: string) => void;
  /** 变化时触发刷新 */
  refreshKey: unknown;
}

export function ThreadSidebar({ activeThreadId, onSelect, refreshKey }: Props) {
  const [threads, setThreads] = useState<ThreadItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await client.threads.search({ limit: 30 });
      setThreads(res as unknown as ThreadItem[]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, refreshKey]);

  return (
    <aside className="sidebar">
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
      {error && <p className="error-text">{error}</p>}
      {!loading && !error && !threads.length && (
        <p className="empty-hint">还没有历史会话，新建一个任务开始创作吧</p>
      )}

      <ul className="thread-list">
        {threads.map((t) => (
          <li key={t.thread_id}>
            <button
              className={`thread-item ${
                t.thread_id === activeThreadId ? "thread-active" : ""
              }`}
              onClick={() => onSelect(t.thread_id)}
            >
              <span className="thread-title">
                {t.values?.task?.slice(0, 40) || "未命名任务"}
              </span>
              <span className="thread-id">{t.thread_id.slice(0, 8)}</span>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  );
}
