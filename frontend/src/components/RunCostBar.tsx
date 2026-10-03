import { useEffect, useRef, useState } from "react";
import type { MultimediaState } from "../types";

interface HealthInfo {
  state?: string;
  [key: string]: unknown;
}

interface Props {
  state: MultimediaState;
  busy: boolean;
  threadId: string | null;
}

/** F-2：顶栏常驻成本与额度仪表（数据全部来自 state 派生 + /settings/health）。 */
export function RunCostBar({ state, busy, threadId }: Props) {
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef<number | null>(null);
  const [health, setHealth] = useState<Record<string, HealthInfo>>({});

  // 耗时：busy 从 false→true 记起点；结束保留最后值
  useEffect(() => {
    if (busy) {
      if (startRef.current === null) startRef.current = Date.now();
      const t = window.setInterval(() => {
        if (startRef.current !== null) {
          setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
        }
      }, 1000);
      return () => window.clearInterval(t);
    }
    return undefined;
  }, [busy]);

  // 后端健康灯：挂载 / 会话变化时拉一次，之后 60s 轮询
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const r = await fetch("/settings/health");
        if (!r.ok) return;
        const data = (await r.json()) as { backend_health?: Record<string, HealthInfo> };
        if (alive && data.backend_health) setHealth(data.backend_health);
      } catch {
        // 健康灯失败静默（不干扰主界面）
      }
    };
    void load();
    const t = window.setInterval(load, 60000);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, [threadId]);

  const scenes = state.scenes ?? [];
  const total = scenes.length;
  const done = scenes.filter((s) => s.final_video_url || s.raw_video_url).length;
  const failed = scenes.filter((s) => s.shot_failed).length;
  // 视频生成尝试次数（成功 1 次/镜 + 失败重试），作为额度消耗的可观测代理
  const videoAttempts = scenes.reduce(
    (acc, s) => (s.final_video_url || s.raw_video_url || s.video_gen_failures ? acc + 1 + (s.video_gen_failures ?? 0) : acc),
    0,
  );
  const rewrites = state.rewrite_count ?? 0;

  const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const ss = String(elapsed % 60).padStart(2, "0");

  const healthClass = (st?: string) => {
    const v = (st ?? "").toLowerCase();
    if (v.includes("healthy") || v.includes("closed")) return "ok";
    if (v.includes("half")) return "warn";
    if (v.includes("open") || v.includes("down") || v.includes("fail")) return "bad";
    return "idle";
  };
  const healthEntries = Object.entries(health).slice(0, 4);

  if (!threadId) return null;

  return (
    <div className="cost-bar" aria-label="运行成本与额度仪表">
      <span className="cost-chip" title="本次任务已运行时间">
        ⏱ {mm}:{ss}
      </span>
      {total > 0 && (
        <>
          <span className="cost-chip" title={`镜头完成进度（失败 ${failed}）`}>
            🎞 {done}/{total} 镜{failed > 0 ? ` · 失败 ${failed}` : ""}
          </span>
          <span className="cost-chip" title="视频生成尝试次数（含失败重试），即额度消耗代理">
            🎬 视频生成 ×{videoAttempts}
          </span>
        </>
      )}
      {rewrites > 0 && (
        <span className="cost-chip" title="人工/自动触发的重写次数">
          ♻ 重写 ×{rewrites}
        </span>
      )}
      {healthEntries.length > 0 && (
        <span className="cost-health" title="各生成后端熔断健康度">
          {healthEntries.map(([name, h]) => (
            <span key={name} className={`cost-dot cost-dot-${healthClass(h.state)}`} title={`${name}: ${h.state ?? "unknown"}`}>
              {name}
            </span>
          ))}
        </span>
      )}
    </div>
  );
}
