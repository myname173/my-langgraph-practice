import { useEffect, useState } from "react";

/**
 * 开场动画（Splash Screen）。
 * 设计参考社区开源实现思路（react-gradient-animation 的极光流动背景、react-fade-in 的逐元素淡入），
 * 但完全自包含、零依赖，与项目现有玻璃拟态 + 紫蓝渐变风格统一。
 *
 * 行为：
 * - 首次挂载播放约 2.2s，随后整体淡出（fade-out）并回调 onDone。
 * - 用 localStorage 记忆「已看过」，避免每次刷新都弹；清缓存或首次访问才出现。
 * - 支持按 Esc / 点击跳过。
 */
const SEEN_KEY = "splash_seen_v1";
const PLAY_MS = 2200;
/** 后端健康检查端点（由 vite 代理 /healthz → 媒体服务 /health） */
const HEALTH_URL = "/healthz";

type Health = "checking" | "ok" | "down";

export function SplashScreen({ onDone }: { onDone: () => void }) {
  const [leaving, setLeaving] = useState(false);
  const [ready, setReady] = useState(false);
  const [health, setHealth] = useState<Health>("checking");

  // 加载完成后进入「可进入」状态；点击或按 Enter 进入，Esc 跳过
  useEffect(() => {
    const t = window.setTimeout(() => setReady(true), PLAY_MS);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setLeaving(true);
      } else if (e.key === "Enter" && ready) {
        setLeaving(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.clearTimeout(t);
      window.removeEventListener("keydown", onKey);
    };
  }, [ready]);

  // 探测后端健康：开场动画期间并发检查服务是否就绪
  useEffect(() => {
    let alive = true;
    const ctrl = new AbortController();
    fetch(HEALTH_URL, { signal: ctrl.signal })
      .then((r) => {
        if (!alive) return;
        setHealth(r.ok ? "ok" : "down");
      })
      .catch(() => {
        if (alive) setHealth("down");
      });
    return () => {
      alive = false;
      ctrl.abort();
    };
  }, []);

  // 淡出动画结束后真正卸载
  const handleAnimEnd = () => {
    if (leaving) {
      try {
        localStorage.setItem(SEEN_KEY, "1");
      } catch {
        /* 忽略隐私模式等写入失败 */
      }
      onDone();
    }
  };

  const enter = () => {
    if (ready) setLeaving(true);
  };

  return (
    <div
      className={`splash ${leaving ? "splash-leave" : ""} ${ready && !leaving ? "splash-ready" : ""}`}
      onClick={enter}
      onAnimationEnd={handleAnimEnd}
      role="presentation"
    >
      <div className="splash-aurora" aria-hidden>
        <span />
        <span />
        <span />
      </div>

      <div className="splash-inner">
        <div className="splash-logo">◈</div>
        <h1 className="splash-title">AI 多模态宣传片生成 Agent</h1>
        <p className="splash-sub">
          一句话创意 → 剧本 · 视觉 · 镜头 · 评审 · 成片
        </p>
        <div className="splash-bar">
          <span className="splash-bar-fill" />
        </div>
        <div className={`splash-health splash-health-${health}`}>
          <span className="splash-health-dot" />
          <span className="splash-health-text">
            {health === "checking"
              ? "正在连接后端服务…"
              : health === "ok"
                ? "后端服务已就绪"
                : "未能连接到后端服务"}
          </span>
        </div>
        <p className="splash-hint">
          {ready ? "点击任意处或按 Enter 进入" : "加载中…（点击或按 Esc 跳过）"}
        </p>
      </div>
    </div>
  );
}

/** 是否应播放开场动画：首次访问（未标记看过）才播放 */
export function shouldPlaySplash(): boolean {
  // 每次刷新都播放开场动画
  return true;
}
