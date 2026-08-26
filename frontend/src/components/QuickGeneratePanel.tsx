import { useState } from "react";

type GenType = "video" | "image";

interface GenResult {
  url: string;
  type: GenType;
  remote?: boolean;
}

export function QuickGeneratePanel() {
  const [prompt, setPrompt] = useState("");
  const [type, setType] = useState<GenType>("video");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<GenResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canGenerate = prompt.trim().length > 0 && !busy;

  async function handleGenerate() {
    if (!canGenerate) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const resp = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt.trim(), type }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        // 402 = 额度耗尽；其余按错误信息展示
        throw new Error(data.detail || `生成失败（HTTP ${resp.status}）`);
      }
      setResult({ url: data.url, type: data.type, remote: data.remote });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="task-launcher quick-gen">
      <h3>
        <span className="section-icon">⚡</span>快速生成
      </h3>
      <p className="quick-gen-hint">
        输入一句话，直接生成单条视频或图片（使用免费额度模型）。
      </p>

      {/* 类型切换 */}
      <div className="seg">
        <button
          className={`seg-btn ${type === "video" ? "seg-active" : ""}`}
          onClick={() => setType("video")}
          type="button"
        >
          🎬 视频
        </button>
        <button
          className={`seg-btn ${type === "image" ? "seg-active" : ""}`}
          onClick={() => setType("image")}
          type="button"
        >
          🖼️ 图片
        </button>
      </div>

      {/* 文字输入 */}
      <textarea
        className="quick-gen-input"
        rows={3}
        placeholder="例如：一只橘猫在窗台上看夕阳，微风拂过窗帘"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void handleGenerate();
        }}
      />

      {/* 生成按钮 */}
      <button
        className="btn btn-generate"
        disabled={!canGenerate}
        onClick={() => void handleGenerate()}
        type="button"
      >
        {busy ? (
          <>
            <span className="spinner" /> 生成中…
          </>
        ) : (
          "立即生成"
        )}
      </button>

      {/* 错误横幅 */}
      {error && (
        <div className="quick-gen-error">
          {error.includes("额度") ? "⚠️ 免费额度已耗尽：" : "⛔ 生成失败："}
          {error.replace(/^(⚠️ 免费额度已耗尽：|⛔ 生成失败：)/, "")}
        </div>
      )}

      {/* 产物预览 */}
      {busy && !error && (
        <div className="quick-gen-skeleton">
          <div className="skeleton-box" />
          <span>正在调用模型渲染，请稍候…</span>
        </div>
      )}

      {result && !busy && (
        <div className="quick-gen-result">
          {result.type === "image" ? (
            <img src={result.url} alt="生成结果" className="quick-gen-media" />
          ) : (
            <video src={result.url} controls className="quick-gen-media" />
          )}
          {result.remote && (
            <span className="quick-gen-note">
              预览链接带时效，建议尽快下载保存。
            </span>
          )}
        </div>
      )}
    </section>
  );
}
