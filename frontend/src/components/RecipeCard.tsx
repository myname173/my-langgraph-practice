// frontend/src/components/RecipeCard.tsx
/**
 * P2-4 镜头配方卡（Recipe Card）面板。
 *
 * 一次展示整任务的「镜头配方」：
 *   - 左：镜头列表（可切换）
 *   - 右：所选镜头的完整配方（可读）+ 微调输入（首帧/尾帧/视频 prompt、参考强度、时长）
 *   - 底部：复制 JSON / 复制 Markdown / 另存文件 / 应用微调并重跑
 *
 * 纯前端 + 后端只读接口，不直接触发生图；只有「应用微调并重跑」会写回 checkpoint
 * 并从该镜重跑（由父组件负责启动续跑）。
 */
import { useEffect, useMemo, useState } from "react";
import type { TaskRecipe } from "../types";
import {
  applyRecipe,
  saveRecipe,
  type RecipePatch,
} from "../lib/recipeClient";

interface Props {
  threadId: string;
  recipe: TaskRecipe;
  markdown: string;
  /** 打开时默认聚焦的镜头下标（从某张镜头卡点进来时用） */
  initialShot?: number;
  /** 是否允许「应用微调并重跑」（任务未完成/可续跑时为 true） */
  canApply?: boolean;
  onClose: () => void;
  /** 微调应用成功后的回调（父组件据此发起续跑 / 刷新状态） */
  onApplied?: (sceneIndex: number, result: unknown) => void;
}

function copyText(text: string): Promise<boolean> {
  // 优先用 Clipboard API；沙箱/无权限时回退 textarea + execCommand。
  return new Promise((resolve) => {
    try {
      if (navigator.clipboard?.writeText) {
        navigator.clipboard.writeText(text).then(
          () => resolve(true),
          () => resolve(false),
        );
        return;
      }
    } catch {
      /* fallthrough */
    }
    resolve(false);
  });
}

const fmt = (v: unknown): string =>
  v === null || v === undefined || v === "" ? "—" : String(v);

export function RecipeCard({
  threadId,
  recipe,
  markdown,
  initialShot = 0,
  canApply = true,
  onClose,
  onApplied,
}: Props) {
  const shots = recipe.shots ?? [];
  const [shotIdx, setShotIdx] = useState(
    shots.some((s) => s.index === initialShot) ? initialShot : shots[0]?.index ?? 0,
  );
  const shot = useMemo(() => shots.find((s) => s.index === shotIdx) ?? null, [shots, shotIdx]);

  const [imagePrompt, setImagePrompt] = useState("");
  const [lastPrompt, setLastPrompt] = useState("");
  const [videoPrompt, setVideoPrompt] = useState("");
  const [refStrength, setRefStrength] = useState("");
  const [duration, setDuration] = useState("");
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [showMd, setShowMd] = useState(false);

  // 切换镜头时把编辑框重置为该镜当前配方值
  useEffect(() => {
    if (!shot) return;
    setImagePrompt(shot.image?.prompt ?? "");
    setLastPrompt(shot.end_frame?.prompt ?? "");
    setVideoPrompt(shot.video?.prompt ?? "");
    setRefStrength(
      shot.image?.ref_strength === null || shot.image?.ref_strength === undefined
        ? ""
        : String(shot.image.ref_strength),
    );
    setDuration(
      shot.video?.duration === null || shot.video?.duration === undefined
        ? ""
        : String(shot.video.duration),
    );
    setToast(null);
  }, [shot]);

  const recipeJson = useMemo(() => JSON.stringify(recipe, null, 2), [recipe]);

  const doCopy = async (text: string, label: string) => {
    const ok = await copyText(text);
    setToast(ok ? `${label}已复制到剪贴板` : `复制受限，请手动选择文本复制（${label}）`);
  };

  const handleSave = async () => {
    setBusy(true);
    try {
      const r = await saveRecipe(threadId);
      setToast(`已另存：${r.latest}`);
    } catch (e) {
      setToast(`另存失败：${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  const buildPatch = (): RecipePatch => {
    const patch: RecipePatch = {};
    const img: RecipePatch["image"] = {};
    if (shot && imagePrompt.trim() && imagePrompt.trim() !== (shot.image?.prompt ?? "")) {
      img.prompt = imagePrompt.trim();
    }
    if (refStrength.trim()) {
      const n = Number(refStrength);
      if (!Number.isNaN(n) && n >= 0 && n <= 1 && n !== shot?.image?.ref_strength) {
        img.ref_strength = n;
      }
    }
    if (Object.keys(img).length) patch.image = img;

    const end: RecipePatch["end_frame"] = {};
    if (shot && lastPrompt.trim() && lastPrompt.trim() !== (shot.end_frame?.prompt ?? "")) {
      end.prompt = lastPrompt.trim();
    }
    if (Object.keys(end).length) patch.end_frame = end;

    const vid: RecipePatch["video"] = {};
    if (shot && videoPrompt.trim() && videoPrompt.trim() !== (shot.video?.prompt ?? "")) {
      vid.prompt = videoPrompt.trim();
    }
    if (duration.trim()) {
      const n = Number(duration);
      if (!Number.isNaN(n) && n > 0 && n !== shot?.video?.duration) vid.duration = n;
    }
    if (Object.keys(vid).length) patch.video = vid;

    return patch;
  };

  const patch = buildPatch();
  const hasPatch = Object.keys(patch).length > 0;

  const handleApply = async () => {
    if (!shot || !hasPatch) return;
    setBusy(true);
    try {
      const r = await applyRecipe(threadId, shot.index, patch);
      setToast("微调已应用，正在从该镜重跑…");
      onApplied?.(shot.index, r);
    } catch (e) {
      setToast(`应用失败：${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="recipe-overlay" role="dialog" aria-modal="true" aria-label="镜头配方卡">
      <div className="recipe-panel">
        <header className="recipe-head">
          <div>
            <div className="recipe-kicker">镜头配方卡 · Shot Recipe</div>
            <h3>{recipe.task || "(未命名任务)"}</h3>
            <div className="recipe-sub">
              thread <code>{threadId}</code> · {shots.length} 个镜头 · 生成于 {fmt(recipe.created_at)}
            </div>
          </div>
          <button type="button" className="btn btn-ghost btn-sm" onClick={onClose}>
            关闭 ✕
          </button>
        </header>

        <div className="recipe-meta">
          <span className="recipe-chip">风格 {fmt(recipe.style?.visual_style)}</span>
          <span className="recipe-chip">调色 {fmt(recipe.post?.color_preset)}</span>
          <span className="recipe-chip">字幕 {fmt(recipe.post?.subtitle_theme)}</span>
          <span className="recipe-chip">包装卡 {recipe.post?.title_card ? "开" : "关"}</span>
        </div>

        <div className="recipe-body">
          <nav className="recipe-shotlist">
            {shots.map((s) => (
              <button
                key={s.index}
                type="button"
                className={`recipe-shot-btn${s.index === shotIdx ? " active" : ""}`}
                onClick={() => setShotIdx(s.index)}
              >
                <strong>镜头 {s.index + 1}</strong>
                <span>{s.script ? s.script.slice(0, 18) : "—"}</span>
              </button>
            ))}
          </nav>

          <div className="recipe-detail">
            {shot ? (
              <>
                <label className="recipe-field">
                  <span>首帧 Prompt（image）</span>
                  <textarea
                    value={imagePrompt}
                    rows={3}
                    onChange={(e) => setImagePrompt(e.target.value)}
                  />
                </label>
                <label className="recipe-field">
                  <span>尾帧 Prompt（end_frame）</span>
                  <textarea
                    value={lastPrompt}
                    rows={2}
                    onChange={(e) => setLastPrompt(e.target.value)}
                  />
                </label>
                <label className="recipe-field">
                  <span>生视频 Prompt（video）</span>
                  <textarea
                    value={videoPrompt}
                    rows={3}
                    onChange={(e) => setVideoPrompt(e.target.value)}
                  />
                </label>
                <div className="recipe-row">
                  <label className="recipe-field recipe-field-inline">
                    <span>首帧参考强度 ref_strength（0–1）</span>
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.05}
                      value={refStrength}
                      onChange={(e) => setRefStrength(e.target.value)}
                    />
                  </label>
                  <label className="recipe-field recipe-field-inline">
                    <span>时长 duration（秒）</span>
                    <input
                      type="number"
                      min={1}
                      step={0.5}
                      value={duration}
                      onChange={(e) => setDuration(e.target.value)}
                    />
                  </label>
                </div>

                <div className="recipe-readonly">
                  <div className="recipe-ro-row">
                    <span>首帧后端</span>
                    <b>{fmt(shot.image?.backend)}</b>
                  </div>
                  <div className="recipe-ro-row">
                    <span>首帧尺寸</span>
                    <b>{fmt(shot.image?.size)}</b>
                  </div>
                  <div className="recipe-ro-row">
                    <span>首帧参考图</span>
                    <b>{(shot.image?.ref_images ?? []).length} 张</b>
                  </div>
                  <div className="recipe-ro-row">
                    <span>视频后端 / 档位</span>
                    <b>
                      {fmt(shot.video?.backend)} / {fmt(shot.video?.quality_tier)}
                    </b>
                  </div>
                  <div className="recipe-ro-row">
                    <span>首尾帧双控</span>
                    <b>{shot.video?.use_first_last_frame ? "是" : "否"}</b>
                  </div>
                  <div className="recipe-ro-row">
                    <span>修正痕迹</span>
                    <b>
                      {fmt(shot.fix?.fix_type)}
                      {shot.fix?.extend_applied ? " · 已延长" : ""}
                    </b>
                  </div>
                </div>
              </>
            ) : (
              <p className="hint">该任务暂无可用配方（可能尚未生成任何镜头）。</p>
            )}
          </div>
        </div>

        {showMd && <pre className="recipe-md">{markdown}</pre>}

        {toast && <div className="recipe-toast">{toast}</div>}

        <footer className="recipe-actions">
          <button type="button" className="btn btn-ghost btn-sm" disabled={busy} onClick={() => doCopy(recipeJson, "配方 JSON")}>
            复制配方 JSON
          </button>
          <button type="button" className="btn btn-ghost btn-sm" disabled={busy} onClick={() => doCopy(markdown, "配方 Markdown")}>
            复制 Markdown
          </button>
          <button type="button" className="btn btn-ghost btn-sm" disabled={busy} onClick={() => setShowMd((v) => !v)}>
            {showMd ? "隐藏全文" : "查看全文"}
          </button>
          <button type="button" className="btn btn-ghost btn-sm" disabled={busy} onClick={() => void handleSave()}>
            另存文件
          </button>
          {canApply && (
            <button
              type="button"
              className="btn btn-primary btn-sm"
              disabled={busy || !hasPatch}
              title={hasPatch ? "把上方改动写回该镜并从这里重跑" : "先修改上方任一字段再应用"}
              onClick={() => void handleApply()}
            >
              {busy ? "处理中…" : "应用微调并重跑此镜"}
            </button>
          )}
        </footer>
      </div>
    </div>
  );
}
