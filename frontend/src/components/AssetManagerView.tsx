import { useRef, useState } from "react";
import type { AssetCategory } from "../types";
import { AssetPicker } from "./AssetPicker";

type Cat = Exclude<AssetCategory, "ignore">;

/** 待登记（命名后入库）的素材项 */
interface DraftAsset {
  /** 前端临时 id */
  key: string;
  /** 图的真实可访问 url（/media/...） */
  url: string;
  /** 展示用文件名 */
  fileLabel: string;
  name: string;
  category: Cat;
  description: string;
  source: "upload" | "library";
}

const CATEGORY_COLOR: Record<Cat, string> = {
  character: "#8B5CF6",
  prop: "#F59E0B",
  environment: "#22C55E",
};

const _ALLOWED_EXT = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"];

export function AssetManagerView({ story }: { story?: string }) {
  const [drafts, setDrafts] = useState<DraftAsset[]>([]);
  const [uploading, setUploading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState<string>("");
  const fileRef = useRef<HTMLInputElement | null>(null);
  // 从素材库挑选时，把勾选的资产名(role_name)记录下来
  const [pickedLib, setPickedLib] = useState<Set<string>>(new Set());
  const [pickedUrls, setPickedUrls] = useState<Record<string, string[]>>({});
  const [catOverride, setCatOverride] = useState<Record<string, Cat>>({});

  const flash = (msg: string) => {
    setToast(msg);
    window.setTimeout(() => setToast(""), 2600);
  };

  // 上传本地图：直接拿到 url，加入待命名草稿
  const handleUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setUploading(true);
    try {
      const fd = new FormData();
      for (const f of Array.from(files)) fd.append("files", f);
      if (story) fd.append("story", story);
      const r = await fetch("/assets/upload", { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok || !data.ok) throw new Error(data.detail || "上传失败");
      const next: DraftAsset[] = data.files.map((f: { url: string; filename: string }) => ({
        key: `${Date.now()}_${Math.random().toString(36).slice(2)}`,
        url: f.url,
        fileLabel: f.filename,
        name: "",
        category: "character",
        description: "",
        source: "upload",
      }));
      setDrafts((d) => [...d, ...next]);
      flash(`已上传 ${next.length} 张，请在右侧命名后登记`);
    } catch (e) {
      flash(`上传失败：${(e as Error).message}`);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  // 从素材库勾选 -> 把被勾选资产加入草稿（库里图可能多张，合并为该资产名一条草稿）
  const onPickToggle = (name: string, urls: string[], checked: boolean) => {
    setPickedLib((s) => {
      const n = new Set(s);
      if (checked) n.add(name);
      else n.delete(name);
      return n;
    });
    setPickedUrls((m) => {
      const n = { ...m };
      if (checked) n[name] = urls;
      else delete n[name];
      return n;
    });
  };

  const addPickedToDrafts = () => {
    const next: DraftAsset[] = [];
    for (const name of pickedLib) {
      const urls = pickedUrls[name] || [];
      const first = urls[0];
      if (!first) continue;
      next.push({
        key: `lib_${name}_${Date.now()}`,
        url: first,
        fileLabel: name,
        name,
        category: catOverride[name] ?? "character",
        description: "",
        source: "library",
      });
    }
    if (next.length === 0) {
      flash("请先在左侧勾选要加入的素材");
      return;
    }
    setDrafts((d) => [...d, ...next]);
    flash(`已加入 ${next.length} 项，请在右侧命名后登记`);
  };

  const updateDraft = (key: string, patch: Partial<DraftAsset>) =>
    setDrafts((d) => d.map((x) => (x.key === key ? { ...x, ...patch } : x)));

  const removeDraft = (key: string) =>
    setDrafts((d) => d.filter((x) => x.key !== key));

  // 批量登记入库：逐条调用 /assets/register
  const saveAll = async () => {
    const unnamed = drafts.filter((d) => !d.name.trim());
    if (unnamed.length) {
      flash("请先为每张素材填写名称");
      return;
    }
    setSaving(true);
    let ok = 0;
    for (const d of drafts) {
      try {
        const r = await fetch("/assets/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            url: d.url,
            name: d.name.trim(),
            category: d.category,
            description: d.description.trim(),
            story: story ?? "",
          }),
        });
        const data = await r.json();
        if (r.ok && data.ok) ok += 1;
      } catch {
        /* 单条失败不影响其他 */
      }
    }
    setSaving(false);
    if (ok > 0) {
      setDrafts([]);
      setPickedLib(new Set());
      setPickedUrls({});
      flash(`已登记 ${ok} 张素材到素材库`);
    } else {
      flash("登记失败，请确认后端媒体服务已启动");
    }
  };

  return (
    <div className="asset-manager">
      <div className="am-head">
        <h2>素材库管理</h2>
        <p className="panel-hint">
          在这里上传你自己的角色 / 道具 / 场景素材，或从已有素材库挑选，命名后「登记入库」。
          登记后的素材会出现在「创建任务 → 已有素材库」中，并在生成视频时作为
          <b>一致性参考锚点</b>参与（不会直接生成视频）。
        </p>
      </div>

      <div className="am-body">
        {/* 左：添加素材（上传 / 选库） */}
        <div className="am-col am-col-add">
          <section className="am-section">
            <h3>1 · 上传本地素材</h3>
            <div
              className="au-drop"
              onClick={() => fileRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                handleUpload(e.dataTransfer.files);
              }}
            >
              <input
                ref={fileRef}
                type="file"
                accept={_ALLOWED_EXT.join(",")}
                multiple
                hidden
                onChange={(e) => handleUpload(e.target.files)}
              />
              <div className="au-drop-inner">
                <span className="au-drop-icon">＋</span>
                <p>点击或拖拽图片到此处上传</p>
                <p className="panel-hint">支持 {_ALLOWED_EXT.join(" / ")}，单张 ≤ 10MB</p>
              </div>
            </div>
            {uploading && <div className="panel-hint">上传中…</div>}
          </section>

          <section className="am-section">
            <h3>2 · 从已有素材库选择</h3>
            <AssetPicker
              selected={pickedLib}
              onToggle={onPickToggle}
              categoryOverride={catOverride}
              onCategoryChange={(name, cat) =>
                setCatOverride((m) => ({ ...m, [name]: cat }))
              }
            />
            <button className="btn-ghost am-add-lib" onClick={addPickedToDrafts}>
              把勾选的加入待命名列表 →
            </button>
          </section>
        </div>

        {/* 右：待命名 + 登记 */}
        <div className="am-col am-col-edit">
          <section className="am-section">
            <h3>3 · 命名并登记入库（{drafts.length}）</h3>
            {drafts.length === 0 ? (
              <div className="panel-hint">
                暂无待登记素材。请先在左侧上传或选择素材。
              </div>
            ) : (
              <div className="am-draft-list">
                {drafts.map((d) => (
                  <div key={d.key} className="am-draft-card">
                    <img src={d.url} alt={d.fileLabel} className="am-draft-thumb" />
                    <div className="am-draft-fields">
                      <div className="am-row">
                        <input
                          className="am-name"
                          placeholder="素材名称（需与剧本称呼一致）"
                          value={d.name}
                          onChange={(e) => updateDraft(d.key, { name: e.target.value })}
                        />
                        <select
                          value={d.category}
                          onChange={(e) =>
                            updateDraft(d.key, { category: e.target.value as Cat })
                          }
                          style={{ borderLeftColor: CATEGORY_COLOR[d.category] }}
                        >
                          <option value="character">角色</option>
                          <option value="prop">道具</option>
                          <option value="environment">场景</option>
                        </select>
                      </div>
                      <input
                        className="am-desc"
                        placeholder="可选描述（如：紫衣、长剑、仙气缭绕）"
                        value={d.description}
                        onChange={(e) => updateDraft(d.key, { description: e.target.value })}
                      />
                      <span className="am-src">
                        来源：{d.source === "upload" ? "本地上传" : "素材库"} · {d.fileLabel}
                      </span>
                    </div>
                    <button
                      className="am-remove"
                      title="移除"
                      onClick={() => removeDraft(d.key)}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
            <div className="am-actions">
              <button className="btn-primary" disabled={saving || drafts.length === 0} onClick={saveAll}>
                {saving ? "登记中…" : "登记入库"}
              </button>
            </div>
          </section>
        </div>
      </div>

      {toast && <div className="am-toast">{toast}</div>}
    </div>
  );
}
