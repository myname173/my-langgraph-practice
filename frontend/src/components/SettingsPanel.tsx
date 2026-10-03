import { useCallback, useEffect, useState } from "react";

interface Props {
  onClose: () => void;
}

type FieldKind = "text" | "secret" | "select";

interface FieldDef {
  key: string;
  label: string;
  kind: FieldKind;
  group: string;
  placeholder?: string;
  hint?: string;
  options?: { value: string; label: string }[];
}

const BACKEND_OPTIONS = [
  { value: "dashscope", label: "dashscope · 阿里通义（万相）" },
  { value: "jimeng", label: "jimeng · 即梦（免费积分）" },
  { value: "google", label: "google · Gemini / Veo" },
  { value: "siliconflow", label: "siliconflow · 硅基流动" },
];

const TIER_OPTIONS = [
  { value: "", label: "standard · 默认档" },
  { value: "fast", label: "fast · 秒级草稿，省额度" },
  { value: "standard", label: "standard · 默认档" },
  { value: "cinema", label: "cinema · 高质感" },
];

/** 与后端 tts.EDGE_TTS_VOICES_ZH 对齐；空值 = 跟随任务启动面板的选择 */
const VOICE_OPTIONS = [
  { value: "", label: "（跟随任务启动面板的选择）" },
  { value: "xiaoxiao", label: "xiaoxiao · 女声 温柔叙事" },
  { value: "yunyang", label: "yunyang · 男声 沉稳解说" },
  { value: "xiaoyi", label: "xiaoyi · 女声 轻快" },
  { value: "yunxi", label: "yunxi · 男声 活泼" },
  { value: "xiaochen", label: "xiaochen · 女声 成熟知性" },
  { value: "xiaomo", label: "xiaomo · 女声 情感丰富" },
];

const FIELDS: FieldDef[] = [
  // ── 后端与档位 ──
  { key: "MULTIMODIA_BACKEND", label: "生图/视频后端", kind: "select", group: "后端与档位", options: BACKEND_OPTIONS },
  { key: "QUALITY_TIER", label: "质量档位", kind: "select", group: "后端与档位", options: TIER_OPTIONS },
  { key: "VIDEO_PROVIDER", label: "视频后端固定覆盖", kind: "text", group: "后端与档位",
    placeholder: "留空=按档位自动；agnes / zhipu / jimeng / dashscope",
    hint: "优先级最高；留空时按质量档位方案自动选择并在后端间容错" },
  // ── API 密钥 ──
  { key: "DASHSCOPE_API_KEY", label: "DashScope API Key", kind: "secret", group: "API 密钥",
    hint: "阿里云百炼；生图/视频/文本 LLM 主通道" },
  { key: "ZHIPU_API_KEY", label: "智谱 API Key", kind: "secret", group: "API 密钥",
    hint: "zhipu-flash 视频（fast 档首选，秒级）" },
  { key: "AGNES_API_KEY", label: "Agnes 凭证", kind: "secret", group: "API 密钥",
    hint: "agnes 免费档；限速 1 次/分钟" },
  { key: "JIMENG_SESSION_ID", label: "即梦 Session ID", kind: "secret", group: "API 密钥",
    placeholder: "即梦网页版 cookie sessionid；多个用英文逗号分隔",
    hint: "即梦每日免费积分（伪装 OpenAI 兼容本地 API）" },
  { key: "GEMINI_API_KEY", label: "Google AI Studio Key", kind: "secret", group: "API 密钥",
    hint: "Imagen / Veo 替代后端" },
  { key: "SILICONFLOW_API_KEY", label: "SiliconFlow API Key", kind: "secret", group: "API 密钥",
    hint: "硅基流动 img2img 锚点生图" },
  { key: "OPENAI_API_KEY", label: "文本 LLM Key", kind: "secret", group: "API 密钥",
    placeholder: "OpenAI 兼容 Key（默认走 DashScope）",
    hint: "分镜/导演/审核等文本推理所用" },
  // ── 视频模型 ──
  { key: "AGNES_VIDEO_MODEL", label: "Agnes 视频模型", kind: "text", group: "视频模型",
    placeholder: "agnes-video-v2.0" },
  { key: "ZHIPU_VIDEO_MODEL", label: "智谱视频模型", kind: "text", group: "视频模型",
    placeholder: "如 cogvideox-flash / cogvideox-2" },
  { key: "DASHSCOPE_VIDEO_MODEL", label: "百炼视频模型（通义万相）", kind: "text", group: "视频模型",
    placeholder: "如 wan2.7-t2v-2026-06-12 / wan3.0-video / wan2.2-i2v-plus / wan2.7-r2v",
    hint: "任意百炼 wan 系列模型名均可：按名称自动路由 t2v / i2v / r2v 链，失败自动回落默认链" },
  { key: "GEMINI_VIDEO_MODEL", label: "Veo 视频模型", kind: "text", group: "视频模型",
    placeholder: "veo-3.1-generate-preview" },
  // ── 生图模型 ──
  { key: "GEMINI_IMAGE_MODEL", label: "Imagen 生图模型", kind: "text", group: "生图模型",
    placeholder: "gemini-2.5-flash-image" },
  { key: "SILICONFLOW_IMAGE_MODEL", label: "硅基生图模型", kind: "text", group: "生图模型",
    placeholder: "Kwai-Kolors/Kolors" },
  { key: "DASHSCOPE_IMAGE_MODEL", label: "通义生图模型", kind: "text", group: "生图模型",
    placeholder: "如 wanx2.1-t2i-turbo" },
  // ── 文本 LLM ──
  { key: "DASHSCOPE_TEXT_MODEL", label: "通义文本模型", kind: "text", group: "文本 LLM",
    placeholder: "如 qwen-max / qwen-plus" },
  { key: "SILICONFLOW_TEXT_MODEL", label: "硅基文本模型", kind: "text", group: "文本 LLM",
    placeholder: "如 deepseek-ai/DeepSeek-V3" },
  { key: "OPENAI_BASE_URL", label: "LLM 接口地址", kind: "text", group: "文本 LLM",
    placeholder: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    hint: "OpenAI 兼容地址，可指向任意中转/本地服务" },
  // ── 语音 / 字幕 ──
  { key: "TTS_VOICE", label: "默认配音音色", kind: "select", group: "语音 / 字幕",
    options: VOICE_OPTIONS,
    hint: "edge-tts 免费引擎，无需密钥；任务启动面板仍可按次覆盖" },
  // ── 生成参数 ──
  { key: "VIDEO_DURATION", label: "默认镜头时长(秒)", kind: "text", group: "生成参数",
    placeholder: "8" },
  { key: "JIMENG_IMAGE_RESOLUTION", label: "即梦生图分辨率", kind: "text", group: "生成参数",
    placeholder: "如 1280x720" },
  { key: "AGNES_MIN_SUBMIT_GAP", label: "Agnes 提交间隔(秒)", kind: "text", group: "生成参数",
    placeholder: "60", hint: "并行模式下多镜头竞争 agnes 时的限速错峰" },
];

const GROUP_ORDER = [
  "后端与档位",
  "API 密钥",
  "视频模型",
  "生图模型",
  "文本 LLM",
  "语音 / 字幕",
  "生成参数",
];

interface HealthInfo {
  state?: string;
  consecutive_failures?: number;
  cooldown_remaining_s?: number;
}

const HEALTH_LABEL: Record<string, string> = {
  healthy: "健康",
  down: "已熔断",
  unknown: "未使用",
};

export function SettingsPanel({ onClose }: Props) {
  // GET 返回的当前值（密钥为打码形态）；edited 仅存用户改动
  const [current, setCurrent] = useState<Record<string, string>>({});
  const [edited, setEdited] = useState<Record<string, string>>({});
  const [cleared, setCleared] = useState<Set<string>>(new Set());
  const [health, setHealth] = useState<Record<string, HealthInfo>>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [probing, setProbing] = useState(false);
  const [probeResult, setProbeResult] = useState<{ ok: boolean; text: string } | null>(null);
  const [probeField, setProbeField] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/settings");
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      setCurrent(data.settings ?? {});
      setHealth(data.backend_health ?? {});
      setEdited({});
      setCleared(new Set());
      setMsg(null);
    } catch (e) {
      setMsg({ ok: false, text: "加载设置失败：" + String(e) });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const setField = useCallback((key: string, value: string) => {
    setEdited((prev) => ({ ...prev, [key]: value }));
  }, []);

  const save = useCallback(async () => {
    setSaving(true);
    setMsg(null);
    try {
      const values: Record<string, string> = {};
      for (const f of FIELDS) {
        if (f.kind === "secret") {
          if (cleared.has(f.key)) values[f.key] = ""; // 显式清除 → 写空（回退 .env 默认）
          else if ((edited[f.key] ?? "").trim() !== "") values[f.key] = edited[f.key].trim();
        } else if (edited[f.key] !== undefined && edited[f.key] !== current[f.key]) {
          values[f.key] = edited[f.key];
        }
      }
      if (!Object.keys(values).length) {
        setMsg({ ok: true, text: "没有改动，无需保存。" });
        return;
      }
      const res = await fetch("/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ values }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail.slice(0, 200));
      }
      const data = await res.json();
      setMsg({
        ok: true,
        text: "已保存并即时生效：" + (data.applied_keys ?? []).join("、"),
      });
      await load();
    } catch (e) {
      setMsg({ ok: false, text: "保存失败：" + String(e) });
    } finally {
      setSaving(false);
    }
  }, [edited, cleared, current, load]);

  // 生图/视频后端连通测试映射：字段 → probe 参数（填 Key → 测试 → 通了参与生图 fallback）
  const PROBE_BY_FIELD: Record<string, { kind: string; apiKeyField: string }> = {
    DASHSCOPE_VIDEO_MODEL: { kind: "dashscope-video", apiKeyField: "DASHSCOPE_API_KEY" },
    DASHSCOPE_IMAGE_MODEL: { kind: "dashscope-image", apiKeyField: "DASHSCOPE_API_KEY" },
    SILICONFLOW_IMAGE_MODEL: { kind: "siliconflow-image", apiKeyField: "SILICONFLOW_API_KEY" },
  };

  /** 零额度探测：验证「模型名 + API Key」可达性（不创建任务、不消耗额度） */
  const runProbe = useCallback(async (fieldKey: string) => {
    const conf = PROBE_BY_FIELD[fieldKey];
    if (!conf) return;
    setProbing(true);
    setProbeField(fieldKey);
    setProbeResult(null);
    try {
      const model = (edited[fieldKey] ?? current[fieldKey] ?? "").trim();
      const apiKey = (edited[conf.apiKeyField] ?? "").trim(); // 仅当用户刚输入新 Key 时随探测发送
      const res = await fetch("/settings/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: conf.kind, model, api_key: apiKey }),
      });
      const data = await res.json();
      if (!res.ok) {
        setProbeResult({ ok: false, text: String(data.detail ?? "HTTP " + res.status) });
        return;
      }
      setProbeResult({ ok: !!data.ok, text: String(data.detail ?? "") });
    } catch (e) {
      setProbeResult({ ok: false, text: "探测请求失败：" + String(e) });
    } finally {
      setProbing(false);
    }
  }, [edited, current]);

  const dirtyCount =
    Object.keys(edited).filter((k) => edited[k] !== current[k]).length + cleared.size;

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="settings-head">
          <h3>
            <span className="section-icon">⚙️</span>生成引擎设置
            <span className="settings-sub">
              密钥与模型名写入 data/user_settings.json，各进程即时生效
            </span>
          </h3>
          <button className="btn btn-ghost" onClick={onClose}>
            关闭
          </button>
        </div>

        {Object.keys(health).length > 0 && (
          <div className="settings-health">
            {Object.entries(health).map(([name, h]) => {
              const cool = h.cooldown_remaining_s ?? 0;
              return (
                <span
                  key={name}
                  className={"meta-chip settings-health-chip health-" + (h.state ?? "unknown")}
                  title={
                    "连续失败 " +
                    (h.consecutive_failures ?? 0) +
                    " 次" +
                    (cool > 0 ? "；冷却剩余 " + cool + "s" : "")
                  }
                >
                  {name}: {HEALTH_LABEL[h.state ?? "unknown"] ?? h.state}
                  {cool > 0 ? " (" + Math.round(cool) + "s)" : ""}
                </span>
              );
            })}
          </div>
        )}

        {loading && <div className="settings-loading">加载中…</div>}

        {!loading &&
          GROUP_ORDER.map((group) => {
            const fields = FIELDS.filter((f) => f.group === group);
            if (!fields.length) return null;
            return (
              <fieldset key={group} className="settings-group">
                <legend>{group}</legend>
                {fields.map((f) => {
                  const src = current[f.key + "__source"] ?? "";
                  const cur = current[f.key] ?? "";
                  const val = edited[f.key] ?? "";
                  const changed =
                    f.kind === "secret"
                      ? cleared.has(f.key) || (edited[f.key] ?? "").trim() !== ""
                      : edited[f.key] !== undefined && edited[f.key] !== cur;
                  return (
                    <div key={f.key} className={"settings-row" + (changed ? " changed" : "")}>
                      <label htmlFor={"set-" + f.key}>
                        {f.label}
                        {src && (
                          <em className={"settings-src src-" + src}>
                            {src === "file" ? "来自设置文件" : "来自 .env"}
                          </em>
                        )}
                      </label>
                      <div className="settings-field">
                        {f.kind === "select" ? (
                          <select
                            id={"set-" + f.key}
                            value={edited[f.key] ?? cur}
                            onChange={(e) => setField(f.key, e.target.value)}
                          >
                            {(f.options ?? []).map((o) => (
                              <option key={o.value} value={o.value}>
                                {o.label}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            id={"set-" + f.key}
                            type={f.kind === "secret" ? "password" : "text"}
                            value={val}
                            placeholder={
                              f.kind === "secret"
                                ? cur
                                  ? "已设置（" + cur + "），留空保持不变"
                                  : "未设置"
                                : f.placeholder ?? ""
                            }
                            autoComplete="off"
                            onChange={(e) => setField(f.key, e.target.value)}
                          />
                        )}
                        {PROBE_BY_FIELD[f.key] && (
                          <button
                            className="btn btn-ghost settings-probe-btn"
                            disabled={probing}
                            title="零额度探测：验证模型名与 API Key 是否可用"
                            onClick={() => void runProbe(f.key)}
                          >
                            {probing ? "探测中…" : "探测"}
                          </button>
                        )}
                        {f.kind === "secret" && (cur || cleared.has(f.key)) && (
                          <label className="settings-clear">
                            <input
                              type="checkbox"
                              checked={cleared.has(f.key)}
                              onChange={(e) =>
                                setCleared((prev) => {
                                  const next = new Set(prev);
                                  if (e.target.checked) next.add(f.key);
                                  else next.delete(f.key);
                                  return next;
                                })
                              }
                            />
                            清除
                          </label>
                        )}
                      </div>
                      {f.hint && <p className="settings-hint">{f.hint}</p>}
                      {PROBE_BY_FIELD[f.key] && probeField === f.key && probeResult && (
                        <p className={"settings-hint settings-probe-result " + (probeResult.ok ? "ok" : "err")}>
                          {probeResult.ok ? "✅ " : "❌ "}
                          {probeResult.text}
                        </p>
                      )}
                    </div>
                  );
                })}
                {group === "语音 / 字幕" && (
                  <p className="settings-hint settings-group-hint">
                    字幕由各镜头真实时长规则生成（SRT），无需模型；配音音色即改即生效，下次合成时使用。
                  </p>
                )}
              </fieldset>
            );
          })}

        <div className="settings-foot">
          {msg && <span className={"settings-msg " + (msg.ok ? "ok" : "err")}>{msg.text}</span>}
          <button className="btn btn-ghost" disabled={loading || saving} onClick={() => void load()}>
            重新加载
          </button>
          <button
            className="btn btn-primary"
            disabled={loading || saving || dirtyCount === 0}
            onClick={() => void save()}
          >
            {saving ? "保存中…" : dirtyCount > 0 ? "保存 " + dirtyCount + " 项改动" : "保存"}
          </button>
        </div>
      </div>
    </div>
  );
}
