// F-6 窄屏审计工具（可复跑）
// ---------------------------------------------------------------------------
// 用 Chrome DevTools Protocol 在真机视口下渲染 mobile-audit-fixture.html，
// 客观测量三处关键区（审核卡片 / 成片播放 / 进度时间轴）与导航抽屉，并按阈值断言。
//
// 用法（在 frontend/ 下，需先 npm run build）：
//     node scripts/audit-mobile.mjs
//     node scripts/audit-mobile.mjs --json     # 只输出 JSON
//
// 退出码：0 = 全部通过；1 = 存在断言失败。
import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync, existsSync, readFileSync, copyFileSync, readdirSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import os from "node:os";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FRONTEND = resolve(__dirname, "..");
const DIST = join(FRONTEND, "dist");
const JSON_ONLY = process.argv.includes("--json");

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  process.env.LOCALAPPDATA + "\\Google\\Chrome\\Application\\chrome.exe",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].filter(Boolean);
const CHROME = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!CHROME) {
  console.error("✗ 未找到 Chrome。可设置环境变量 CHROME_PATH 指定可执行文件。");
  process.exit(2);
}
if (!existsSync(join(DIST, "index.html"))) {
  console.error("✗ 未找到 dist/。请先在 frontend/ 运行 `npm run build`。");
  process.exit(2);
}

// 准备临时工作目录：夹具 + 打包后的 CSS
const WORK = join(os.tmpdir(), "f6-mobile-audit");
mkdirSync(join(WORK, "assets"), { recursive: true });
const cssFile = readdirSync(join(DIST, "assets")).find((f) => f.endsWith(".css"));
if (!cssFile) {
  console.error("✗ dist/assets 下未找到 CSS 产物。");
  process.exit(2);
}
copyFileSync(join(DIST, "assets", cssFile), join(WORK, "assets", "app.css"));
copyFileSync(join(__dirname, "mobile-audit-fixture.html"), join(WORK, "fixture.html"));
const FILE_URL = "file:///" + join(WORK, "fixture.html").replace(/\\/g, "/");

const PORT = 9444;
const UD = join(WORK, "chrome-profile");
mkdirSync(UD, { recursive: true });

const chrome = spawn(CHROME, [
  "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
  "--remote-debugging-port=" + PORT, "--user-data-dir=" + UD,
  "--no-first-run", "--no-default-browser-check", "about:blank",
], { stdio: "ignore" });
const cleanup = () => { try { chrome.kill(); } catch {} };
process.on("exit", cleanup);

async function findTarget() {
  for (let i = 0; i < 60; i++) {
    try {
      const j = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const t = j.find((x) => x.type === "page" && x.webSocketDebuggerUrl);
      if (t) return t;
    } catch {}
    await sleep(250);
  }
  throw new Error("Chrome CDP 目标未就绪");
}

const ws = new WebSocket((await findTarget()).webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
let seq = 0;
const pend = new Map();
const ev = [];
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pend.has(m.id)) {
    const p = pend.get(m.id); pend.delete(m.id);
    m.error ? p.rej(new Error(JSON.stringify(m.error))) : p.res(m.result);
  } else if (m.method) ev.push(m.method);
};
const send = (method, params = {}) =>
  new Promise((res, rej) => { const id = ++seq; pend.set(id, { res, rej }); ws.send(JSON.stringify({ id, method, params })); });

await send("Page.enable");
await send("Runtime.enable");

const VIEWPORTS = [
  { name: "mobile_390", width: 390, height: 844, dsf: 2, mobile: true },
  { name: "mobile_430", width: 430, height: 932, dsf: 2, mobile: true },
  { name: "tablet_768", width: 768, height: 1024, dsf: 2, mobile: true },
  { name: "laptop_1280", width: 1280, height: 900, dsf: 1, mobile: false },
];

const results = [];
for (const vp of VIEWPORTS) {
  await send("Emulation.setDeviceMetricsOverride", {
    width: vp.width, height: vp.height, deviceScaleFactor: vp.dsf, mobile: vp.mobile,
  });
  await send("Page.navigate", { url: FILE_URL });
  for (let i = 0; i < 40; i++) { await sleep(150); if (ev.includes("Page.loadEventFired")) break; }
  await sleep(300);
  const r = await send("Runtime.evaluate", {
    expression: `document.getElementById("audit-result")?.textContent || "NO_RESULT"`,
    returnByValue: true,
  });
  const m = String(r.result.value || "").match(/AUDIT_JSON_START([\s\S]*?)AUDIT_JSON_END/);
  let data = null;
  try { data = m ? JSON.parse(m[1]) : null; } catch {}
  results.push({ viewport: vp.name, results: data });
  ev.length = 0;
}

// ── 断言 ──────────────────────────────────────────────────────────────
const checks = [];
const add = (vp, name, ok, detail) => checks.push({ vp, name, ok: !!ok, detail });
for (const { viewport, results: rs } of results) {
  if (!rs || !rs.length) { add(viewport, "测量结果可取", false, "NO_RESULT"); continue; }
  const open = rs[0], closed = rs[1] || rs[0];
  const isNarrow = ["mobile_390", "mobile_430", "tablet_768"].includes(viewport);

  add(viewport, "无横向溢出", !open.hOverflow, `docScrollW=${open.docScrollW} vw=${open.vw}`);
  add(viewport, "无溢出元素", (open.overflowing || []).length === 0, JSON.stringify(open.overflowing || []));
  add(viewport, "成片播放撑满内容宽", open.finalMovieW > 0 && open.finalMovieW >= Math.min(300, open.vw * 0.6), `w=${open.finalMovieW}`);
  add(viewport, "成片链接 ≥44px（窄屏）", !isNarrow || open.finalLinkH >= 44, `h=${open.finalLinkH}`);

  if (isNarrow) {
    // 44px 触控最小尺寸只对窄屏/触控成立；桌面 .btn 保持既有 39px 设计
    add(viewport, "审核按钮 ≥44px 高", open.actionBtnH >= 44, `h=${open.actionBtnH}`);
    add(viewport, "审核按钮纵向堆叠", open.actionRowDir === "column", open.actionRowDir);
    add(viewport, "修正方式单列", !open.fixRowCols.includes("  "), open.fixRowCols);
    add(viewport, "候选条可横向滚动", open.candStripOverflowX === "auto", open.candStripOverflowX);
    add(viewport, "时间轴解除高度锁", open.timelineMaxH === "none", open.timelineMaxH);
    add(viewport, "侧栏为抽屉", open.sidebarPos === "fixed", open.sidebarPos);
    add(viewport, "汉堡按钮可见", open.navToggleDisplay !== "none", open.navToggleDisplay);
    add(viewport, "抽屉关闭时移出视口", /-3\d\d|-2\d\d/.test(closed.sidebarTransform), closed.sidebarTransform);
  } else {
    add(viewport, "桌面侧栏保持文档流", open.sidebarPos === "static", open.sidebarPos);
    add(viewport, "桌面汉堡按钮隐藏", open.navToggleDisplay === "none", open.navToggleDisplay);
    add(viewport, "桌面时间轴保留高度锁", open.timelineMaxH !== "none", open.timelineMaxH);
    add(viewport, "桌面审核按钮可用（≥36px）", open.actionBtnH >= 36, `h=${open.actionBtnH}`);
    add(viewport, "桌面按钮保持横排", open.actionRowDir === "row", open.actionRowDir);
  }
  if (viewport !== "laptop_1280") {
    add(viewport, "输入控件字号 ≥16px（防 iOS 缩放）", parseFloat(open.textareaFont) >= 16, open.textareaFont);
  }
}

const failed = checks.filter((c) => !c.ok);
writeFileSync(join(WORK, "audit-report.json"), JSON.stringify({ results, checks }, null, 2));

// 同时落一份人类可读报告（便于离线查看 / 编码无关）
const lines = ["=== F-6 窄屏审计 ==="];
let cur = "";
for (const c of checks) {
  if (c.vp !== cur) { cur = c.vp; lines.push("\n[" + cur + "]"); }
  lines.push(`  ${c.ok ? "OK  " : "FAIL"} ${c.name}${c.ok ? "" : "  <- " + c.detail}`);
}
lines.push(`\n断言：${checks.length - failed.length}/${checks.length} 通过`);
lines.push("明细：" + join(WORK, "audit-report.json"));
writeFileSync(join(WORK, "audit-report.txt"), lines.join("\n"), "utf8");

if (JSON_ONLY) {
  console.log(JSON.stringify({ checks, failed: failed.length }, null, 2));
} else {
  console.log(lines.join("\n"));
}
ws.close();
cleanup();
process.exit(failed.length ? 1 : 0);
