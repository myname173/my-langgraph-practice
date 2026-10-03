// F-6 CSS 分层校验：确认「令牌 → 布局 → 组件 → 响应式」四层齐备、顺序正确，
// 且构建产物已把四层内联为单一 CSS（关键选择器都在）。
//
// 用法（frontend/ 下，需先 npm run build）：node scripts/check-css-layers.mjs
import { readFileSync, existsSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const FRONTEND = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SRC = join(FRONTEND, "src");
const LAYERS = ["tokens", "layout", "components", "responsive"];

const out = [];
let fail = 0;
const ck = (name, ok, detail = "") => { if (!ok) fail++; out.push(`  ${ok ? "OK  " : "FAIL"} ${name}${ok ? "" : "  <- " + detail}`); };

// 1) 入口
const entryPath = join(SRC, "styles.css");
ck("styles.css 入口存在", existsSync(entryPath));
const entry = readFileSync(entryPath, "utf8");
const imports = [...entry.matchAll(/@import\s+"\.\/styles\/([a-z]+)\.css"/g)].map((m) => m[1]);
ck("入口为纯 @import 形式（无裸规则）", !/\n\s*[.#a-z\[*:][^{}]*\{[^}]*\}/i.test(entry.replace(/\/\*[\s\S]*?\*\//g, "")), entry.slice(0, 120));
ck("入口导入顺序 = tokens→layout→components→responsive",
  JSON.stringify(imports) === JSON.stringify(LAYERS), JSON.stringify(imports));

// 2) 各层文件
const bodies = {};
for (const l of LAYERS) {
  const p = join(SRC, "styles", `${l}.css`);
  const ok = existsSync(p) && readFileSync(p, "utf8").trim().length > 0;
  ck(`styles/${l}.css 存在且非空`, ok, p);
  if (ok) bodies[l] = readFileSync(p, "utf8");
}

// 3) 分层内容归属
ck("tokens.css 含设计令牌（--bg-0 / --radius）", /--bg-0\s*:/.test(bodies.tokens || "") && /--radius\s*:/.test(bodies.tokens || ""));
ck("layout.css 含 App shell（.app / .sidebar / .main）", /\.app\s*\{/.test(bodies.layout || "") && /\.sidebar\s*\{/.test(bodies.layout || "") && /\.main\s*\{/.test(bodies.layout || ""));
ck("components.css 含审核卡与时间轴", /\.approval-card/.test(bodies.components || "") && /\.timeline-list/.test(bodies.components || ""));
ck("responsive.css 含三档断点", /max-width:\s*900px/.test(bodies.responsive || "") && /max-width:\s*768px/.test(bodies.responsive || "") && /max-width:\s*430px/.test(bodies.responsive || ""));
ck("responsive.css 含触控目标规则（pointer: coarse）", /pointer:\s*coarse/.test(bodies.responsive || ""));

// 4) 构建产物内联校验
const distAssets = join(FRONTEND, "dist", "assets");
let built = "";
if (existsSync(distAssets)) {
  const cssName = readdirSync(distAssets).find((f) => f.endsWith(".css"));
  if (cssName) built = readFileSync(join(distAssets, cssName), "utf8");
}
ck("存在构建产物 CSS", built.length > 0, distAssets);
if (built) {
  // 产物中不应残留 @import（说明已内联）
  ck("产物无残留 @import（已内联）", !/@import\s+"\.\/styles\//.test(built));
  ck("产物含 tokens 层（--bg-0）", /--bg-0\s*:/.test(built));
  ck("产物含 layout 层（.sidebar）", /\.sidebar\s*\{/.test(built));
  ck("产物含 components 层（.approval-card）", /\.approval-card/.test(built));
  ck("产物含 responsive 层（max-width: 900px）", /max-width:\s*900px/.test(built));
}

// 5) 响应式层应位于产物末尾（层叠优先级）
if (built) {
  const iResp = built.lastIndexOf("@media (max-width: 900px)");
  const iToken = built.indexOf("--bg-0");
  ck("响应式规则位于令牌之后（顺序正确）", iToken >= 0 && iResp > iToken, `token@${iToken} resp@${iResp}`);
}

console.log("=== F-6 CSS 分层校验 ===");
console.log(out.join("\n"));
console.log(fail === 0 ? "\n==> 全部通过" : `\n==> ${fail} 项失败`);
process.exit(fail ? 1 : 0);
