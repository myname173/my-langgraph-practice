# webapp.py
"""
Savant SWE Agent — 控制台前端
用 Streamlit 构建，支持本地开发与演示。
生产部署请使用 src/api.py (FastAPI + SSE)。

启动命令：
  streamlit run webapp.py
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import logging
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.agent.swe.graph import graph
from src.agent.swe.tools import WORKSPACE_DIR, GLOBAL_STATS, SKILLS_DIR
from src.agent.swe.evolution import parse_skill_metadata, DRAFT_DIR

load_dotenv()

logger = logging.getLogger("SWE_WebApp")

# ══════════════════════════════════════════════════════
# 页面配置（必须是第一个 st 调用）
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Savant SWE Agent",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════
# 全局样式
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── 字体 & 基底 ────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
code, pre, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── 主背景 ─────────────────────────────── */
.stApp {
    background: #0b0f1a;
}
section[data-testid="stSidebar"] {
    background: #0e1420;
    border-right: 1px solid #1e2840;
}

/* ── 顶部标题带 ──────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 60%, #0a192f 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 28px 16px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, #00d4ff22 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.7rem;
    color: #e0f4ff;
    letter-spacing: -0.02em;
    margin: 0;
}
.hero-sub {
    color: #4a8fad;
    font-size: 0.82rem;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── 流水线节点 ──────────────────────────── */
.pipeline-row {
    display: flex;
    gap: 6px;
    align-items: center;
    flex-wrap: wrap;
    margin: 8px 0;
}
.pipe-node {
    display: flex; align-items: center; gap: 5px;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid #1e2840;
    background: #111827;
    color: #4a5568;
    white-space: nowrap;
    transition: all 0.2s;
}
.pipe-node.active {
    background: #0a2540;
    border-color: #00d4ff;
    color: #00d4ff;
    box-shadow: 0 0 12px #00d4ff33;
    animation: pulse-glow 1.5s ease-in-out infinite;
}
.pipe-node.done {
    background: #0a1f14;
    border-color: #00e676;
    color: #00e676;
}
.pipe-arrow {
    color: #1e2840;
    font-size: 0.8rem;
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 8px #00d4ff33; }
    50%       { box-shadow: 0 0 18px #00d4ff66; }
}

/* ── 状态卡片（审批请求） ────────────────── */
.approval-card {
    background: linear-gradient(135deg, #1a1000, #1f1500);
    border: 1px solid #b7791f;
    border-left: 4px solid #f6ad55;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 14px 0;
}
.approval-card h4 { color: #f6ad55; margin: 0 0 6px; font-size: 1rem; }
.approval-card p  { color: #c9a96e; margin: 0; font-size: 0.88rem; }

/* ── 进度条 ─────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00d4ff, #0070f3);
}

/* ── 指标卡片 ────────────────────────────── */
.metric-card {
    background: #111827;
    border: 1px solid #1e2840;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.metric-card .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e0f4ff;
    line-height: 1.1;
}
.metric-card .lbl {
    font-size: 0.7rem;
    color: #4a5568;
    margin-top: 3px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── 标签组 ─────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #0e1420;
    border-bottom: 1px solid #1e2840;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    color: #4a5568;
    font-size: 0.82rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    padding: 8px 18px;
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    background: #0a1a2e !important;
}

/* ── 技能徽章 ────────────────────────────── */
.skill-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin-left: 6px;
    vertical-align: middle;
}
.badge-scaffold { background: #1a2a1a; color: #52c41a; border: 1px solid #237804; }
.badge-debug    { background: #1a1a2a; color: #597ef7; border: 1px solid #2f54eb; }
.badge-test     { background: #2a1a2a; color: #eb2f96; border: 1px solid #c41d7f; }
.badge-config   { background: #2a2a1a; color: #fadb14; border: 1px solid #d4b106; }
.badge-deploy   { background: #1a2a2a; color: #13c2c2; border: 1px solid #0e9494; }
.badge-misc     { background: #1e1e1e; color: #8c8c8c; border: 1px solid #434343; }

/* ── 训练奖励条 ─────────────────────────── */
.reward-bar-wrap {
    background: #111827;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    margin: 4px 0 2px;
}
.reward-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #ff4d4f, #ffa940, #52c41a);
    transition: width 0.4s ease;
}

/* ── 信息提示框（初学者引导） ────────────── */
.guide-box {
    background: #0a1a30;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #00d4ff;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.82rem;
    color: #7cb9d8;
    line-height: 1.6;
}
.guide-box strong { color: #b0d8ef; }

/* ── 输入框 ─────────────────────────────── */
.stTextInput input, .stTextArea textarea {
    background: #111827 !important;
    border: 1px solid #1e2840 !important;
    color: #e0f4ff !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 0 2px #00d4ff22 !important;
}

/* ── 按钮 ────────────────────────────────── */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #0070f3, #00d4ff);
    border: none;
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    border-radius: 8px;
    transition: all 0.2s;
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px #0070f340;
}

/* ── 展开框 ─────────────────────────────── */
.streamlit-expanderHeader {
    background: #111827 !important;
    border: 1px solid #1e2840 !important;
    border-radius: 8px !important;
    color: #a0b4c8 !important;
    font-size: 0.84rem !important;
}

/* ── 侧边栏标题 ─────────────────────────── */
.sidebar-section-title {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #2d4a6b;
    font-weight: 600;
    margin: 16px 0 6px;
}

/* ── 任务清单条目 ────────────────────────── */
.task-item {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid #1a2030;
    font-size: 0.82rem;
    color: #8090a8;
    line-height: 1.4;
}
.task-item.current { color: #e0f4ff; }
.task-item.done    { color: #2d5a3d; text-decoration: line-through; }

/* ── 聊天消息覆盖 ────────────────────────── */
[data-testid="stChatMessage"] {
    background: #0e1420 !important;
    border: 1px solid #1e2840 !important;
    border-radius: 10px !important;
    margin: 6px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# 常量 & 辅助
# ══════════════════════════════════════════════════════
_THREAD_ID_RE = re.compile(r"^[\w\-]{1,64}$")
_FILE_PREVIEW_MAX_BYTES = 256 * 1024

NODES_META = {
    "planner":            ("📋", "需求拆解",   "将任务分解为可执行的原子步骤"),
    "index_builder":      ("🔍", "索引构建",   "建立代码三层智能索引（AST + 语义 + BM25）"),
    "coder":              ("💻", "代码编写",   "调用工具完成编码"),
    "tools":              ("🛠️", "工具执行",   "执行文件读写、命令运行等操作"),
    "task_manager":       ("🗺️", "进度更新",   "更新待办清单"),
    "summarizer":         ("🧠", "记忆压缩",   "压缩超长历史，节省 Token"),
    "reviewer":           ("🔍", "质量审查",   "验证任务是否真正完成"),
    "evolution_reflect":  ("🔬", "演化反思",   "分析本次任务是否值得固化为技能"),
    "evolution_generate": ("🧬", "技能生成",   "生成可复用的 Python 技能脚本"),
    "evolution_verify":   ("✅", "技能验证",   "AST 检查 + 沙盒运行 __test__()"),
    "trajectory_export":  ("📚", "轨迹导出",   "将执行轨迹保存为训练数据"),
}

BADGE_CLASSES = {
    "scaffold": "badge-scaffold", "debug": "badge-debug",
    "test": "badge-test", "config": "badge-config",
    "deploy": "badge-deploy", "misc": "badge-misc",
}


def get_safe_state(config: dict) -> dict:
    """安全地从 checkpointer 读取当前图状态。"""
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            raise ValueError
        return state.values
    except Exception:
        return {
            "todo_list": [], "completed_tasks": [],
            "iteration_count": 0, "summary": "",
            "status": "coding", "reviewer_reject_count": 0,
            "evolution_skill_draft": "", "evolution_report": "",
            "code_index_ready": False, "repo_map": "",
            "trajectory_id": "", "training_export_path": "",
        }


def reward_color(reward: float) -> str:
    if reward >= 0.6: return "#52c41a"
    if reward >= 0.3: return "#ffa940"
    return "#ff4d4f"


def _make_init_state(prompt: str, max_iterations: int = 25) -> dict:
    """
    构造全新任务的初始状态。
    max_iterations 默认 25，planner_node 会根据任务类型动态调整（最高 35）。
    Bug 5 修复：不再强制覆盖为 25，允许调用者传入更大值。
    """
    return {
        "messages": [HumanMessage(content=prompt)],
        "task_description": prompt,
        "plan": [], "todo_list": [], "completed_tasks": [],
        "summary": "",
        "workspace": str(WORKSPACE_DIR.absolute()),
        "iteration_count": 0, "max_iterations": max_iterations,
        "status": "coding", "test_passed": False,
        "reviewer_reject_count": 0,
        "evolution_skill_draft": "", "evolution_report": "",
        "code_index_ready": False, "repo_map": "",
        "trajectory_id": "", "training_export_path": "",
    }


# ══════════════════════════════════════════════════════
# Session State 初始化
# ══════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_guide" not in st.session_state:
    st.session_state.show_guide = True
# 熔断标志：circuit breaker 触发后置 True，阻止路由继续自动流转
if "agent_terminated" not in st.session_state:
    st.session_state.agent_terminated = False
# 重置时自动生成的新 thread_id，覆盖输入框的值
if "override_thread_id" not in st.session_state:
    st.session_state.override_thread_id = None


# ══════════════════════════════════════════════════════
# 侧边栏
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:12px 0 8px;'>
        <div style='font-family:Syne,sans-serif;font-weight:800;font-size:1.1rem;color:#e0f4ff;'>
            🛰️ Savant SWE
        </div>
        <div style='font-size:0.7rem;color:#2d4a6b;font-family:JetBrains Mono,monospace;margin-top:2px;'>
            AI 软件工程师控制台
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 会话 ID ──────────────────────────────────
    st.markdown("<div class='sidebar-section-title'>会话配置</div>", unsafe_allow_html=True)
    raw_thread_id = st.text_input(
        "会话 ID",
        value="session_001",
        help="同一个会话 ID 可以恢复上次未完成的任务。修改 ID 可以开启新对话。",
        label_visibility="collapsed",
        placeholder="会话 ID，如 my_project_001",
    )
    if not _THREAD_ID_RE.match(raw_thread_id):
        st.error("⚠️ 只允许字母、数字、下划线、连字符，长度 1~64")
        st.stop()

    # 重置后使用自动生成的新 ID，否则使用输入框的值
    thread_id = st.session_state.override_thread_id or raw_thread_id
    config = {"configurable": {"thread_id": thread_id}}
    state_values = get_safe_state(config)

    # ── 任务进度 ──────────────────────────────────
    st.markdown("<div class='sidebar-section-title'>任务进度</div>", unsafe_allow_html=True)
    todo = state_values.get("todo_list", [])
    done = state_values.get("completed_tasks", [])
    total = len(todo) + len(done)
    prog = (len(done) / total) if total > 0 else 0.0

    st.progress(prog)
    st.caption(f"{'✅' if prog == 1.0 and total > 0 else '⏳'} 已完成 **{len(done)}** / 总计 **{total}** 步")

    # ── 实时指标 4 格 ─────────────────────────────
    st.markdown("<div class='sidebar-section-title'>运行指标</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='val'>{state_values.get('iteration_count', 0)}</div>
            <div class='lbl'>迭代轮次</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='val'>{len(list(SKILLS_DIR.glob("*.py")))}</div>
            <div class='lbl'>已积累技能</div>
        </div>""", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='val'>{GLOBAL_STATS['tavily_count']}<span style='font-size:.9rem;color:#4a5568'>/{GLOBAL_STATS['max_tavily']}</span></div>
            <div class='lbl'>搜索配额</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        traj_count = 0
        traj_path = WORKSPACE_DIR / "_training_data" / "trajectories.jsonl"
        if traj_path.exists():
            with open(traj_path) as _f:
                traj_count = sum(1 for l in _f if l.strip())
        st.markdown(f"""
        <div class='metric-card'>
            <div class='val'>{traj_count}</div>
            <div class='lbl'>训练轨迹</div>
        </div>""", unsafe_allow_html=True)

    # ── 代码索引状态 ──────────────────────────────
    st.markdown("<div class='sidebar-section-title'>代码智能索引</div>", unsafe_allow_html=True)
    code_index_ready = state_values.get("code_index_ready", False)
    if code_index_ready:
        st.markdown("✅ &nbsp;<span style='color:#00e676;font-size:.82rem;'>三层索引已就绪</span>", unsafe_allow_html=True)
        try:
            from src.agent.swe.code_index import get_index_stats
            s = get_index_stats()
            if s.get("status") != "not_built":
                st.caption(
                    f"📦 {s.get('chunk_count',0)} 块 &nbsp;│&nbsp; "
                    f"向量 {'✓' if s.get('semantic_ready') else '✗'} "
                    f"BM25 {'✓' if s.get('bm25_ready') else '✗'} "
                    f"图谱 {'✓' if s.get('graph_ready') else '✗'}"
                )
        except Exception:
            pass
    else:
        st.markdown("⏳ &nbsp;<span style='color:#ffd54f;font-size:.82rem;'>等待首次任务启动后自动构建</span>", unsafe_allow_html=True)

    # ── 当前任务清单 ──────────────────────────────
    if todo or done:
        st.markdown("<div class='sidebar-section-title'>任务蓝图</div>", unsafe_allow_html=True)
        if todo:
            st.markdown(f"""
            <div style='background:#0a1a2e;border:1px solid #1e3a5f;border-left:3px solid #00d4ff;
                        border-radius:8px;padding:10px 14px;margin:4px 0;font-size:.82rem;color:#b0d8ef;'>
                🎯 <strong>当前目标</strong><br>{todo[0]}
            </div>""", unsafe_allow_html=True)
        if len(todo) > 1:
            with st.expander(f"后续 {len(todo)-1} 项任务"):
                for t in todo[1:]:
                    st.markdown(f"<div class='task-item'>⏳ {t}</div>", unsafe_allow_html=True)
        if done:
            with st.expander(f"已完成 {len(done)} 项"):
                for d in done:
                    st.markdown(f"<div class='task-item done'>✅ {d}</div>", unsafe_allow_html=True)

    # ── Evolution 快报 ────────────────────────────
    evolution_report = state_values.get("evolution_report", "")
    if evolution_report:
        st.markdown("<div class='sidebar-section-title'>技能进化快报</div>", unsafe_allow_html=True)
        if "✅" in evolution_report:
            st.success("🧬 新技能已固化入库！查看「技能进化」标签了解详情。")
        elif "❌" in evolution_report:
            st.error("🧬 技能生成失败，见「技能进化」标签")
        else:
            st.info("🔍 本次任务已分析，无可复用技能")


# ══════════════════════════════════════════════════════
# 主区域顶部：标题 + 流水线
# ══════════════════════════════════════════════════════
state_raw = graph.get_state(config)
next_node = state_raw.next[0] if (state_raw and state_raw.next) else "idle"

# 标题横幅
st.markdown(f"""
<div class='hero-banner'>
    <p class='hero-title'>🛰️ Savant SWE Agent</p>
    <p class='hero-sub'>AI 软件工程师 · 会话 <code style='color:#00d4ff'>{thread_id}</code>
        &nbsp;·&nbsp; 状态
        <code style='color:{"#00e676" if state_values.get("status")=="success"
                    else "#ff4d4f" if state_values.get("status")=="failed"
                    else "#ffd54f"}'>{state_values.get("status","idle")}</code>
    </p>
</div>
""", unsafe_allow_html=True)

# 流水线可视化
MAIN_FLOW  = ["planner", "index_builder", "coder", "tools", "task_manager", "summarizer", "reviewer"]
EVO_FLOW   = ["evolution_reflect", "evolution_generate", "evolution_verify", "trajectory_export"]

def _pipe_nodes_html(node_ids: list, next_node: str, done_nodes: list = None) -> str:
    parts = []
    for i, nid in enumerate(node_ids):
        icon, label, _ = NODES_META.get(nid, ("·", nid, ""))
        cls = "active" if nid == next_node else ""
        parts.append(f"<div class='pipe-node {cls}'>{icon} {label}</div>")
        if i < len(node_ids) - 1:
            parts.append("<span class='pipe-arrow'>→</span>")
    return "".join(parts)

with st.expander("🛤️ 实时执行流水线", expanded=True):
    st.markdown(f"""
    <div style='margin-bottom:6px;font-size:.7rem;color:#2d4a6b;text-transform:uppercase;letter-spacing:.1em;'>主流程</div>
    <div class='pipeline-row'>{_pipe_nodes_html(MAIN_FLOW, next_node)}</div>
    <div style='margin:10px 0 6px;font-size:.7rem;color:#2d4a6b;text-transform:uppercase;letter-spacing:.1em;'>
        技能进化 & 训练导出（任务成功后自动触发）
    </div>
    <div class='pipeline-row'>{_pipe_nodes_html(EVO_FLOW, next_node)}</div>
    """, unsafe_allow_html=True)

    if next_node != "idle" and next_node in NODES_META:
        icon, label, desc = NODES_META[next_node]
        st.markdown(f"""
        <div style='margin-top:10px;padding:8px 14px;background:#0a1a2e;border-radius:6px;
                    border:1px solid #1e3a5f;font-size:.8rem;color:#7cb9d8;'>
            {icon} <strong style='color:#b0d8ef;'>{label}</strong> &nbsp;·&nbsp; {desc}
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# 终态 UI 辅助函数
# ══════════════════════════════════════════════════════

def _do_reset():
    """生成新的 thread_id，清空消息历史，解除熔断标志。"""
    import time
    new_tid = f"session_{int(time.time())}"
    st.session_state.override_thread_id = new_tid
    st.session_state.messages = []
    st.session_state.agent_terminated = False


def _render_new_task_button():
    """
    成功后的按钮行。
    Bug fix: chat_input 不能放在 st.columns() 内，移到顶层。
    """
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("🔄 开始新任务（新会话）", use_container_width=True):
        _do_reset()
        st.rerun()
    # chat_input 必须在顶层，不能嵌套在 columns 中
    prompt = st.chat_input("继续输入新需求，或点击上方按钮重置会话...", key="main_input")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.agent_terminated = False
        with st.chat_message("user"):
            st.markdown(prompt)
        run_agent_ui(_make_init_state(prompt))
        st.rerun()


def _render_failed_ui(config: dict):
    """
    任务失败 / 熔断时的终态 UI。
    提供三条出路：重新开始（新 thread）、调整后重试（同 thread）、手动输入。
    """
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a0a0a,#1f0f0f);
                border:1px solid #7f1d1d;border-left:4px solid #ef4444;
                border-radius:10px;padding:18px 22px;margin:10px 0;'>
        <div style='color:#fca5a5;font-weight:700;font-size:1rem;margin-bottom:6px;'>
            😔 任务未能完成
        </div>
        <div style='color:#f87171;font-size:.84rem;line-height:1.6;'>
            Agent 已达到最大迭代次数或被熔断保护停止。<br>
            请选择下方的处理方式继续。
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='guide-box'>
        💡 <strong>常见原因与解决方法</strong><br>
        &nbsp;·&nbsp; <strong>任务太复杂</strong>：把大任务拆成几个小需求，逐步完成<br>
        &nbsp;·&nbsp; <strong>描述不够具体</strong>：加上技术栈、文件路径、预期输出等细节<br>
        &nbsp;·&nbsp; <strong>环境问题</strong>：检查工作区目录权限和依赖是否已安装<br>
        &nbsp;·&nbsp; <strong>API 超时</strong>：检查终端日志，或稍等片刻再重试
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 开始新任务（新会话）", use_container_width=True, type="primary",
                     key="reset_new_session"):
            _do_reset()
            st.rerun()
    with col_b:
        if st.button("♻️ 继续未完成的任务", use_container_width=True, key="retry_same"):
            st.session_state.agent_terminated = False
            todo_left = state_values.get("todo_list", [])
            done_left = state_values.get("completed_tasks", [])
            try:
                _resume_msg = (
                    "[系统] 请继续完成剩余任务，不要重新规划，从第一项待办任务继续。"
                    " 待办: " + str(todo_left) + " 已完成: " + str(done_left)
                )
                graph.update_state(
                    config,
                    {
                        "status": "coding",
                        "iteration_count": 0,
                        "messages": [HumanMessage(content=_resume_msg)],
                    },
                    as_node="coder",
                )
            except Exception as _e:
                logger.warning(f"续接状态写入失败: {_e}")
            run_agent_ui(None, run_config=config)
            st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    prompt = st.chat_input("或直接输入修改后的新需求，点击发送重新开始...", key="main_input_failed")
    if prompt:
        _do_reset()          # 开新会话，避免带着旧状态
        # 用新 config（override_thread_id 刚写入）重新初始化
        new_config = {"configurable": {"thread_id": st.session_state.override_thread_id}}
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # 直接运行（不能调用 run_agent_ui 因为 config 还是旧的，用新 config）
        try:
            run_agent_ui(_make_init_state(prompt), run_config=new_config)
        except Exception as e:
            st.error(f"❌ 运行异常：{e}")
        st.rerun()


# ══════════════════════════════════════════════════════
# 核心运行函数
# ══════════════════════════════════════════════════════
def run_agent_ui(input_data=None, run_config=None):
    """
    Bug 6 修复：接受可选的 run_config 参数，防止闭包捕获旧的 config。
    _render_failed_ui 重置会话后需要传入新 config，否则会向旧 thread 发请求。
    """
    effective_config = run_config if run_config is not None else config
    no_tool_count = 0
    try:
        with st.status("🚀 Agent 正在处理任务...", expanded=True) as status:
            for event in graph.stream(input_data, config=effective_config, stream_mode="updates"):
                node_name = list(event.keys())[0]
                node_output = event[node_name]

                if "messages" in node_output:
                    new_msg = node_output["messages"][-1]
                    msg_content = getattr(new_msg, "content", "")
                    if msg_content and not any(
                        m["content"] == msg_content for m in st.session_state.messages
                    ):
                        st.session_state.messages.append(
                            {"role": "assistant", "content": msg_content}
                        )
                        with st.chat_message("assistant"):
                            st.markdown(msg_content)

                # 节点提示
                if node_name == "index_builder":
                    st.toast("🔍 代码智能索引构建中...", icon="📦")
                elif node_name == "trajectory_export":
                    st.toast("📚 执行轨迹已保存为训练数据", icon="💾")
                elif node_name in ("evolution_reflect", "evolution_generate", "evolution_verify"):
                    _, lbl, _ = NODES_META.get(node_name, ("", node_name, ""))
                    st.toast(f"🧬 {lbl}...", icon="🔬")
                elif node_name == "coder" and "messages" in node_output:
                    last_m = node_output["messages"][-1]
                    if hasattr(last_m, "tool_calls") and last_m.tool_calls:
                        no_tool_count = 0
                        for tc in last_m.tool_calls:
                            st.toast(f"🛠️ 调用工具: {tc['name']}", icon="🔧")
                    else:
                        no_tool_count += 1

                # ★ 修复：熔断时强制写入 failed 状态，阻止路由再次进入 coder
                if no_tool_count > 3:
                    st.error("🚨 监测到 Agent 无进展，已自动熔断。")
                    st.caption("点击下方「🔄 开始新任务」按钮可重新开始，或修改任务描述后再试。")
                    st.session_state.agent_terminated = True
                    try:
                        graph.update_state(
                            effective_config,
                            {
                                "status": "failed",
                                "messages": [AIMessage(
                                    content="[系统熔断] Agent 连续无进展，任务已终止。"
                                )],
                            },
                        )
                    except Exception as _ue:
                        logger.warning(f"熔断状态写入失败（无影响）: {_ue}")
                    break

            status.update(label="✅ 当前阶段处理完成", state="complete", expanded=False)
    except Exception as e:
        st.error(f"❌ 运行异常：{str(e)}")
        st.caption("💡 提示：请检查 .env 文件中的 API Key 是否正确配置，或查看终端日志排查原因。")


# ══════════════════════════════════════════════════════
# 主标签页
# ══════════════════════════════════════════════════════
TAB_LABELS = [
    "💬 交互中心",
    "🧠 记忆矩阵",
    "🗺️ Repo Map",
    "🧬 技能进化",
    "📚 训练数据",
    "📁 工作区看板",
]
tab_chat, tab_summary, tab_repomap, tab_evolution, tab_training, tab_workspace = st.tabs(TAB_LABELS)


# ══════════════════════════════════════════════════════
# Tab 1 · 交互中心
# ══════════════════════════════════════════════════════
with tab_chat:

    # 新手引导（可关闭）
    if st.session_state.show_guide and not st.session_state.messages:
        st.markdown("""
        <div class='guide-box'>
            <strong>👋 欢迎使用 Savant SWE Agent</strong><br><br>
            在下方输入框描述你的编程需求，Agent 会自动帮你 <strong>拆解任务 → 编写代码 → 运行测试 → 验证结果</strong>。<br><br>
            💡 <strong>示例需求：</strong><br>
            &nbsp;·&nbsp; 创建一个 FastAPI 应用，包含用户注册和登录接口<br>
            &nbsp;·&nbsp; 帮我写一个爬取某网站数据的 Python 脚本<br>
            &nbsp;·&nbsp; 给 workspace/app.py 添加错误处理和日志记录<br><br>
            ✋ <strong>人工审批：</strong>Agent 在执行敏感操作（写文件、运行命令）前会暂停等待你确认，确保安全可控。
        </div>
        """, unsafe_allow_html=True)
        if st.button("我已了解，不再显示", key="close_guide"):
            st.session_state.show_guide = False
            st.rerun()

    # 聊天历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── 状态路由 ────────────────────────────────
    current_status = state_values.get("status", "")
    is_terminated = st.session_state.agent_terminated

    # ★ 修复核心：终态（failed/success）或熔断标志置位时，
    #   无论 state_raw.next 是什么，都不再自动流转，直接显示终态 UI。
    if current_status in ("failed",) or is_terminated:
        _render_failed_ui(config)

    elif current_status == "success" and not (state_raw and state_raw.next):
        st.success("🎉 任务已成功完成！你可以继续输入新需求，或查看右侧各标签了解详情。")
        _render_new_task_button()

    elif state_raw and state_raw.next:
        node_now = state_raw.next[0]

        # 人工审批
        if node_now == "tools":
            last_message = state_raw.values["messages"][-1]
            tc_count = len(last_message.tool_calls) if hasattr(last_message, "tool_calls") else 0
            st.markdown(f"""
            <div class='approval-card'>
                <h4>✋ 等待你的确认</h4>
                <p>Agent 准备执行 <strong>{tc_count}</strong> 项操作，请审查后决定是否批准。</p>
            </div>""", unsafe_allow_html=True)

            if last_message.content:
                st.info(f"💭 **Agent 的意图**：{last_message.content}")

            for i, tc in enumerate(last_message.tool_calls, 1):
                with st.expander(f"操作 {i}：`{tc['name']}`", expanded=True):
                    st.markdown(f"<div class='guide-box'>{NODES_META.get('tools', ('','',''))[2]}</div>",
                                unsafe_allow_html=True)
                    st.json(tc["args"])

            col_approve, col_reject = st.columns(2)
            with col_approve:
                if st.button("🚀 批准并执行", use_container_width=True, type="primary"):
                    run_agent_ui(None)
                    st.rerun()
            with col_reject:
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                feedback = st.text_input(
                    "驳回原因（可选）",
                    placeholder="例如：请改用更安全的方式...",
                    key="reject_feedback",
                )
                if st.button("❌ 驳回，重新考虑", use_container_width=True):
                    graph.update_state(
                        config,
                        {"messages": [HumanMessage(
                            content=f"用户驳回操作。建议：{feedback or '请重新检查逻辑。'}"
                        )]},
                    )
                    run_agent_ui(None)
                    st.rerun()

        # Evolution / trajectory_export 自动流转
        elif node_now in ("evolution_reflect", "evolution_generate", "evolution_verify", "trajectory_export"):
            _, lbl, desc = NODES_META.get(node_now, ("", node_now, ""))
            with st.spinner(f"🧬 {lbl} 自动执行中... （{desc}）"):
                run_agent_ui(None)
                st.rerun()

        # 其他节点自动流转
        else:
            _, lbl, _ = NODES_META.get(node_now, ("", node_now, ""))
            with st.spinner(f"⚙️ {lbl} 自动推进中..."):
                run_agent_ui(None)
                st.rerun()

    else:
        # ── 空闲：检测是否有未完成工作，决定续接还是全新启动
        existing_todo = state_values.get("todo_list", [])
        existing_done = state_values.get("completed_tasks", [])
        has_unfinished = bool(existing_todo)

        if has_unfinished:
            total_steps = len(existing_todo) + len(existing_done)
            _cur = existing_todo[0] if existing_todo else "无"
            st.markdown(
                f"<div style='background:#0d1f10;border:1px solid #1a4d1a;"
                f"border-left:4px solid #52c41a;border-radius:10px;"
                f"padding:14px 18px;margin:6px 0;'>"
                f"<div style='color:#95d97a;font-weight:700;font-size:.95rem;margin-bottom:4px;'>"
                f"▶ 检测到未完成任务（{len(existing_done)}/{total_steps} 步已完成）</div>"
                f"<div style='color:#6db55a;font-size:.82rem;'>当前目标：{_cur}</div></div>",
                unsafe_allow_html=True,
            )
            col_resume, col_abandon = st.columns([2, 1])
            with col_resume:
                if st.button("▶️ 继续执行未完成的任务",
                             use_container_width=True, type="primary", key="resume_unfinished"):
                    try:
                        _msg = (
                            "[系统] 请继续完成剩余任务，不要重新规划。"
                            " 待办: " + str(existing_todo) + " 已完成: " + str(existing_done)
                        )
                        graph.update_state(
                            config,
                            {"status": "coding", "iteration_count": 0,
                             "messages": [HumanMessage(content=_msg)]},
                            as_node="coder",
                        )
                        st.session_state.agent_terminated = False
                    except Exception as _e:
                        logger.warning(f"续接写入失败: {_e}")
                    run_agent_ui(None)
                    st.rerun()
            with col_abandon:
                if st.button("🗑️ 放弃，开始新任务",
                             use_container_width=True, key="abandon_unfinished"):
                    _do_reset()
                    st.rerun()

        _placeholder = (
            "补充说明、追加需求，或按上方按钮直接继续..."
            if has_unfinished else
            "描述你的编程需求，Agent 会自动规划并执行..."
        )
        prompt = st.chat_input(_placeholder, key="main_input")
        if prompt:
            st.session_state.agent_terminated = False
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            if has_unfinished:
                try:
                    _supp = (
                        "用户追加说明: " + prompt + "\n\n"
                        "[系统] 请结合用户补充，继续完成剩余任务（不要重新规划）。"
                        " 待办: " + str(existing_todo) + " 已完成: " + str(existing_done)
                    )
                    graph.update_state(
                        config,
                        {"status": "coding", "iteration_count": 0,
                         "messages": [HumanMessage(content=_supp)]},
                        as_node="coder",
                    )
                except Exception as _e:
                    logger.warning(f"追加说明注入失败，降级为全新启动: {_e}")
                    run_agent_ui(_make_init_state(prompt))
                    st.rerun()
                run_agent_ui(None)
            else:
                run_agent_ui(_make_init_state(prompt))
            st.rerun()


# ══════════════════════════════════════════════════════
# Tab 2 · 记忆矩阵
# ══════════════════════════════════════════════════════
with tab_summary:
    st.markdown("### 🧠 长期记忆矩阵")
    st.markdown("""
    <div class='guide-box'>
        当对话历史超过 <strong>15,000 Token</strong> 时，Summarizer 节点会自动将历史压缩为精炼摘要，
        帮助 Agent 在长任务中保持连贯的「记忆」，避免遗忘早期上下文。
    </div>""", unsafe_allow_html=True)

    summary_text = state_values.get("summary", "")
    if summary_text:
        st.markdown(f"""
        <div style='background:#0e1420;border:1px solid #1e2840;border-radius:10px;
                    padding:18px 22px;font-size:.84rem;color:#a0b4c8;line-height:1.7;
                    font-family:Syne,sans-serif;'>
        {summary_text}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:40px;color:#2d4a6b;font-size:.9rem;'>
            💤 暂无压缩记忆<br>
            <span style='font-size:.78rem;'>上下文超出阈值后此处将自动出现摘要</span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# Tab 3 · Repo Map
# ══════════════════════════════════════════════════════
with tab_repomap:
    st.markdown("### 🗺️ 代码库符号地图（Repo Map）")
    st.markdown("""
    <div class='guide-box'>
        Repo Map 是 Agent 的「代码地图」：通过 AST 解析 + PageRank 算法，
        将工作区所有函数和类按<strong>重要性</strong>排序，帮助 Agent 快速定位关键代码，而不是逐行搜索。
    </div>""", unsafe_allow_html=True)

    if not state_values.get("code_index_ready"):
        st.info("⏳ 代码索引尚未构建。启动任意一个任务后，`index_builder` 节点会在第一步自动构建索引。")
    else:
        col_ctrl, col_stat = st.columns([3, 1])
        with col_ctrl:
            map_query = st.text_input(
                "按功能关键词过滤（可选）",
                placeholder="例如：authentication  /  database  /  test",
                key="repo_map_query",
            )
            max_tok = st.slider("地图 Token 上限", 500, 5000, 2000, 250,
                                help="Token 数越大，包含的符号越多，但也会消耗更多上下文空间。")
        with col_stat:
            try:
                from src.agent.swe.code_index import get_index_stats
                idx_stats = get_index_stats()
                st.metric("代码块数", idx_stats.get("chunk_count", 0))
                st.metric("解析方式", "tree-sitter" if idx_stats.get("tree_sitter") else "regex")
            except Exception:
                pass

        if st.button("🔄 刷新地图", use_container_width=True):
            st.rerun()

        repo_map = state_values.get("repo_map", "")
        try:
            from src.agent.swe.code_index import get_repo_map_str, get_index_stats
            if get_index_stats().get("status") != "not_built":
                repo_map = get_repo_map_str(query=map_query, max_tokens=max_tok)
        except Exception as e:
            st.warning(f"无法实时获取 Repo Map：{e}，显示上次快照。")

        if repo_map and len(repo_map) > 10:
            st.code(repo_map, language="text")
        else:
            st.markdown("""
            <div style='text-align:center;padding:30px;color:#2d4a6b;font-size:.88rem;'>
                工作区暂无 Python 文件，或索引尚未就绪。
            </div>""", unsafe_allow_html=True)

        # 实时代码搜索
        st.divider()
        st.markdown("#### 🔍 实时代码搜索")
        st.markdown("""
        <div class='guide-box'>
            支持三种搜索模式：<strong>auto（推荐）</strong>自动选择；
            <strong>semantic</strong> 用自然语言描述功能；
            <strong>keyword</strong> 精确匹配函数名/变量名。
        </div>""", unsafe_allow_html=True)

        col_sq, col_sm = st.columns([3, 1])
        with col_sq:
            search_q = st.text_input("搜索代码...", placeholder="例如：处理用户登录的函数  /  database connection", key="live_search")
        with col_sm:
            search_mode = st.radio("模式", ["auto", "semantic", "keyword"], horizontal=False, key="search_mode")

        if search_q:
            try:
                from src.agent.swe.code_index import search_code_index
                results = search_code_index(search_q, mode=search_mode, top_k=6)
                if results:
                    for c in results:
                        with st.expander(f"`{c.display_name}` — {c.file_path} 第 {c.start_line} 行"):
                            st.code(c.body[:700], language="python")
                            if c.docstring:
                                st.caption(f"📝 {c.docstring[:150]}")
                            if hasattr(c, "calls") and c.calls:
                                st.caption(f"📞 调用：{', '.join(c.calls[:5])}")
                else:
                    st.write("未找到相关代码，请尝试换一种描述方式或切换搜索模式。")
            except Exception as e:
                st.error(f"搜索失败：{e}")


# ══════════════════════════════════════════════════════
# Tab 4 · 技能进化
# ══════════════════════════════════════════════════════
with tab_evolution:
    st.markdown("### 🧬 Capability Evolution Loop")
    st.markdown("""
    <div class='guide-box'>
        每次任务成功完成后，Agent 会自动分析本次解决方案是否具有<strong>复用价值</strong>。
        如果是，会将核心逻辑固化为带有自验证函数的 Python 脚本，存入技能库（<code>skills/</code>）。
        下次遇到类似需求时，Agent 会优先复用技能，减少重复劳动。
    </div>""", unsafe_allow_html=True)

    col_report, col_library = st.columns([1, 1])

    with col_report:
        st.markdown("#### 📋 本次演化报告")
        evo_rpt = state_values.get("evolution_report", "")
        if evo_rpt:
            if "✅" in evo_rpt:
                st.success(evo_rpt)
            elif "❌" in evo_rpt:
                st.error(evo_rpt)
            elif "⚠️" in evo_rpt:
                st.warning(evo_rpt)
            else:
                st.info(evo_rpt)
        else:
            st.markdown("""
            <div style='text-align:center;padding:30px;color:#2d4a6b;font-size:.85rem;'>
                💤 演化报告将在任务成功完成后生成
            </div>""", unsafe_allow_html=True)

        # 草稿预警
        draft_files = list(DRAFT_DIR.glob("*.py")) if DRAFT_DIR.exists() else []
        if draft_files:
            st.warning(f"⚠️ 发现 {len(draft_files)} 个未完成草稿（验证失败或中途中断）")
            for df in draft_files:
                st.caption(f"· {df.name}")

    with col_library:
        st.markdown("#### 💪 技能库全览")
        skill_files = sorted(SKILLS_DIR.glob("*.py"))
        if not skill_files:
            st.markdown("""
            <div style='text-align:center;padding:24px;color:#2d4a6b;font-size:.84rem;'>
                📭 技能库当前为空<br>
                <span style='font-size:.75rem;'>完成第一个任务后，Agent 会自动评估并积累可复用技能</span>
            </div>""", unsafe_allow_html=True)
        else:
            by_cat: dict = {}
            for f in skill_files:
                m = parse_skill_metadata(f)
                by_cat.setdefault(m.get("category", "misc"), []).append((f, m))

            for cat, items in sorted(by_cat.items()):
                badge_cls = BADGE_CLASSES.get(cat, "badge-misc")
                st.markdown(f"<span class='skill-badge {badge_cls}'>{cat.upper()}</span>",
                            unsafe_allow_html=True)
                for f, meta in items:
                    with st.expander(f"**{meta.get('name', f.name)}** — {meta.get('description', '无描述')}"):
                        col_m, col_c = st.columns([1, 2])
                        with col_m:
                            st.markdown(f"- **版本**：v{meta.get('version', '1.0')}")
                            st.markdown(f"- **创建**：{meta.get('created_at', '未知')}")
                            st.markdown(f"- **类别**：{cat}")
                        with col_c:
                            try:
                                code_txt = f.read_text(encoding="utf-8")
                                preview = code_txt[:600] + ("..." if len(code_txt) > 600 else "")
                                st.code(preview, language="python")
                            except Exception:
                                pass


# ══════════════════════════════════════════════════════
# Tab 5 · 训练数据（全新）
# ══════════════════════════════════════════════════════
with tab_training:
    st.markdown("### 📚 LlamaFactory 训练数据中心")
    st.markdown("""
    <div class='guide-box'>
        Agent 每次运行都会自动将执行轨迹保存到 <code>workspace/_training_data/</code>。
        在这里，你可以 <strong>查看积累情况</strong>、<strong>生成训练数据集</strong>，并 <strong>一键生成 LlamaFactory 训练配置</strong>，
        用这些真实任务数据对你的私有模型进行后训练（SFT / DPO / GRPO）。
    </div>""", unsafe_allow_html=True)

    training_dir = WORKSPACE_DIR / "_training_data"

    # ── 统计概览 ──────────────────────────────────
    st.markdown("#### 📊 数据积累概览")

    try:
        from src.agent.swe.training.trajectory_logger import load_trajectories
        from src.agent.swe.training.reward_computer import annotate_trajectories

        records = load_trajectories(WORKSPACE_DIR)
        records = annotate_trajectories(records)

        total_t   = len(records)
        success_t = sum(1 for r in records if r.status == "success")
        failed_t  = sum(1 for r in records if r.status == "failed")
        avg_rew   = round(sum(r.reward for r in records) / max(total_t, 1), 3)
        evolved_t = sum(1 for r in records if r.evolved_skill_name)

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        for col, val, lbl in [
            (mc1, total_t,   "总轨迹"),
            (mc2, success_t, "成功任务"),
            (mc3, failed_t,  "失败任务"),
            (mc4, f"{avg_rew:.3f}", "平均奖励"),
            (mc5, evolved_t, "含技能轨迹"),
        ]:
            col.markdown(f"""
            <div class='metric-card'>
                <div class='val'>{val}</div>
                <div class='lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"暂无训练数据：{e}")
        records = []
        total_t = 0

    # ── 轨迹列表 ──────────────────────────────────
    if records:
        st.markdown("#### 🕐 最近轨迹记录")
        st.markdown("""
        <div class='guide-box'>
            奖励分是 Agent 表现的综合评分（-1 到 1），
            <span style='color:#52c41a'>绿色</span>表示高质量任务，
            <span style='color:#ffa940'>橙色</span>表示中等，
            <span style='color:#ff4d4f'>红色</span>表示失败。只有高质量轨迹才会被纳入 SFT 训练数据。
        </div>""", unsafe_allow_html=True)

        show_count = st.slider("显示最近 N 条", 5, min(50, total_t), min(10, total_t), key="traj_show")
        recent = list(reversed(records))[:show_count]

        for rec in recent:
            rew_color = reward_color(rec.reward)
            bar_width = int(max(0, min(100, (rec.reward + 1) / 2 * 100)))
            status_icon = "✅" if rec.status == "success" else "❌" if rec.status == "failed" else "⏳"
            skill_badge = f"<span class='skill-badge badge-deploy'>🧬 {rec.evolved_skill_name}</span>" if rec.evolved_skill_name else ""

            with st.expander(
                f"{status_icon} `{rec.trajectory_id}` &nbsp;·&nbsp; {rec.task_description[:60]}... &nbsp;·&nbsp; 奖励 {rec.reward:.3f}",
                expanded=False,
            ):
                col_info, col_reward = st.columns([2, 1])
                with col_info:
                    st.markdown(f"""
                    <div style='font-size:.82rem;color:#8090a8;line-height:1.8;'>
                        <div>📅 <strong>时间</strong>：{rec.timestamp[:19].replace('T',' ')}</div>
                        <div>🔁 <strong>迭代轮次</strong>：{rec.iteration_count} / {rec.max_iterations}</div>
                        <div>🛠️ <strong>工具调用</strong>：{len(rec.tool_call_steps)} 次</div>
                        <div>🧪 <strong>测试通过</strong>：{'是 ✅' if rec.test_passed else '否 ❌'}</div>
                        <div>📋 <strong>完成步骤</strong>：{len(rec.completed_tasks)} 项</div>
                        {f'<div>🧬 <strong>关联技能</strong>：{rec.evolved_skill_name}</div>' if rec.evolved_skill_name else ''}
                    </div>""", unsafe_allow_html=True)
                with col_reward:
                    st.markdown(f"""
                    <div style='text-align:center;padding:10px;'>
                        <div style='font-size:1.8rem;font-weight:700;color:{rew_color};font-family:JetBrains Mono,monospace;'>{rec.reward:.3f}</div>
                        <div style='font-size:.65rem;color:#4a5568;margin:4px 0 8px;text-transform:uppercase;letter-spacing:.1em;'>综合奖励</div>
                        <div class='reward-bar-wrap'>
                            <div class='reward-bar-fill' style='width:{bar_width}%;'></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    if rec.reward_breakdown:
                        for k, v in rec.reward_breakdown.items():
                            if k != "total":
                                st.caption(f"{k}: {v:+.3f}")

    st.divider()

    # ── Pipeline 触发 ─────────────────────────────
    st.markdown("#### ⚙️ 生成训练数据集")
    st.markdown("""
    <div class='guide-box'>
        点击下方按钮，Pipeline 会自动将所有轨迹转换为 LlamaFactory 可直接使用的 JSONL 格式：<br>
        <strong>SFT</strong>（成功轨迹监督微调）、<strong>DPO</strong>（成功/失败对比偏好学习）、
        <strong>GRPO</strong>（全量轨迹+奖励分强化学习）、<strong>Skills SFT</strong>（技能库代码生成）。
    </div>""", unsafe_allow_html=True)

    with st.form("pipeline_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            sel_formats = st.multiselect(
                "选择要导出的格式",
                ["sft", "dpo", "grpo", "skills"],
                default=["sft", "dpo", "skills"],
                help="SFT=监督微调  DPO=偏好学习  GRPO=强化学习  skills=技能代码生成",
            )
            min_rew_sft = st.slider(
                "SFT 数据最低奖励阈值",
                0.0, 1.0, 0.4, 0.05,
                help="只有奖励分 ≥ 此值的轨迹才会被纳入 SFT 训练数据。建议 0.4～0.6。",
            )
        with col_f2:
            max_dpo = st.number_input("DPO 最大训练对数", 50, 5000, 500, 50)
            enrich_idx = st.checkbox(
                "用 Three-Index 增强训练指令",
                value=True,
                help="为每条 SFT 指令注入 Repo Map 上下文，帮助模型学会利用代码结构信息。",
            )

        submitted = st.form_submit_button("🚀 生成训练数据集", use_container_width=True, type="primary")

    if submitted:
        if not sel_formats:
            st.error("请至少选择一种导出格式。")
        elif total_t == 0 and "skills" not in sel_formats:
            st.warning("⚠️ 暂无轨迹数据，请先运行几个任务积累数据。可仅导出 skills 格式。")
        else:
            with st.spinner("🔄 数据 Pipeline 运行中，请稍候..."):
                try:
                    from src.agent.swe.training.data_pipeline import run_pipeline
                    report = run_pipeline(
                        workspace_dir=WORKSPACE_DIR,
                        formats=sel_formats,
                        min_reward_sft=min_rew_sft,
                        max_dpo_pairs=max_dpo,
                        enrich_with_index=enrich_idx,
                    )
                    st.success("✅ 训练数据集生成完成！")
                    cnt = report.get("counts", {})
                    c1, c2, c3, c4 = st.columns(4)
                    for col, key, lbl in [
                        (c1, "sft_samples",   "SFT 样本"),
                        (c2, "dpo_pairs",     "DPO 对"),
                        (c3, "grpo_samples",  "GRPO 样本"),
                        (c4, "skills_samples","Skills 样本"),
                    ]:
                        if key in cnt:
                            col.metric(lbl, cnt[key])
                    st.caption(f"📂 输出目录：`{report['training_dir']}`")
                except Exception as e:
                    st.error(f"Pipeline 运行失败：{e}")

    # ── 已导出文件 ────────────────────────────────
    exported = []
    for fname in ["sft_success.jsonl", "dpo_pairs.jsonl", "grpo_all.jsonl", "skills_sft.jsonl", "dataset_info.json"]:
        fpath = training_dir / fname
        if fpath.exists():
            exported.append((fname, fpath))

    if exported:
        st.markdown("#### 📂 已导出文件")
        for fname, fpath in exported:
            col_n, col_s, col_p = st.columns([3, 1, 4])
            col_n.markdown(f"`{fname}`")
            col_s.caption(f"{fpath.stat().st_size // 1024} KB")
            if fname.endswith(".json"):
                with col_p.expander("预览"):
                    try:
                        st.json(json.loads(fpath.read_text()))
                    except Exception:
                        pass

    st.divider()

    # ── LlamaFactory 配置生成器 ───────────────────
    st.markdown("#### 🏋️ 生成 LlamaFactory 训练配置")
    st.markdown("""
    <div class='guide-box'>
        填写下方信息后，点击按钮将自动生成可直接运行的 LlamaFactory YAML 配置文件。<br>
        生成后复制启动命令，在安装了 LlamaFactory 的环境中执行即可开始训练。
    </div>""", unsafe_allow_html=True)

    with st.form("llama_config_form"):
        col_lm1, col_lm2 = st.columns(2)
        with col_lm1:
            lm_mode = st.selectbox(
                "训练范式",
                ["sft", "dpo", "grpo"],
                help="SFT：监督微调（最常用，效果稳定）\nDPO：偏好对比学习\nGRPO：奖励驱动强化学习",
            )
            lm_model = st.text_input(
                "基础模型路径或名称",
                placeholder="例：Qwen/Qwen2.5-7B-Instruct",
                help="支持 HuggingFace 模型名或本地路径。",
            )
        with col_lm2:
            lm_template = st.selectbox(
                "对话模板",
                ["qwen", "llama3", "deepseek", "mistral", "yi", "baichuan2"],
                help="必须与模型系列匹配，否则训练效果会变差。",
            )
            lm_output = st.text_input(
                "配置文件输出目录",
                value="./llamafactory_runs",
            )

        lm_submitted = st.form_submit_button("🛠️ 生成训练配置", use_container_width=True, type="primary")

    if lm_submitted:
        if not lm_model.strip():
            st.error("请填写基础模型路径或名称。")
        else:
            try:
                from src.agent.swe.training.llamafactory_config import generate_and_save_config
                cfg_path = generate_and_save_config(
                    mode=lm_mode,
                    model_name_or_path=lm_model.strip(),
                    data_dir=training_dir,
                    output_dir=Path(lm_output) / lm_mode,
                    template=lm_template,
                )
                st.success(f"✅ 配置文件已生成：`{cfg_path}`")
                st.markdown("**启动训练命令（复制到终端执行）：**")
                st.code(f"pip install llamafactory\nllamafactory-cli train {cfg_path}", language="bash")

                with st.expander("📄 查看配置文件内容"):
                    st.code(cfg_path.read_text(), language="yaml")
            except Exception as e:
                st.error(f"配置生成失败：{e}")


# ══════════════════════════════════════════════════════
# Tab 6 · 工作区看板
# ══════════════════════════════════════════════════════
with tab_workspace:
    st.markdown("### 📁 工作区实时监控")
    st.markdown("""
    <div class='guide-box'>
        工作区（<code>workspace/</code>）是 Agent 的沙盒目录，所有生成的代码文件都存放在此。
        左侧是文件树，右侧可以预览任意文件内容。
    </div>""", unsafe_allow_html=True)

    col_tree, col_view = st.columns([1, 2])

    _IGNORE_DIRS = {".git", "__pycache__", "node_modules", "_evolution_drafts", ".venv"}

    with col_tree:
        st.markdown("**📂 文件结构**")

        def render_file_tree(startpath: Path):
            items = []
            for path in sorted(startpath.rglob("*")):
                if any(x in path.parts for x in _IGNORE_DIRS):
                    continue
                depth = len(path.relative_to(startpath).parts)
                if depth > 6:  # 限制深度
                    continue
                icon = "📄" if path.is_file() else "📁"
                indent = "&nbsp;" * (depth - 1) * 4
                color = "#7cb9d8" if path.is_file() else "#4a8fad"
                items.append(
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:.75rem;"
                    f"color:{color};line-height:1.7;'>{indent}{icon} {path.name}</div>"
                )
            if items:
                st.markdown("\n".join(items), unsafe_allow_html=True)
            else:
                st.caption("工作区暂无文件")

        render_file_tree(WORKSPACE_DIR)

    with col_view:
        files = sorted([
            str(p.relative_to(WORKSPACE_DIR))
            for p in WORKSPACE_DIR.rglob("*")
            if p.is_file() and not any(x in p.parts for x in _IGNORE_DIRS)
        ])

        if files:
            selected = st.selectbox("选择文件预览", files, key="ws_file_select")
            try:
                file_path = WORKSPACE_DIR / selected
                file_size = file_path.stat().st_size

                # 文件信息条
                st.markdown(f"""
                <div style='display:flex;gap:16px;font-size:.75rem;color:#4a5568;
                            font-family:JetBrains Mono,monospace;margin-bottom:8px;'>
                    <span>📄 {selected}</span>
                    <span>💾 {file_size // 1024 if file_size >= 1024 else file_size}{'KB' if file_size >= 1024 else 'B'}</span>
                </div>""", unsafe_allow_html=True)

                if file_size > _FILE_PREVIEW_MAX_BYTES:
                    st.warning(f"文件较大，仅显示前 {_FILE_PREVIEW_MAX_BYTES // 1024} KB。")

                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(_FILE_PREVIEW_MAX_BYTES)

                ext = Path(selected).suffix.lower()
                lang_map = {
                    ".py": "python", ".md": "markdown", ".json": "json",
                    ".yaml": "yaml", ".yml": "yaml", ".sh": "bash",
                    ".js": "javascript", ".ts": "typescript",
                    ".html": "html", ".css": "css", ".txt": "text",
                }
                lang = lang_map.get(ext, "text")
                st.code(content, language=lang)

            except Exception as e:
                st.error(f"读取文件失败：{e}")
        else:
            st.markdown("""
            <div style='text-align:center;padding:40px;color:#2d4a6b;font-size:.88rem;'>
                📭 工作区暂无文件<br>
                <span style='font-size:.75rem;'>开始一个任务后，Agent 生成的代码将出现在这里</span>
            </div>""", unsafe_allow_html=True)
