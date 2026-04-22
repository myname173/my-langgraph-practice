# web_app.py
"""
Streamlit 本地开发 / 演示界面。
生产部署请使用 src/api.py (FastAPI)。
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
from pathlib import Path
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.swe.graph import graph
from src.agent.swe.tools import WORKSPACE_DIR, GLOBAL_STATS, SKILLS_DIR
from src.agent.swe.evolution import parse_skill_metadata, DRAFT_DIR
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="Savant SWE Pro Console",
    page_icon="🛰️",
    layout="wide",
)

st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    }
    .status-card {
        background-color: #1e2130;
        border-left: 5px solid #00f2fe;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .step-active   { color: #00f2fe; font-weight: bold; border-bottom: 2px solid #00f2fe; }
    .step-inactive { color: #4e525e; }
    .index-ready   { color: #00e676; }
    .index-pending { color: #ffd54f; }
    </style>
""", unsafe_allow_html=True)

_THREAD_ID_RE = re.compile(r"^[\w\-]{1,64}$")
_FILE_PREVIEW_MAX_BYTES = 256 * 1024  # 256 KB

# ==========================================
# Session 初始化
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []


def get_safe_state(config: dict) -> dict:
    """安全地从 checkpointer 读取当前图状态，失败时返回空默认值。"""
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            raise ValueError("state is empty")
        return state.values
    except Exception:
        return {
            "todo_list": [],
            "completed_tasks": [],
            "iteration_count": 0,
            "summary": "",
            "status": "coding",
            "reviewer_reject_count": 0,
            "evolution_skill_draft": "",
            "evolution_report": "",
            "code_index_ready": False,
            "repo_map": "",
        }


# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.title("🛰️ 领航员控制台")

    raw_thread_id = st.text_input("会话 ID", value="swe_pro_001")
    if not _THREAD_ID_RE.match(raw_thread_id):
        st.error("⚠️ 会话 ID 只允许字母、数字、下划线、连字符，长度 1~64")
        st.stop()
    thread_id = raw_thread_id
    config = {"configurable": {"thread_id": thread_id}}

    state_values = get_safe_state(config)

    st.subheader("📊 总体完成度")
    todo = state_values.get("todo_list", [])
    done = state_values.get("completed_tasks", [])
    total = len(todo) + len(done)
    progress_val = (len(done) / total) if total > 0 else 0

    st.progress(progress_val)
    st.caption(f"已完成 {len(done)} / 总计 {total} 项任务")

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tavily", f"{GLOBAL_STATS['tavily_count']}/{GLOBAL_STATS['max_tavily']}")
    with col2:
        st.metric("迭代轮次", state_values.get("iteration_count", 0))
    with col3:
        skill_count = len(list(SKILLS_DIR.glob("*.py")))
        st.metric("🧬 技能", skill_count)

    st.divider()

    # ★ 新增：代码索引状态面板
    st.subheader("🔍 代码智能索引")
    code_index_ready = state_values.get("code_index_ready", False)
    if code_index_ready:
        st.markdown("<span class='index-ready'>✅ 三层索引已就绪</span>", unsafe_allow_html=True)
        try:
            from src.agent.swe.code_index import get_index_stats
            stats = get_index_stats()
            if stats.get("status") != "not_built":
                st.caption(
                    f"📦 {stats.get('chunk_count', 0)} 个代码块 | "
                    f"向量:{'✓' if stats.get('semantic_ready') else '✗'} | "
                    f"BM25:{'✓' if stats.get('bm25_ready') else '✗'} | "
                    f"图:{'✓' if stats.get('graph_ready') else '✗'}"
                )
        except Exception:
            pass
    else:
        st.markdown("<span class='index-pending'>⏳ 索引未构建（任务启动后自动构建）</span>", unsafe_allow_html=True)

    st.divider()

    st.subheader("📍 任务蓝图")
    if done:
        with st.expander("查看已完成项目"):
            for d in done:
                st.write(f"✅ {d}")
    if todo:
        st.info(f"🎯 当前任务: {todo[0]}")
        if len(todo) > 1:
            with st.expander("后续任务"):
                for t in todo[1:]:
                    st.write(f"⏳ {t}")
    else:
        st.caption("暂无待处理任务")

    evolution_report = state_values.get("evolution_report", "")
    if evolution_report:
        st.divider()
        if "✅" in evolution_report:
            st.success("🧬 Evolution 完成")
        elif "❌" in evolution_report:
            st.error("🧬 Evolution 失败")
        else:
            st.info("🧬 Evolution 已分析")


# ==========================================
# 主界面 Pipeline 可视化
# ==========================================
NODES_MAP = {
    "planner":            "📋 需求拆解",
    "index_builder":      "🔍 索引构建",   # ★ 新增
    "coder":              "💻 代码编写",
    "tools":              "🛠️ 工具执行",
    "reviewer":           "🔍 质量审查",
    "summarizer":         "🧠 记忆压缩",
    "task_manager":       "🗺️ 进度更新",
    "evolution_reflect":  "🔬 演化反思",
    "evolution_generate": "🧬 技能生成",
    "evolution_verify":   "✅ 技能验证",
}

state_raw = graph.get_state(config)
next_node = state_raw.next[0] if (state_raw and state_raw.next) else "idle"

st.write("### 🛤️ 实时位置流水线")
# ★ 新增 index_builder 到主流程节点列表
main_nodes = ["planner", "index_builder", "coder", "tools", "reviewer", "summarizer", "task_manager"]
evo_nodes = ["evolution_reflect", "evolution_generate", "evolution_verify"]

cols = st.columns(len(main_nodes))
for i, node_id in enumerate(main_nodes):
    with cols[i]:
        name = NODES_MAP[node_id]
        if next_node == node_id:
            st.markdown(f"<div class='step-active'>{name}</div>", unsafe_allow_html=True)
            st.caption("● 正在此处...")
        else:
            st.markdown(f"<div class='step-inactive'>{name}</div>", unsafe_allow_html=True)

st.caption("🧬 Capability Evolution Loop（任务成功后触发）")
evo_cols = st.columns(len(evo_nodes))
for i, node_id in enumerate(evo_nodes):
    with evo_cols[i]:
        name = NODES_MAP[node_id]
        if next_node == node_id:
            st.markdown(f"<div class='step-active'>{name}</div>", unsafe_allow_html=True)
            st.caption("● 正在此处...")
        else:
            st.markdown(f"<div class='step-inactive'>{name}</div>", unsafe_allow_html=True)

st.divider()


# ==========================================
# 核心运行逻辑
# ==========================================
def run_agent_ui(input_data=None):
    """流式运行 Agent 并渲染到 UI。"""
    try:
        no_tool_count = 0
        with st.status("🚀 Savant 正在处理任务...", expanded=True) as status:
            for event in graph.stream(input_data, config=config, stream_mode="updates"):
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

                # ★ index_builder 进度提示
                if node_name == "index_builder":
                    st.toast("🔍 代码智能索引构建中...", icon="📦")

                if node_name in ("evolution_reflect", "evolution_generate", "evolution_verify"):
                    st.toast(f"🧬 {NODES_MAP.get(node_name, node_name)}...", icon="🔬")

                if node_name == "coder" and "messages" in node_output:
                    last_m = node_output["messages"][-1]
                    if hasattr(last_m, "tool_calls") and last_m.tool_calls:
                        no_tool_count = 0
                        for tc in last_m.tool_calls:
                            st.toast(f"🛠️ 准备执行工具: {tc['name']}", icon="🔧")
                    else:
                        no_tool_count += 1

                if no_tool_count > 3:
                    st.error("🚨 监测到 Agent 陷入死循环或无进展，已自动熔断。")
                    break

            status.update(label="✅ 当前阶段处理完成", state="complete", expanded=False)

    except Exception as e:
        st.error(f"❌ 运行异常: {str(e)}")


# ==========================================
# Tabs（新增 Repo Map 标签）
# ==========================================
tab_chat, tab_summary, tab_repomap, tab_evolution, tab_workspace = st.tabs([
    "💬 交互中心", "🧠 记忆矩阵", "🗺️ Repo Map", "🧬 技能进化", "📁 工作区看板"
])

# ==========================================
# Tab 1：交互中心
# ==========================================
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    current_state = state_raw

    if current_state and current_state.next:
        node_now = current_state.next[0]

        if node_now == "tools":
            last_message = current_state.values["messages"][-1]
            st.markdown(
                f"""<div class='status-card'>
                    <h4>✋ 人工审批请求</h4>
                    <p>Agent 已准备好执行以下 <b>{len(last_message.tool_calls)}</b> 项操作。</p>
                </div>""",
                unsafe_allow_html=True,
            )
            if last_message.content:
                st.info(f"**意图分析**: {last_message.content}")

            for tc in last_message.tool_calls:
                with st.expander(f"🛠️ 待操作: {tc['name']}", expanded=True):
                    st.json(tc["args"])

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🚀 批准并执行", use_container_width=True, type="primary"):
                    run_agent_ui(None)
                    st.rerun()
            with c2:
                feedback = st.text_input("如有疑问，请输入修改意见：")
                if st.button("❌ 驳回请求", use_container_width=True):
                    graph.update_state(
                        config,
                        {"messages": [HumanMessage(
                            content=f"用户驳回操作。建议：{feedback if feedback else '请重新检查逻辑。'}"
                        )]},
                    )
                    run_agent_ui(None)
                    st.rerun()

        elif node_now in ("evolution_reflect", "evolution_generate", "evolution_verify"):
            with st.spinner(f"🧬 {NODES_MAP.get(node_now, node_now)} 自动执行中..."):
                run_agent_ui(None)
                st.rerun()

        else:
            with st.spinner(f"Agent 正在从 {NODES_MAP.get(node_now, node_now)} 自动流转..."):
                run_agent_ui(None)
                st.rerun()

    else:
        if prompt := st.chat_input("输入新的需求或追加指令..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            init_data = {
                "messages": [HumanMessage(content=prompt)],
                "task_description": prompt,
                "plan": [],
                "todo_list": [],
                "completed_tasks": [],
                "summary": "",
                "workspace": str(WORKSPACE_DIR.absolute()),
                "iteration_count": 0,
                "max_iterations": 25,
                "status": "coding",
                "test_passed": False,
                "reviewer_reject_count": 0,
                "evolution_skill_draft": "",
                "evolution_report": "",
                "code_index_ready": False,   # ★ 新增
                "repo_map": "",              # ★ 新增
            }
            run_agent_ui(init_data)
            st.rerun()


# ==========================================
# Tab 2：记忆矩阵
# ==========================================
with tab_summary:
    st.subheader("🧠 长期记忆矩阵")
    summary_text = state_values.get("summary", "")
    if summary_text:
        st.info(summary_text)
        st.caption("此摘要由 Summarizer 节点在历史消息超过 15,000 Token 时自动生成。")
    else:
        st.write("暂无压缩记忆。当上下文超过 15,000 Token 时，Summarizer 节点会自动触发。")


# ==========================================
# Tab 3：Repo Map（★ 新增）
# ==========================================
with tab_repomap:
    st.subheader("🗺️ 代码库符号地图 (Repo Map)")

    if not state_values.get("code_index_ready"):
        st.info("代码索引尚未构建。启动一个任务后，index_builder 节点会自动构建索引。")
    else:
        col_controls, col_stats = st.columns([3, 1])
        with col_controls:
            map_query = st.text_input(
                "按关键词过滤（可选）",
                placeholder="例如：authentication、database、test",
                key="repo_map_query",
            )
            max_tok = st.slider("最大 Token 数", 500, 5000, 2000, 250)

        with col_stats:
            try:
                from src.agent.swe.code_index import get_index_stats
                stats = get_index_stats()
                st.metric("代码块数", stats.get("chunk_count", 0))
                st.metric("解析方式", "tree-sitter" if stats.get("tree_sitter") else "regex")
            except Exception:
                pass

        if st.button("🔄 刷新 Repo Map", use_container_width=True):
            st.rerun()

        # 显示 Repo Map
        repo_map = state_values.get("repo_map", "")
        try:
            from src.agent.swe.code_index import get_repo_map_str, get_index_stats
            if get_index_stats().get("status") != "not_built":
                repo_map = get_repo_map_str(query=map_query, max_tokens=max_tok)
        except Exception as e:
            st.warning(f"无法从引擎获取 Repo Map: {e}，显示状态快照。")

        if repo_map and len(repo_map) > 10:
            st.code(repo_map, language="text")
        else:
            st.write("Repo Map 为空（工作区无 Python 文件或索引未就绪）。")

        # 实时符号搜索
        st.divider()
        st.markdown("#### 🔍 实时代码搜索")
        search_q = st.text_input("搜索代码（语义 / 关键字 / 自动）", key="live_search")
        search_mode = st.radio("搜索模式", ["auto", "semantic", "keyword"], horizontal=True)
        if search_q:
            try:
                from src.agent.swe.code_index import search_code_index
                results = search_code_index(search_q, mode=search_mode, top_k=6)
                if results:
                    for c in results:
                        with st.expander(f"{c.display_name}  ({c.file_path} L{c.start_line})"):
                            st.code(c.body[:600], language="python")
                            if c.docstring:
                                st.caption(c.docstring[:150])
                else:
                    st.write("未找到相关代码。")
            except Exception as e:
                st.error(f"搜索失败: {e}")


# ==========================================
# Tab 4：技能进化
# ==========================================
with tab_evolution:
    st.subheader("🧬 Capability Evolution Loop")

    col_report, col_library = st.columns([1, 1])

    with col_report:
        st.markdown("#### 📋 本次演化报告")
        evolution_report = state_values.get("evolution_report", "")
        if evolution_report:
            if "✅" in evolution_report:
                st.success(evolution_report)
            elif "❌" in evolution_report:
                st.error(evolution_report)
            elif "⚠️" in evolution_report:
                st.warning(evolution_report)
            else:
                st.info(evolution_report)
        else:
            st.write("演化报告将在任务成功完成后出现。")

    with col_library:
        st.markdown("#### 💪 技能库全览")
        skill_files = sorted(SKILLS_DIR.glob("*.py"))
        if not skill_files:
            st.write("技能库当前为空，完成第一个任务后将自动积累。")
        else:
            by_category: dict = {}
            for f in skill_files:
                m = parse_skill_metadata(f)
                cat = m.get("category", "misc")
                by_category.setdefault(cat, []).append((f, m))

            category_icons = {
                "scaffold": "🏗️", "debug": "🔧", "test": "🧪",
                "config": "⚙️", "deploy": "🚀", "misc": "📦",
            }
            for cat, skills_in_cat in sorted(by_category.items()):
                icon = category_icons.get(cat, "📦")
                st.markdown(f"**{icon} {cat.upper()}**")
                for f, meta in skills_in_cat:
                    with st.expander(f"{meta.get('name', f.name)} — {meta.get('description', '无描述')}"):
                        col_meta, col_code = st.columns([1, 2])
                        with col_meta:
                            st.markdown(f"- **版本**: v{meta.get('version', '1.0')}")
                            st.markdown(f"- **创建**: {meta.get('created_at', '未知')}")
                        with col_code:
                            try:
                                code = f.read_text(encoding="utf-8")
                                st.code(code[:800] + ("..." if len(code) > 800 else ""), language="python")
                            except Exception:
                                pass

    draft_files = list(DRAFT_DIR.glob("*.py")) if DRAFT_DIR.exists() else []
    if draft_files:
        st.divider()
        st.warning(f"⚠️ 发现 {len(draft_files)} 个未完成的演化草稿：")
        for df in draft_files:
            st.text(f"  - {df.name}")


# ==========================================
# Tab 5：工作区看板
# ==========================================
with tab_workspace:
    st.subheader("📁 工作区实时监控")
    col_tree, col_view = st.columns([1, 2])

    with col_tree:
        st.caption("📂 文件树")

        def list_files(startpath: Path):
            for path in sorted(startpath.rglob("*")):
                if any(x in path.parts for x in [".git", "__pycache__", "node_modules", "_evolution_drafts"]):
                    continue
                depth = len(path.relative_to(startpath).parts)
                icon = "📄" if path.is_file() else "📁"
                st.text(f"{'  ' * (depth - 1)}{icon} {path.name}")

        list_files(WORKSPACE_DIR)

    with col_view:
        files = [
            str(p.relative_to(WORKSPACE_DIR))
            for p in WORKSPACE_DIR.rglob("*")
            if p.is_file()
            and not any(x in p.parts for x in [".git", "__pycache__", "_evolution_drafts"])
        ]
        if files:
            selected = st.selectbox("预览文件内容", sorted(files))
            try:
                file_path = WORKSPACE_DIR / selected
                file_size = file_path.stat().st_size
                if file_size > _FILE_PREVIEW_MAX_BYTES:
                    st.warning(f"文件过大 ({file_size // 1024} KB)，只显示前 {_FILE_PREVIEW_MAX_BYTES // 1024} KB。")
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(_FILE_PREVIEW_MAX_BYTES)
                lang = (
                    "python" if selected.endswith(".py")
                    else "markdown" if selected.endswith(".md")
                    else "json" if selected.endswith(".json")
                    else "text"
                )
                st.code(content, language=lang)
            except Exception as e:
                st.error(f"读取文件失败: {e}")
        else:
            st.caption("工作区暂无文件。")
