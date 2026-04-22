# src/agent/swe/tools.py
import ast
import atexit
import logging
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# API 保护与缓存（加 maxsize 防止无界增长）
_SEARCH_CACHE: dict = {}
_URL_CACHE: dict = {}
_CACHE_MAXSIZE = 200
GLOBAL_STATS = {
    "tavily_count": 0,
    "max_tavily": 15,
}

logger = logging.getLogger("SWE_Tools")

# 可选重依赖
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False

# 精确 Token 计数（无 tiktoken 时降级为字符估算）
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def _count_str_tokens(text: str) -> int:
        return len(_ENC.encode(text))

    TIKTOKEN_AVAILABLE = True
except ImportError:
    def _count_str_tokens(text: str) -> int:  # type: ignore[misc]
        return len(text) // 4

    TIKTOKEN_AVAILABLE = False


def _cache_set(cache: dict, key: str, value: str) -> None:
    """带 maxsize 的简单 LRU 写入（dict 在 Python 3.7+ 保持插入顺序）。"""
    if key in cache:
        del cache[key]
    elif len(cache) >= _CACHE_MAXSIZE:
        oldest = next(iter(cache))
        del cache[oldest]
    cache[key] = value


# ==========================================
# 工作区与技能目录
# ==========================================
WORKSPACE_DIR = Path(
    os.getenv(
        "SWE_WORKSPACE_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../workspace"))
    )
)
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

SKILLS_DIR = WORKSPACE_DIR / "skills"
SKILLS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_path(file_path: str) -> Path:
    """
    将相对/绝对路径统一解析到 WORKSPACE_DIR 内，拒绝路径逃逸攻击。
    resolve() 解析 symlink + .. 跳转，彻底消除路径穿越。
    """
    if os.path.isabs(file_path):
        if file_path.startswith("/workspace/"):
            file_path = file_path.replace("/workspace/", "", 1)
        else:
            raise ValueError(f"禁止使用绝对路径，请使用相对路径: {file_path}")

    resolved = (WORKSPACE_DIR / file_path).resolve()
    workspace_resolved = WORKSPACE_DIR.resolve()
    if not str(resolved).startswith(str(workspace_resolved) + os.sep) and resolved != workspace_resolved:
        raise ValueError(f"路径逃逸攻击被拦截: {file_path}")
    return resolved


# ==========================================
# 1. Docker 沙盒管理器
# ==========================================
class DockerSandbox:
    def __init__(self, workspace_dir: Path, image: str = "python:3.10"):
        self.workspace_dir = workspace_dir
        self.image = image
        self.container_name = f"swe_agent_sandbox_{os.getpid()}"
        self.container = None
        self.client = None

        if DOCKER_AVAILABLE:
            try:
                self.client = docker.from_env()
                self.client.ping()
            except Exception as e:
                logger.warning(f"⚠️ 无法连接到 Docker 引擎，沙盒执行工具将被禁用。({e})")
                self.client = None

    def start(self):
        if self.container or not self.client:
            return
        try:
            try:
                self.client.images.get(self.image)
            except docker.errors.ImageNotFound:
                logger.info(f"正在拉取镜像 {self.image}...")
                self.client.images.pull(self.image)

            logger.info(f"🚀 启动 Docker 沙盒: {self.container_name}")
            self.container = self.client.containers.run(
                self.image,
                command="tail -f /dev/null",
                name=self.container_name,
                volumes={str(self.workspace_dir): {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                detach=True,
                auto_remove=True,
            )
            logger.info("📦 沙盒中预装 Linter (flake8)...")
            self.container.exec_run("pip install flake8", detach=True)
        except Exception as e:
            logger.error(f"启动 Docker 容器失败: {e}")
            raise

    def execute(self, command: str, timeout: int = 60) -> str:
        if not self.client:
            return "Error: Docker SDK 未安装或 Docker 引擎未启动。"
        self.start()
        try:
            cmd_list = ["bash", "-c", f"cd /workspace && timeout {timeout}s bash -c {repr(command)}"]
            exit_code, output = self.container.exec_run(cmd_list, workdir="/workspace")
            out_str = output.decode("utf-8", errors="replace").strip()
            if exit_code == 124:
                return f"Error: 命令执行超时 (>{timeout}s)。\nOutput:\n{out_str}"
            return f"Return Code: {exit_code}\nOutput:\n{out_str}"
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def cleanup(self):
        if self.container:
            logger.info(f"🧹 清理沙盒: {self.container_name}")
            try:
                self.container.stop()
            except Exception:
                pass
            self.container = None


sandbox = DockerSandbox(WORKSPACE_DIR)
atexit.register(sandbox.cleanup)


# ==========================================
# 2. 隐式 Linter 钩子
# ==========================================
def _auto_lint(file_path: str) -> str:
    """
    后置钩子：对刚修改的 .py 文件进行自动语法检查。
    使用 normalize_path 防止路径穿越。
    """
    if not file_path.endswith(".py"):
        return ""

    try:
        full_path = normalize_path(file_path)
    except ValueError as e:
        return f"\n\n🚨 [路径安全] 拒绝访问: {e}"

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            ast.parse(f.read(), filename=full_path.name)
    except SyntaxError as e:
        return f"\n\n🚨 [致命语法错误] 行 {e.lineno}: {e.msg}\n请立即使用 edit_file 修复！"

    if sandbox.container:
        workspace_file = f"/workspace/{file_path}"
        exit_code, output = sandbox.container.exec_run(
            ["flake8", workspace_file],
            workdir="/workspace",
        )
        out_str = output.decode("utf-8").strip()
        if exit_code != 0 and out_str:
            critical = [ln for ln in out_str.split("\n") if " F" in ln or " E9" in ln]
            if critical:
                return (
                    f"\n\n⚠️ [Linter 警告] 发现潜在逻辑错误:\n{chr(10).join(critical)}\n"
                    f"请检查是否遗漏了 import 或拼写错误。"
                )

    return "\n\n✅ [Linter] 语法检查通过。"


# ==========================================
# 3. 文件操作工具（集成代码索引增量更新钩子）
# ==========================================
@tool
def read_file(file_path: str) -> str:
    """读取指定文件的内容（带行号）。文件过大时自动截断中间部分以节省 Token。"""
    try:
        full_path = normalize_path(file_path)
        if not full_path.exists():
            return f"Error: 文件 '{file_path}' 不存在。"
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")

        MAX_LINES = 250
        if len(lines) > MAX_LINES:
            head = lines[:100]
            tail = lines[-100:]
            head_str = "\n".join([f"{i+1:4d} | {line}" for i, line in enumerate(head)])
            tail_str = "\n".join(
                [f"{i+1:4d} | {line}" for i, line in enumerate(tail, start=len(lines) - 100)]
            )
            hidden = len(lines) - 200
            return (
                f"--- {file_path} (共 {len(lines)} 行) ---\n{head_str}\n\n"
                f"...[中间 {hidden} 行已折叠。如需查看，请 execute_command 运行"
                f" 'sed -n \"100,150p\" {file_path}']...\n\n{tail_str}"
            )

        numbered = "\n".join([f"{i+1:4d} | {line}" for i, line in enumerate(lines)])
        return f"--- {file_path} ---\n{numbered}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """创建新文件或完全覆盖写入文件。写入后自动更新代码智能索引。"""
    try:
        full_path = normalize_path(file_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        lint_result = _auto_lint(file_path)

        # [新增] 写入后触发代码索引增量更新（仅对 Python 文件）
        if file_path.endswith(".py"):
            _trigger_index_update(file_path)

        return f"Success: 文件 '{file_path}' 已成功写入。{lint_result}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


class EditFileArgs(BaseModel):
    file_path: str = Field(..., description="要修改的文件相对路径")
    start_line: int = Field(..., description="要替换的起始行号（从 1 开始，包含该行）")
    end_line: int = Field(..., description="要替换的结束行号（包含该行）")
    replace_text: str = Field(..., description="新的代码块内容。如果只是想删除代码，可以传空字符串。")


@tool(args_schema=EditFileArgs)
def edit_file(file_path: str, start_line: int, end_line: int, replace_text: str) -> str:
    """通过指定行号范围来局部修改现有文件。修改后自动更新代码智能索引。"""
    try:
        full_path = normalize_path(file_path)
        if not full_path.exists():
            return f"Error: 文件 '{file_path}' 不存在。"

        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")

        if start_line < 1 or start_line > len(lines):
            return f"Error: start_line ({start_line}) 无效。文件共有 {len(lines)} 行。"
        if end_line < start_line:
            return f"Error: end_line ({end_line}) 不能小于 start_line ({start_line})。"

        start_idx = start_line - 1
        end_idx = min(end_line, len(lines))
        new_lines = replace_text.split("\n") if replace_text else []
        lines = lines[:start_idx] + new_lines + lines[end_idx:]

        with open(full_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        lint_result = _auto_lint(file_path)

        # [新增] 编辑后触发代码索引增量更新
        if file_path.endswith(".py"):
            _trigger_index_update(file_path)

        return f"Success: 文件 '{file_path}' 第 {start_line}~{end_line} 行修改成功。{lint_result}"
    except Exception as e:
        return f"Error editing file: {str(e)}"


def _trigger_index_update(file_path: str) -> None:
    """
    触发代码索引增量更新（非阻塞，后台线程执行）。
    只有在索引已构建的情况下才执行，避免任务开始前的冗余解析。
    """
    import threading
    from src.agent.swe.code_index import update_file_index, get_index_stats

    stats = get_index_stats()
    if stats.get("status") == "not_built":
        return  # 索引尚未构建，跳过

    def _do_update():
        try:
            update_file_index(file_path)
        except Exception as e:
            logger.debug(f"代码索引增量更新失败（不影响主流程）: {e}")

    t = threading.Thread(target=_do_update, daemon=True)
    t.start()


# ==========================================
# 4. 终端执行工具
# ==========================================
@tool
def execute_command(command: str) -> str:
    """在安全的 Docker 沙盒环境中执行 Bash/终端命令。"""
    if ("npm start" in command or "python main.py" in command) and "&" not in command:
        return "Error: 检测到可能是阻塞命令。请在命令末尾添加 ' &' 以在后台运行，例如 'npm start &'"
    result = sandbox.execute(command)

    MAX_LEN = 1500
    if len(result) > MAX_LEN:
        head = result[:500]
        tail = result[-1000:]
        return f"{head}\n\n...[输出过长，中间 {len(result) - MAX_LEN} 个字符已截断]...\n\n{tail}"

    return result


# ==========================================
# 5. 代码库搜索工具（保留 grep 版本作为 fallback）
# ==========================================
@tool
def search_codebase(query: str) -> str:
    """
    在整个工作区代码库中全局搜索指定的字符串或关键字（精确 grep 搜索）。
    注意：如果需要语义搜索或符号级搜索，请使用 search_code 工具。
    """
    try:
        results = []
        for root, dirs, files in os.walk(WORKSPACE_DIR):
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d not in ("_evolution_drafts",)
            ]
            for file in files:
                if file.startswith("."):
                    continue
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if query in line:
                                rel = file_path.relative_to(WORKSPACE_DIR)
                                results.append(f"{rel}:{i+1}: {line.rstrip()}")
                except Exception:
                    pass

        if not results:
            return f"未找到包含 '{query}' 的代码。"

        MAX_RESULTS = 15
        output = "\n".join(results[:MAX_RESULTS])
        if len(results) > MAX_RESULTS:
            output += f"\n\n...[结果过多，已截断。共找到 {len(results)} 处，请尝试更精确的 query]..."
        return output
    except Exception as e:
        return f"搜索失败: {str(e)}"


# ==========================================
# 6. 网络搜索与网页读取
# ==========================================
@tool
def search_web(query: str) -> str:
    """搜索互联网获取技术文档、报错解决方案或第三方库用法。"""
    if query in _SEARCH_CACHE:
        logger.info(f"🗂️ [命中缓存] 搜索: {query}")
        return f"[命中缓存] 网络搜索结果:\n\n{_SEARCH_CACHE[query]}"

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        if GLOBAL_STATS["tavily_count"] >= GLOBAL_STATS["max_tavily"]:
            logger.warning("⚠️ Tavily API 调用次数已达上限，降级为 DuckDuckGo。")
        else:
            logger.info(f"🔍 [Tavily] 正在深度搜索: {query}")
            try:
                GLOBAL_STATS["tavily_count"] += 1
                resp = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_api_key,
                        "query": query,
                        "search_depth": "basic",
                        "max_results": 3,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                if not results:
                    return "未找到相关搜索结果。"
                formatted = [
                    f"标题: {r.get('title')}\n内容: {r.get('content')}\n链接: {r.get('url')}"
                    for r in results
                ]
                result_str = "\n---\n".join(formatted)
                _cache_set(_SEARCH_CACHE, query, result_str)
                return f"🔥 网络搜索结果 (Tavily AI):\n\n{result_str}"
            except Exception as e:
                logger.warning(f"Tavily 报错，尝试降级: {e}")

    if not SEARCH_AVAILABLE:
        return "Error: 搜索工具不可用（未安装 duckduckgo_search）。"
    try:
        results = DDGS().text(query, max_results=3)
        formatted = [
            f"标题: {r.get('title')}\n摘要: {r.get('body')}\n链接: {r.get('href')}"
            for r in results
        ]
        return "网络搜索结果 (DuckDuckGo):\n\n" + "\n---\n".join(formatted)
    except Exception as e:
        return f"搜索失败: {str(e)}"


@tool
def read_url(url: str) -> str:
    """
    读取指定 URL 网页的完整内容并解析为纯文本。
    当搜索结果摘要不够用，需要查阅完整官方文档或教程时必须使用。
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"Error: 不支持的 URL scheme '{parsed.scheme}'，仅允许 http/https。"
        if not parsed.netloc:
            return "Error: 无效的 URL，缺少域名。"
    except Exception:
        return "Error: URL 格式无效。"

    if url in _URL_CACHE:
        logger.info(f"🗂️ [命中缓存] 读取网页: {url}")
        return _URL_CACHE[url]

    logger.info(f"🌐 正在抓取网页: {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        if BS4_AVAILABLE:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.extract()
            text = soup.get_text(separator="\n")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            text = "\n".join(lines)
        else:
            text = response.text

        MAX_LEN = 8000
        if len(text) > MAX_LEN:
            text = f"{text[:6000]}\n\n...[省略 {len(text) - MAX_LEN} 字符]...\n\n{text[-2000:]}"

        result_str = f"--- 网页内容: {url} ---\n{text}"
        _cache_set(_URL_CACHE, url, result_str)
        return result_str
    except Exception as e:
        return f"读取网页失败: {str(e)}"


# ==========================================
# 7. 技能库工具
# ==========================================
def _parse_skill_metadata_for_tool(filepath: Path) -> dict:
    """解析技能文件的 YAML frontmatter，返回元数据字典。"""
    meta = {"name": filepath.name, "description": "无描述", "category": "misc", "version": "1.0"}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        in_fm = False
        for line in lines[:15]:
            stripped = line.strip()
            if stripped == "# ---":
                in_fm = not in_fm
                continue
            if in_fm and stripped.startswith("# ") and ": " in stripped:
                kv = stripped[2:]
                key, val = kv.split(": ", 1)
                meta[key.strip()] = val.strip()
    except Exception:
        pass
    return meta


@tool
def list_skills() -> str:
    """
    列出当前工作区积累的所有可复用技能 (Skills) 脚本，展示名称、类别、描述和版本信息。
    遇到常规复杂配置任务时，优先查看并复用已有技能，避免重复造轮子。
    """
    try:
        skill_files = sorted(SKILLS_DIR.glob("*.py"))
        if not skill_files:
            return "当前技能库为空。技能将在任务成功完成后由 Evolution Loop 自动积累。"

        by_category: dict = {}
        for f in skill_files:
            m = _parse_skill_metadata_for_tool(f)
            cat = m.get("category", "misc")
            by_category.setdefault(cat, []).append(m)

        category_icons = {
            "scaffold": "🏗️", "debug": "🔧", "test": "🧪",
            "config": "⚙️", "deploy": "🚀", "misc": "📦",
        }
        lines = ["💪 已掌握的肌肉记忆 (技能库):\n"]
        for cat, skills in sorted(by_category.items()):
            icon = category_icons.get(cat, "📦")
            lines.append(f"{icon} [{cat.upper()}]")
            for s in skills:
                lines.append(
                    f"   • {s['name']}  —  {s.get('description', '无描述')}"
                    f"  (v{s.get('version', '1.0')})"
                )
        lines.append(f"\n共 {len(skill_files)} 个技能。使用 run_skill('文件名') 直接执行。")
        return "\n".join(lines)
    except Exception as e:
        return f"列出技能失败: {str(e)}"


@tool
def run_skill(skill_name: str, args: str = "") -> str:
    """
    执行指定的技能脚本。
    参数:
        skill_name: 技能脚本的文件名，例如 'setup_db.py'
        args: 传递给该脚本的命令行参数（可选，如 '--env prod'）
    """
    if (
        os.sep in skill_name
        or "/" in skill_name
        or "\\" in skill_name
        or not skill_name.endswith(".py")
        or ".." in skill_name
    ):
        return f"Error: 非法的 skill_name '{skill_name}'。只允许形如 'setup_db.py' 的纯文件名。"

    skill_path = SKILLS_DIR / skill_name
    try:
        resolved = skill_path.resolve()
        if not str(resolved).startswith(str(SKILLS_DIR.resolve())):
            return f"Error: 路径逃逸攻击被拦截: {skill_name}"
    except Exception as e:
        return f"Error: 路径解析失败: {e}"

    if not skill_path.exists():
        return f"Error: 技能 '{skill_name}' 不存在。请先使用 list_skills 查看可用技能。"

    command = f"python skills/{skill_name} {args}".strip()
    return sandbox.execute(command)


# ==========================================
# 8. ★ 三层代码智能搜索工具（新增）
# ==========================================

@tool
def search_code(query: str, mode: str = "auto") -> str:
    """
    【智能代码搜索】三索引融合搜索，远优于简单 grep。

    mode 参数：
      "auto"     — 自动选择（含符号特征 → 偏关键字；自然语言描述 → 偏语义）【推荐】
      "semantic" — 纯语义向量搜索（适合：「实现 X 功能的函数」「处理错误的代码」）
      "keyword"  — 纯 BM25 关键字搜索（适合：精确函数名 / 变量名）

    相比 search_codebase（grep），此工具能理解语义、按代码结构切块、支持自然语言查询。
    """
    from src.agent.swe.code_index import search_code_index, get_index_stats

    stats = get_index_stats()
    if stats.get("status") == "not_built" or not stats.get("chunk_count", 0):
        return (
            "代码索引尚未构建（或工作区暂无 Python 文件）。\n"
            "请先使用 search_codebase 工具，或等待 index_builder 节点完成构建。"
        )

    # 合法 mode 校验
    valid_modes = ("auto", "semantic", "keyword")
    if mode not in valid_modes:
        mode = "auto"

    try:
        chunks = search_code_index(query, mode=mode, top_k=8)
    except Exception as e:
        return f"代码搜索异常（降级提示）: {e}\n请使用 search_codebase 工具作为替代。"

    if not chunks:
        return f"未找到与 '{query}' 相关的代码（mode={mode}）。请尝试换 mode 或使用 search_codebase。"

    lines = [f"🔍 代码智能搜索结果（query='{query}', mode={mode}）：\n"]
    for i, c in enumerate(chunks, 1):
        lines.append(f"{'─' * 50}")
        lines.append(f"[{i}] {c.display_name}  ({c.chunk_type})")
        lines.append(f"    📁 {c.file_path}  L{c.start_line}~{c.end_line}")
        lines.append(f"    ✏️ {c.signature}")
        if c.docstring:
            lines.append(f"    📝 {c.docstring[:120]}")
        if c.calls:
            lines.append(f"    📞 调用: {', '.join(c.calls[:6])}")
        lines.append("")

    lines.append(f"💡 使用 get_symbol_context('函数名') 获取完整源码和调用关系。")
    return "\n".join(lines)


@tool
def get_repo_map(query: str = "", max_tokens: int = 2000) -> str:
    """
    【代码库符号地图】返回工作区所有文件的函数/类清单，按重要性（PageRank）排序。

    用途：
      - 任务开始时快速了解代码库全局结构
      - 找到"最该看哪个文件"的答案
      - 理解跨文件的调用关系

    query 参数（可选）：提供后，相关文件/函数会排在前面。
    max_tokens 参数：控制输出长度，避免塞满上下文。
    """
    from src.agent.swe.code_index import get_repo_map_str, get_index_stats

    stats = get_index_stats()
    if stats.get("status") == "not_built":
        return (
            "代码索引尚未构建。工作区可能为空，或 index_builder 节点尚未运行。\n"
            "一旦有文件写入，索引将自动更新。"
        )

    try:
        return get_repo_map_str(query=query, max_tokens=max_tokens)
    except Exception as e:
        return f"Repo Map 生成失败: {e}"


@tool
def get_symbol_context(symbol_name: str) -> str:
    """
    【符号上下文查询】获取指定函数/类的完整信息：
      - 完整源码
      - 函数签名与文档
      - 调用了哪些函数（callees）
      - 被哪些函数调用（callers）

    相当于代码编辑器的「跳转到定义」+「查找所有引用」。
    适合在修改某个函数前，先了解其完整上下文和影响范围。
    """
    from src.agent.swe.code_index import get_symbol_context_str, get_index_stats

    stats = get_index_stats()
    if stats.get("status") == "not_built":
        return "代码索引尚未构建。请先使用 search_codebase 工具。"

    if not symbol_name or not symbol_name.strip():
        return "Error: symbol_name 不能为空。"

    try:
        return get_symbol_context_str(symbol_name.strip())
    except Exception as e:
        return f"符号查询失败: {e}"


# ==========================================
# 导出工具列表
# ==========================================
TOOLS = [
    read_file,
    write_file,
    edit_file,
    execute_command,
    search_code,          # ★ 新增：三索引智能搜索（主要代码搜索工具）
    get_repo_map,         # ★ 新增：Repo Map
    get_symbol_context,   # ★ 新增：符号上下文
    search_codebase,      # 保留：grep 精确搜索（作为 fallback）
    search_web,
    read_url,
    list_skills,
    run_skill,
]
