# src/agent/swe/code_index.py
"""
Three-Index Code Intelligence Engine
=====================================
将 Agent Brain Three-Index 架构 + Aider Repo Map 的核心思路集成到本项目，
彻底替换掉原有的 grep 方式搜索。

三层索引：
  1. AST 结构索引  — tree-sitter 解析，按函数/类切块，保留完整元数据
  2. 语义向量索引  — sentence-transformers + numpy 余弦相似度，理解自然语言查询
  3. BM25 关键字索引 — rank_bm25，精确符号名 / 标识符匹配

Repo Map（来自 Aider 方案）：
  — NetworkX 构建调用图，PageRank 评估符号重要性
  — 结合查询相关性生成"最相关符号地图"，压缩进 context

所需包（均可选，缺失时逐级降级）：
  pip install tree-sitter-languages rank_bm25 sentence-transformers networkx
"""

import ast as pyast
import hashlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("SWE_CodeIndex")

# ==========================================
# 可选依赖检测（逐级降级）
# ==========================================

try:
    from tree_sitter_languages import get_language, get_parser as _ts_get_parser
    _TS_PY_LANG = get_language("python")
    _TS_PY_PARSER = _ts_get_parser("python")
    TREE_SITTER_AVAILABLE = True
except Exception:
    TREE_SITTER_AVAILABLE = False
    _TS_PY_LANG = None
    _TS_PY_PARSER = None

try:
    from sentence_transformers import SentenceTransformer
    _EMBED_MODEL_NAME = os.getenv("SWE_EMBED_MODEL", "all-MiniLM-L6-v2")
    # 延迟加载，避免 import 时就下载模型
    _embed_model: Optional[SentenceTransformer] = None
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    _embed_model = None

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


def _get_embed_model() -> Optional["SentenceTransformer"]:
    """懒加载 embedding 模型（线程安全）。"""
    global _embed_model
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    if _embed_model is None:
        try:
            logger.info(f"正在加载 embedding 模型: {_EMBED_MODEL_NAME} ...")
            _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
            logger.info("Embedding 模型加载完成。")
        except Exception as e:
            logger.warning(f"无法加载 embedding 模型，语义搜索将不可用: {e}")
            return None
    return _embed_model


# ==========================================
# 数据结构
# ==========================================

@dataclass
class CodeChunk:
    """一个代码单元（函数、方法或类）。"""
    chunk_id: str           # 唯一 ID，格式：file_path::name::start_line
    file_path: str          # 相对 workspace 的路径
    chunk_type: str         # "function" | "method" | "class"
    name: str               # 函数/类名
    signature: str          # 完整签名，如 "def foo(x: int) -> str"
    docstring: str          # 文档字符串（最多 300 字符）
    body: str               # 完整源码（最多 3000 字符，用于向量化）
    start_line: int         # 1-indexed
    end_line: int           # 1-indexed
    calls: list[str] = field(default_factory=list)   # 调用的函数名
    language: str = "python"
    parent_class: str = ""  # 对于方法：所属类名

    @property
    def embed_text(self) -> str:
        """用于向量化的文本：名称 + 签名 + 文档 + 少量正文。"""
        parts = [self.name, self.signature]
        if self.docstring:
            parts.append(self.docstring[:200])
        # 正文取前 400 字符
        parts.append(self.body[:400])
        return "\n".join(parts)

    @property
    def display_name(self) -> str:
        if self.parent_class:
            return f"{self.parent_class}.{self.name}"
        return self.name


# ==========================================
# 1. AST 解析器
# ==========================================

def _node_text(node) -> str:
    return node.text.decode("utf-8") if node.text else ""


def _extract_calls_from_node(node) -> list[str]:
    """递归提取 tree-sitter 节点中所有函数调用名。"""
    calls: list[str] = []

    def _walk(n):
        if n.type == "call":
            for child in n.children:
                if child.type == "identifier":
                    calls.append(_node_text(child))
                elif child.type == "attribute":
                    # 取 attribute 节点的最后一个 identifier
                    for sub in reversed(child.children):
                        if sub.type == "identifier":
                            calls.append(_node_text(sub))
                            break
                break  # 只取 function 部分第一个 identifier
        for child in n.children:
            _walk(child)

    _walk(node)
    return calls


def _extract_docstring_from_source(source: str) -> str:
    """用 Python 内置 ast 从函数/类源码中提取文档字符串。"""
    try:
        tree = pyast.parse(source)
        for node in pyast.walk(tree):
            if isinstance(node, (pyast.FunctionDef, pyast.AsyncFunctionDef, pyast.ClassDef)):
                ds = pyast.get_docstring(node)
                return (ds or "")[:300]
    except Exception:
        pass
    return ""


def _ts_parse_python(source: str, rel_path: str) -> list[CodeChunk]:
    """使用 tree-sitter 解析 Python 文件，提取所有函数和类。"""
    chunks: list[CodeChunk] = []
    source_bytes = source.encode("utf-8")
    source_lines = source.split("\n")

    try:
        tree = _TS_PY_PARSER.parse(source_bytes)
    except Exception as e:
        logger.debug(f"tree-sitter 解析失败 {rel_path}: {e}")
        return []

    def _visit(node, parent_class: str = ""):
        if node.type in ("function_definition", "decorated_definition"):
            # 处理 @decorator + function_definition 的组合
            actual_node = node
            if node.type == "decorated_definition":
                for child in node.children:
                    if child.type == "function_definition":
                        actual_node = child
                        break
                if actual_node is node:
                    return  # 不是函数，跳过

            chunk = _build_function_chunk(actual_node, source_lines, source, rel_path, parent_class)
            if chunk:
                chunks.append(chunk)
                # 递归查找嵌套函数（在函数体内）
                for child in actual_node.children:
                    if child.type == "block":
                        for stmt in child.children:
                            _visit(stmt, parent_class)

        elif node.type == "class_definition":
            class_chunk = _build_class_chunk(node, source_lines, source, rel_path)
            if class_chunk:
                chunks.append(class_chunk)

            # 提取类名，递归处理方法
            class_name = ""
            for child in node.children:
                if child.type == "identifier":
                    class_name = _node_text(child)
                    break

            for child in node.children:
                if child.type == "block":
                    for stmt in child.children:
                        _visit(stmt, parent_class=class_name)
        else:
            for child in node.children:
                _visit(child, parent_class)

    _visit(tree.root_node)
    return chunks


def _build_function_chunk(
    node, source_lines: list[str], source: str, rel_path: str, parent_class: str
) -> Optional[CodeChunk]:
    """从 tree-sitter function_definition 节点构建 CodeChunk。"""
    name = ""
    params_text = ""
    return_type_text = ""

    for child in node.children:
        if child.type == "identifier" and not name:
            name = _node_text(child)
        elif child.type == "parameters":
            params_text = _node_text(child)
        elif child.type == "type":
            return_type_text = _node_text(child)

    if not name or name.startswith("_") and name.startswith("__") and name.endswith("__"):
        # 保留 dunder 方法
        pass
    if not name:
        return None

    start_line = node.start_point[0]   # 0-indexed
    end_line = node.end_point[0]       # 0-indexed
    body_lines = source_lines[start_line: end_line + 1]
    body_source = "\n".join(body_lines)

    signature = f"def {name}{params_text}"
    if return_type_text:
        signature += f" -> {return_type_text}"

    calls = _extract_calls_from_node(node)
    docstring = _extract_docstring_from_source(body_source)

    chunk_id = f"{rel_path}::{name}::{start_line + 1}"
    return CodeChunk(
        chunk_id=chunk_id,
        file_path=rel_path,
        chunk_type="method" if parent_class else "function",
        name=name,
        signature=signature,
        docstring=docstring,
        body=body_source[:3000],
        start_line=start_line + 1,
        end_line=end_line + 1,
        calls=list(set(calls))[:30],
        language="python",
        parent_class=parent_class,
    )


def _build_class_chunk(
    node, source_lines: list[str], source: str, rel_path: str
) -> Optional[CodeChunk]:
    """从 tree-sitter class_definition 节点构建 CodeChunk。"""
    name = ""
    bases_text = ""

    for child in node.children:
        if child.type == "identifier" and not name:
            name = _node_text(child)
        elif child.type == "argument_list":
            bases_text = _node_text(child)

    if not name:
        return None

    start_line = node.start_point[0]
    end_line = node.end_point[0]
    body_lines = source_lines[start_line: end_line + 1]
    body_source = "\n".join(body_lines)

    signature = f"class {name}"
    if bases_text:
        signature += f"({bases_text})"

    docstring = _extract_docstring_from_source(body_source)
    chunk_id = f"{rel_path}::class::{name}::{start_line + 1}"

    return CodeChunk(
        chunk_id=chunk_id,
        file_path=rel_path,
        chunk_type="class",
        name=name,
        signature=signature,
        docstring=docstring,
        body=body_source[:1500],  # 类体较长，截短
        start_line=start_line + 1,
        end_line=end_line + 1,
        calls=[],
        language="python",
        parent_class="",
    )


def _regex_parse_python(source: str, rel_path: str) -> list[CodeChunk]:
    """
    Fallback：用正则表达式解析 Python 文件。
    精度低于 tree-sitter，但不依赖额外包。
    """
    chunks: list[CodeChunk] = []
    lines = source.split("\n")
    current_class = ""
    class_indent = -1

    func_re = re.compile(r'^(\s*)(?:async\s+)?def\s+(\w+)\s*(\([^)]*\))\s*(?:->\s*([^:]+))?\s*:')
    class_re = re.compile(r'^(\s*)class\s+(\w+)\s*(?:\(([^)]*)\))?\s*:')

    for i, line in enumerate(lines):
        cm = class_re.match(line)
        if cm:
            indent = len(cm.group(1))
            if indent <= class_indent:
                current_class = ""
                class_indent = -1
            current_class = cm.group(2)
            class_indent = indent

            docstring = ""
            if i + 1 < len(lines) and '"""' in lines[i + 1]:
                docstring = lines[i + 1].strip().strip('"')

            sig = f"class {cm.group(2)}"
            if cm.group(3):
                sig += f"({cm.group(3)})"
            chunk_id = f"{rel_path}::class::{cm.group(2)}::{i + 1}"
            chunks.append(CodeChunk(
                chunk_id=chunk_id, file_path=rel_path, chunk_type="class",
                name=cm.group(2), signature=sig, docstring=docstring,
                body="\n".join(lines[i:min(i + 50, len(lines))]),
                start_line=i + 1, end_line=min(i + 50, len(lines)),
            ))
            continue

        fm = func_re.match(line)
        if fm:
            indent = len(fm.group(1))
            if class_indent >= 0 and indent <= class_indent:
                current_class = ""
                class_indent = -1

            name = fm.group(2)
            params = fm.group(3) or "()"
            ret = (fm.group(4) or "").strip()
            sig = f"def {name}{params}"
            if ret:
                sig += f" -> {ret}"

            # 提取函数体（向下找直到缩进变浅）
            body_lines = [line]
            for j in range(i + 1, min(i + 100, len(lines))):
                next_line = lines[j]
                if next_line.strip() == "":
                    body_lines.append(next_line)
                elif len(next_line) - len(next_line.lstrip()) <= indent and next_line.strip():
                    break
                else:
                    body_lines.append(next_line)

            body = "\n".join(body_lines)
            docstring = _extract_docstring_from_source(body)
            # 简单提取调用（非常粗糙）
            call_re = re.compile(r'\b(\w+)\s*\(')
            calls = list(set(call_re.findall(body)))[:20]

            chunk_id = f"{rel_path}::{name}::{i + 1}"
            chunks.append(CodeChunk(
                chunk_id=chunk_id, file_path=rel_path,
                chunk_type="method" if current_class else "function",
                name=name, signature=sig, docstring=docstring,
                body=body[:3000], start_line=i + 1, end_line=i + len(body_lines),
                calls=calls, language="python", parent_class=current_class,
            ))

    return chunks


def parse_file(abs_path: Path, workspace_dir: Path) -> list[CodeChunk]:
    """解析单个文件，返回 CodeChunk 列表。"""
    try:
        rel_path = str(abs_path.relative_to(workspace_dir))
        source = abs_path.read_text(encoding="utf-8", errors="replace")

        if abs_path.suffix == ".py":
            if TREE_SITTER_AVAILABLE:
                chunks = _ts_parse_python(source, rel_path)
            else:
                chunks = _regex_parse_python(source, rel_path)
            return chunks

    except Exception as e:
        logger.debug(f"解析文件失败 {abs_path}: {e}")
    return []


# ==========================================
# 2. 语义向量索引（sentence-transformers + numpy）
# ==========================================

class _SemanticIndex:
    """内存中的向量索引，使用余弦相似度。"""

    def __init__(self):
        self._chunks: list[CodeChunk] = []
        self._matrix: Optional[np.ndarray] = None  # (n_chunks, embed_dim)
        self._lock = threading.Lock()

    def _embed_texts(self, texts: list[str]) -> Optional[np.ndarray]:
        model = _get_embed_model()
        if model is None or not texts:
            return None
        try:
            vecs = model.encode(texts, show_progress_bar=False, batch_size=32)
            # L2 归一化
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return (vecs / norms).astype(np.float32)
        except Exception as e:
            logger.warning(f"向量化失败: {e}")
            return None

    def build(self, chunks: list[CodeChunk]) -> None:
        texts = [c.embed_text for c in chunks]
        matrix = self._embed_texts(texts)
        with self._lock:
            self._chunks = list(chunks)
            self._matrix = matrix
        if matrix is not None:
            logger.info(f"语义索引构建完成：{len(chunks)} 个 chunk，维度={matrix.shape[1]}")
        else:
            logger.info(f"语义索引：embedding 不可用，将降级到 BM25")

    def update_file(self, file_path: str, new_chunks: list[CodeChunk]) -> None:
        """增量更新：删除该文件的旧 chunk，插入新 chunk。"""
        with self._lock:
            keep_mask = [c.file_path != file_path for c in self._chunks]
            kept_chunks = [c for c, k in zip(self._chunks, keep_mask) if k]
            kept_matrix = (
                self._matrix[keep_mask] if self._matrix is not None and any(keep_mask) else None
            )

        if not new_chunks:
            with self._lock:
                self._chunks = kept_chunks
                self._matrix = kept_matrix
            return

        new_texts = [c.embed_text for c in new_chunks]
        new_matrix = self._embed_texts(new_texts)

        with self._lock:
            self._chunks = kept_chunks + new_chunks
            if new_matrix is not None and kept_matrix is not None:
                self._matrix = np.vstack([kept_matrix, new_matrix])
            elif new_matrix is not None:
                self._matrix = new_matrix
            else:
                self._matrix = kept_matrix

    def search(self, query: str, top_k: int = 8) -> list[tuple[CodeChunk, float]]:
        with self._lock:
            chunks = self._chunks
            matrix = self._matrix

        if matrix is None or not chunks:
            return []

        q_vec = self._embed_texts([query])
        if q_vec is None:
            return []

        scores = (matrix @ q_vec[0]).tolist()
        ranked = sorted(
            [(c, s) for c, s in zip(chunks, scores)],
            key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


# ==========================================
# 3. BM25 关键字索引
# ==========================================

def _tokenize_code(text: str) -> list[str]:
    """
    代码专用分词：
    - 保留字母数字 token
    - camelCase → 拆分（FooBar → foo bar）
    - snake_case → 拆分（foo_bar → foo bar）
    """
    # 先 camelCase 拆分
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # 找所有 word token（包含下划线）
    words = re.findall(r"[a-zA-Z_]\w*", text)
    result = []
    for w in words:
        # 按下划线再拆
        parts = [p.lower() for p in w.split("_") if len(p) >= 2]
        result.extend(parts)
    return result


class _BM25Index:
    def __init__(self):
        self._chunks: list[CodeChunk] = []
        self._bm25: Optional["BM25Okapi"] = None
        self._lock = threading.Lock()

    def _rebuild(self, chunks: list[CodeChunk]) -> None:
        if not BM25_AVAILABLE or not chunks:
            return
        corpus = [_tokenize_code(f"{c.name} {c.signature} {c.docstring} {c.body[:500]}") for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def build(self, chunks: list[CodeChunk]) -> None:
        with self._lock:
            self._chunks = list(chunks)
            self._rebuild(self._chunks)
        logger.info(f"BM25 索引构建完成：{len(chunks)} 个 chunk")

    def update_file(self, file_path: str, new_chunks: list[CodeChunk]) -> None:
        with self._lock:
            kept = [c for c in self._chunks if c.file_path != file_path]
            self._chunks = kept + new_chunks
            self._rebuild(self._chunks)

    def search(self, query: str, top_k: int = 8) -> list[tuple[CodeChunk, float]]:
        with self._lock:
            if self._bm25 is None or not self._chunks:
                return []
            tokens = _tokenize_code(query)
            if not tokens:
                return []
            scores = self._bm25.get_scores(tokens)
            ranked = sorted(
                zip(self._chunks, scores.tolist()),
                key=lambda x: x[1], reverse=True
            )
            return [(c, s) for c, s in ranked[:top_k] if s > 0]

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


# ==========================================
# 4. 依赖图 + PageRank（Repo Map）
# ==========================================

class _DependencyGraph:
    def __init__(self):
        self._chunks: list[CodeChunk] = []
        self._name_to_chunks: dict[str, list[CodeChunk]] = {}
        self._pagerank: dict[str, float] = {}
        self._lock = threading.Lock()

    def build(self, chunks: list[CodeChunk]) -> None:
        if not NETWORKX_AVAILABLE:
            with self._lock:
                self._chunks = chunks
            return

        G = nx.DiGraph()

        # 建立 name → chunk 映射（同名函数取第一个）
        name_map: dict[str, CodeChunk] = {}
        for c in chunks:
            G.add_node(c.chunk_id)
            name_map.setdefault(c.name, c)
            if c.parent_class:
                name_map.setdefault(c.display_name, c)

        # 加边：调用关系
        for c in chunks:
            for called_name in c.calls:
                target = name_map.get(called_name)
                if target and target.chunk_id != c.chunk_id:
                    G.add_edge(c.chunk_id, target.chunk_id)

        # PageRank
        try:
            pr = nx.pagerank(G, alpha=0.85, max_iter=100)
        except Exception:
            pr = {c.chunk_id: 1.0 / max(len(chunks), 1) for c in chunks}

        # 规范化到 [0, 1]
        if pr:
            max_pr = max(pr.values()) or 1.0
            pr = {k: v / max_pr for k, v in pr.items()}

        name_to_chunks: dict[str, list[CodeChunk]] = {}
        for c in chunks:
            name_to_chunks.setdefault(c.name, []).append(c)

        with self._lock:
            self._chunks = chunks
            self._pagerank = pr
            self._name_to_chunks = name_to_chunks

    def get_pagerank(self, chunk_id: str) -> float:
        return self._pagerank.get(chunk_id, 0.0)

    def get_callers(self, target_name: str) -> list[CodeChunk]:
        """找出所有调用了 target_name 的 chunk。"""
        with self._lock:
            return [c for c in self._chunks if target_name in c.calls]

    def get_callees(self, chunk: CodeChunk) -> list[CodeChunk]:
        """找出 chunk 调用的所有已知 chunk。"""
        with self._lock:
            result = []
            for called in chunk.calls:
                result.extend(self._name_to_chunks.get(called, []))
            return result[:10]  # 限制数量

    def update_file(self, file_path: str, new_chunks: list[CodeChunk]) -> None:
        with self._lock:
            kept = [c for c in self._chunks if c.file_path != file_path]
        self.build(kept + new_chunks)

    def get_file_scores(self) -> dict[str, float]:
        """按 PageRank 汇总每个文件的重要性得分。"""
        scores: dict[str, float] = {}
        with self._lock:
            for c in self._chunks:
                scores[c.file_path] = scores.get(c.file_path, 0.0) + self._pagerank.get(c.chunk_id, 0.0)
        return scores


# ==========================================
# 5. CodeIndexEngine（三层索引总调度）
# ==========================================

# 支持的文件扩展名
_SUPPORTED_EXTENSIONS = {".py"}  # 未来可扩展 .js .ts .go 等

# 忽略的目录
_IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".mypy_cache", ".pytest_cache", "_evolution_drafts",
}


class CodeIndexEngine:
    """
    三层索引总调度引擎。
    线程安全，支持增量更新。
    """

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self._semantic = _SemanticIndex()
        self._bm25 = _BM25Index()
        self._graph = _DependencyGraph()
        self._all_chunks: list[CodeChunk] = []
        self._is_ready = False
        self._lock = threading.Lock()
        self._build_time: float = 0.0

    # ---- 构建 ----

    def build(self) -> str:
        """全量构建所有索引，返回初始 repo map。"""
        t0 = time.time()
        files = self._collect_files()
        all_chunks: list[CodeChunk] = []
        for f in files:
            all_chunks.extend(parse_file(f, self.workspace_dir))

        with self._lock:
            self._all_chunks = all_chunks

        self._semantic.build(all_chunks)
        self._bm25.build(all_chunks)
        self._graph.build(all_chunks)

        with self._lock:
            self._is_ready = True
            self._build_time = time.time() - t0

        logger.info(
            f"代码索引构建完成：{len(files)} 个文件，{len(all_chunks)} 个 chunk，"
            f"耗时 {self._build_time:.2f}s"
        )
        return self.get_repo_map()

    def update_file(self, file_path: str) -> None:
        """
        增量更新单个文件的索引。
        file_path 为相对 workspace 的路径。
        """
        abs_path = self.workspace_dir / file_path
        if not abs_path.exists():
            # 文件已删除，从索引中移除
            self._semantic.update_file(file_path, [])
            self._bm25.update_file(file_path, [])
            self._graph.update_file(file_path, [])
            with self._lock:
                self._all_chunks = [c for c in self._all_chunks if c.file_path != file_path]
            return

        new_chunks = parse_file(abs_path, self.workspace_dir)

        self._semantic.update_file(file_path, new_chunks)
        self._bm25.update_file(file_path, new_chunks)
        self._graph.update_file(file_path, new_chunks)

        with self._lock:
            self._all_chunks = [
                c for c in self._all_chunks if c.file_path != file_path
            ] + new_chunks

        logger.debug(f"增量更新：{file_path}，{len(new_chunks)} 个 chunk")

    # ---- 搜索 ----

    def search(self, query: str, mode: str = "auto", top_k: int = 8) -> list[CodeChunk]:
        """
        三模式搜索。
        mode:
          "semantic" — 语义向量搜索（自然语言描述）
          "keyword"  — BM25 关键字搜索（精确符号名）
          "auto"     — 自动选择：query 中含有下划线/大小写混合 → keyword，否则 → semantic + keyword 融合
        """
        if not self._is_ready:
            return []

        if mode == "keyword":
            results = self._bm25.search(query, top_k * 2)
            return [c for c, _ in results][:top_k]

        elif mode == "semantic":
            results = self._semantic.search(query, top_k * 2)
            return [c for c, _ in results][:top_k]

        else:  # auto
            is_symbol = bool(re.search(r"[_A-Z][a-z]|[a-z][A-Z]|_\w", query))

            sem_results = self._semantic.search(query, top_k)
            kw_results = self._bm25.search(query, top_k)

            # 融合：用 RRF（Reciprocal Rank Fusion）
            scores: dict[str, float] = {}
            all_chunks_map: dict[str, CodeChunk] = {}

            for rank, (c, _) in enumerate(sem_results):
                scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + (1.0 / (rank + 60))
                all_chunks_map[c.chunk_id] = c

            kw_weight = 1.5 if is_symbol else 1.0
            for rank, (c, _) in enumerate(kw_results):
                scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + kw_weight * (1.0 / (rank + 60))
                all_chunks_map[c.chunk_id] = c

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [all_chunks_map[cid] for cid, _ in ranked[:top_k]]

    # ---- Repo Map ----

    def get_repo_map(self, query: str = "", max_tokens: int = 2000) -> str:
        """
        生成 Repo Map：按重要性排序的符号地图。
        若提供 query，结合语义相关性进行重新排序。
        """
        with self._lock:
            all_chunks = list(self._all_chunks)

        if not all_chunks:
            return "（工作区暂无代码文件）"

        # 按文件分组
        file_chunks: dict[str, list[CodeChunk]] = {}
        for c in all_chunks:
            file_chunks.setdefault(c.file_path, []).append(c)

        # 计算文件得分
        file_scores = self._graph.get_file_scores()

        # 若有 query，结合语义相关性提升相关文件
        if query:
            sem_results = self._semantic.search(query, top_k=20)
            for rank, (c, score) in enumerate(sem_results):
                boost = score * 2.0
                file_scores[c.file_path] = file_scores.get(c.file_path, 0.0) + boost

        # 对文件排序
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        # 也包含 file_scores 中没有的文件（得分为 0）
        remaining = [f for f in file_chunks if f not in file_scores]
        all_files_ordered = [f for f, _ in sorted_files if f in file_chunks] + remaining

        lines: list[str] = []
        token_estimate = 0
        CHARS_PER_TOKEN = 4

        for file_path in all_files_ordered:
            chunks = file_chunks.get(file_path, [])
            if not chunks:
                continue

            # 文件头
            file_header = f"\n📦 {file_path}"
            lines.append(file_header)
            token_estimate += len(file_header) // CHARS_PER_TOKEN

            # 按 PageRank 排序函数/类
            sorted_chunks = sorted(
                chunks,
                key=lambda c: self._graph.get_pagerank(c.chunk_id),
                reverse=True,
            )

            for c in sorted_chunks:
                # 跳过只是 class 体本身（方法已单独列出）
                if c.chunk_type == "class" and any(
                    x.parent_class == c.name for x in chunks
                ):
                    # 只显示类签名
                    entry = f"  ├── {c.signature}"
                    lines.append(entry)
                    token_estimate += len(entry) // CHARS_PER_TOKEN
                    continue

                pr = self._graph.get_pagerank(c.chunk_id)
                bar = "■" * int(pr * 4) + "□" * (4 - int(pr * 4))
                entry = f"  ├── {c.signature}  [{bar}] L{c.start_line}"
                if c.docstring:
                    entry += f"\n  │     {c.docstring[:80]}"
                lines.append(entry)
                token_estimate += len(entry) // CHARS_PER_TOKEN

            if token_estimate * CHARS_PER_TOKEN > max_tokens * CHARS_PER_TOKEN:
                lines.append("\n  ... [更多文件已截断，增加 max_tokens 以查看]")
                break

        return "\n".join(lines)

    # ---- 符号上下文 ----

    def get_symbol_context(self, symbol_name: str) -> str:
        """返回符号的定义、调用者和被调用者（完整上下文）。"""
        with self._lock:
            all_chunks = list(self._all_chunks)

        # 精确匹配 + 模糊匹配
        exact = [c for c in all_chunks if c.name == symbol_name or c.display_name == symbol_name]
        if not exact:
            # 不区分大小写 + 包含匹配
            symbol_lower = symbol_name.lower()
            exact = [
                c for c in all_chunks
                if symbol_lower in c.name.lower() or symbol_lower in c.display_name.lower()
            ][:3]

        if not exact:
            return f"未在代码索引中找到符号 '{symbol_name}'。请确认拼写，或先使用 search_code 搜索。"

        lines = []
        for chunk in exact[:2]:  # 最多展示 2 个同名符号
            lines.append(f"📍 {chunk.display_name}")
            lines.append("─" * 50)
            lines.append(f"📁 文件: {chunk.file_path}  (L{chunk.start_line}~{chunk.end_line})")
            lines.append(f"🏷️ 类型: {chunk.chunk_type}")
            lines.append(f"✏️ 签名: {chunk.signature}")
            if chunk.docstring:
                lines.append(f"📝 文档: {chunk.docstring[:200]}")

            if chunk.calls:
                lines.append(f"\n📞 调用: {', '.join(chunk.calls[:10])}")

            callers = self._graph.get_callers(chunk.name)
            if callers:
                caller_names = [c.display_name for c in callers[:5]]
                lines.append(f"🔙 被调用于: {', '.join(caller_names)}")

            callees = self._graph.get_callees(chunk)
            if callees:
                callee_names = list({c.display_name for c in callees[:5]})
                lines.append(f"🔗 调用的函数: {', '.join(callee_names)}")

            lines.append(f"\n💻 源码:")
            lines.append("```python")
            lines.append(chunk.body[:1500])
            if len(chunk.body) > 1500:
                lines.append("... [已截断]")
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    # ---- 工具方法 ----

    def _collect_files(self) -> list[Path]:
        """收集工作区内所有支持的源码文件。"""
        files = []
        for root, dirs, filenames in os.walk(self.workspace_dir):
            dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS and not d.startswith(".")]
            for fn in filenames:
                p = Path(root) / fn
                if p.suffix in _SUPPORTED_EXTENSIONS:
                    files.append(p)
        return files

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "chunk_count": len(self._all_chunks),
            "semantic_ready": self._semantic.chunk_count > 0,
            "bm25_ready": self._bm25.chunk_count > 0,
            "graph_ready": NETWORKX_AVAILABLE,
            "tree_sitter": TREE_SITTER_AVAILABLE,
            "embeddings": SENTENCE_TRANSFORMERS_AVAILABLE,
            "build_time_s": round(self._build_time, 2),
        }


# ==========================================
# 全局单例 + 公开 API
# ==========================================

# 延迟初始化：模块导入时不立刻创建引擎，避免 WORKSPACE_DIR 还未确定
_engine: Optional[CodeIndexEngine] = None
_engine_lock = threading.Lock()


def _get_engine(workspace_dir: Optional[Path] = None) -> CodeIndexEngine:
    """获取（或创建）全局单例引擎。"""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            if workspace_dir is None:
                from src.agent.swe.tools import WORKSPACE_DIR
                workspace_dir = WORKSPACE_DIR
            _engine = CodeIndexEngine(workspace_dir)
    return _engine


def build_workspace_index(workspace_dir: Optional[Path] = None) -> str:
    """
    全量构建代码索引，返回初始 repo map 字符串。
    在 index_builder_node 中调用。
    """
    engine = _get_engine(workspace_dir)
    return engine.build()


def update_file_index(file_path: str) -> None:
    """
    增量更新单个文件的索引。
    在 write_file / edit_file 工具执行后调用。
    """
    engine = _get_engine()
    if engine.is_ready:
        engine.update_file(file_path)


def search_code_index(query: str, mode: str = "auto", top_k: int = 8) -> list[CodeChunk]:
    """搜索代码索引，返回 CodeChunk 列表。"""
    return _get_engine().search(query, mode=mode, top_k=top_k)


def get_repo_map_str(query: str = "", max_tokens: int = 2000) -> str:
    """获取 Repo Map 字符串。"""
    return _get_engine().get_repo_map(query=query, max_tokens=max_tokens)


def get_symbol_context_str(symbol_name: str) -> str:
    """获取符号的完整上下文。"""
    return _get_engine().get_symbol_context(symbol_name)


def get_index_stats() -> dict[str, Any]:
    """返回索引统计信息。"""
    if _engine is None:
        return {"status": "not_built"}
    return _engine.stats
