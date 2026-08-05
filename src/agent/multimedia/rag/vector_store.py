# src/agent/multimedia/rag/vector_store.py
"""
ChromaDB 向量库封装
====================
管理 ChromaDB persistent client 和 3 个知识 Collection。

Collection 设计：
    - cinematography_knowledge: 摄影/布光/剪辑专业知识
    - prompt_examples: 高质量 prompt 样本（社区验证）
    - film_references: 影视场景参考描述

设计原则：
    - 数据持久化到 data/chroma_db/，项目根目录下
    - 自定义 EmbeddingFunction 包装 DashScope text-embedding-v3
    - get_or_create 语义，多次初始化不丢数据
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb import Collection, PersistentClient

from ..tools.text_embedding import get_text_embedding, get_text_embeddings


# ============================================================
# 1. 存储路径
# ============================================================

# data/chroma_db/ 在项目根目录下（与 src/ 同级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # my-langgraph-practice-main/
_CHROMA_DIR = _PROJECT_ROOT / "data" / "chroma_db"


# ============================================================
# 2. 自定义 Embedding Function（适配 ChromaDB 接口）
# ============================================================

class DashScopeEmbeddingFunction:
    """
    ChromaDB EmbeddingFunction 适配器。
    内部调用 DashScope text-embedding-v3，对外符合 ChromaDB 的 __call__ 协议。
    """

    def __init__(self, dimensions: int = 1024):
        self._dimensions = dimensions

    def name(self) -> str:
        """ChromaDB v1.5+ 要求的名称标识。"""
        return "dashscope_text_embedding"

    def embed_query(self, input=None, **kwargs) -> List[List[float]]:
        """
        ChromaDB v1.5+ query 时调用的 embedding 接口。
        ChromaDB 内部以 embed_query(input=[query_text1, ...]) 方式调用，
        input 是一个字符串列表（即使单条查询也是列表）。
        返回 List[List[float]]（每条文本对应一个向量）。

        失败策略：跳过空/无效输入，仅返回成功 embedding 的条目。
        如果全部失败则抛出 RuntimeError，防止零向量污染检索结果。
        """
        if input is None:
            input = kwargs.get("query", [])

        if isinstance(input, str):
            input = [input]

        results = []
        for text in input:
            if not text or not isinstance(text, str):
                continue
            vec = get_text_embedding(text, dimensions=self._dimensions)
            if vec is not None:
                results.append(vec)

        if not results:
            raise RuntimeError(
                "All query embeddings failed — cannot return zero vectors "
                "(would pollute retrieval with false matches)"
            )
        return results

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        ChromaDB 调用的标准接口（用于 ingest/add/upsert）。

        失败策略：过滤掉 embedding 失败的条目，仅返回成功向量。
        如果全部失败则抛出 RuntimeError，阻止零向量写入 ChromaDB。
        注意：返回长度可能 < 输入长度，ChromaDB 会因此中止当前 upsert，
        这是期望行为——防止部分文档以零向量污染索引。
        """
        results = get_text_embeddings(input, dimensions=self._dimensions)
        valid = [vec for vec in results if vec is not None]

        if not valid:
            raise RuntimeError(
                f"All {len(input)} embeddings failed in batch — refusing to store "
                "zero vectors in ChromaDB (would pollute similarity search)"
            )

        n_failed = len(input) - len(valid)
        if n_failed > 0:
            print(f"    [!] Embedding: {n_failed}/{len(input)} items failed, skipped")

        return valid


# ============================================================
# 3. Client + Collection 管理
# ============================================================

_client: Optional[PersistentClient] = None
_embedding_fn: Optional[DashScopeEmbeddingFunction] = None

# Collection 名称 → 用途描述
COLLECTION_DEFINITIONS: Dict[str, str] = {
    "cinematography_knowledge": "摄影/布光/剪辑专业知识片段",
    "prompt_examples": "高质量 AI 生成 prompt 样本",
    "film_references": "影视场景参考描述",
}


def get_chroma_client() -> PersistentClient:
    """获取或创建 ChromaDB persistent client。"""
    global _client
    if _client is None:
        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = PersistentClient(path=str(_CHROMA_DIR))
    return _client


def get_embedding_function() -> DashScopeEmbeddingFunction:
    """获取共享的 embedding function 实例。"""
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = DashScopeEmbeddingFunction()
    return _embedding_fn


def get_or_create_collection(name: str) -> Collection:
    """
    获取或创建指定名称的 Collection。

    Args:
        name: Collection 名称（必须在 COLLECTION_DEFINITIONS 中定义）

    Returns:
        ChromaDB Collection 对象
    """
    if name not in COLLECTION_DEFINITIONS:
        raise ValueError(
            f"未知的 collection '{name}'，"
            f"可选: {list(COLLECTION_DEFINITIONS.keys())}"
        )

    client = get_chroma_client()
    embedding_fn = get_embedding_function()

    collection = client.get_or_create_collection(
        name=name,
        embedding_function=embedding_fn,
        metadata={"description": COLLECTION_DEFINITIONS[name]},
    )
    return collection


def get_all_collections() -> Dict[str, Collection]:
    """获取所有 3 个 Collection 对象。"""
    return {name: get_or_create_collection(name) for name in COLLECTION_DEFINITIONS}


def get_collection_stats() -> Dict[str, Dict[str, Any]]:
    """
    获取所有 Collection 的统计信息。

    Returns:
        {
            "cinematography_knowledge": {"count": 15, "description": "..."},
            "prompt_examples": {"count": 50, "description": "..."},
            "film_references": {"count": 30, "description": "..."},
        }
    """
    stats = {}
    for name, desc in COLLECTION_DEFINITIONS.items():
        try:
            col = get_or_create_collection(name)
            stats[name] = {
                "count": col.count(),
                "description": desc,
            }
        except Exception as e:
            stats[name] = {
                "count": -1,
                "description": desc,
                "error": str(e),
            }
    return stats


def reset_collection(name: str) -> None:
    """
    清空指定 Collection（删除后重建）。
    用于 --force 模式下的数据重建。
    """
    client = get_chroma_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass  # collection 可能不存在
    get_or_create_collection(name)
    print(f"    [*] Collection '{name}' 已清空重建")
