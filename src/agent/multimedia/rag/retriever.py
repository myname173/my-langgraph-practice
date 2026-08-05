# src/agent/multimedia/rag/retriever.py
"""
RAG 知识检索模块
=================
根据 scene script + 意图，从 3 个 ChromaDB collection 检索相关知识，
格式化为导演可读的文本块注入 prompt。

检索策略：
    - cinematography_knowledge: top 3（摄影/布光专业知识）
    - prompt_examples: top 2（高质量 prompt 样本）
    - film_references: top 2（影视参考描述）
    - 过滤距离 > 0.6 的低相关结果

设计原则：
    - 静默降级：ChromaDB 不可用时返回空 dict，不抛异常
    - 不增加 LLM 调用，纯检索开销（< 100ms）
    - 输出格式化文本块可直接注入 prompt
"""

from typing import Any, Dict, List, Optional

from .vector_store import get_or_create_collection


# ============================================================
# 1. 配置
# ============================================================

# 各 collection 的检索参数
_RETRIEVAL_CONFIG = {
    "cinematography_knowledge": {
        "top_k": 3,
        "max_distance": 1.0,   # 1024-dim L2 distance: <1.0 = good match
        "label": "Cinematography Knowledge",
    },
    "prompt_examples": {
        "top_k": 2,
        "max_distance": 1.0,   # prompt 匹配
        "label": "High-Quality Prompt Examples",
    },
    "film_references": {
        "top_k": 2,
        "max_distance": 1.1,   # film_references 数据量小，稍放宽
        "label": "Film Scene References",
    },
}

# 场景意图 → 相关知识分类（用于 metadata where 过滤）
# 每个意图映射到最相关的 knowledge category，提升检索精度
_INTENT_CATEGORY_MAP = {
    "world_building": ["camera", "composition", "lighting", "lens"],
    "character_introduction": ["camera", "lighting", "composition"],
    "action": ["camera", "editing", "transition", "pacing"],
    "emotional": ["mood", "color", "lighting", "lens"],
}

# 所有已知分类（用于 fallback 全量检索）
_ALL_CATEGORIES = [
    "camera", "lighting", "composition", "editing", "transition",
    "pacing", "mood", "color", "lens", "style", "prompt_keywords", "general",
]


# ============================================================
# 2. 查询构建
# ============================================================

def _build_query(script: str, global_setting: str, intent: str) -> str:
    """
    构建检索查询文本。

    将 scene script（核心）、intent（分类）和 global_setting（风格）
    组合为一条语义丰富的查询。
    """
    parts = [script]

    if intent:
        parts.append(f"scene intent: {intent}")

    # global_setting 取前 150 字符（避免过长查询）
    if global_setting:
        setting_excerpt = global_setting[:150].strip()
        parts.append(f"visual style: {setting_excerpt}")

    return " | ".join(parts)


# ============================================================
# 3. 单 Collection 检索
# ============================================================

def _query_collection(
    collection_name: str,
    query: str,
    top_k: int,
    max_distance: float,
    where_filter: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """
    查询单个 Collection 并过滤低分结果。

    支持可选的 metadata where 过滤：先尝试带过滤的精确检索，
    如果结果为空则回退到无过滤的全量检索（保证召回率）。

    Args:
        collection_name: Collection 名称
        query: 查询文本
        top_k: 最大返回数
        max_distance: 最大 L2 距离阈值
        where_filter: ChromaDB where 过滤条件（可选）

    Returns:
        [{"document": str, "distance": float, "metadata": dict}, ...]
    """
    try:
        collection = get_or_create_collection(collection_name)

        if collection.count() == 0:
            return []

        # ── 策略：filter-first, fallback-unfiltered ──
        items: List[Dict[str, Any]] = []

        # Step 1: 如果有 where_filter，先尝试精确检索
        if where_filter:
            items = _do_query(collection, query, top_k, max_distance,
                              where=where_filter)

        # Step 2: 精确检索为空 → 回退全量检索
        if not items:
            items = _do_query(collection, query, top_k, max_distance,
                              where=None)

        return items

    except Exception as e:
        print(f"    [!] RAG 检索失败 ({collection_name}): {e}")
        return []


def _do_query(
    collection,
    query: str,
    top_k: int,
    max_distance: float,
    where: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """
    执行单次 ChromaDB query 并过滤低分结果。

    Args:
        collection: ChromaDB Collection 对象
        query: 查询文本
        top_k: 最大返回数
        max_distance: 距离阈值
        where: 可选的 metadata 过滤条件

    Returns:
        匹配的文档列表
    """
    query_kwargs: Dict[str, Any] = {
        "query_texts": [query],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    items = []
    if results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        dists = results["distances"][0] if results["distances"] else [0] * len(docs)
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)

        for doc, dist, meta in zip(docs, dists, metas):
            if dist <= max_distance:
                items.append({
                    "document": doc,
                    "distance": dist,
                    "metadata": meta,
                })

    return items


# ============================================================
# 4. 结果格式化
# ============================================================

def _format_rag_block(
    cinematography_refs: List[Dict[str, Any]],
    prompt_refs: List[Dict[str, Any]],
    film_refs: List[Dict[str, Any]],
) -> str:
    """
    将检索结果格式化为导演可读的文本块。

    输出格式：
        [RAG Reference — Cinematography Knowledge]
        1. (camera) 内容...
        2. (lighting) 内容...

        [RAG Reference — High-Quality Prompt Examples]
        1. [style=cinematic] 内容...

        [RAG Reference — Film Scene References]
        1. 内容...
    """
    sections = []

    # 摄影知识（子切块后每个 chunk ~600 字符，保留前 500 字符）
    if cinematography_refs:
        lines = []
        for i, ref in enumerate(cinematography_refs, 1):
            category = ref.get("metadata", {}).get("category", "general")
            doc_text = ref["document"][:500].strip()
            lines.append(f"    {i}. [{category}] {doc_text}")
        sections.append(
            "[RAG Reference — Cinematography Knowledge]\n"
            + "\n".join(lines)
        )

    # Prompt 样本
    if prompt_refs:
        lines = []
        for i, ref in enumerate(prompt_refs, 1):
            style = ref.get("metadata", {}).get("style", "general")
            score = ref.get("metadata", {}).get("quality_score", 0)
            # prompt 样本保留前 400 字符
            doc_text = ref["document"][:400].strip()
            lines.append(f"    {i}. [style={style}, quality={score:.1f}] {doc_text}")
        sections.append(
            "[RAG Reference — High-Quality Prompt Examples]\n"
            + "\n".join(lines)
        )

    # 影视参考
    if film_refs:
        lines = []
        for i, ref in enumerate(film_refs, 1):
            title = ref.get("metadata", {}).get("film_title", "")
            doc_text = ref["document"][:300].strip()
            prefix = f"[{title}] " if title else ""
            lines.append(f"    {i}. {prefix}{doc_text}")
        sections.append(
            "[RAG Reference — Film Scene References]\n"
            + "\n".join(lines)
        )

    if not sections:
        return "[RAG Reference — No relevant references retrieved]"

    return "\n\n".join(sections)


# ============================================================
# 5. 主入口
# ============================================================

def retrieve_rag_context(
    script: str,
    global_setting: str,
    intent: str = "",
    top_k: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    根据 scene script + 意图，从 3 个 collection 检索相关知识。

    这是 RAG 模块的核心入口函数，由 visual_context_builder_node 调用。

    Args:
        script: 当前镜头的剧本脚本
        global_setting: 全局视觉设定
        intent: 场景意图分类（world_building/character_introduction/action/emotional）
        top_k: 每个 collection 的最大检索数（会被 _RETRIEVAL_CONFIG 覆盖）
        verbose: 是否打印检索日志

    Returns:
        {
            "cinematography_refs": List[Dict],   # 摄影知识片段
            "prompt_refs": List[Dict],           # 高质量 prompt 样本
            "film_refs": List[Dict],             # 影视参考描述
            "total_retrieved": int,              # 总检索数
            "formatted_rag_block": str,          # 格式化后的完整 RAG 文本块
        }
        如果 ChromaDB 不可用，返回空 dict（静默降级）
    """
    try:
        query = _build_query(script, global_setting, intent)

        if verbose:
            print(f"    [*] RAG 检索: query={query[:80]}...")

        # ── 构建 metadata where 过滤条件 ──
        # cinematography_knowledge 支持按 category 过滤
        where_filter = None
        intent_categories = _INTENT_CATEGORY_MAP.get(intent)
        if intent_categories:
            # ChromaDB where: category IN [cam1, cam2, ...]
            where_filter = {"category": {"$in": intent_categories}}
            if verbose:
                print(f"    [*] RAG 过滤: intent={intent} → categories={intent_categories}")

        # 分别检索 3 个 collection
        cine_config = _RETRIEVAL_CONFIG["cinematography_knowledge"]
        cinematography_refs = _query_collection(
            "cinematography_knowledge",
            query,
            top_k=cine_config["top_k"],
            max_distance=cine_config["max_distance"],
            where_filter=where_filter,
        )

        prompt_config = _RETRIEVAL_CONFIG["prompt_examples"]
        prompt_refs = _query_collection(
            "prompt_examples",
            query,
            top_k=prompt_config["top_k"],
            max_distance=prompt_config["max_distance"],
        )

        film_config = _RETRIEVAL_CONFIG["film_references"]
        film_refs = _query_collection(
            "film_references",
            query,
            top_k=film_config["top_k"],
            max_distance=film_config["max_distance"],
        )

        total = len(cinematography_refs) + len(prompt_refs) + len(film_refs)

        if verbose:
            print(f"    [*] RAG 结果: cinematography={len(cinematography_refs)}, "
                  f"prompts={len(prompt_refs)}, films={len(film_refs)} (total={total})")

        # 格式化
        formatted_block = _format_rag_block(cinematography_refs, prompt_refs, film_refs)

        return {
            "cinematography_refs": cinematography_refs,
            "prompt_refs": prompt_refs,
            "film_refs": film_refs,
            "total_retrieved": total,
            "formatted_rag_block": formatted_block,
        }

    except Exception as e:
        if verbose:
            print(f"    [!] RAG 检索整体失败（静默降级）: {e}")
        return {}
