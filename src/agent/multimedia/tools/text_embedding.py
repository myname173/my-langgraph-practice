# src/agent/multimedia/tools/text_embedding.py
"""
DashScope 文本 Embedding 封装
==============================
为 RAG 检索层提供文本向量化能力。
使用 DashScope text-embedding-v3 模型，与现有 image embedding 共用 API key / base_url。

设计原则：
    - 复用 embedding.py 的 httpx + proxy bypass 模式
    - API 失败时返回 None，不中断管线（静默降级）
    - 支持单条和批量输入
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
import httpx
from openai import OpenAI

load_dotenv()

# 复用与 embedding.py 相同的 httpx 客户端配置（绕过代理问题）
_http_client = httpx.Client(trust_env=False)

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """延迟初始化 OpenAI 客户端，避免模块导入时失败。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", os.getenv("DASHSCOPE_API_KEY")),
            base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            http_client=_http_client,
        )
    return _client


# DashScope text-embedding-v3 默认维度
DEFAULT_EMBEDDING_MODEL = os.getenv("DASHSCOPE_TEXT_EMBEDDING_MODEL", "text-embedding-v3")
DEFAULT_DIMENSIONS = 1024


def get_text_embedding(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = DEFAULT_DIMENSIONS,
) -> Optional[List[float]]:
    """
    获取单条文本的 embedding 向量。

    Args:
        text: 输入文本（建议 < 8000 字符）
        model: DashScope embedding 模型名
        dimensions: 输出向量维度

    Returns:
        向量列表，API 失败时返回 None
    """
    if not text or not text.strip():
        return None

    try:
        client = _get_client()
        response = client.embeddings.create(
            model=model,
            input=text.strip(),
            dimensions=dimensions,
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"    [!] text embedding 失败: {e}")
        return None


def get_text_embeddings(
    texts: List[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = DEFAULT_DIMENSIONS,
    batch_size: int = 20,
) -> List[Optional[List[float]]]:
    """
    批量获取文本 embedding 向量。

    DashScope 单次最多处理 25 条，这里默认 batch_size=20 留余量。

    Args:
        texts: 输入文本列表
        model: DashScope embedding 模型名
        dimensions: 输出向量维度
        batch_size: 每批最大条数

    Returns:
        与输入等长的向量列表，失败的条目为 None
    """
    if not texts:
        return []

    results: List[Optional[List[float]]] = [None] * len(texts)

    try:
        client = _get_client()

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            # 过滤空文本，记录原始索引
            valid_indices = []
            valid_texts = []
            for i, t in enumerate(batch):
                if t and t.strip():
                    valid_indices.append(start + i)
                    valid_texts.append(t.strip())

            if not valid_texts:
                continue

            response = client.embeddings.create(
                model=model,
                input=valid_texts,
                dimensions=dimensions,
            )

            for item in response.data:
                idx = valid_indices[item.index]
                results[idx] = item.embedding

    except Exception as e:
        print(f"    [!] batch text embedding 失败: {e}")

    return results
