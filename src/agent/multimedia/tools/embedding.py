# src/agent/multimedia/tools/embedding.py
import os
from typing import List
import numpy as np
from dotenv import load_dotenv
import requests

load_dotenv()

_session = requests.Session()
_session.trust_env = False

# DashScope 原生 API 端点（multimodal-embedding-v1 不支持 OpenAI 兼容模式）
_EMBEDDING_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
_DEFAULT_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "multimodal-embedding-v1")
_DEFAULT_DIMENSION = 1024


def get_image_embedding(image_url: str, model: str = _DEFAULT_MODEL, dimension: int = _DEFAULT_DIMENSION) -> List[float]:
    """
    获取图片 embedding（通过 DashScope 原生 API）。
    multimodal-embedding-v1 不支持 OpenAI 兼容模式，必须走原生端点。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY"))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": {
            "contents": [
                {"image": image_url}
            ]
        },
        "parameters": {
            "dimension": dimension,
        }
    }

    response = _session.post(_EMBEDDING_URL, headers=headers, json=payload, timeout=30)
    if response.status_code != 200:
        raise Exception(f"Image embedding API error ({response.status_code}): {response.text[:200]}")

    data = response.json()
    embeddings = data.get("output", {}).get("embeddings", [])
    if not embeddings:
        raise Exception(f"Image embedding returned empty: {data}")
    return embeddings[0]["embedding"]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def compute_similarity(current_url: str, reference_url: str) -> float:
    emb1 = get_image_embedding(current_url)
    emb2 = get_image_embedding(reference_url)
    return cosine_similarity(emb1, emb2)
