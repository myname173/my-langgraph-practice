# src/agent/multimedia/tools/embedding.py
import os
from typing import List
import numpy as np
from dotenv import load_dotenv
import httpx
from openai import OpenAI

load_dotenv()

http_client = httpx.Client(trust_env=False)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", os.getenv("DASHSCOPE_API_KEY")),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    http_client=http_client
)


def get_image_embedding(image_url: str) -> List[float]:
    """
    获取图片 embedding。
    如果你的平台对多模态 embedding 的入参格式有差异，只需要改这里。
    """
    response = client.embeddings.create(
        model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", "multimodal-embedding-v1"),
        input=[{"image_url": image_url}]
    )
    return response.data[0].embedding


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
