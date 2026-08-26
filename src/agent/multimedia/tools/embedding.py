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


def _to_dashscope_image(src: str) -> str:
    """把参考图统一为 DashScope multimodal-embedding 可接受的形态。

    DashScope embedding 的 image 字段只接受 http(s) URL 或 base64 data URI，
    **不接受本地绝对路径**（否则 requests 报 "No connection adapters" 并白烧积分）。
    这里对本地路径 / /media/ 前缀统一内联为 base64 data URI，与即梦 _as_inline 对齐。
    """
    if not src:
        return src
    if src.startswith("data:"):
        return src
    if src.startswith(("http://", "https://")):
        # localhost / 127.0.0.1 回环地址 DashScope 云端无法回源，需同机 fetch 内联
        from urllib.parse import urlparse
        _host = (urlparse(src).hostname or "").lower()
        if _host in ("localhost", "127.0.0.1"):
            try:
                import base64 as _b64
                _r = requests.get(src, timeout=30)
                _r.raise_for_status()
                _ct = _r.headers.get("Content-Type", "image/jpeg")
                if not _ct.startswith("image/"):
                    _ct = "image/jpeg"
                return f"data:{_ct};base64," + _b64.b64encode(_r.content).decode("ascii")
            except Exception as _e:
                print(f"    [WARN][embedding] 回环 URL 内联失败，原样下发: {_e}")
        return src
    # 本地绝对路径 或 项目自定义 /media/<rel> 前缀 -> 读本地文件转 base64
    _path = src
    if src.startswith("/media/"):
        # /media/<rel> -> <PROJECT_ROOT>/output/<rel>
        _rel = src[len("/media/"):]
        # embedding.py -> tools -> multimedia -> agent -> src -> <root>  (5 层)
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        _path = os.path.join(_root, "output", _rel)
    if os.path.isfile(_path):
        import base64 as _b64
        try:
            with open(_path, "rb") as _f:
                _b = _f.read()
            _ext = os.path.splitext(_path)[1].lower().lstrip(".")
            _ct = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                   "webp": "image/webp", "gif": "image/gif"}.get(_ext, "image/png")
            return f"data:{_ct};base64," + _b64.b64encode(_b).decode("ascii")
        except Exception as _e:
            print(f"    [WARN][embedding] 本地图转内联失败，原样下发: {_e}")
    return src


def get_image_embedding(image_url: str, model: str = _DEFAULT_MODEL, dimension: int = _DEFAULT_DIMENSION) -> List[float]:
    """
    获取图片 embedding（通过 DashScope 原生 API）。
    multimodal-embedding-v1 不支持 OpenAI 兼容模式，必须走原生端点。
    """
    # 本地路径 / /media/ 前缀 / 回环 URL -> 统一内联为 base64 data URI，
    # 避免把本地路径裸发给 DashScope（会报 No connection adapters 并浪费积分）。
    image_url = _to_dashscope_image(image_url)
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
