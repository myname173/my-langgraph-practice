# src/agent/multimedia/rag/ingestors/prompt_scraper.py
"""
Web Prompt 爬虫 — Civitai 高质量 prompt 采集
==============================================
从 Civitai 公开 API 采集电影感/影视风格的高质量 prompt 样本。

数据流：
    Civitai API → 过滤/清洗 → 本地 JSON 缓存 → ChromaDB prompt_examples

设计原则：
    - 限速：每次请求间隔 1 秒，避免被封
    - 缓存：首次爬取结果存为 JSON，后续灌入直接读缓存
    - 过滤：只保留有完整 prompt 且长度 > 50 字符的高质量样本
    - 安全：过滤 NSFW 内容
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..vector_store import get_or_create_collection


# ============================================================
# 1. 配置
# ============================================================

_CIVITAI_API_BASE = "https://civitai.com/api/v1"
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "cache"
_CACHE_FILE = _CACHE_DIR / "civitai_prompts.json"

# 搜索标签（电影感相关）
SEARCH_TAGS = [
    "cinematic",
    "film",
    "movie",
    "cinematography",
    "dramatic lighting",
    "epic",
    "trailer",
]

# 每次搜索获取的图片数
_IMAGES_PER_TAG = 20
# 请求间隔（秒）
_REQUEST_DELAY = 1.0
# prompt 最短长度
_MIN_PROMPT_LENGTH = 50


# ============================================================
# 2. Civitai API 调用
# ============================================================

def _fetch_images_by_tag(tag: str, limit: int = _IMAGES_PER_TAG) -> List[Dict[str, Any]]:
    """
    从 Civitai API 按标签获取图片及其 prompt。

    Returns:
        包含 prompt、meta 等字段的图片列表
    """
    url = f"{_CIVITAI_API_BASE}/images"
    params = {
        "limit": min(limit, 100),  # API 限制最大 100
        "sort": "Most Reactions",
        "period": "AllTime",
        "tag": tag,
        "nsfw": "false",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", [])
    except Exception as e:
        print(f"    [!] Civitai API 请求失败 (tag={tag}): {e}")
        return []


def _extract_prompt_from_image(image: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    从 Civitai 图片对象中提取 prompt 信息。

    Returns:
        {"prompt": str, "negative_prompt": str, "meta": dict, "stats": dict}
        或 None（如果不符合条件）
    """
    meta = image.get("meta", {}) or {}
    prompt = meta.get("prompt", "")

    # 过滤：必须有 prompt 且够长
    if not prompt or len(prompt) < _MIN_PROMPT_LENGTH:
        return None

    # 过滤：NSFW 检查
    if image.get("nsfwLevel", 0) and image.get("nsfwLevel", 0) > 1:
        return None

    return {
        "prompt": prompt.strip(),
        "negative_prompt": meta.get("negativePrompt", ""),
        "sampler": meta.get("samplerName", ""),
        "model": meta.get("Model", meta.get("model", "")),
        "stats": {
            "reactions": image.get("stats", {}).get("reactionCountAllTime", 0),
            "comments": image.get("stats", {}).get("commentCountAllTime", 0),
        },
        "tags": image.get("tagNames", []),
        "image_id": image.get("id", ""),
    }


# ============================================================
# 3. 采集 + 缓存
# ============================================================

def scrape_civitai_prompts(
    tags: List[str] = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    从 Civitai 采集高质量 prompt。

    Args:
        tags: 搜索标签列表
        use_cache: 是否使用本地缓存（如果缓存存在且非空）
        verbose: 是否打印进度

    Returns:
        清洗后的 prompt 列表
    """
    tags = tags or SEARCH_TAGS

    # 检查缓存
    if use_cache and _CACHE_FILE.exists():
        cached = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        if cached:
            if verbose:
                print(f"    [*] 使用缓存: {_CACHE_FILE.name} ({len(cached)} 条)")
            return cached

    all_prompts: List[Dict[str, Any]] = []
    seen_ids = set()

    if verbose:
        print(f"    [*] 开始采集 {len(tags)} 个标签...")

    for tag in tags:
        if verbose:
            print(f"        标签: {tag}...", end=" ")

        images = _fetch_images_by_tag(tag)
        count = 0

        for img in images:
            extracted = _extract_prompt_from_image(img)
            if extracted and extracted["image_id"] not in seen_ids:
                seen_ids.add(extracted["image_id"])
                extracted["search_tag"] = tag
                all_prompts.append(extracted)
                count += 1

        if verbose:
            print(f"提取 {count} 条 (共 {len(images)} 张)")

        time.sleep(_REQUEST_DELAY)

    if verbose:
        print(f"    [*] 共采集 {len(all_prompts)} 条不重复 prompt")

    # 保存缓存
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(
        json.dumps(all_prompts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if verbose:
        print(f"    [*] 已缓存到 {_CACHE_FILE}")

    return all_prompts


# ============================================================
# 4. 写入 ChromaDB
# ============================================================

def _infer_style(tags: List[str], prompt: str) -> str:
    """从标签和 prompt 推断风格类型。"""
    text = " ".join(tags).lower() + " " + prompt.lower()[:200]
    if any(w in text for w in ["cyberpunk", "neon", "futuristic"]):
        return "cyberpunk"
    if any(w in text for w in ["fantasy", "magic", "dragon", "wizard"]):
        return "fantasy"
    if any(w in text for w in ["horror", "dark", "gothic", "scary"]):
        return "horror"
    if any(w in text for w in ["anime", "manga", "cartoon"]):
        return "anime"
    return "cinematic"


def _infer_scene_type(prompt: str) -> str:
    """从 prompt 推断场景类型。"""
    text = prompt.lower()[:300]
    if any(w in text for w in ["portrait", "character", "person", "woman", "man"]):
        return "character"
    if any(w in text for w in ["landscape", "city", "mountain", "ocean", "forest"]):
        return "environment"
    if any(w in text for w in ["battle", "fight", "action", "explosion", "combat"]):
        return "action"
    return "general"


def ingest_prompts(verbose: bool = True) -> int:
    """
    采集并灌入 prompt 样本到 ChromaDB。

    Returns:
        写入的 chunk 总数
    """
    prompts = scrape_civitai_prompts(verbose=verbose)
    if not prompts:
        if verbose:
            print("    [!] 未采集到任何 prompt，跳过灌入")
        return 0

    collection = get_or_create_collection("prompt_examples")
    count = 0

    for i, p in enumerate(prompts):
        # 用 prompt 文本作为文档（ChromaDB 自动 embedding）
        doc_text = p["prompt"]
        if p.get("negative_prompt"):
            doc_text += f"\n\nNegative: {p['negative_prompt']}"

        # 质量分：基于社区反应数归一化到 0-1
        reactions = p.get("stats", {}).get("reactions", 0)
        quality_score = min(reactions / 100.0, 1.0)

        style = _infer_style(p.get("tags", []), p["prompt"])
        scene_type = _infer_scene_type(p["prompt"])

        chunk_id = f"prompt_{p.get('image_id', i):06d}"

        collection.upsert(
            ids=[chunk_id],
            documents=[doc_text],
            metadatas=[{
                "source_type": "civitai",
                "quality_score": quality_score,
                "style": style,
                "scene_type": scene_type,
                "sampler": p.get("sampler", ""),
                "model": p.get("model", "")[:50],  # 截断过长的模型名
            }],
        )
        count += 1

    if verbose:
        print(f"    ✅ 共灌入 {count} 条 prompt 到 prompt_examples")

    return count
