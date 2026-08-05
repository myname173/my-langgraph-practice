# src/agent/multimedia/rag/ingestors/film_knowledge_scraper.py
"""
影视知识文本爬虫
=================
从摄影/影视教程网站采集专业知识，写入 ChromaDB film_references collection。

数据源策略（独立开发者友好，零登录）：
    1. 预置的高质量影视知识文本（内嵌在代码中，保证最小可用数据集）
    2. 可选的 Web 采集（从指定 URL 列表抓取文章）

设计原则：
    - 内嵌数据保证即使网络不可用也有基础数据
    - Web 采集失败不中断流程
    - 缓存到本地 JSON
"""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..vector_store import get_or_create_collection


# ============================================================
# 1. 内嵌影视知识（保底数据集）
# ============================================================

_BUILTIN_FILM_KNOWLEDGE: List[Dict[str, str]] = [
    {
        "title": "Three-Act Trailer Structure",
        "category": "trailer_structure",
        "content": (
            "Professional game trailers follow a three-act structure adapted from film: "
            "Act 1 (World Establishment) opens with wide establishing shots, slow camera "
            "movements, and atmospheric lighting to immerse viewers in the game world. "
            "Act 2 (Character & Conflict) introduces protagonists through medium shots and "
            "close-ups, escalating tension with faster cuts and dynamic camera work. "
            "Act 3 (Climax & Resolution) delivers rapid-fire action montage, peak emotional "
            "intensity, then a final beat of silence or a single powerful image for the title card. "
            "The transition between acts should feel natural — use audio cues, color shifts, "
            "or match cuts to bridge sections."
        ),
    },
    {
        "title": "Color Temperature and Emotional Impact",
        "category": "color_theory",
        "content": (
            "Color temperature directly influences viewer emotion in cinematic content. "
            "Cool tones (5000K-7000K, blue-cyan) evoke isolation, technology, melancholy, "
            "and futuristic settings — ideal for sci-fi and cyberpunk trailers. "
            "Warm tones (2700K-4000K, amber-gold) create intimacy, nostalgia, comfort, "
            "and heroic warmth — perfect for fantasy and emotional scenes. "
            "Neutral tones (4000K-5000K) feel documentary-like and objective. "
            "Color contrast between scenes (cool-to-warm shift) signals narrative transitions. "
            "Teal-and-orange grading (complementary colors) is the industry standard for "
            "blockbuster trailers because it simultaneously pushes skin tones warm against "
            "cool backgrounds, making characters pop."
        ),
    },
    {
        "title": "Camera Movement Psychology",
        "category": "camera_psychology",
        "content": (
            "Camera movement carries subconscious emotional weight. "
            "Dolly in (pushing toward subject): creates intimacy, escalating tension, or "
            "revelation — the viewer is drawn closer to the subject's emotional state. "
            "Dolly out (pulling away): isolation, abandonment, or revealing context — "
            "the viewer gains perspective but loses connection. "
            "Tracking shot (lateral movement): journey, pursuit, or parallel observation. "
            "Crane up: transcendence, freedom, overview, or divine perspective. "
            "Crane down: descent into danger, oppression, or grounding. "
            "Handheld: urgency, realism, chaos, documentary feel. "
            "Steadicam: immersive, dreamlike, first-person presence. "
            "The speed of movement matters: slow = contemplation, medium = narrative, "
            "fast = excitement or panic."
        ),
    },
    {
        "title": "Lighting Ratios and Mood",
        "category": "lighting_design",
        "content": (
            "Lighting ratio (key-to-fill ratio) is the primary tool for mood control. "
            "Low ratio (1:1 to 2:1): flat, even lighting — comedy, beauty, utopian settings. "
            "Medium ratio (3:1 to 4:1): natural, balanced — drama, adventure, standard cinematic. "
            "High ratio (8:1+): deep shadows, noir, horror, moral ambiguity. "
            "Practical lighting (visible light sources in frame) adds authenticity and depth. "
            "Motivated lighting (light direction justified by scene elements like windows, lamps) "
            "maintains viewer immersion. "
            "Rim lighting separates subjects from backgrounds — essential for dark scenes. "
            "Volumetric lighting (visible light rays through fog/dust/smoke) adds atmosphere "
            "and dimensionality, particularly effective in establishing shots."
        ),
    },
    {
        "title": "Shot Composition for Trailers",
        "category": "composition",
        "content": (
            "Trailer composition follows intensified cinema rules. "
            "Rule of thirds: place subjects at intersection points for dynamic balance. "
            "Leading lines: use architecture, roads, or light beams to guide viewer's eye. "
            "Frame within a frame: doorways, windows, or arches create depth and focus. "
            "Negative space: vast empty areas around subjects emphasize scale or loneliness. "
            "Symmetry: conveys power, order, or unease (when slightly broken). "
            "Depth layering: foreground element + mid-ground subject + background context. "
            "For AI-generated trailer frames, include composition keywords in prompts: "
            "'rule of thirds composition', 'leading lines', 'foreground bokeh', "
            "'cinematic framing', 'depth of field' to guide generation quality."
        ),
    },
    {
        "title": "Pacing and Cut Rhythm",
        "category": "editing",
        "content": (
            "Cut rhythm directly controls viewer heart rate. "
            "Slow pacing (3-5 second holds): contemplation, grandeur, tension building. "
            "Medium pacing (1.5-3 seconds): narrative flow, standard storytelling. "
            "Fast pacing (0.5-1.5 seconds): action, excitement, chaos, climax energy. "
            "Ultra-fast cuts (<0.5 seconds): montage effect, sensory overload, trailer finale. "
            "Rhythm variation is key: never maintain one pace for too long. "
            "The 'breathing' pattern — fast cuts followed by a held beat — creates emotional peaks. "
            "Match cuts (similar composition between shots) allow faster cutting without confusion. "
            "Jump cuts create disorientation — use sparingly for horror or psychological tension. "
            "Cross-cutting (alternating between two storylines) builds tension toward convergence."
        ),
    },
    {
        "title": "Sound Design Principles for Visual Planning",
        "category": "audio_visual",
        "content": (
            "While trailers are primarily visual in AI generation, planning shots with "
            "implied audio rhythm improves visual cohesion. "
            "Bass drops align with impact shots (close-ups of impacts, explosions). "
            "Rising tones pair with ascending crane shots or dolly-ins. "
            "Silence before the climax shot amplifies its visual impact. "
            "Rhythmic cutting synced to implied beat creates professional feel. "
            "Visual silence (still frames, wide empty shots) corresponds to audio pauses. "
            "When planning shot sequences, consider the implied audio arc: "
            "quiet opening → building rhythm → peak intensity → sudden silence → final impact."
        ),
    },
    {
        "title": "AI Prompt Engineering for Cinematic Quality",
        "category": "prompt_engineering",
        "content": (
            "Key techniques for cinematic AI image generation prompts: "
            "Always specify camera/lens: 'shot on 35mm lens', '85mm portrait', 'anamorphic'. "
            "Include lighting direction: 'rim-lit from behind', 'golden hour side lighting'. "
            "Add film stock reference: 'Kodak Vision3 500T', 'Fuji Eterna', 'film grain'. "
            "Specify composition: 'rule of thirds', 'low angle shot', 'over-the-shoulder'. "
            "Use negative space keywords: 'minimalist background', 'isolated subject'. "
            "Atmospheric effects: 'volumetric fog', 'lens flare', 'bokeh', 'light rays'. "
            "Color grading hints: 'teal and orange color grading', 'desaturated cool tones'. "
            "Quality boosters: 'cinematic lighting', 'professional photography', '8K detailed'. "
            "Avoid: overly complex scenes, too many subjects, contradictory lighting descriptions."
        ),
    },
]


# ============================================================
# 2. Web 采集（可选扩展）
# ============================================================

# 预定义的高质量影视知识文章 URL（经过筛选的、无需登录的页面）
_ARTICLE_URLS: List[Dict[str, str]] = [
    # 可在后续扩展更多 URL
]

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "cache"
_CACHE_FILE = _CACHE_DIR / "film_knowledge.json"


def _fetch_article(url: str) -> Optional[str]:
    """抓取网页文章并提取正文文本。"""
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (compatible; FilmKnowledgeBot/1.0)"
        })
        resp.raise_for_status()

        # 简单的 HTML → 文本提取
        text = resp.text
        # 移除 script 和 style 标签
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL)
        # 移除 HTML 标签
        text = re.sub(r"<[^>]+>", " ", text)
        # 清理空白
        text = re.sub(r"\s+", " ", text).strip()
        # 取前 3000 字符（避免过长）
        return text[:3000] if len(text) > 100 else None
    except Exception as e:
        print(f"    [!] 文章抓取失败 ({url}): {e}")
        return None


def _chunk_article(text: str, title: str, max_chunk: int = 500) -> List[str]:
    """将长文章按段落切块。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chunk:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [c for c in chunks if len(c) > 50]


# ============================================================
# 3. 灌入入口
# ============================================================

def ingest_film_knowledge(verbose: bool = True) -> int:
    """
    灌入影视知识到 ChromaDB film_references collection。

    数据来源：
        1. 内嵌保底知识（始终可用）
        2. Web 采集文章（如果 URL 列表非空）

    Returns:
        写入的 chunk 总数
    """
    collection = get_or_create_collection("film_references")
    count = 0

    # ── Part 1: 内嵌知识 ──
    if verbose:
        print(f"\n--- 🎬 [影视知识灌入] 内嵌知识: {len(_BUILTIN_FILM_KNOWLEDGE)} 条 ---")

    for i, item in enumerate(_BUILTIN_FILM_KNOWLEDGE):
        chunk_id = f"builtin_film_{i:03d}"
        collection.upsert(
            ids=[chunk_id],
            documents=[item["content"]],
            metadatas=[{
                "source_type": "builtin",
                "film_title": item["title"],
                "category": item.get("category", "general"),
                "scene_description": item["title"],
            }],
        )
        count += 1
        if verbose:
            print(f"    [*] {item['title']} ({item.get('category', 'general')})")

    # ── Part 2: Web 采集 ──
    if _ARTICLE_URLS:
        if verbose:
            print(f"\n    [*] Web 采集: {len(_ARTICLE_URLS)} 个 URL...")

        for article_info in _ARTICLE_URLS:
            url = article_info["url"]
            title = article_info.get("title", url[:50])

            text = _fetch_article(url)
            if not text:
                continue

            chunks = _chunk_article(text, title)
            for j, chunk in enumerate(chunks):
                # 使用 hashlib 替代内置 hash()：Python 3.3+ 的 hash() 有随机种子，
                # 跨进程不稳定，会导致 ChromaDB 中出现重复 chunk ID
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                chunk_id = f"web_{url_hash}_{j:02d}"
                collection.upsert(
                    ids=[chunk_id],
                    documents=[chunk],
                    metadatas=[{
                        "source_type": "web",
                        "film_title": title,
                        "category": article_info.get("category", "general"),
                        "scene_description": f"{title} (chunk {j+1}/{len(chunks)})",
                    }],
                )
                count += 1

            time.sleep(0.5)

    if verbose:
        print(f"    ✅ 共灌入 {count} 条影视知识到 film_references")

    return count
