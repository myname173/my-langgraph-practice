# src/agent/multimedia/rag/ingestors/curated_prompt_ingestor.py
"""
网络精选 Prompt 灌入器
========================
读取 vision_knowledge/web_curated/ 下的高质量 markdown 文件，
按 ## 标题切块后写入 ChromaDB 的 prompt_examples 和 film_references collection。

数据来源：
    - 联网搜索并人工筛选的高质量 AI 短剧/视频/游戏提示词
    - 涵盖运镜、电影级场景、游戏风格、prompt 工程方法论

与 markdown_ingestor 的区别：
    - markdown_ingestor → 写入 cinematography_knowledge（摄影知识）
    - 本模块 → 写入 prompt_examples（prompt 样本）+ film_references（方法论）

切块策略：
    - 按 ## 标题切分
    - 代码块 (```) 保持完整不拆分
    - 每个 chunk 保留 metadata: style, scene_type, source_file
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

from ..vector_store import get_or_create_collection


# 网络精选文件所在目录
_CURATED_DIR = Path(__file__).resolve().parent.parent.parent / "vision_knowledge" / "web_curated"

# 文件 → 目标 collection 映射
_FILE_COLLECTION_MAP: Dict[str, str] = {
    "camera_movements.md": "prompt_examples",
    "cinematic_prompts.md": "prompt_examples",
    "game_visual_prompts.md": "prompt_examples",
    "prompt_framework.md": "film_references",
}

# 从文件名推断的默认 style 和 scene_type
_FILE_META: Dict[str, Dict[str, str]] = {
    "camera_movements.md": {"style": "cinematic", "scene_type": "camera_technique"},
    "cinematic_prompts.md": {"style": "cinematic", "scene_type": "general"},
    "game_visual_prompts.md": {"style": "game_trailer", "scene_type": "general"},
    "prompt_framework.md": {"style": "methodology", "scene_type": "general"},
}

# 从 heading 关键词推断 style
_STYLE_KEYWORDS: List[Tuple[List[str], str]] = [
    (["cyberpunk", "霓虹", "赛博"], "cyberpunk"),
    (["fantasy", "奇幻", "魔法", "elven", "dragon"], "fantasy"),
    (["horror", "恐怖", "gothic", "哥特", "吸血鬼"], "horror"),
    (["post-apocalyptic", "废土", "末世", "survivor"], "post_apocalyptic"),
    (["sci-fi", "太空", "space", "星际", "warship"], "sci_fi"),
    (["xianxia", "仙侠", "武侠", "wuxia", "修仙"], "xianxia"),
    (["racing", "竞速", "赛车", "motorsport"], "racing"),
    (["stealth", "潜行", "谍战", "espionage"], "stealth"),
    (["anime", "动漫", "manga"], "anime"),
    (["水墨", "国风", "ink painting"], "chinese_ink"),
    (["产品", "product", "commercial"], "commercial"),
    (["复古", "retro", "胶片", "film", "vlog"], "retro"),
]

# 从 heading 关键词推断 scene_type
_SCENE_KEYWORDS: List[Tuple[List[str], str]] = [
    (["运镜", "camera", "movement", "dolly", "crane", "orbit", "tracking", "pan"], "camera_technique"),
    (["environment", "环境", "场景", "landscape", "城市", "forest", "森林"], "environment"),
    (["action", "动作", "战斗", "combat", "battle"], "action"),
    (["character", "角色", "人物", "portrait"], "character"),
    (["product", "产品", "commercial"], "product"),
    (["trailer", "预告片", "节奏", "pacing", "节奏模板"], "trailer_template"),
    (["framework", "框架", "方法论", "methodology", "权重"], "methodology"),
    (["prompt", "提示词", "prompt engineering"], "prompt_engineering"),
]


def _infer_style(heading: str) -> str:
    """从标题推断视觉风格。"""
    h = heading.lower()
    for keywords, style in _STYLE_KEYWORDS:
        for kw in keywords:
            if kw.lower() in h:
                return style
    return "cinematic"


def _infer_scene_type(heading: str) -> str:
    """从标题推断场景类型。"""
    h = heading.lower()
    for keywords, stype in _SCENE_KEYWORDS:
        for kw in keywords:
            if kw.lower() in h:
                return stype
    return "general"


def _split_by_h2(content: str) -> List[Dict[str, str]]:
    """按 ## 标题切分 markdown，保持代码块完整。"""
    chunks = []
    pattern = re.compile(r"^## ", re.MULTILINE)
    splits = pattern.split(content)
    headings = re.findall(r"^## (.+)$", content, re.MULTILINE)

    for i, heading in enumerate(headings):
        section = splits[i + 1].strip() if i + 1 < len(splits) else ""
        if not section or len(section) < 30:
            continue
        chunks.append({
            "heading": heading.strip(),
            "content": f"## {heading.strip()}\n\n{section}",
        })

    return chunks


def ingest_curated_prompts(verbose: bool = True) -> int:
    """
    读取 web_curated/ 下所有精选 markdown，灌入对应 ChromaDB collection。

    Returns:
        写入的 chunk 总数
    """
    if not _CURATED_DIR.exists():
        if verbose:
            print(f"    [!] web_curated 目录不存在: {_CURATED_DIR}")
        return 0

    md_files = sorted(_CURATED_DIR.glob("*.md"))
    if not md_files:
        if verbose:
            print(f"    [!] web_curated 下没有 md 文件")
        return 0

    total_chunks = 0

    if verbose:
        print(f"\n--- 🌐 [网络精选灌入] 发现 {len(md_files)} 个精选文件 ---")

    for md_file in md_files:
        filename = md_file.name
        target_collection = _FILE_COLLECTION_MAP.get(filename, "prompt_examples")
        file_meta = _FILE_META.get(filename, {"style": "general", "scene_type": "general"})

        collection = get_or_create_collection(target_collection)
        content = md_file.read_text(encoding="utf-8")
        chunks = _split_by_h2(content)

        if verbose:
            print(f"    [*] {filename} → {target_collection}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            style = _infer_style(chunk["heading"])
            if style == "cinematic":
                style = file_meta.get("style", "cinematic")

            scene_type = _infer_scene_type(chunk["heading"])
            if scene_type == "general":
                scene_type = file_meta.get("scene_type", "general")

            chunk_id = f"curated_{md_file.stem}_{i:02d}"

            collection.upsert(
                ids=[chunk_id],
                documents=[chunk["content"]],
                metadatas=[{
                    "source_type": "web_curated",
                    "source_file": filename,
                    "quality_score": 0.85,
                    "style": style,
                    "scene_type": scene_type,
                    "heading": chunk["heading"],
                    "chunk_index": i,
                }],
            )
            total_chunks += 1

        if verbose:
            styles = {}
            for c in chunks:
                s = _infer_style(c["heading"])
                styles[s] = styles.get(s, 0) + 1
            print(f"        风格分布: {styles}")

    if verbose:
        print(f"    ✅ 共灌入 {total_chunks} 个精选 chunks")

    return total_chunks
