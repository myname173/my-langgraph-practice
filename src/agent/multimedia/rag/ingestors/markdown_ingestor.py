# src/agent/multimedia/rag/ingestors/markdown_ingestor.py
"""
Markdown 知识灌入器
====================
将 vision_knowledge/ 下的 markdown 文件按 ## 标题切块，
对超长 section 进行子切块（500-800 字符），写入 ChromaDB 的
cinematography_knowledge collection。

切块策略：
    - 按 ## 标题切分，每个 section 为一个基础 chunk
    - 超过 MAX_CHUNK_CHARS 的 section 按段落边界子切块（~600 字符）
    - 子切块之间保留 ~100 字符重叠，防止上下文断裂
    - 短 chunk（< 30 字符）跳过
    - category 从标题关键词推断
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

from ..vector_store import get_or_create_collection


# 知识文件所在目录
_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "vision_knowledge"

# 子切块参数：超过 MAX_CHUNK_CHARS 的 section 会被拆分为 TARGET_CHUNK_CHARS 大小的子块
MAX_CHUNK_CHARS = 800
TARGET_CHUNK_CHARS = 600
OVERLAP_CHARS = 100

# 标题关键词 → category 映射
_CATEGORY_RULES: List[Tuple[List[str], str]] = [
    (["shot", "camera", "angle", "movement", "lens", "组合", "景别", "机位", "运镜"], "camera"),
    (["light", "backlit", "silhouette", "time-of-day", "atmospheric", "光", "影", "逆光"], "lighting"),
    (["prompt", "keyword", "关键词", "descriptor", "prompt engineering", "quality booster", "style anchor", "negative prompt"], "prompt_keywords"),
    (["trailer", "pattern", "pacing", "transition", "arc", "节奏", "转场", "模板"], "editing"),
    (["composition", "构图", "frame", "golden ratio", "rule of thirds", "leading lines"], "composition"),
    (["color", "palette", "hue", "saturation", "色彩", "调色", "color theory"], "color"),
    (["mood", "atmosphere", "emotion", "feeling", "氛围", "情绪", "tension", "dread"], "mood"),
    (["style", "genre", "fantasy", "cyberpunk", "horror", "steampunk", "xianxia", "风格", "类型"], "style"),
    (["focal length", "depth of field", "bokeh", "lens flare", "optic", "aperture", "景深", "焦段"], "lens"),
    (["cut", "dissolve", "wipe", "match cut", "smash cut", "剪辑", "过渡"], "transition"),
    (["rhythm", "editing", "montage", "crescendo", "build-up", "剪辑节奏"], "pacing"),
]


def _infer_category(heading: str) -> str:
    """从标题推断知识分类。"""
    heading_lower = heading.lower()
    for keywords, category in _CATEGORY_RULES:
        for kw in keywords:
            if kw in heading_lower:
                return category
    return "general"


def _sub_chunk_text(text: str, target: int = TARGET_CHUNK_CHARS,
                    overlap: int = OVERLAP_CHARS) -> List[str]:
    """
    将长文本按段落边界切分为 ~target 字符的子块，相邻子块保留 ~overlap 字符重叠。

    切分规则：
    - 优先在双换行（段落边界）处切分
    - 如果单个段落超过 target，在 target 位置强制切分
    - 每个子块尾部保留前一个子块的最后 overlap 字符作为上下文重叠

    Args:
        text: 待切分的原始文本
        target: 目标子块大小（字符数）
        overlap: 相邻子块的重叠字符数

    Returns:
        子块文本列表
    """
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            continue
        para_len = len(para_stripped) + 2  # +2 for \n\n join

        # 单个段落就超过 target → 先把已有的 current 收尾，再强制切分长段落
        if para_len > target and not current_parts:
            # 强制按 target 长度切分超长段落
            start = 0
            while start < len(para_stripped):
                end = min(start + target, len(para_stripped))
                chunks.append(para_stripped[start:end])
                start = end - overlap if end < len(para_stripped) else end
            continue

        # 加入当前段落会超过 target → 收尾当前块
        if current_len + para_len > target and current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(chunk_text)
            # 保留尾部 overlap 字符作为下一块的前缀
            tail = chunk_text[-overlap:] if len(chunk_text) > overlap else ""
            current_parts = [tail, para_stripped] if tail else [para_stripped]
            current_len = len(tail) + 2 + para_len if tail else para_len
        else:
            current_parts.append(para_stripped)
            current_len += para_len

    # 收尾最后一块
    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return [c for c in chunks if len(c) > 30]


def _split_markdown_by_h2(content: str) -> List[Dict[str, str]]:
    """
    按 ## 标题切分 markdown 文件，超长 section 自动子切块。

    Returns:
        [{"heading": "标题文本", "content": "子块文本（含标题前缀）", "sub_index": int}, ...]
    """
    raw_sections: List[Tuple[str, str]] = []  # (heading, body_text)
    pattern = re.compile(r"^## ", re.MULTILINE)
    splits = pattern.split(content)
    headings = re.findall(r"^## (.+)$", content, re.MULTILINE)

    for i, heading in enumerate(headings):
        body = splits[i + 1].strip() if i + 1 < len(splits) else ""
        if not body or len(body) < 30:
            continue
        raw_sections.append((heading.strip(), body))

    # 对每个 section 做子切块
    chunks: List[Dict[str, str]] = []
    for heading, body in raw_sections:
        if len(body) <= MAX_CHUNK_CHARS:
            # 短 section：直接作为一个 chunk
            chunks.append({
                "heading": heading,
                "content": f"## {heading}\n\n{body}",
                "sub_index": 0,
            })
        else:
            # 长 section：按段落边界子切块
            sub_chunks = _sub_chunk_text(body)
            for si, sub_text in enumerate(sub_chunks):
                chunks.append({
                    "heading": heading,
                    "content": f"## {heading}\n\n{sub_text}",
                    "sub_index": si,
                })

    return chunks


def ingest_markdown_files(verbose: bool = True) -> int:
    """
    读取 vision_knowledge/ 下所有 .md 文件，切块后写入 ChromaDB。

    Args:
        verbose: 是否打印进度信息

    Returns:
        写入的 chunk 总数
    """
    collection = get_or_create_collection("cinematography_knowledge")
    total_chunks = 0

    md_files = sorted(_KNOWLEDGE_DIR.rglob("*.md"))
    if not md_files:
        print(f"    [!] 未找到知识文件: {_KNOWLEDGE_DIR}")
        return 0

    if verbose:
        print(f"\n--- 📚 [Markdown 灌入] 发现 {len(md_files)} 个知识文件 ---")

    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        chunks = _split_markdown_by_h2(content)

        if verbose:
            # 统计子切块效果
            n_sections = len({c["heading"] for c in chunks})
            print(f"    [*] {md_file.name}: {n_sections} sections → {len(chunks)} chunks "
                  f"(子切块: {len(chunks) - n_sections} 额外块)")

        # 用相对路径构建 ID，避免子目录文件 ID 冲突
        rel_stem = str(md_file.relative_to(_KNOWLEDGE_DIR).with_suffix("")).replace("/", "_").replace("\\", "_")

        for i, chunk in enumerate(chunks):
            category = _infer_category(chunk["heading"])
            sub_idx = chunk.get("sub_index", 0)
            # ID 格式: {file}_{section_idx}_{sub_idx}_{category}
            chunk_id = f"{rel_stem}_{i:02d}s{sub_idx}_{category}"

            # 写入 ChromaDB
            collection.upsert(
                ids=[chunk_id],
                documents=[chunk["content"]],
                metadatas=[{
                    "source_type": "markdown",
                    "source_file": md_file.name,
                    "relative_path": str(md_file.relative_to(_KNOWLEDGE_DIR)),
                    "category": category,
                    "heading": chunk["heading"],
                    "chunk_index": i,
                    "sub_index": sub_idx,
                }],
            )
            total_chunks += 1

        if verbose:
            categories = [_infer_category(c["heading"]) for c in chunks]
            cat_summary = {}
            for c in categories:
                cat_summary[c] = cat_summary.get(c, 0) + 1
            print(f"        分类: {cat_summary}")

    if verbose:
        print(f"    ✅ 共灌入 {total_chunks} 个 chunks 到 cinematography_knowledge")

    return total_chunks
