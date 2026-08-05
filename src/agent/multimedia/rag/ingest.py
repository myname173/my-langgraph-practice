# src/agent/multimedia/rag/ingest.py
"""
RAG 知识灌入 CLI 入口
======================
用法：
    python -m src.agent.multimedia.rag.ingest              # 灌入全部数据源
    python -m src.agent.multimedia.rag.ingest --source md    # 只灌入 markdown
    python -m src.agent.multimedia.rag.ingest --source gen   # LLM 生成知识 + 灌入（首次推荐）
    python -m src.agent.multimedia.rag.ingest --source web   # 网络精选 prompt（推荐，高质量人工筛选数据）
    python -m src.agent.multimedia.rag.ingest --source gen web films  # 推荐组合：生成知识 + 网络精选 + 影视知识
    python -m src.agent.multimedia.rag.ingest --force        # 清空后重建
    python -m src.agent.multimedia.rag.ingest --stats        # 只查看统计
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中（支持 python -m 方式运行）
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.agent.multimedia.rag.vector_store import (
    get_collection_stats,
    reset_collection,
    COLLECTION_DEFINITIONS,
)
from src.agent.multimedia.rag.ingestors.markdown_ingestor import ingest_markdown_files
from src.agent.multimedia.rag.ingestors.prompt_scraper import ingest_prompts
from src.agent.multimedia.rag.ingestors.film_knowledge_scraper import ingest_film_knowledge
from src.agent.multimedia.rag.ingestors.knowledge_generator import ingest_generated_knowledge
from src.agent.multimedia.rag.ingestors.curated_prompt_ingestor import ingest_curated_prompts


# ============================================================
# 灌入器注册表
# ============================================================

INGESTOR_REGISTRY = {
    "md": {
        "name": "Markdown 知识文件",
        "collections": ["cinematography_knowledge"],
        "run": ingest_markdown_files,
    },
    "gen": {
        "name": "LLM 生成知识（首次运行较慢，会调用 LLM 生成 10 个主题文件）",
        "collections": ["cinematography_knowledge"],
        "run": ingest_generated_knowledge,
    },
    "web": {
        "name": "网络精选 Prompt（人工筛选的高质量 AI 短剧/视频/游戏提示词）",
        "collections": ["prompt_examples", "film_references"],
        "run": ingest_curated_prompts,
    },
    "prompts": {
        "name": "Web Prompt 爬虫 (Civitai)",
        "collections": ["prompt_examples"],
        "run": ingest_prompts,
    },
    "films": {
        "name": "影视知识",
        "collections": ["film_references"],
        "run": ingest_film_knowledge,
    },
}


def run_all_ingestors(sources=None, force=False, verbose=True):
    """运行指定的灌入器。"""
    if force:
        collections_to_reset = set()
        for key, info in INGESTOR_REGISTRY.items():
            if sources is None or key in sources:
                collections_to_reset.update(info["collections"])
        for col_name in collections_to_reset:
            reset_collection(col_name)

    total = 0
    for key, info in INGESTOR_REGISTRY.items():
        if sources is not None and key not in sources:
            continue
        if verbose:
            print(f"\n{'='*50}")
            print(f"  灌入: {info['name']}")
            print(f"{'='*50}")
        count = info["run"](verbose=verbose)
        total += count

    if verbose:
        print(f"\n{'='*50}")
        print(f"  灌入完成，总计: {total} chunks")
        print(f"{'='*50}")
        print_stats()

    return total


def print_stats():
    """打印所有 Collection 的统计信息。"""
    stats = get_collection_stats()
    print("\n--- 📊 Collection 统计 ---")
    for name, info in stats.items():
        count = info["count"]
        desc = info["description"]
        status = f"{count} 条" if count >= 0 else f"错误: {info.get('error', '?')}"
        print(f"    {name}: {status} ({desc})")


def main():
    parser = argparse.ArgumentParser(description="RAG 知识库灌入工具")
    parser.add_argument(
        "--source", "-s",
        choices=list(INGESTOR_REGISTRY.keys()),
        nargs="+",
        default=None,
        help="指定灌入的数据源（默认全部）",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="清空 Collection 后重建",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="只显示统计信息，不灌入",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="减少输出",
    )

    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    run_all_ingestors(
        sources=args.source,
        force=args.force,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
