# RAG 知识检索层

为 director 节点提供基于 ChromaDB + DashScope text-embedding 的语义检索增强，丰富图像/视频生成提示词。

## 架构

```
vision_knowledge/*.md ─┐
Civitai prompts ────────┤── ingest.py ──→ ChromaDB (data/chroma_db/)
Film knowledge ─────────┘                        │
                                                 ▼
                                    retriever.py (语义检索)
                                                 │
                                    rag_context  ▼
                              visual_context_builder → director_node
```

RAG 是叠加层，不替换现有规则系统。检索失败时静默降级，不影响管线运行。

## 3 个 Collection

| Collection | 数据源 | 用途 |
|---|---|---|
| `cinematography_knowledge` | `vision_knowledge/` 下的 md 文件 | 摄影/布光/运镜知识 |
| `prompt_examples` | Civitai API 爬取 | 高质量 prompt 样本 |
| `film_references` | 内置影视知识 + 可选网页爬取 | 影视参考描述 |

## 灌入数据

```bash
# 灌入全部数据源
python -m src.agent.multimedia.rag.ingest

# LLM 生成知识 + 灌入（首次推荐，会自动调用 LLM 生成 10 个专业主题的知识文件）
python -m src.agent.multimedia.rag.ingest --source gen

# 只灌入已有的 markdown 知识（不含 LLM 生成的）
python -m src.agent.multimedia.rag.ingest --source md

# 只灌入 Civitai prompt
python -m src.agent.multimedia.rag.ingest --source prompts

# 只灌入影视知识
python -m src.agent.multimedia.rag.ingest --source films

# 清空后重建
python -m src.agent.multimedia.rag.ingest --force

# 查看统计（不灌入）
python -m src.agent.multimedia.rag.ingest --stats

# 减少输出
python -m src.agent.multimedia.rag.ingest --quiet
```

### 知识生成器独立使用

```bash
# 查看所有可生成的主题
python -m src.agent.multimedia.rag.ingestors.knowledge_generator --list

# 只生成某个主题（序号从 1 开始）
python -m src.agent.multimedia.rag.ingestors.knowledge_generator --only 3

# 强制重新生成全部（覆盖已有文件）
python -m src.agent.multimedia.rag.ingestors.knowledge_generator --force
```

生成器会将知识文件写入 `vision_knowledge/generated/` 目录，已存在的文件不会被重复生成（除非加 `--force`）。

## 文件结构

```
rag/
├── __init__.py
├── vector_store.py          # ChromaDB 初始化 + Collection 管理
├── retriever.py             # 语义检索入口
├── ingest.py                # 灌入 CLI 入口
├── ingestors/
│   ├── __init__.py
│   ├── markdown_ingestor.py # md 文件切块灌入（支持 generated/ 子目录）
│   ├── knowledge_generator.py  # LLM 批量生成专业知识文件
│   ├── prompt_scraper.py    # Civitai API 爬虫
│   └── film_knowledge_scraper.py  # 影视知识（内置 + 可选爬取）
└── README.md
```

## 添加自定义知识源

1. 在 `ingestors/` 下新建灌入器文件，实现 `ingest_xxx(verbose=True) -> int` 函数
2. 在 `vector_store.py` 的 `COLLECTION_DEFINITIONS` 中注册新 collection（如需）
3. 在 `ingest.py` 的 `INGESTOR_REGISTRY` 中注册新灌入器
4. 在 `retriever.py` 的 `_RETRIEVAL_CONFIG` 中添加检索配置

## 依赖

- `chromadb>=0.5.0` — 向量数据库
- DashScope `text-embedding-v3` — 文本向量化（复用项目已有的 API key）

## 注意事项

- ChromaDB 数据存储在 `data/chroma_db/`，首次运行需先执行灌入
- Civitai 爬虫有 1s 限速，大量数据时请耐心等待
- 所有灌入器都有本地 JSON 缓存（`data/cache/`），重复运行不会重复请求 API
- 检索结果 distance > 0.5~0.6 的会被自动过滤，只保留高相关内容
