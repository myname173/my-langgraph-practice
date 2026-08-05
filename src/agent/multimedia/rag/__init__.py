# src/agent/multimedia/rag/__init__.py
"""
RAG (Retrieval-Augmented Generation) 模块
==========================================
为 LangGraph trailer 管线提供基于 ChromaDB 的语义检索能力。

子模块：
    - vector_store: ChromaDB 客户端封装 + Collection 管理
    - retriever: 知识检索 + 格式化
    - ingest: 数据灌入入口脚本
    - ingestors/: 各类数据源灌入器
"""
