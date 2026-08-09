# src/agent/graph.py
"""
统一图入口。

实际的多模态宣传片生成工作流定义在 `src/agent/multimedia/graph.py`
（编译产物名为 `multimedia_agent`）。为兼容：
  - `langgraph.json` 中 `"agent": "src.agent.graph:graph"` 的配置
  - `tests/` 中 `from agent.graph import graph` / `from agent import graph` 的引用

在此处重新导出编译后的图为 `graph`，作为对外暴露的单一入口。
"""
from .multimedia.graph import multimedia_agent as graph

__all__ = ["graph"]
