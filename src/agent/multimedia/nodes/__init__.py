# src/agent/multimedia/nodes/__init__.py
"""multimedia 图的节点实现包（P2-3 拆分）。

由原 graph.py 单体按 stage 拆出：
  common    跨 stage 常量 / 纯工具 / 风格与 prompt 工具箱
  scripting 脚本与规划（视觉上下文 / 镜头策略 / 序列编排 / showrunner）
  design    视觉设计（参考图生成 / 资产匹配）
  director  导演（电影级评审 / 精修 / 影棚 / 首尾帧导演 / 运镜）
  render    渲染（关键帧 / 尾帧 / 视频生成 + 审核闸口）
  assemble  成片（推进 / 中止 / 拼接 / 草稿审片 / 混音）
  routing   路由决策
  parallel  并行 / 链式扇出机制
  wiring    图组装与编译
"""
import sys

# 根治：在 GBK 等非 UTF-8 控制台下，含 emoji/中文的 print 会抛 UnicodeEncodeError
# 并静默打断重试/降级逻辑。强制 stdout/stderr 以 UTF-8 容错输出。
# （与原 graph.py 顶部保持一致的运行期行为）
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
