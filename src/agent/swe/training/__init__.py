# src/agent/swe/training/__init__.py
"""
SWE Agent × LlamaFactory Training Integration
==============================================
将 Agent 执行轨迹转换为 LlamaFactory 可消费的后训练数据。

快速使用:
  from src.agent.swe.training.data_pipeline import run_pipeline
  from pathlib import Path
  report = run_pipeline(workspace_dir=Path("./workspace"))

模块说明:
  trajectory_logger   — 轨迹捕获与持久化
  reward_computer     — 奖励信号计算（Outcome + 效率 + 工具准确率）
  data_pipeline       — 全量 Pipeline（轨迹 → SFT/DPO/GRPO JSONL）
  llamafactory_config — LlamaFactory YAML 配置文件生成器
"""
