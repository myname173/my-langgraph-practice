# src/agent/swe/training/run_training_pipeline.py
"""
SWE Agent 训练数据流水线（一键可复现）
======================================

将「导入公开 SWE 数据 → 导出训练集 → 生成 LlamaFactory 配置」
三步串联为单个命令，保证从原始数据到可训练配置的端到端可复现。

依赖：
  - import_public_data.py   （导入公开数据集 → trajectories.jsonl）
  - data_pipeline.py        （trajectories → sft/dpo/grpo 导出）
  - llamafactory_config.py  （生成 LlamaFactory YAML 配置）

使用方式：
  # 默认：从 SWE-bench_Lite 导入并生成 SFT + GRPO 配置
  python -m src.agent.swe.training.run_training_pipeline \
      --model deepseek-ai/deepseek-coder-7b-instruct

  # 指定本地权重 / 其他模型
  python -m src.agent.swe.training.run_training_pipeline \
      --model /path/to/Qwen2.5-Coder-7B-Instruct \
      --output-dir ./llamafactory_runs

说明：
  - 按用户决策，DPO 不纳入默认流程（无带标签的偏好对数据）。
    如需单独生成 DPO 配置，可手动调用 llamafactory_config.py。
"""

import argparse
import logging
import sys
from pathlib import Path

from . import import_public_data as importer
from . import data_pipeline as pipeline
from . import llamafactory_config as lf_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SWE_TrainingPipeline")


def run(
    model_name_or_path: str,
    source: str = "swebench",
    data_dir: Path = Path("./workspace/_training_data"),
    output_dir: Path = Path("./llamafactory_runs"),
    max_records: int = 100,
    template: str = None,
) -> None:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # ── Step 1: 导入公开数据 → 转换 → 重新导出（sft/dpo/grpo + dataset_info）──
    logger.info("=" * 60)
    logger.info("Step 1/3  导入公开 SWE 数据并导出训练集")
    logger.info("=" * 60)
    max_success = min(max_records, 100)
    max_failed = 0  # SWE-bench 仅正样本，无失败/偏好数据（按决策放弃 DPO）
    report = importer.import_and_export(
        source=source,
        max_success=max_success,
        max_failed=max_failed,
        workspace_dir=data_dir.parent if data_dir.name == "_training_data" else data_dir,
    )

    counts = report.get("counts", {})
    summary = {
        "sft_success": counts.get("sft_samples", 0),
        "dpo_pairs": counts.get("dpo_pairs", 0),
        "grpo_all": counts.get("grpo_samples", 0),
    }

    # ── Step 2: 生成 LlamaFactory 配置 ──
    logger.info("=" * 60)
    logger.info("Step 2/3  生成 LlamaFactory 配置 (SFT + GRPO)")
    logger.info("=" * 60)

    sft_path = lf_config.generate_and_save_config(
        mode="sft",
        model_name_or_path=model_name_or_path,
        data_dir=data_dir,
        output_dir=output_dir / "sft_v1",
        template=template,
    )
    logger.info(f"SFT 配置 → {sft_path}")

    grpo_path = lf_config.generate_and_save_config(
        mode="grpo",
        model_name_or_path=model_name_or_path,
        data_dir=data_dir,
        output_dir=output_dir / "grpo_v1",
        template=template,
    )
    logger.info(f"GRPO 配置 → {grpo_path}")

    # ── 汇总 ──
    logger.info("=" * 60)
    logger.info("流水线完成 ✅")
    logger.info(f"  SFT 样本 : {summary['sft_success']}")
    logger.info(f"  DPO 样本 : {summary['dpo_pairs']}  (按决策跳过训练)")
    logger.info(f"  GRPO 样本: {summary['grpo_all']}")
    logger.info("")
    logger.info("启动训练:")
    logger.info(f"  llamafactory-cli train {sft_path}")
    logger.info(f"  llamafactory-cli train {grpo_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SWE Agent 训练数据流水线")
    parser.add_argument(
        "--model",
        required=True,
        help="训练用的 base model（本地路径或 HF 名称）。"
             "例如 deepseek-ai/deepseek-coder-7b-instruct 或 Qwen/Qwen2.5-Coder-7B",
    )
    parser.add_argument("--source", default="swebench", choices=["swebench"])
    parser.add_argument("--data-dir", default="./workspace/_training_data")
    parser.add_argument("--output-dir", default="./llamafactory_runs")
    parser.add_argument("--max-records", type=int, default=100,
                        help="最多导入的公开样本数")
    parser.add_argument("--template", default=None,
                        help="LlamaFactory 模板，留空自动按模型名推导")
    args = parser.parse_args()

    try:
        run(
            model_name_or_path=args.model,
            source=args.source,
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            max_records=args.max_records,
            template=args.template,
        )
    except Exception as e:
        logger.error(f"流水线失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
