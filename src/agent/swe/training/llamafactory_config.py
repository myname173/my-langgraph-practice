# src/agent/swe/training/llamafactory_config.py
"""
LlamaFactory Config Generator
==============================
自动生成 LlamaFactory 的 YAML 训练配置文件，无需手动编写。

支持三种训练范式：
  1. SFT (Supervised Fine-tuning)  — 在成功轨迹上做监督微调
  2. DPO (Direct Preference Optimization) — 成功/失败轨迹对比训练
  3. GRPO (Group Relative Policy Optimization) — 全量轨迹 + 奖励信号

使用方式（CLI）：
  python -m src.agent.swe.training.llamafactory_config \
    --mode sft --model Qwen/Qwen2.5-7B-Instruct \
    --data-dir ./workspace/_training_data \
    --output-dir ./llamafactory_runs/sft_v1

使用方式（程序调用）：
  from src.agent.swe.training.llamafactory_config import generate_and_save_config
  generate_and_save_config(mode="sft", model_name_or_path="Qwen/Qwen2.5-7B-Instruct", ...)

注意：
  - 不修改 LlamaFactory 源码，仅通过 YAML 配置 + dataset_info.json 集成
  - 生成的 YAML 可直接用于：llamafactory-cli train config.yaml
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger("SWE_LlamaFactoryConfig")


# ==========================================
# 默认超参数（可覆盖）
# ==========================================

SFT_DEFAULTS = {
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "all",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "dataset": "swe_agent_sft",
    "template": "qwen",                    # 按使用的模型调整：qwen / llama3 / deepseek
    "cutoff_len": 4096,
    "max_samples": 2000,
    "overwrite_cache": True,
    "preprocessing_num_workers": 4,
    "output_dir": "./saves/sft_v1",
    "logging_steps": 10,
    "save_steps": 200,
    "plot_loss": True,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": "2.0e-4",
    "num_train_epochs": 3.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "fp16": True,                          # A100/H100 改为 bf16=True
    "ddp_timeout": 180000000,
    # WandB 集成（可选，按需取消注释）
    # "report_to": "wandb",
    # "run_name": "swe-agent-sft",
}

DPO_DEFAULTS = {
    "stage": "dpo",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "all",
    "lora_rank": 16,
    "lora_alpha": 32,
    "pref_beta": 0.1,
    "pref_loss": "sigmoid",               # 可选 "ipo" / "orpo"
    "dataset": "swe_agent_dpo",
    "template": "qwen",
    "cutoff_len": 4096,
    "max_samples": 1000,
    "overwrite_cache": True,
    "output_dir": "./saves/dpo_v1",
    "logging_steps": 10,
    "save_steps": 100,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": "5.0e-5",
    "num_train_epochs": 2.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "fp16": True,
}

GRPO_DEFAULTS = {
    "stage": "grpo",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "all",
    "lora_rank": 16,
    "lora_alpha": 32,
    "reward_model": None,                  # 使用内置奖励分而非外部 RM
    "dataset": "swe_agent_grpo",
    "template": "qwen",
    "cutoff_len": 4096,
    "max_samples": 3000,
    "overwrite_cache": True,
    "output_dir": "./saves/grpo_v1",
    "logging_steps": 10,
    "save_steps": 200,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": "1.0e-5",
    "num_train_epochs": 2.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "fp16": True,
}

# 自定义 Metrics 回调（注入 tool_call_accuracy / test_pass_rate）
CUSTOM_METRICS_CALLBACK = """
# custom_callbacks.py（放到 LlamaFactory 根目录，通过 --callbacks 加载）
# 计算 SWE Agent 专属指标：工具调用准确率、测试通过率、迭代效率
import json, re
from transformers import TrainerCallback

class SWEAgentMetricsCallback(TrainerCallback):
    \"\"\"
    向 WandB/TensorBoard 注入 SWE Agent 专属训练指标。
    在 trainer.predict() 输出的 predictions 中提取：
      - tool_call_accuracy: 工具调用成功率（正确调用 / 总调用）
      - test_pass_rate:     包含 TASK_COMPLETED 的预测比例
      - iteration_efficiency: 1 - mean(iterations / max_iterations)
    \"\"\"

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        # 从 eval_loss 辅助字段中读取自定义指标（由 compute_metrics 注入）
        for k in ["tool_call_accuracy", "test_pass_rate", "iteration_efficiency"]:
            if k in metrics:
                # LlamaFactory 的 trainer 会自动 log 到 WandB
                pass

def compute_swe_metrics(eval_preds):
    \"\"\"
    传入 compute_metrics 的函数。
    eval_preds.predictions: List[str]，模型生成的文本
    eval_preds.label_ids:   List[str]，参考答案
    \"\"\"
    predictions, labels = eval_preds
    if not isinstance(predictions[0], str):
        return {}

    task_completed = sum(1 for p in predictions if "TASK_COMPLETED" in p)
    test_pass_rate = task_completed / max(len(predictions), 1)

    # 工具调用格式合法性
    valid_tool_calls = 0
    total_tool_calls = 0
    for pred in predictions:
        matches = re.findall(r'\\{\"name\":\\s*\"(\\w+)\"', pred)
        total_tool_calls += len(matches)
        valid_tool_calls += sum(
            1 for m in matches
            if m in ["execute_command", "write_file", "read_file",
                     "search_code", "edit_file", "search_codebase"]
        )
    tool_call_accuracy = valid_tool_calls / max(total_tool_calls, 1)

    return {
        "test_pass_rate": round(test_pass_rate, 4),
        "tool_call_accuracy": round(tool_call_accuracy, 4),
    }
"""


# ==========================================
# 配置生成函数
# ==========================================

def generate_sft_config(
    model_name_or_path: str,
    data_dir: Path,
    output_dir: str = "./saves/sft_v1",
    template: str = "qwen",
    overrides: Optional[Dict] = None,
) -> Dict:
    """生成 SFT 训练配置字典。"""
    config = {**SFT_DEFAULTS}
    config["model_name_or_path"] = model_name_or_path
    config["dataset_dir"] = str(data_dir)
    config["output_dir"] = output_dir
    config["template"] = template
    if overrides:
        config.update(overrides)
    # 移除 None 值（YAML 中不应出现 null 字段）
    return {k: v for k, v in config.items() if v is not None}


def generate_dpo_config(
    model_name_or_path: str,
    data_dir: Path,
    output_dir: str = "./saves/dpo_v1",
    template: str = "qwen",
    adapter_name_or_path: Optional[str] = None,  # 可以在 SFT checkpoint 上继续 DPO
    overrides: Optional[Dict] = None,
) -> Dict:
    """生成 DPO 训练配置字典。"""
    config = {**DPO_DEFAULTS}
    config["model_name_or_path"] = model_name_or_path
    config["dataset_dir"] = str(data_dir)
    config["output_dir"] = output_dir
    config["template"] = template
    if adapter_name_or_path:
        config["adapter_name_or_path"] = adapter_name_or_path
    if overrides:
        config.update(overrides)
    return {k: v for k, v in config.items() if v is not None}


def generate_grpo_config(
    model_name_or_path: str,
    data_dir: Path,
    output_dir: str = "./saves/grpo_v1",
    template: str = "qwen",
    overrides: Optional[Dict] = None,
) -> Dict:
    """生成 GRPO 训练配置字典。"""
    config = {**GRPO_DEFAULTS}
    config["model_name_or_path"] = model_name_or_path
    config["dataset_dir"] = str(data_dir)
    config["output_dir"] = output_dir
    config["template"] = template
    if overrides:
        config.update(overrides)
    return {k: v for k, v in config.items() if v is not None}


def generate_and_save_config(
    mode: str,
    model_name_or_path: str,
    data_dir: Path,
    output_dir: Path,
    template: str = "qwen",
    adapter_name_or_path: Optional[str] = None,
    overrides: Optional[Dict] = None,
    also_write_callback: bool = True,
) -> Path:
    """
    生成 YAML 配置并写入磁盘，返回配置文件路径。
    同时在 output_dir 下写入 custom_callbacks.py 模板。

    参数:
      mode      训练范式：sft / dpo / grpo
      overrides 额外覆盖参数，例如 {"num_train_epochs": 5, "bf16": True}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "sft":
        config = generate_sft_config(
            model_name_or_path, data_dir,
            output_dir=str(output_dir / "model"),
            template=template, overrides=overrides,
        )
    elif mode == "dpo":
        config = generate_dpo_config(
            model_name_or_path, data_dir,
            output_dir=str(output_dir / "model"),
            template=template,
            adapter_name_or_path=adapter_name_or_path,
            overrides=overrides,
        )
    elif mode == "grpo":
        config = generate_grpo_config(
            model_name_or_path, data_dir,
            output_dir=str(output_dir / "model"),
            template=template, overrides=overrides,
        )
    else:
        raise ValueError(f"不支持的 mode: {mode}。请选择 sft / dpo / grpo")

    config_path = output_dir / f"{mode}_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logger.info(f"✅ {mode.upper()} 配置已写入: {config_path}")

    # 写入 README
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        f"# SWE Agent {mode.upper()} Training\n\n"
        f"## 启动训练\n\n"
        f"```bash\n"
        f"# 安装 LlamaFactory\n"
        f"pip install llamafactory\n\n"
        f"# 运行训练\n"
        f"llamafactory-cli train {config_path}\n"
        f"```\n\n"
        f"## 数据集\n\n"
        f"数据集位于 `{data_dir}`，已注册到 `dataset_info.json`。\n\n"
        f"## 自定义 Metrics\n\n"
        f"`custom_callbacks.py` 提供 `tool_call_accuracy` 和 `test_pass_rate` 等 SWE 专属指标，\n"
        f"在 LlamaFactory 配置中添加 `--callbacks custom_callbacks.SWEAgentMetricsCallback` 启用。\n",
        encoding="utf-8",
    )

    # 写入自定义 Metrics 回调模板
    if also_write_callback:
        cb_path = output_dir / "custom_callbacks.py"
        cb_path.write_text(CUSTOM_METRICS_CALLBACK, encoding="utf-8")
        logger.info(f"自定义 Metrics 回调已写入: {cb_path}")

    return config_path


# ==========================================
# CLI 入口
# ==========================================

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="生成 LlamaFactory 训练配置")
    parser.add_argument("--mode", choices=["sft", "dpo", "grpo"], required=True)
    parser.add_argument("--model", required=True, help="模型名称或路径，如 Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data-dir", default="./workspace/_training_data")
    parser.add_argument("--output-dir", default="./llamafactory_runs")
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--adapter", default=None, help="SFT checkpoint 路径（DPO 继续训练用）")
    args = parser.parse_args()

    try:
        config_path = generate_and_save_config(
            mode=args.mode,
            model_name_or_path=args.model,
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir) / args.mode,
            template=args.template,
            adapter_name_or_path=args.adapter,
        )
        print(f"✅ 配置已生成: {config_path}")
        print(f"\n启动训练命令:\n  llamafactory-cli train {config_path}")
    except Exception as e:
        logger.error(f"配置生成失败: {e}")
        sys.exit(1)
