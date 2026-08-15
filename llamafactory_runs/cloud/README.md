# 云端 7B 正式训练（deepseek-coder-7b-instruct + LoRA）

本目录存放面向**正式训练目标**的配置与说明。本地开发机（如 GTX 1650 Ti 4GB）显存不足以微调 7B 模型，需在云端 GPU 环境运行。

## 推荐云端环境

| 环境 | 显存 | 说明 |
|------|------|------|
| Colab Pro (A100) | 40GB | 最简单，notebook 直接跑 |
| AutoDL (A10-24G) | 24GB | 性价比高，按量计费 |
| 阿里云/腾讯云 GN7 (T4/A10) | 16-24GB | 企业级，需自建 |

> 7B 模型 fp16 推理约需 14GB 显存，LoRA 训练还需优化器/梯度显存，建议 **>= 16GB** 显存。

## 运行步骤

```bash
# 1. 准备数据（在任何机器，纯 CPU 即可）：见 scripts/prepare_training_data.ps1
# 2. 在云端 GPU 机器执行：
pip install llamafactory
llamafactory-cli train llamafactory_runs/sft_v1/sft_config.yaml
```

或在 Windows 云端用 `scripts/train_cloud.ps1`（含 CUDA 环境校验）。

## 关键参数（与 sft_v1/sft_config.yaml 对应）

- `model_name_or_path: deepseek-ai/deepseek-coder-7b-instruct` — 代码领域 SOTA 开源基座
- `stage: sft` + `finetuning_type: lora` + `lora_rank: 16` — LoRA 在大幅降成本同时保留基座能力，rank=16 是代码任务常用起点
- `cutoff_len: 4096` — SWE 轨迹含长上下文（repo 描述 + patch），4096 覆盖绝大多数样本
- `lora_dropout: 0.05` — 小样本（数百条）下防过拟合的关键正则项
- `per_device_train_batch_size: 1` + `gradient_accumulation_steps: 8` — 小显存下用梯度累积模拟大 batch
- `num_train_epochs: 3` — 小数据集避免过多 epoch 导致过拟合，配合 early stopping 观察 loss 平台

## 过拟合应对（面试常问）

1. **数据层**：失败样本（rejected）仅进 DPO/GRPO，不进 SFT（`min_reward=0.4` 过滤），避免噪声污染 SFT。
2. **训练层**：LoRA 冻结基座 + dropout + 早停；监控 `eval_loss` 与自定义 `tool_call_accuracy` 指标不回升即停。
3. **评估层**：在 SWE-bench_Verified 子集上对比 baseline（基座）与 finetuned 的 resolve rate，量化增益。
