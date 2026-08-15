# SWE Agent SFT 本地 smoke_test（小模型验证流程）

本配置使用 `Qwen2.5-0.5B-Instruct` + 4bit 量化 + `max_steps: 3`，
可在 **4GB 显存机器（如 GTX 1650 Ti）** 跑通，用于验证「数据→训练」链路非空壳。
**它不产出可用模型**，正式训练请用 `llamafactory_runs/sft_v1`（云端 7B）。

## 启动训练

```bash
# 安装 LlamaFactory
pip install llamafactory

# 运行本地验证（无需 GPU 高端卡）
llamafactory-cli train workspace\llamafactory_runs\smoke_test\sft_config.yaml
```

## 数据集

数据集位于 `workspace\_training_data`，已注册到 `dataset_info.json`。

## 自定义 Metrics

`custom_callbacks.py` 提供 `tool_call_accuracy` 和 `test_pass_rate` 等 SWE 专属指标，
在 LlamaFactory 配置中添加 `--callbacks custom_callbacks.SWEAgentMetricsCallback` 启用。
