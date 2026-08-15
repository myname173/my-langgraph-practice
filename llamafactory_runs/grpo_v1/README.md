# SWE Agent GRPO Training

## 启动训练

```bash
# 安装 LlamaFactory
pip install llamafactory

# 运行训练
llamafactory-cli train llamafactory_runs\grpo_v1\grpo_config.yaml
```

## 数据集

数据集位于 `workspace\_training_data`，已注册到 `dataset_info.json`。

## 自定义 Metrics

`custom_callbacks.py` 提供 `tool_call_accuracy` 和 `test_pass_rate` 等 SWE 专属指标，
在 LlamaFactory 配置中添加 `--callbacks custom_callbacks.SWEAgentMetricsCallback` 启用。
