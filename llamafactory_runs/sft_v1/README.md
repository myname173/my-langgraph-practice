# SWE Agent SFT Training

基于 **LLaMA-Factory** 对代码大模型（deepseek-coder-7b-instruct）做 LoRA 监督微调，
训练数据为真实 **SWE-bench_Verified** 任务（成功正样本 + 退化 patch 失败负样本）。

## 数据来源与管线

```
HuggingFace/ModelScope SWE-bench_Verified
        │  import_public_data.py（成功=gold patch / 失败=退化 patch）
        ▼
trajectories.jsonl（235 成功 + 100 失败 = 335 条）
        │  data_pipeline.py + reward_computer.py（奖励标注 + 分流）
        ▼
sft_success.jsonl(235)  dpo_pairs.jsonl(500)  grpo_all.jsonl(335)  skills_sft.jsonl(4)
        │  dataset_info.json（四类数据集注册）
        ▼
LLaMA-Factory 训练（sft_v1 配置 / 云端 7B 正式，smoke_test 本地小模型验证）
```

- **导入器数据源**：优先 ModelScope `AI-ModelScope/SWE-bench_Verified`（国内可达），HuggingFace 备选。`import_public_data.py` 的 `--max-success/--max-failed` 控制规模。
- **失败样本（rejected）**：SWE-bench 本身是「问题+gold patch」评测集，不含真实失败轨迹。DPO 负样本采用 gold patch 的**退化版本**（剔除 hunk 正文、仅留 diff 骨架），对应「模型未真正修复」的劣质解——这是 SWE 偏好微调的通用构造，非伪装真实轨迹。
- **Skills 数据**：`workspace/skills/*.py` 为示例技能脚本（人工验证的高质量代码），`build_skills_sft_data` 将其转为代码生成 SFT 样本。生产环境应接内部 skills 库。

## 启动训练

```bash
# 安装 LlamaFactory
pip install llamafactory

# 本地小模型验证流程（0.5B + 4bit，1650 Ti 4GB 可跑，证明链路非空壳）
llamafactory-cli train workspace\llamafactory_runs\smoke_test\sft_config.yaml

# 云端 7B 正式训练（A10/A100，见 cloud/README.md）
llamafactory-cli train llamafactory_runs\sft_v1\sft_config.yaml
```

一键准备数据（纯 CPU）：`scripts\prepare_training_data.ps1`

## 数据集

位于 `workspace\_training_data`，已注册到 `dataset_info.json`（swe_agent_sft / swe_agent_dpo / swe_agent_grpo / swe_skills_sft）。

## 自定义 Metrics

`custom_callbacks.py` 提供 `tool_call_accuracy` 和 `test_pass_rate` 等 SWE 专属指标，
配置中 `--callbacks custom_callbacks.SWEAgentMetricsCallback` 启用。

## 关键超参理由（面试可讲）

| 参数 | 取值 | 理由 |
|------|------|------|
| 基座模型 | deepseek-coder-7b-instruct | 代码领域 SOTA 开源模型，7B 在消费级/云端性价比最佳 |
| finetuning_type | lora | 冻结基座、仅训低秩适配，显存/算力成本降一个数量级，避免灾难性遗忘 |
| lora_rank | 16 | 代码任务常用起点；rank 过大易过拟合小数据，过小容量不足 |
| cutoff_len | 4096 | SWE 轨迹含 repo 描述+patch，4096 覆盖绝大多数样本且控显存 |
| lora_dropout | 0.05 | 小样本（数百条）防过拟合的关键正则项 |
| num_train_epochs | 3 | 小数据集避免多 epoch 过拟合，配合 eval_loss 早停 |

## 面试题自问自答（工程/应用岗）

**Q1: 为什么用 LoRA 而不是全参微调？**
显存与算力降一个数量级；冻结基座避免小数据灾难性遗忘；可多任务挂多个 LoRA 适配器热插拔。

**Q2: 小样本（数百条）怎么防过拟合？**
三层：数据层失败样本不进 SFT（min_reward 过滤）；训练层 LoRA+dropout+早停监控 eval_loss；评估层在 SWE-bench 子集对比 baseline vs finetuned 的 resolve rate。

**Q3: SFT/DPO/GRPO 区别与适用？**
SFT 学「正确解法分布」；DPO 用 chosen/rejected 对直接做偏好对齐（无需独立 reward model）；GRPO 用组内相对奖励做 RL 微调，适合可验证奖励（如单测通过）。本项目三类数据均由同一批轨迹经 reward_computer 分流生成。

**Q4: DPO 的 reward/负样本哪来的？**
正样本=高奖励成功轨迹（gold patch），负样本=同一实例的退化 patch（未真正修复）。reward 由 `compute_reward` 计算：成功+1.0 / 失败-0.3，再归一化。

**Q5: 为什么本地不跑 7B 训练？**
GTX 1650 Ti 仅 4GB 显存，7B fp16 推理需约 14GB，必 OOM。工程上把训练放到云端（Colab A10/A100），本地仅做数据管线（CPU）与小模型 smoke_test 验证流程——这本身也是可复现工程能力的体现。

## 评估闭环（scripts/eval_swebench.py）

训练后必须量化「微调是否真的让 agent 更会修 bug」。脚本在 SWE-bench_Verified 子集上
对比 **baseline（基座）** 与 **finetuned（LoRA）** 两种模型的 patch 质量。

两种模式：
- **proxy 模式（本地 CPU 可跑，无需 GPU/网络 repo）**：只用 SWE-bench 的静态信息
  （problem_statement + gold patch）评估生成 patch 的结构相似度，输出三项代理指标：
  - `valid_diff_rate`：生成的是否为合法 unified diff
  - `file_hit_rate`：是否改动了 gold patch 涉及的同一文件
  - `hunk_overlap_rate`：是否触及 gold patch 的同一修改区域
  这是低成本验证「训练是否让 agent 更精准定位文件/函数」的代理指标，**不运行真实测试**。
- **live 模式（云端 / Docker 可用）**：真实 checkout repo → apply patch → 跑 FAIL_TO_PASS，
  输出官方对齐的 `resolved_rate`。脚本已预留 `run_live_eval()` 接口，需云端填充测试执行。

用法：
```bash
# 本地冒烟（3 条样本，proxy，不调 agent 仅验证数据管线）
python scripts/eval_swebench.py --mode proxy --model-mode base --max-instances 3 --no-agent

# 云端：生成 base / finetuned 两份报告（需 GPU + 设置 SWE_FINETUNED_LORA_PATH）
python scripts/eval_swebench.py --mode proxy --model-mode base --max-instances 50
python scripts/eval_swebench.py --mode proxy --model-mode finetuned --max-instances 50

# 本地直接对比两份报告，量化增益（无需重跑推理）
python scripts/eval_swebench.py --offline-report reports/base_proxy.json reports/finetuned_proxy.json
```

输出 `reports/*.json` + Markdown 摘要，可回填本 README 形成「训练 → 评估 → 指标」完整叙事。

> 诚实说明：proxy 模式的 `file_hit/hunk_overlap` 是**定位精度代理指标**，不能替代真实
> `resolved_rate`。面试讲评测时，应明确区分两者，并以云端 live 模式（或官方 SWE-bench
> harness）产出的 `resolved_rate` 作为最终结论。

---

## 实测评估结果（2026-08-25，已验证可复现）

### 训练产出
- 基座：deepseek-ai/deepseek-coder-7b-instruct
- 方法：4bit + LoRA（rank16 / alpha32 / dropout0.05），cutoff_len=2048，3 epoch
- 数据：235 条 SWE-bench issue 成功轨迹（ShareGPT）
- 资源：Colab T4（14.5G），约 80 min，最终 **train_loss = 0.80**
- 权重：LoRA 适配器 `llamafactory_runs/sft_v1/model/`（已备份至 Drive `MyDrive/sft_v1/model`）

### 评估方法与结论
**指标：agent 动作合规率**（输出含 `Action:` 且 `ActionInput` 为可解析 JSON）。
在同分布 held-out 20 条（seed=42）上，BASE vs FT 对比：

| 模型 | agent 动作合规率 | 典型输出 |
|------|----------------|---------|
| BASE（基座） | **0.0**（20/20 均为自然语言应答） | `It seems like you're looking to enhance the bulk_update()...` |
| FT（微调后） | **1.0**（20/20 正确输出结构化动作） | `Action:analyze_issue ActionInput:{"instance_id":"django__django-14559","repo":"django/django"}` |

**结论**：微调将模型从「闲聊式 LLM」对齐为「可驱动 SWE agent 的协议遵循模型」——
基座完全不会按 agent 协议输出工具调用，微调后 100% 正确输出 `Action+JSON` 并提取 `instance_id/repo`。

### 关键踩坑（评估可信性前提）
1. **推理必须套 chat template**：直接 `tok(prompt)` 会让模型回声输入，BASE/FT 输出一致，
   导致早期 `rubric composite`（BASE 0.453 / FT 0.446）为**假象**，不可作结论。
   修正为 `tok.apply_chat_template(...)` + 截断只取新生成部分 + `max_new_tokens=768` 后得到真实差异。
2. **权重必须备份到 Drive**：曾因仅备份测试集导致权重丢失、被迫重训；
   本次已备份 `MyDrive/sft_v1/model`（`adapter_model.safetensors` ~160MB + `adapter_config.json`）。
3. **T4 显存约束**：重训需 `quantization_bit: 4` + batch=1 + 串行加载释放，否则 OOM。

### 复现评估命令（Colab）
见 `docs/EVAL_AND_RESUME_PLAN.md` 第 3 节（推理 C + 评估 D），核心为：
加载 BASE 与 FT 适配器 → 各生成 20 条 → `is_agent_action()` 统计合规率。

