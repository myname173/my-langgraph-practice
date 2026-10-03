# SFT 微调工作 事实总结

> 生成时间：2026-09-27
> 原则：**只写仓库内可核验的内容**。每条都标注证据等级，避免任何口头/推测内容混入。
>
> 证据等级说明：
> - 【实证】= 本仓库代码或数据文件可直接验证（已实际读取/统计）
> - 【文档】= 仅见于本仓库 README/docs 的记载，**未找到原始产物**（如训练日志、指标 json、权重文件）
> - 【未验证】= 无任何证据，不应对外声称

---

## 0. 一句话结论

本仓库包含一条**可运行的 SFT 数据→训练→评估管线**：以权威基准 **SWE-bench_Verified** 的 issue 为任务源，构造「成功=gold patch / 失败=退化 patch」的样本，用 **LLaMA-Factory + 4bit LoRA** 微调 `deepseek-coder-7b-instruct`，并以「agent 动作合规率」为指标做前后对比。

**需要明确的边界**（很重要）：
- 训练/评估的**原始产物不在本仓库**（无权重、无训练日志、无指标 json），相关数字仅见于文档记载 → 【文档】
- 训练用的「轨迹」是**构造的 2 步样本**，不是 agent 真实多步执行轨迹 → 【实证，见 §2.3】

---

## 1. 代码资产清单（全部【实证】）

| 文件 | 职责 |
|---|---|
| `src/agent/swe/training/trajectory_logger.py` | 定义 `TrajectoryRecord` / `ToolCallStep`；`extract_trajectory_from_state()` 从 LangGraph 消息重建轨迹；`save_trajectory()` 追加写 JSONL |
| `src/agent/swe/graph.py:466` | `trajectory_export_node`：SWE agent 图的**终止节点**，success 与 failed 两条路由都汇到它，落盘真实运行轨迹 |
| `src/agent/swe/training/reward_computer.py` | `compute_reward()` 四维离线奖励；`split_by_reward()`；`build_dpo_pairs()` |
| `src/agent/swe/training/import_public_data.py` | 三个适配器 `sweagent` / `nemotron` / `swebench`；`convert_swebench_instance()`；`_degrade_patch()` |
| `src/agent/swe/training/data_pipeline.py` | 导出 SFT / DPO / GRPO / Skills 四类数据集 + 生成 `dataset_info.json` |
| `src/agent/swe/training/llamafactory_config.py` | 生成 LLaMA-Factory 的 SFT / GRPO YAML 配置 |
| `src/agent/swe/training/run_training_pipeline.py` | 一键串联「导入 → 导出 → 生成配置」 |
| `scripts/eval_swebench.py` | 评估脚本：`proxy` 模式（本地可跑）+ `live` 模式（云端预留） |
| `scripts/prepare_training_data.ps1` | 纯 CPU 一键准备训练数据 |

---

## 2. 数据：来源、构造方式、实际规模

### 2.1 数据来源【实证】

`import_public_data.py` 明确定义数据源，CLI `--source` 的 choices 为 **`sweagent` / `nemotron` / `swebench`**（默认 `swebench`）。

- `swebench` 走 `_iter_swebench()`，从 **ModelScope** 下载
  `AI-ModelScope/SWE-bench_Verified` 的 test parquet（HuggingFace 为备选）。
- 代码注释原文：**「SWE-bench 真实数据源（仅 SFT 正样本，无 synth 兜底，避免合成假数据污染训练集）」**。
- **更正记录**：我曾提示"确认是否走了 `synth` 合成源"。该提示**不成立**——CLI 无 `synth` 选项，`import_and_export()` 对未知 source 直接 `raise ValueError`。文件中虽残留 `_synth_*` 函数，但**不可达**（死代码）。

### 2.2 样本构造方式【实证】

`convert_swebench_instance()` 把每条 SWE-bench 实例转成 **2 步**记录：

1. `analyze_issue`（`instance_id` + `repo`）
2. `submit_patch`（正样本塞 **gold patch**；负样本塞 **退化 patch**）

任务描述格式：`Repository: <repo>\nInstance: <instance_id>\n\n<problem_statement>`。

**负样本的构造**（`_degrade_patch()`）：保留 diff 元数据行（`diff --git` / `---` / `+++` / `@@` / `index`），**丢弃全部 `+`/`-` 改动行**，得到「看起来像修了、实则未修改」的最弱解。
> 代码内注释明确说明：SWE-bench 本身不含 agent 失败轨迹，这是 SWE 偏好微调的通用构造方式，不是伪装成真实失败轨迹。

### 2.3 关键事实：这些是"构造轨迹"，不是 agent 真实执行轨迹【实证】

- 直接统计 `workspace/_training_data/trajectories.jsonl`：
  - 总记录 **335** 条
  - `status`: **success 235 / failed 100**
  - `test_passed`: **True 235 / False 100**
  - **`tool_call_steps` 长度分布：全部 = 2**（335/335）
  - `summary` 中 `source=swebench`：335/335
- 即：**任务描述来自权威基准（真实 issue），但"轨迹/步骤"是代码构造的 2 步模板**，二者是两个不同层面的真实性，不能混为一谈。

### 2.4 导出数据集实际规模【实证：逐行统计】

| 文件 | 行数 | 说明 |
|---|---|---|
| `trajectories.jsonl` | 335 | 原始轨迹（235 成功 + 100 失败） |
| `sft_success.jsonl` | **235** | ShareGPT 格式，`reward >= 0.4` 过滤后 |
| `dpo_pairs.jsonl` | **500** | `{prompt, chosen, rejected}` |
| `grpo_all.jsonl` | **335** | `{prompt, response, reward}` |
| `skills_sft.jsonl` | **4** | 由 `workspace/skills/*.py` 转成的代码生成正样本 |

四类数据集均注册在 `workspace/_training_data/dataset_info.json`（`swe_agent_sft` / `swe_agent_dpo` / `swe_agent_grpo` / `swe_skills_sft`）。

> 注：`scripts/prepare_training_data.ps1` 的默认参数是 `--max-success 200 --max-failed 100`，与实际文件中的 235/100 不完全一致，说明实际导入时参数被覆盖或多次运行过；此处以**文件实际统计**为准。

### 2.5 SFT 样本结构【实证】

`sft_success.jsonl` 单条为 ShareGPT 多轮：
- `system`：`你是一个顶级的全栈开发工程师 Agent。根据任务描述，使用可用工具逐步完成编码任务。完成后回复 TASK_COMPLETED。`
- `human`：完整任务描述
- `function_call` / `observation` / `gpt`：工具调用与结果、最终 `submit_patch`

---

## 3. 奖励设计【实证】

`reward_computer.py` 的四维离线奖励（**不依赖 LLM/网络**）：

| 维度 | 区间 | 权重/规则 |
|---|---|---|
| `test_reward` | — | success+test_passed → **1.0**；success 但未过测 → 0.4；failed → **-0.3** |
| `efficiency_bonus` | ±0.3 | `1 - iteration_count/max_iterations`，围绕 0.5 归一 |
| `tool_accuracy` | 0~0.2 | 工具调用成功率 |
| `completion_bonus` | 0~0.2 | 步骤完成比例 |
| `total` | clamp 到 **[-1, 1]** | 四项求和后截断 |

按 reward 分流：`>=0.5` 正样本 / `<0.0` 负样本 / 其余中性；SFT 仅取 `reward >= 0.4`。

---

## 4. 训练配置与实际运行

### 4.1 仓库内的 SFT 配置【实证：`llamafactory_runs/sft_v1/sft_config.yaml`】

| 参数 | 值 |
|---|---|
| 基座 | `deepseek-ai/deepseek-coder-7b-instruct` |
| 方式 | `lora`（`lora_target: all`） |
| LoRA | `rank=16` / `alpha=32` / `dropout=0.05` |
| 模板 | `deepseek` |
| `cutoff_len` | 4096 |
| batch | `per_device=2` × `grad_accum=8` |
| 学习率 | `2e-4`，cosine，`warmup_ratio=0.1` |
| epoch | 3 |
| 精度 | `fp16` |
| 数据集 | `swe_agent_sft` |

### 4.2 实际训练运行【文档：`docs/EVAL_AND_RESUME_PLAN.md` + `sft_v1/README.md`】

> 以下数字**未在本仓库找到原始产物**（无 `trainer_log.jsonl`、无 `all_results.json`、无 `model/` 目录），仅见于文档记载。

- 环境：Colab T4（14.5G）
- 为适配显存，实际运行时相对仓库配置有改动：`cutoff_len` 降到 **2048**、`per_device_train_batch_size` 降到 **1**、增加 `quantization_bit: 4` 与 `gradient_checkpointing: true`
- 训练数据：`sft_success.jsonl` **235 条全部使用**
- 时长：约 **80 分钟**
- 最终 **train_loss = 0.80**
- 权重：LoRA 适配器（`adapter_model.safetensors` ~160MB + `adapter_config.json`），文档称已备份至 Drive `MyDrive/sft_v1/model`

**明确声明【实证】**：本仓库内**不存在** `llamafactory_runs/sft_v1/model/` 目录，也没有任何训练日志文件；训练产物在 Colab/Drive 侧，仓库内无法复核。

### 4.3 本地冒烟配置【实证】

`workspace/llamafactory_runs/smoke_test/sft_config.yaml`：`Qwen/Qwen2.5-0.5B-Instruct` + `quantization_bit: 4` + `max_steps: 3`，用于在 4GB 显存机器验证「数据→训练」链路可跑通。文档明确：**该配置不产出可用模型**。

### 4.4 GRPO【实证：配置存在 / 未验证：是否运行】

`llamafactory_runs/grpo_v1/grpo_config.yaml` 存在（LoRA，`batch=1` × `grad_accum=16`，`lr=1e-5`，2 epoch，数据集 `swe_agent_grpo`）。
**没有任何证据表明 GRPO 训练实际执行过** → 不应声称"跑过 GRPO"。

---

## 5. 评估

### 5.1 评估工具【实证】

`scripts/eval_swebench.py`：
- `--mode proxy`：本地 CPU 可跑，输出 `valid_diff_rate`（合法 unified diff 率）、`file_hit_rate`（是否命中 gold 同一文件）、`hunk_overlap_rate`（是否触及 gold 同一修改区域）
- `--mode live`：`run_live_eval()`，文档标注为**云端预留接口**，需 Docker 执行 FAIL_TO_PASS 才能得到官方 `resolved_rate`
- `--offline-report`：直接对比两份已生成报告

### 5.2 实际评估结论【文档：`docs/EVAL_AND_RESUME_PLAN.md`】

- 指标定义：**agent 动作合规率** —— 输出含 `Action:` 且 `ActionInput` 为可解析 JSON
- 样本：同分布 held-out **20 条（seed=42）**
- 结果：**基座 0.0 → 微调后 1.0**（20/20 正确输出 `Action:analyze_issue` + 合法 JSON，并正确提取 `instance_id`/`repo`）
- 结论：微调把「闲聊式 LLM」对齐为「可驱动 SWE agent 的协议遵循模型」

### 5.3 重要：一次被证伪的假结论（评估可信性前提）【文档】

- 早期结果 `rubric composite BASE 0.453 / FT 0.446` 是**假象**，原因是推理时**未套 chat template**，模型只是回声 prompt，BASE/FT 输出几乎相同。
- 修正方式：`tok.apply_chat_template(...)` + 截断只取新生成部分 + `max_new_tokens=768`，才得到真实差异。
- 这是本实验最有价值的工程教训之一。

### 5.4 评估数字的证据等级声明

| 数字 | 证据等级 |
|---|---|
| proxy 指标（valid_diff/file_hit/hunk_overlap） | 【实证：脚本存在】但**未找到实际运行报告** `reports/*.json` |
| `resolved_rate`（官方指标） | 【未验证】live 模式为预留接口，未见执行结果 |
| agent 动作合规率 0.0 → 1.0 | 【文档】仅见于 `EVAL_AND_RESUME_PLAN.md`，**无 `eval_metrics.json` / `eval_predictions.json` 原始文件** |

---

## 6. 可写 / 不可写 边界（对外表述用）

### ✅ 可以写（有依据）

- 基于**权威基准 SWE-bench_Verified** 的 issue 构建训练数据（正样本=gold patch，负样本=退化 patch）
- 4bit + LoRA（rank16/alpha32）微调 `deepseek-coder-7b-instruct`，约 80min / train_loss 0.80（**注明来自实验记录**）
- 设计「agent 动作合规率」指标，基座 0% → 微调后 100%（20 条 held-out）
- 复用 `TrajectoryRecord` 数据契约与四维奖励逻辑（`trajectory_export_node` 确实接在 SWE agent 图上）
- 修正推理链路（chat template）以保证评估可信，并识别出旧 rubric 结论为假象

### ❌ 不能写（会穿）

- 「用 agent 真实执行轨迹训练」——实际训练数据是 2 步构造样本（**除非**改成用 `sweagent` 源的真实轨迹重训）
- 「微调后修 bug 能力提升 / resolved_rate 提升」——评估指标只是**格式合规率**，非 resolve
- 「跑过 GRPO / DPO 训练」——只有配置文件，无运行证据
- 「达到官方 SWE-bench 评测结果」——live 模式未执行
- 把本实验说成「agent 项目的线上数据飞轮」——训练数据来自公开基准导入，与线上执行无数据流

---

## 7. 复现命令（仓库内【实证】）

```bash
# 1) 准备数据（纯 CPU）：从 ModelScope 拉 SWE-bench_Verified → 导出四类数据集
python -m src.agent.swe.training.import_public_data --source swebench --max-success 200 --max-failed 100
python -m src.agent.swe.training.data_pipeline --workspace ./workspace

# 2) 一键管线（导入 + 导出 + 生成 LLaMA-Factory 配置）
python -m src.agent.swe.training.run_training_pipeline --model deepseek-ai/deepseek-coder-7b-instruct

# 3) 本地冒烟（0.5B + 4bit + 3 步，验证链路）
llamafactory-cli train workspace/llamafactory_runs/smoke_test/sft_config.yaml

# 4) 正式训练（需云端 GPU，>=16GB）
llamafactory-cli train llamafactory_runs/sft_v1/sft_config.yaml

# 5) 评估（proxy 本地 / live 需 Docker）
python scripts/eval_swebench.py --mode proxy --model-mode base --max-instances 3
```

---

## 8. 证据索引（便于面试时被追问）

| 结论 | 去哪个文件核对 |
|---|---|
| 轨迹/样本是 2 步构造 | `src/agent/swe/training/import_public_data.py`：`convert_swebench_instance()` |
| 负样本退化方式 | 同上：`_degrade_patch()` |
| 无 synth 兜底 | 同上：CLI choices + 第 552 行注释 |
| 数据规模 235/335/500 | `workspace/_training_data/*.jsonl`（逐行统计） |
| 四维奖励 | `src/agent/swe/training/reward_computer.py` |
| LoRA 超参 | `llamafactory_runs/sft_v1/sft_config.yaml` |
| 实际训练数字（80min/loss0.80） | `docs/EVAL_AND_RESUME_PLAN.md`（无原始产物） |
| 0.0→1.0 指标 | 同上（无原始产物） |
| 评估脚本能力 | `scripts/eval_swebench.py` |
| 轨迹落盘接在 agent 图上 | `src/agent/swe/graph.py:466` |

---

## 9. 更正记录（诚实交代）

1. 我曾建议"确认是否用了 `synth` 合成数据源"——**撤回**。CLI 无该选项，代码明确无反 synth 兜底，数据为真实 SWE-bench_Verified。
2. 我曾把本实验描述为与 agent 评估闭环连成"微调数据飞轮"——**过度连接**。实际关系仅为：训练侧**复用**了 agent 的 `TrajectoryRecord` 数据契约与奖励逻辑；训练数据本身来自公开基准导入，与线上执行无数据流。
