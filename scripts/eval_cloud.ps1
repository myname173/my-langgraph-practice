# 云端评估脚本（在 GPU 机器/Colab/AutoDL 上运行，本地 1650 Ti 跑不了 7B 推理）
# 用法：在云端机器执行  powershell -File scripts/eval_cloud.ps1
# 作用：生成 baseline 与 finetuned 两份 eval 报告，并离线对比量化增益。
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot | Split-Path -Parent
$py  = "python"

# 环境校验
Write-Host "==> 环境校验 ..."
& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print('GPU:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    Write-Host "    本机无 CUDA GPU，无法做 7B 推理评估。请移步云端（Colab A10/A100 或 AutoDL）。"
    exit 1
}

# 安装依赖
Write-Host "==> 安装依赖 (pandas/pyarrow) ..."
& $py -m pip install pandas pyarrow

# 1) baseline（基座模型）
Write-Host "==> 评估 baseline (base) ..."
& $py scripts/eval_swebench.py --mode proxy --model-mode base --max-instances 50
if ($LASTEXITCODE -ne 0) { Write-Host "base 评估失败"; exit 1 }

# 2) finetuned（需先由 train_cloud.ps1 产出 LoRA 权重，并设置 SWE_FINETUNED_LORA_PATH）
$lora = $env:SWE_FINETUNED_LORA_PATH
if (-not $lora) {
    Write-Host "!! 未设置 SWE_FINETUNED_LORA_PATH，跳过 finetuned 评估。"
    Write-Host "   请先运行 train_cloud.ps1，然后："
    Write-Host "   `$env:SWE_FINETUNED_LORA_PATH='<lora输出目录>'; powershell -File scripts/eval_cloud.ps1"
    exit 0
}
Write-Host "==> 评估 finetuned (LoRA: $lora) ..."
& $py scripts/eval_swebench.py --mode proxy --model-mode finetuned --max-instances 50
if ($LASTEXITCODE -ne 0) { Write-Host "finetuned 评估失败"; exit 1 }

# 3) 离线对比
Write-Host "==> 对比 base vs finetuned 增益 ..."
& $py scripts/eval_swebench.py --offline-report reports/base_proxy.json reports/finetuned_proxy.json
