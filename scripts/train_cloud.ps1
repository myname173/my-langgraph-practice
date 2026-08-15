# 云端 7B 正式训练启动脚本（在 GPU 机器/Colab/AutoDL 上运行，本机 1650 Ti 无法跑）
# 用法：在云端机器执行  powershell -File scripts/train_cloud.ps1  （Linux 用 bash 等效命令见同目录 README）
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot | Split-Path -Parent
$py  = "python"

# 环境校验：必须有可用 CUDA GPU 且显存足够（7B + LoRA 建议 >= 16GB）
Write-Host "==> 环境校验 ..."
& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print('GPU:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    Write-Host "    本机无 CUDA GPU，无法进行 7B 训练。请移步云端（Colab A10/A100 或 AutoDL），参考 llamafactory_runs/cloud/README.md"
    exit 1
}

Write-Host "==> 安装 LLaMA-Factory ..."
& $py -m pip install llamafactory
& $py -c "import llamafactory; print('llamafactory OK')"

Write-Host "==> 启动 SFT 训练 (deepseek-coder-7b-instruct + LoRA) ..."
& $py -m llamafactory.cli train (Join-Path $root "llamafactory_runs/sft_v1/sft_config.yaml")

Write-Host "==> 训练结束，exit=$LASTEXITCODE"
