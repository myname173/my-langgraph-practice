# 使用项目专用 .venv313 (Python 3.13) 跑通小模型 SFT（Qwen2.5-0.5B + 4bit QLoRA）
# 目标：在 GTX 1650 Ti (4GB) 上验证整条 LLaMA-Factory SFT 管线可跑通
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot | Split-Path -Parent
$venvPy = Join-Path $root ".venv313/Scripts/python.exe"
$cfg = Join-Path $root "workspace/llamafactory_runs/smoke_test/sft_config.yaml"

if (-not (Test-Path $venvPy)) {
    Write-Host "找不到 .venv313，请先运行: py -3.13 -m venv .venv313" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $cfg)) {
    Write-Host "配置不存在: $cfg" -ForegroundColor Red
    exit 1
}

Write-Host "==> 启动小模型 SFT (max_steps=3, Qwen2.5-0.5B, 4bit QLoRA)" -ForegroundColor Cyan
& $venvPy -m llamafactory.cli train $cfg
