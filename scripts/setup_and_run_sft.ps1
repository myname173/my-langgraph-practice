# 一键：装依赖 + 跑通小模型 SFT（Qwen2.5-0.5B + 4bit QLoRA, max_steps=3）
# 用法：在 PowerShell 中 cd 到项目根目录，执行 .\scripts\setup_and_run_sft.ps1
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot | Split-Path -Parent
$venvPy = Join-Path $root ".venv313/Scripts/python.exe"
$cli = Join-Path $root ".venv313/Scripts/llamafactory-cli.exe"
$cfg = Join-Path $root "workspace/llamafactory_runs/smoke_test/sft_config.yaml"

if (-not (Test-Path $venvPy)) {
    Write-Host "找不到 .venv313，请先运行: py -3.13 -m venv .venv313" -ForegroundColor Red
    exit 1
}

# 1) torch（若缺失）
$torchOk = & $venvPy -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "==> 安装 torch (cu124) ..." -ForegroundColor Cyan
    & $venvPy -m pip install torch --index-url https://download.pytorch.org/whl/cu124
}

# 2) llamafactory（若缺失）
$lfOk = & $venvPy -c "import llamafactory" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "==> 安装 llamafactory ..." -ForegroundColor Cyan
    & $venvPy -m pip install llamafactory
}

# 3) 跑训练（用 python -m 入口，避免依赖 cli 可执行文件）
Write-Host "==> 启动小模型 SFT ..." -ForegroundColor Green
& $venvPy -m llamafactory.cli train $cfg
