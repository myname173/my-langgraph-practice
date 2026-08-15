# 修复版：装 llamafactory（忽略 pip 警告）+ 跑小模型 SFT
# torch 已装好，这里跳过 torch
$root = $PSScriptRoot | Split-Path -Parent
$venvPy = Join-Path $root ".venv313/Scripts/python.exe"
$cfg = Join-Path $root "workspace/llamafactory_runs/smoke_test/sft_config.yaml"

# 1) 装 llamafactory（不开启 ErrorActionPreference Stop，避免 pip 警告中断）
Write-Host "==> 安装 llamafactory ..."
& $venvPy -m pip install llamafactory *> "logs_lf_install.txt"
$code = $LASTEXITCODE
Write-Host "pip llamafactory exit code: $code"

# 2) 校验
& $venvPy -c "import llamafactory; print('llamafactory OK')" *> "logs_lf_check.txt"
if ($LASTEXITCODE -ne 0) {
    Write-Host "llamafactory 导入失败，详见 logs_lf_install.txt"
    exit 1
}

# 3) 跑训练
Write-Host "==> 启动小模型 SFT ..."
& $venvPy -m llamafactory.cli train $cfg *> "logs_sft_train.txt"
Write-Host "==> SFT 训练结束，exit code: $LASTEXITCODE"
