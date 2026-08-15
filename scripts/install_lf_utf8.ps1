# 用 PYTHONUTF8=1 重装 llamafactory（修复 Windows GBK 下 numpy meson 编译崩溃）
$root = $PSScriptRoot | Split-Path -Parent
$env:PYTHONUTF8 = "1"
$venvPy = Join-Path $root ".venv313/Scripts/python.exe"
$cfg = Join-Path $root "workspace/llamafactory_runs/smoke_test/sft_config.yaml"

# 升级构建工具
& $venvPy -m pip install --upgrade pip wheel setuptools *> "logs_lf_install2.txt"
Add-Content -Encoding UTF8 "logs_lf_install2.txt" "upgrade exit: $LASTEXITCODE"

# 装 llamafactory（PYTHONUTF8=1 已继承）
& $venvPy -m pip install llamafactory *> "logs_lf_install2.txt"
Add-Content -Encoding UTF8 "logs_lf_install2.txt" "llamafactory install exit: $LASTEXITCODE"

# 校验
& $venvPy -c "import llamafactory; print('llamafactory OK', llamafactory.__version__)" *> "logs_lf_check2.txt"
Add-Content -Encoding UTF8 "logs_lf_check2.txt" "check exit: $LASTEXITCODE"

# 跑训练
& $venvPy -m llamafactory.cli train $cfg *> "logs_sft_train2.txt"
Add-Content -Encoding UTF8 "logs_sft_train2.txt" "train exit: $LASTEXITCODE"
