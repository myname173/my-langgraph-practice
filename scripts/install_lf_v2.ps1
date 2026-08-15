# 稳健安装：固化 torch，PYTHONUTF8=1，高重试，分两步（本体 + 依赖）
$root = $PSScriptRoot | Split-Path -Parent
$env:PYTHONUTF8 = "1"
$venvPy = Join-Path $root ".venv313/Scripts/python.exe"
$cfg = Join-Path $root "workspace/llamafactory_runs/smoke_test/sft_config.yaml"
$log = Join-Path $root "logs_lf_v2.txt"

function Log($msg) { Add-Content -Encoding UTF8 $log "[$(Get-Date -Format 'HH:mm:ss')] $msg" }

Log "start: 固化 torch"
# 固化已装 torch，避免被升级到 2.13（更大且易断）
& $venvPy -m pip install "torch==2.6.0+cu124" --extra-index-url https://download.pytorch.org/whl/cu124 --retries 20 *>> $log
Log "torch pin exit: $LASTEXITCODE"

Log "step1: 装 llamafactory 本体 (--no-deps)"
& $venvPy -m pip install --no-deps llamafactory --retries 20 *>> $log
Log "llamafactory no-deps exit: $LASTEXITCODE"

Log "step2: 装 llamafactory 依赖"
& $venvPy -m pip install "numpy" "transformers>=4.45" "peft>=0.13" "accelerate>=0.34" "bitsandbytes>=0.43" "datasets>=3.0" "trl>=0.12" "sentencepiece" "protobuf" "modelscope" "gradio" "swanlab" "deepmerge" "fire" "jinja2" "safetensors" "tokenizers" "einops" "scikit-learn" "scipy" "pandas" "tqdm" "tyro" --retries 20 *>> $log
Log "deps exit: $LASTEXITCODE"

Log "check import"
& $venvPy -c "import llamafactory; print('llamafactory OK', llamafactory.__version__)" *>> $log
Log "check exit: $LASTEXITCODE"

Log "train start"
& $venvPy -m llamafactory.cli train $cfg *>> $log
Log "train exit: $LASTEXITCODE"
