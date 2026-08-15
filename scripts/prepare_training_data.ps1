# 本地一键准备训练数据（纯 CPU，无需 GPU）
# 步骤：备份现有数据 -> 从 ModelScope 拉取 SWE-bench_Verified 成功/失败样本 -> 跑 data_pipeline 生成 SFT/DPO/GRPO/Skills
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot | Split-Path -Parent
$py  = "python"   # 直接用系统 python；如需 venv 改为 .venv313/Scripts/python.exe

Write-Host "==> [1/3] 备份现有训练数据 ..."
$bak = Join-Path $root "workspace/_training_data_bak_auto"
if (Test-Path (Join-Path $root "workspace/_training_data")) {
    if (-not (Test-Path $bak)) {
        Copy-Item -Recurse -Force (Join-Path $root "workspace/_training_data") $bak
        Write-Host "    已备份到 $bak"
    } else {
        Write-Host "    备份已存在，跳过: $bak"
    }
}

Write-Host "==> [2/3] 从 ModelScope 拉取 SWE-bench_Verified 数据 (默认 200 成功 / 100 失败) ..."
& $py -m src.agent.swe.training.import_public_data --source swebench --max-success 200 --max-failed 100
if ($LASTEXITCODE -ne 0) {
    Write-Host "    数据导入失败（请检查网络：需可访问 modelscope.cn），exit=$LASTEXITCODE"
    exit 1
}

Write-Host "==> [3/3] 运行 data_pipeline 生成 SFT/DPO/GRPO/Skills ..."
& $py -m src.agent.swe.training.data_pipeline --workspace ./workspace
if ($LASTEXITCODE -ne 0) {
    Write-Host "    pipeline 失败，exit=$LASTEXITCODE"
    exit 1
}

Write-Host "==> 完成。训练数据已就绪于 workspace/_training_data/"
Write-Host "    下一步：本地小模型验证执行 scripts/run_lf_and_train.ps1；7B 正式训练见 llamafactory_runs/cloud/README.md"
