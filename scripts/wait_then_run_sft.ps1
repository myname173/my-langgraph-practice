# 守卫脚本：等当前 torch 下载进程退出 -> 装 llamafactory -> 跑小模型 SFT
# 全部后台执行，日志落盘到 logs_sft_*.txt
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot | Split-Path -Parent
$venvPy = Join-Path $root ".venv313/Scripts/python.exe"
$cfg = Join-Path $root "workspace/llamafactory_runs/smoke_test/sft_config.yaml"
$torchPidFile = Join-Path $root "logs_torch_pid.txt"

# 1) 等 torch 进程退出
if (Test-Path $torchPidFile) {
    $pidNum = [int](Get-Content -Encoding ASCII $torchPidFile)
    $proc = Get-Process -Id $pidNum -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "等待 torch 下载进程 (pid=$pidNum) 退出 ..."
        $proc.WaitForExit()
        Write-Host "torch 进程已退出，等待 5s 让安装收尾 ..."
        Start-Sleep -Seconds 5
    }
}

# 2) 装 llamafactory
Write-Host "==> 安装 llamafactory ..."
& $venvPy -m pip install llamafactory 2>&1 | Out-Host

# 3) 跑训练
Write-Host "==> 启动小模型 SFT ..."
& $venvPy -m llamafactory.cli train $cfg 2>&1 | Out-Host
Write-Host "==> SFT 流程结束"
