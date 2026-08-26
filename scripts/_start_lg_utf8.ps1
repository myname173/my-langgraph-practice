$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
# 用全新的 checkpoint 库，绕开旧库中残留的"幽灵 run"状态（避免 An internal error occurred）
$env:MULTIMEDIA_CHECKPOINT_DB = "./data/multimedia_checkpoints_clean.sqlite"
$ROOT = "c:/Users/13682/Desktop/my-langgraph-practice-main"
Set-Location $ROOT
Write-Host "PYTHONUTF8=$env:PYTHONUTF8 PYTHONIOENCODING=$env:PYTHONIOENCODING"
Write-Host "MULTIMEDIA_CHECKPOINT_DB=$env:MULTIMEDIA_CHECKPOINT_DB"
# 前台启动 langgraph（此脚本由 Start-Process 调用，输出重定向到日志）
uv run langgraph dev --port 2024 --no-reload
