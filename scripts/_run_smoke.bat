@echo off
setlocal
set SMOKE=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d c:\Users\13682\Desktop\my-langgraph-practice-main
uv run python scripts/run_xianxia_real.py > data\xianxia_smoke.log 2>&1
