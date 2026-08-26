@echo off
cd /d c:\Users\13682\Desktop\my-langgraph-practice-main
set PYTHONPATH=c:\Users\13682\Desktop\my-langgraph-practice-main\.venv\Lib\site-packages
set PYTHONDONTWRITEBYTECODE=1
C:\Users\13682\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe _probe_strength.py > logs_probe.txt 2>&1
echo DONE
