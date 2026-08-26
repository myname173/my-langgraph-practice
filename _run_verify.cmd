@echo off
cd /d c:\Users\13682\Desktop\my-langgraph-practice-main
set PYTHONPATH=c:\Users\13682\Desktop\my-langgraph-practice-main\.venv\Lib\site-packages
set PYTHONDONTWRITEBYTECODE=1
C:\Users\13682\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe _verify_jimeng.py > logs_verify.txt 2>&1
echo DONE_VERIFY
