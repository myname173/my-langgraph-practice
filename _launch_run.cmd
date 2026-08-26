@echo off
cd /d c:\Users\13682\Desktop\my-langgraph-practice-main
del logs_run_real.txt 2>nul
start "" /min powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\_run_real.ps1
echo DONE_RUN_LAUNCH
