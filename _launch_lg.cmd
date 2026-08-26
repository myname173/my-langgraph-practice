@echo off
cd /d c:\Users\13682\Desktop\my-langgraph-practice-main
del logs_lg_job.txt 2>nul
start "" /min powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\_start_lg_job.ps1
echo DONE_LAUNCH
