@echo off
set REUSE_THREAD=01a006d5-1e51-7420-8ca0-b38896bf18d6
cd /d C:\Users\13682\Desktop\my-langgraph-practice-main
start "" /b python scripts/_launch_xianxia_rerun.py > workspace\_reuse_out.txt 2> workspace\_reuse_err.txt
echo LAUNCHED
