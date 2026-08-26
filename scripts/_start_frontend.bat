@echo off
cd /d c:\Users\13682\Desktop\my-langgraph-practice-main\frontend
set VITE_LANGGRAPH_PROXY_TARGET=http://127.0.0.1:2026
set VITE_MEDIA_PROXY_TARGET=http://127.0.0.1:8900
call "C:\Program Files\nodejs\npm.cmd" run dev
