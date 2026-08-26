import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
p = r"c:/Users/13682/Desktop/my-langgraph-practice-main/_cleanup2.py"
if os.path.exists(p):
    os.remove(p)
    print("removed" if not os.path.exists(p) else "fail")
else:
    print("not exist")
