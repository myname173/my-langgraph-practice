"""批量驱动 multimedia_agent 跑多个真实故事任务，自动 approve HITL，跑后汇总。

用法：
    # 读取 stories.json（数组，每项 {"name":..., "task":...}）逐条跑
    LANGGRAPH_API_MODE=0 python scripts/run_stories.py --stories stories.json

说明：
    - 每条故事独立 thread，interrupt 自动 {"action":"approve"}，直到 END。
    - 运行指标由 graph.py 内的 run_metrics 埋点写入 data/run_metrics.jsonl。
    - 全部跑完自动调用 summarize_runs 打印真实统计。
"""
import argparse
import json
import os
import sys
import time

# Windows 控制台默认 gbk，pipeline 日志可能含非 gbk 字符（emoji 等），统一用 utf-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 本地脚本模式必须带 checkpointer 才能 HITL resume/get_state。
# 关闭 LANGGRAPH_API_MODE（.env 里是 1，给 LangGraph API 平台用的），否则 graph.py
# 会 compile() 不带 checkpointer，导致 get_state 抛 "No checkpointer set"。
# 必须在 import graph / 任何 load_dotenv() 之前设置成空串：
#   - 空串使 bool("") == False（注意 "0" 也是 True，不能用 "0"）；
#   - 各 tools 在 import 时调用 load_dotenv()，默认 override=False，不会覆盖已存在的空串。
os.environ["LANGGRAPH_API_MODE"] = ""

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.multimedia.graph import multimedia_agent, make_thread_config  # noqa: E402
from langgraph.types import Command  # noqa: E402


def _is_interrupt(snap):
    tasks = getattr(snap, "tasks", None) or ()
    for t in tasks:
        if getattr(t, "interrupts", None):
            return True
    return False


def run_one(task: str, idx: int):
    tid = f"story_{idx}_{int(time.time())}"
    cfg = make_thread_config(tid)
    print(f"\n===== [{idx}] thread={tid} 启动 =====")
    try:
        multimedia_agent.invoke({"task": task}, cfg)
    except Exception as e:  # 单条失败不影响其他
        print(f"  [!] 首调异常：{e}")
        return
    loops = 0
    while True:
        loops += 1
        if loops > 80:
            print("  [!] 超过最大循环次数，中止本条"); break
        snap = multimedia_agent.get_state(cfg)
        if not _is_interrupt(snap):
            if not getattr(snap, "next", None):
                break
        try:
            multimedia_agent.invoke(Command(resume={"action": "approve"}), cfg)
        except Exception as e:
            print(f"  [!] resume 异常：{e}")
            break
        time.sleep(0.3)
    final = multimedia_agent.get_state(cfg)
    values = getattr(final, "values", {}) or {}
    fm = values.get("final_movie_with_audio") or values.get("final_movie_path")
    print(f"  -> final_movie: {fm}")
    print(f"  -> error_log: {values.get('error_log', '')[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", required=True, help="故事 JSON 数组文件")
    args = ap.parse_args()

    with open(args.stories, "r", encoding="utf-8") as f:
        stories = json.load(f)
    print(f"共 {len(stories)} 条故事，开始批量运行...")

    for i, s in enumerate(stories, 1):
        run_one(s.get("task", ""), i)

    # 汇总真实指标
    print("\n===== 汇总真实运行指标 =====")
    try:
        from scripts.summarize_runs import main as _summ
        sys.argv = ["summarize_runs.py", "--top"]
        _summ()
    except Exception as e:
        print(f"  [!] 汇总失败：{e}")


if __name__ == "__main__":
    main()
