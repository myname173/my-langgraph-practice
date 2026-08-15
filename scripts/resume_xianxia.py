#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
审核中断后续跑（复刻前端 InterruptApprovalCard 的 approve 行为）。

前端路径: useAgentStream.resumeRun(decision) -> client.runs.stream(tid, ASSISTANT_ID, {command:{resume:decision}})
本脚本: 读取 _xianxia_thread.json 的 thread_id，用 Command(resume={"action":"approve"}) 续跑。

用法:
  python scripts/resume_xianxia.py [thread_id] [approve|reject]
  不传 thread_id 则从 workspace/_xianxia_thread.json 读取（由 run_xianxia_full.py 写入）
  不传 action 默认 approve
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LG_BASE = "http://localhost:2024"
ASSISTANT_ID = "fe096781-5601-53d2-b2f6-0d3403f7e9ca"


async def main():
    tid = None
    action = "approve"
    args = sys.argv[1:]
    if args:
        if args[0] in ("approve", "reject", "rewrite", "edit_prompt"):
            action = args[0]
        else:
            tid = args[0]
            if len(args) > 1:
                action = args[1]

    if not tid:
        meta_path = ROOT / "workspace" / "_xianxia_thread.json"
        if not meta_path.is_file():
            print("错误: 未提供 thread_id 且未找到 workspace/_xianxia_thread.json")
            sys.exit(1)
        tid = json.loads(meta_path.read_text(encoding="utf-8"))["thread_id"]

    decision = {"action": action}
    if action == "reject":
        decision["reason"] = "人工否决"

    from langgraph_sdk import get_client
    from langchain_core.messages import Command
    client = get_client(url=LG_BASE, api_key="")

    print(f"[resume] thread_id={tid} action={action}")
    async for chunk in client.runs.stream(
        tid, ASSISTANT_ID,
        command=Command(resume=decision),
        stream_mode=["values", "updates"],
    ):
        if chunk.event == "updates":
            for node, out in (chunk.data or {}).items():
                if node.startswith("__"):
                    continue
                if isinstance(out, dict) and "__interrupt__" in out:
                    print(f"[interrupt] 节点 {node} 再次触发审核中断（需再次 resume）")
                else:
                    print(f"  -> 完成节点: {node}")
        elif chunk.event == "values":
            vals = chunk.data or {}
            node = vals.get("__node__") if isinstance(vals, dict) else None
            if node:
                print(f"  -> 节点: {node}")
        elif chunk.event == "error":
            print(f"[error] {chunk.data}")
            break
    print("[resume] 流结束。若再次中断，请再次运行本脚本。")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
