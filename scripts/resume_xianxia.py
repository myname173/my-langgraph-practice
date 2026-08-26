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
import os as _os
LG_BASE = _os.getenv("LG_BASE_URL", "http://localhost:2024")  # 可用环境变量覆盖端口
ASSISTANT_ID = _os.getenv("LG_ASSISTANT_ID")  # 留空则自动从线程元数据获取


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
    client = get_client(url=LG_BASE, api_key="")

    # 未显式指定 assistant_id 时，从线程元数据自动解析
    aid = ASSISTANT_ID
    if not aid:
        try:
            th = await client.threads.get(tid)
            aid = (th.get("metadata") or {}).get("assistant_id")
        except Exception as e:
            print(f"[warn] 读取线程元数据失败: {e}")
        if not aid:
            # 退回到默认 assistant（graph_id=agent）
            asst = await client.assistants.search()
            aid = asst[0]["assistant_id"] if asst else None
    if not aid:
        print("错误: 无法确定 assistant_id，请设置环境变量 LG_ASSISTANT_ID")
        sys.exit(1)

    print(f"[resume] thread_id={tid} action={action} assistant_id={aid}")
    async for chunk in client.runs.stream(
        tid, aid,
        command={"resume": decision},
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
