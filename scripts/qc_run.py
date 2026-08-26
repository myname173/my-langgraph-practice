"""成片质量巡检 CLI 入口。

复用项目既有脚本风格：
  - 用 ``langgraph_sdk`` 异步拉取 thread state（对齐 _poll_progress.py）。
  - 用项目内 qc 模块做只读巡检（对齐 qc.py）。
  - 支持环境变量 ``TID`` 或命令行 ``argv[1]`` 传入 thread_id。
  - 落盘 JSON 到 ``output/_qc_<thread_id>.json``，并打印分级日志。

用法：
    python scripts/qc_run.py                 # 用环境变量 TID
    python scripts/qc_run.py <thread_id>     # 用命令行参数
    TID=<thread_id> python scripts/qc_run.py
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# 允许从 scripts/ 直接运行，复用 src 下的 qc 模块
LG_BASE = os.getenv("LG_BASE", "http://localhost:2024")
TID = os.getenv("TID", "01a006b5-d87f-7471-b329-45ac7b2aa27e")


async def _fetch_state(thread_id: str) -> Dict[str, Any]:
    """异步拉取指定 thread 的完整状态。

    Args:
        thread_id: 待拉取的 thread 标识。

    Returns:
        thread state 的 values 字典；失败时返回空字典（QC 降级为产物巡检）。
    """
    try:
        from langgraph_sdk import get_client

        client = get_client(url=LG_BASE, api_key="")
        state = await client.threads.get_state(thread_id)
        values: Any = state.get("values", {})
        return values or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 拉取 thread state 失败（降级为纯产物巡检）: {exc}")
        return {}


async def main() -> int:
    """主流程：拉 state → 巡检 → 落盘 → 打印报告。

    Returns:
        进程退出码（PASS=0，FAIL=1）。
    """
    thread_id = sys.argv[1] if len(sys.argv) > 1 else TID
    if not thread_id or thread_id.startswith("01a006b5"):
        print("[WARN] 未指定有效 TID，使用默认占位（仅做产物巡检）")

    from agent.multimedia.tools.qc import (
        check_thread,
        render_report,
        save_report,
    )

    state = await _fetch_state(thread_id)
    report = check_thread(thread_id, state=state if state else None)

    text = render_report(report)
    print(text)

    out_path = save_report(report)
    print(f"[QC] 报告已落盘: {out_path}")

    return 0 if report["summary"] == "PASS" else 1


if __name__ == "__main__":
    try:
        code = asyncio.run(main())
        sys.exit(code)
    except KeyboardInterrupt:
        sys.exit(130)
