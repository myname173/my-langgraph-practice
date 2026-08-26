#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动连续 approve 续跑：遇到 interrupt 审核门就自动 approve，直到流程结束
或达到最大续跑次数。用于无人值守跑完整个多场景生成流水线。

容错设计:
  - 客户端设置 connect/read 超时，避免无限等待导致卡死
  - 所有网络调用带指数退避重试（应对后端视频生成长任务期间的瞬时超时/断连）
  - runs.stream 中途超时断开时，回到轮询重新查状态，而非退出

用法:
  LG_BASE_URL=http://localhost:2025 python scripts/resume_auto.py <thread_id> [max_resume=200]
"""
import sys, os, asyncio, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LG_BASE = os.getenv("LG_BASE_URL", "http://localhost:2024")
ASSISTANT_ID = os.getenv("LG_ASSISTANT_ID")
TID = sys.argv[1] if len(sys.argv) > 1 else None
MAX_RESUME = int(sys.argv[2]) if len(sys.argv) > 2 else 200

# 超时与重试配置（环境变量可覆盖）
HTTP_TIMEOUT = float(os.getenv("LG_HTTP_TIMEOUT", "120"))      # 单次读超时(秒)
MAX_RETRY = int(os.getenv("LG_MAX_RETRY", "5"))                 # 单次调用最大重试
BUSY_POLL = int(os.getenv("LG_BUSY_POLL", "15"))                # busy 轮询间隔(秒)
STREAM_IDLE = float(os.getenv("LG_STREAM_IDLE", "300"))         # stream 无数据最大空闲(秒)

if not TID:
    meta = ROOT / "workspace" / "_xianxia_thread.json"
    if meta.is_file():
        TID = __import__("json").loads(meta.read_text(encoding="utf-8"))["thread_id"]
if not TID:
    print("错误: 未提供 thread_id")
    sys.exit(1)


def _is_transient(err):
    """判断是否为可重试的瞬时网络错误。"""
    name = type(err).__name__
    return name in ("ReadTimeout", "ConnectTimeout", "ConnectError",
                    "TimeoutException", "ClientDisconnected", "RemoteProtocolError",
                    "ReadError", "WriteError", "ConnectionClosed")


async def _call(fn, *args, **kwargs):
    """带指数退避重试的网络调用包装。"""
    last = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last = e
            if not _is_transient(e):
                raise
            wait = min(2 ** attempt, 30)
            print("  [retry %d/%d] %s: %s (%.0fs 后重试)" % (
                attempt, MAX_RETRY, type(e).__name__, str(e)[:120], wait))
            await asyncio.sleep(wait)
    raise last


async def main():
    from langgraph_sdk import get_client
    # 设置 connect/read 超时，避免长任务期间无限等待
    c = get_client(url=LG_BASE, api_key="", timeout=HTTP_TIMEOUT)
    aid = ASSISTANT_ID
    if not aid:
        th = await _call(c.threads.get, TID)
        aid = (th.get("metadata") or {}).get("assistant_id")
    if not aid:
        asst = await _call(c.assistants.search)
        aid = asst[0]["assistant_id"]

    approved = 0
    for i in range(MAX_RESUME):
        # 线程级 status 才是权威的 interrupted 标志（get_state().status 可能为 None）
        try:
            th = await _call(c.threads.get, TID)
        except Exception as e:  # noqa: BLE001
            print("[%d] 查询线程失败(放弃重试): %s" % (i + 1, e))
            await asyncio.sleep(BUSY_POLL)
            continue
        status = th.get("status")

        if status == "interrupted":
            nxt = (await _call(c.threads.get_state, TID)).get("next") or []
            print("[%d] 中断于 %s -> approve" % (i + 1, nxt))
            # stream 期间若后端长任务导致读超时，捕获后回到轮询（不退出）
            try:
                async for chunk in c.runs.stream(
                    TID, aid,
                    command={"resume": {"action": "approve"}},
                    stream_mode=["updates"],
                    timeout=HTTP_TIMEOUT,
                ):
                    if chunk.event == "error":
                        print("  [error]", chunk.data)
                        break
            except Exception as e:  # noqa: BLE001
                if _is_transient(e):
                    print("  [stream-timeout] %s，回到轮询等待后端..." % type(e).__name__)
                    await asyncio.sleep(BUSY_POLL)
                    continue
                print("  [stream-error] %s" % e)
                return
            approved += 1
            continue
        elif status == "busy":
            # 后端正在自动跑（视频生成等耗时环节），轮询等待
            print("[%d] busy 中，等待后端推进... (已 approve %d 次)" % (i + 1, approved))
            await asyncio.sleep(BUSY_POLL)
            continue
        elif status == "completed":
            print("[done] 流水线已完成 (共 approve %d 次)" % approved)
            return
        else:
            print("[done] status=%s (共 approve %d 次)" % (status, approved))
            return
    print("[warn] 达到最大续跑次数 %d，仍未结束 (已 approve %d 次)" % (MAX_RESUME, approved))


if __name__ == "__main__":
    asyncio.run(main())

