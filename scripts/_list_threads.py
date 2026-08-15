# -*- coding: utf-8 -*-
import asyncio
from langgraph_sdk import get_client

TARGET = "01a0020f-e136-7c62-848a-c667a7cf8962"

async def main():
    client = get_client(url="http://localhost:2024")
    threads = await client.threads.search(limit=50, metadata=None)
    print(f"=== 当前服务内线程总数: {len(threads)} ===")
    for t in threads:
        tid = t.get("thread_id")
        flag = " <-- 目标" if tid.startswith("01a0020f") else ""
        print(f"- {tid} | status={t.get('status')} | created={t.get('created_at')}{flag}")
    # 直接尝试 get 目标线程
    try:
        t = await client.threads.get(TARGET)
        print("\n目标线程存在:", t.get("status"))
    except Exception as e:
        print(f"\n目标线程 {TARGET} 不存在:", e)

asyncio.run(main())
