import asyncio
from langgraph_sdk import get_client

NEW = "6c988ec4-b5db-474b-8fe0-a49809f3e1ed"
RUN = "01a034cb-474b-7cb0-afb8-2f76ce19c438"

async def main():
    client = get_client(url="http://localhost:2024")
    try:
        await client.runs.cancel(NEW, RUN)
        print("CANCELLED")
    except Exception as e:
        print("CANCEL_ERR", repr(e))

asyncio.run(main())
