import asyncio

from sqlalchemy import text

from algobet.infrastructure.database import async_session_scope


async def show():
    async with async_session_scope() as session:
        r = await session.execute(
            text("""
                  SELECT id, algorithm, created_at FROM model_versions
                  ORDER BY created_at DESC LIMIT 5
              """)
        )
        for row in r.mappings():
            print(row["id"], row["algorithm"], str(row["created_at"])[:19])


asyncio.run(show())
