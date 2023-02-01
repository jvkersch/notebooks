import asyncio
import calendar
import os

import aiohttp


def months():
    y = 1997
    m = 4
    while y < 2023:
        yield (y, calendar.month_name[m])
        m += 1
        if m == 13:
            m = 1
            y += 1


async def get_archive(session, y, m):
    url = f"https://stat.ethz.ch/pipermail/r-devel/{y}-{m}.txt.gz"
    fname = f"{y}-{m}.txt.gz"
    if os.path.exists(fname):
        return

    print("Getting", url)
    async with session.get(url) as resp:
        assert resp.status == 200
        with open(fname, "wb") as fd:
            async for chunk in resp.content.iter_any():
                fd.write(chunk)
        print("Wrote", fname)


async def main():
    async with aiohttp.ClientSession() as session:
        for y, m in months():
            await get_archive(session, y, m)


if __name__ == "__main__":
    asyncio.run(main())
