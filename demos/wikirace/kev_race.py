"""The tournament.py races with Kev: one 255-option choice question per page, no knockout.

Kev (https://github.com/jaredpalmer/kev) serves TypeSafe's /v1/systemone API, so JevLM
drives it unchanged. Its pointer head reads up to 255 options at once, so a page takes
one request per 255 links, then one more among the chunk winners.

On an RTX 4070 Laptop (8 GB), Kev-0.8B in bf16 reached all 5 targets in 17 hops, at
0.39 s of picking per hop. tournament.py's SemIf picker, on Qwen3-VL-4B, took 19 hops at
4.74 s per hop.

Usage: python kev_race.py   (needs `python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009`)
"""

import asyncio
import sys

import dspy
from pick255 import one_question
from tournament import RACES, race
from wikirace import links

from llm_schema_lite.dspy_integration import JevAdapter, JevLM


async def main(model: str) -> None:
    dspy.configure(
        adapter=JevAdapter(),
        lm=JevLM("kev-latest", url="http://127.0.0.1:8009/v1/systemone", cache=False),
    )
    reached = total_hops = total_secs = 0.0
    for start, target in RACES:
        target, _ = links(target)
        path, secs = await race(start, target, one_question)
        done, hops = path[-1] == target, len(path) - 1
        print(
            f"{model} {start} -> {target}: {'reached' if done else 'gave up'} in {hops} hops, "
            f"{secs:.1f}s picking ({secs / hops:.2f}s/hop): {' -> '.join(path)}",
            flush=True,
        )
        reached, total_hops, total_secs = reached + done, total_hops + hops, total_secs + secs
    print(
        f"{model} reached {reached:.0f}/{len(RACES)}, {total_hops:.0f} hops, "
        f"{total_secs:.0f}s picking, {total_secs / total_hops:.2f}s per hop"
    )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "kev"))
