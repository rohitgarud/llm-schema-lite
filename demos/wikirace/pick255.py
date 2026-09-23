"""One 255-option SemIf question per page vs one yes/no question per link.

For each (page, target) pair, both pickers choose the next link; a pick scores when the
chosen article links straight to the target. Pages over 255 links run in chunks of 255,
then a final question among the chunk winners.

Usage: python pick255.py   (needs llama-server on :8089 with >= 8k context per slot)
"""

import asyncio
import time
from typing import Literal

import dspy
from wikirace import Hop, links

from llm_schema_lite.dspy_integration import JevAdapter, SemIfLM

PAIRS = [
    ("Rubber duck", "Albert Einstein"),
    ("Pizza", "World War II"),
    ("Banana", "Moon"),
    ("Chess", "Ancient Egypt"),
    ("Guitar", "Photosynthesis"),
    ("Volcano", "William Shakespeare"),
    ("Coffee", "Isaac Newton"),
    ("Basketball", "Roman Empire"),
    ("Honey bee", "Computer"),
    ("Bicycle", "Jupiter"),
]


def pick_sig(options: list[str]) -> type[dspy.Signature]:
    return dspy.Signature(
        {
            "target_article": (str, dspy.InputField()),
            "next_link": (
                Literal[tuple(options)],
                dspy.OutputField(desc="Which link is the best next click toward the target?"),
            ),
        },
        "You are navigating Wikipedia toward a target article.",
    )


async def one_question(target: str, cands: list[str]) -> str:
    async def best(opts: list[str]) -> str:
        if len(opts) == 1:
            return opts[0]
        p = await dspy.Predict(pick_sig(opts)).acall(target_article=target)
        return p.next_link

    chunks = [cands[i : i + 255] for i in range(0, len(cands), 255)]
    winners = await asyncio.gather(*(best(c) for c in chunks))
    return await best(list(dict.fromkeys(winners)))


async def per_link(target: str, cands: list[str]) -> str:
    sem, pred = asyncio.Semaphore(4), dspy.Predict(Hop)

    async def one(link: str) -> float:
        async with sem:
            p = await pred.acall(target_article=target, candidate_link=link)
            return p.jev["mentions"]["noul"]

    ps = await asyncio.gather(*(one(c) for c in cands))
    return max(zip(ps, cands, strict=True))[1]


async def main() -> None:
    lm = SemIfLM(
        "openai/qwen3vl", api_base="http://localhost:8089/v1", api_key="local", cache=False
    )
    dspy.configure(lm=lm, adapter=JevAdapter())
    score = {"one_question": [0, 0.0], "per_link": [0, 0.0]}
    for page, target in PAIRS:
        page, cands = links(page)
        target, _ = links(target)
        cands = [c for c in cands if c != target]
        for name, picker in [("one_question", one_question), ("per_link", per_link)]:
            t0 = time.perf_counter()
            choice = await picker(target, cands)
            dt = time.perf_counter() - t0
            hit = target in links(choice)[1]
            score[name][0] += hit
            score[name][1] += dt
            print(
                f"{page} -> {target} [{len(cands)}] {name}: {choice} "
                f"{'HIT' if hit else '-'} {dt:.1f}s"
            )
    for name, (hits, secs) in score.items():
        print(f"{name:<13} {hits}/{len(PAIRS)} picks link to the target, {secs:.0f}s total")


if __name__ == "__main__":
    asyncio.run(main())
