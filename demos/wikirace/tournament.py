"""Wikirace with a knockout: links in groups of 16, the top few of each group go through.

One SemIf question per group ("which link is the best next click?") over 16 lettered
options, SemIf's native size. Each group's top ``KEEP`` links go to the next round until
16 or fewer are left; one last question picks the hop. 595 links take 38 + 5 + 1
requests instead of 595 yes/no questions, but about as many prompt tokens, which is what
the GPU spends its time on. Links are shuffled first, as small models favour the first
letters.

Usage: python tournament.py   (needs llama-server on :8089, e.g. -np 16 --kv-unified)
"""

import asyncio
import math
import random
import time
from collections.abc import Awaitable, Callable

import dspy
from pick255 import pick_sig
from wikirace import Hop, links

from llm_schema_lite.dspy_integration import JevAdapter, SemIfLM

GROUP, KEEP = 16, 2
RACES = [
    ("Banana", "Albert Einstein"),
    ("Rubber duck", "World War II"),
    ("Chess", "Ancient Egypt"),
    ("Guitar", "Photosynthesis"),
    ("Coffee", "Isaac Newton"),
]
API = {"api_base": "http://localhost:8089/v1", "api_key": "local", "cache": False}
PICK_LM = SemIfLM("openai/qwen3vl", top_logprobs=50, **API)  # all 16 letters, not top 20
LINK_LM = SemIfLM("openai/qwen3vl", question_first=True, **API)  # question shared by every link
SLOTS = asyncio.Semaphore(16)  # one per llama-server slot


async def ranked(target: str, options: list[str]) -> list[str]:
    """One question over up to 16 links, best first."""
    async with SLOTS:
        with dspy.context(lm=PICK_LM):
            p = await dspy.Predict(pick_sig(options)).acall(target_article=target)
    probs = p.jev["next_link"]["probabilities"]
    return sorted(options, key=lambda o: -probs.get(o, 0.0))


async def tournament(target: str, cands: list[str]) -> str:
    cands = random.Random(0).sample(cands, len(cands))
    while len(cands) > GROUP:
        n = math.ceil(len(cands) / GROUP)
        groups = [cands[i::n] for i in range(n)]  # n near-equal groups
        tops = await asyncio.gather(*(ranked(target, g) for g in groups))  # 8-16 links each
        cands = [link for top in tops for link in top[:KEEP]]
    return (await ranked(target, cands))[0] if len(cands) > 1 else cands[0]


async def per_link(target: str, cands: list[str]) -> str:
    async def one(link: str) -> float:
        async with SLOTS:
            with dspy.context(lm=LINK_LM):
                p = await dspy.Predict(Hop).acall(target_article=target, candidate_link=link)
        return float(p.jev["mentions"]["noul"])

    ps = await asyncio.gather(*(one(c) for c in cands))
    return max(zip(ps, cands, strict=True))[1]


async def race(
    start: str, target: str, pick: Callable[[str, list[str]], Awaitable[str]], max_hops: int = 10
) -> tuple[list[str], float]:
    """Path taken, and seconds spent picking (Wikipedia calls excluded)."""
    path, seen, picking, current = [], set(), 0.0, start
    for _ in range(max_hops):
        current, cands = links(current)
        path.append(current)
        seen.add(current)
        if current == target:
            break
        if target in cands:
            path.append(target)
            break
        t0 = time.perf_counter()
        current = await pick(target, [c for c in cands if c not in seen])
        picking += time.perf_counter() - t0
    return path, picking


async def main() -> None:
    dspy.configure(adapter=JevAdapter())
    totals: dict[str, list[float]] = {"tournament": [0, 0, 0], "per_link": [0, 0, 0]}
    for start, target in RACES:
        target, _ = links(target)
        for name, pick in [("tournament", tournament), ("per_link", per_link)]:
            path, secs = await race(start, target, pick)
            done = path[-1] == target
            hops = len(path) - 1
            print(
                f"{name:<10} {start} -> {target}: {'reached' if done else 'gave up'} in "
                f"{hops} hops, {secs:.1f}s picking ({secs / hops:.2f}s/hop): {' -> '.join(path)}",
                flush=True,
            )
            totals[name][0] += done
            totals[name][1] += hops
            totals[name][2] += secs
    for name, (done, hops, secs) in totals.items():
        print(
            f"{name:<10} reached {done:.0f}/{len(RACES)}, {hops:.0f} hops, {secs:.0f}s picking, "
            f"{secs / hops:.2f}s per hop"
        )


if __name__ == "__main__":
    asyncio.run(main())
