"""SemIf plays Wikiracing: reach a target Wikipedia article by following links only.

Every link on the page gets one yes/no question, "does this bring you closer?", and the
walker follows the link with the highest P(yes). Hundreds of links per page are scored
in parallel; the shared prefix (target, current page) stays in llama.cpp's KV cache.

Usage: python wikirace.py "Start" "Target" [max_hops]   (needs llama-server on :8089)
"""

import asyncio
import sys
import time

import dspy
import requests

from llm_schema_lite.dspy_integration import JevAdapter, SemIfLM

API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "llm-schema-lite-demo/0.1 (https://github.com/rohitgarud/llm-schema-lite)"}


class Hop(dspy.Signature):
    """You are navigating Wikipedia toward a target article."""

    # No "current page" field: with it, a 4B model scored links by the page it was on.
    target_article: str = dspy.InputField()
    candidate_link: str = dspy.InputField()
    mentions: bool = dspy.OutputField(
        desc="Would the candidate article probably mention or link to the target article?"
    )


def links(title: str) -> tuple[str, list[str]]:
    """Resolve redirects and return (canonical title, article links on that page)."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "links",
        "plnamespace": 0,
        "pllimit": "max",
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    }
    out, canon = [], title
    while True:
        for wait in (1, 2, 4, 8, 0):  # Wikipedia's API sheds load with 503s now and then
            resp = requests.get(API, params=params, headers=UA, timeout=30)
            if resp.ok or not wait:
                break
            time.sleep(wait)
        resp.raise_for_status()
        r = resp.json()
        page = r["query"]["pages"][0]
        canon = page["title"]
        out += [link["title"] for link in page.get("links", [])]
        if "continue" not in r:
            return canon, out
        params.update(r["continue"])


async def score(pred: dspy.Predict, target: str, cands: list[str]) -> list[float]:
    sem = asyncio.Semaphore(16)  # one per llama-server slot (-np 16)

    async def one(link: str) -> float:
        async with sem:
            p = await pred.acall(target_article=target, candidate_link=link)
            return p.jev["mentions"]["noul"]  # P(True)

    return await asyncio.gather(*(one(c) for c in cands))


async def race(start: str, target: str, max_hops: int = 15) -> list[str]:
    pred = dspy.Predict(Hop)
    target, _ = links(target)
    path, seen = [], set()
    current = start
    for _ in range(max_hops):
        current, cands = links(current)
        path.append(current)
        seen.add(current)
        if current == target:
            break
        if target in cands:
            print(f"  {current} -> {target} (link on page)", flush=True)
            path.append(target)
            break
        cands = [c for c in cands if c not in seen]
        t0 = time.perf_counter()
        ps = await score(pred, target, cands)
        best = sorted(zip(ps, cands, strict=True), reverse=True)
        top = ", ".join(f"{c} {p:.2f}" for p, c in best[:3])
        print(
            f"  {current}: {len(cands)} links in {time.perf_counter() - t0:.1f}s -> {top}",
            flush=True,
        )
        current = best[0][1]
    return path


if __name__ == "__main__":
    lm = SemIfLM(
        "openai/qwen3vl",
        api_base="http://localhost:8089/v1",
        api_key="local",
        cache=False,
        question_first=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    dspy.configure(lm=lm, adapter=JevAdapter())
    start, target = sys.argv[1], sys.argv[2]
    path = asyncio.run(race(start, target, int(sys.argv[3]) if len(sys.argv) > 3 else 15))
    status = "reached" if path[-1] == links(target)[0] else "gave up"
    print(f"{status} in {len(path) - 1} hops: {' -> '.join(path)}")
