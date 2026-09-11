"""Paired bootstrap intervals for the difference between adapters in one accuracy CSV.

Every adapter in an accuracy CSV scored the same cases, so resampling cases -- not rows --
keeps each case's scores together, and the interval is for the difference itself. It
covers case-sampling noise at this n, not run-to-run noise: Ollama moves a cell by up to
0.02 on the same code, so a win whose interval ends that close to zero is a tie. Across
many adapters and files, about one comparison in twenty excludes zero by chance.

    python -m benchmarking.dspy_adapters.paired results/accuracy-pii-*.csv --baseline json
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

Pair = tuple[float, float]

# Micro-averaged like the report's aggregate: summed numerators over summed denominators,
# failed cases included with their full denominators. Pre-recall CSVs have `field` only.
METRICS: dict[str, Callable[[dict[str, str]], Pair]] = {
    "recall": lambda r: (float(r["recall_matched"]), float(r["recall_total"])),
    "field": lambda r: (float(r["matched"]), float(r["total"])),
    "invented": lambda r: (float(r["invented"]), float(r["total"]) - float(r["recall_total"])),
}


def _rate(pairs: Sequence[Pair], idx: Sequence[int]) -> float | None:
    den = sum(pairs[i][1] for i in idx)
    return sum(pairs[i][0] for i in idx) / den if den else None


def compare(
    path: Path, baseline: str, metric: str = "recall", resamples: int = 5000, seed: int = 0
) -> Iterator[tuple[str, float, float, float, float]]:
    """Yield ``(adapter, rate, baseline_rate, lo, hi)`` for each adapter but ``baseline``.

    ``lo``/``hi`` bound the 95% interval of ``rate - baseline_rate``. An adapter whose
    cases have no denominator (recall on all-null gold) is skipped.
    """
    cells: dict[str, dict[str, Pair]] = defaultdict(dict)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cells[row["adapter"]][row["case_id"]] = METRICS[metric](row)
    if baseline not in cells:
        raise SystemExit(f"error: {path} has no {baseline!r} rows; it has {', '.join(cells)}")
    base = cells.pop(baseline)
    for adapter, scores in cells.items():
        ids = sorted(scores.keys() & base.keys())
        a, b = [scores[i] for i in ids], [base[i] for i in ids]
        every = range(len(ids))
        rate_a, rate_b = _rate(a, every), _rate(b, every)
        if rate_a is None or rate_b is None:
            continue
        rng = random.Random(seed)
        diffs = []
        for _ in range(resamples):
            idx = rng.choices(every, k=len(ids))
            da, db = _rate(a, idx), _rate(b, idx)
            if da is not None and db is not None:
                diffs.append(da - db)
        cuts = statistics.quantiles(diffs, n=40, method="inclusive")
        yield adapter, rate_a, rate_b, cuts[0], cuts[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired bootstrap CIs for adapter differences.")
    parser.add_argument("csv", nargs="+", type=Path, help="Per-case accuracy CSVs.")
    parser.add_argument("--baseline", default="json", help="Adapter id to compare against.")
    parser.add_argument("--metric", choices=METRICS, default="recall")
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args(argv)
    print(f"| cell | adapter | {args.metric} | `{args.baseline}` | diff | 95% CI | |")
    print("|---|---|---|---|---|---|---|")
    for path in args.csv:
        for adapter, ra, rb, lo, hi in compare(path, args.baseline, args.metric, args.resamples):
            verdict = "higher" if lo > 0 else "lower" if hi < 0 else "tie"
            print(
                f"| {path.stem.removeprefix('accuracy-')} | `{adapter}` | {ra:.3f} | {rb:.3f} "
                f"| {ra - rb:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {verdict} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
