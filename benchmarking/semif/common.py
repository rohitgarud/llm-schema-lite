"""Rows and metrics shared by the SemIf runtime and backend benchmarks (no torch needed)."""

from __future__ import annotations

import math
import statistics
from typing import Any

from benchmarking.jevbench.run import load_tasks
from benchmarking.semif.parity import mean_family_balanced_accuracy
from llm_schema_lite.dspy_integration.adapters.semif_lm import _options


def hard_rows() -> list[dict[str, Any]]:
    """JevBench's public hard tier (long states) as SemIf rows, labelled where gold maps."""
    rows = []
    for task in load_tasks():
        if task["tier"] != "hard":
            continue
        q = task["question"]
        opts = _options({"type": q["type"], "criteria": q.get("criteria")})
        gold = {"yes": "true", "no": "false"}.get(str(task["expected"]), str(task["expected"]))
        rows.append(
            {
                "id": task["id"],
                "family": task["family"],
                "state": task["state"],
                "question": q["instructions"],
                "options": [{"id": k, "description": v} for k, v in opts.items()],
                "label": list(opts).index(gold) if gold in opts else None,
            }
        )
    return rows


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    weights = [math.exp(v - top) for v in values]
    return [w / sum(weights) for w in weights]


def calibration(rows: list[dict[str, Any]], probs: dict[str, list[float]]) -> dict[str, float]:
    """Gold-label NLL and 10-bin ECE of the top choice."""
    nll = [-math.log(max(probs[r["id"]][r["label"]], 1e-12)) for r in rows]
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    for r in rows:
        p = probs[r["id"]]
        best = max(range(len(p)), key=p.__getitem__)
        bins[min(int(p[best] * 10), 9)].append((p[best], best == r["label"]))
    ece = sum(
        len(b)
        / len(rows)
        * abs(statistics.mean(c for c, _ in b) - statistics.mean(h for _, h in b))
        for b in bins
        if b
    )
    return {"gold_nll": statistics.mean(nll), "ece": ece}


def summarize(
    rows: list[dict[str, Any]],
    probs: dict[str, list[float]],
    times: list[float],
    reference: dict[str, list[float]] | None,
    baseline: dict[str, list[float]],
) -> dict[str, Any]:
    ids = [r["id"] for r in rows]
    labelled = [r for r in rows if r["label"] is not None]
    options = {r["id"]: [o["id"] for o in r["options"]] for r in rows}

    def argmax(p: list[float]) -> int:
        return max(range(len(p)), key=p.__getitem__)

    def agreement(other: dict[str, list[float]]) -> float:
        return sum(argmax(probs[i]) == argmax(other[i]) for i in ids) / len(ids)

    def drift(other: dict[str, list[float]]) -> float:
        return statistics.mean(
            max(abs(a - b) for a, b in zip(probs[i], other[i], strict=True)) for i in ids
        )

    kl = statistics.mean(
        sum(
            b * math.log(max(b, 1e-12) / max(a, 1e-12))
            for a, b in zip(probs[i], baseline[i], strict=True)
        )
        for i in ids
    )
    ms = sorted(t * 1000 for t in times)
    parity = (
        {}
        if reference is None
        else {
            "argmax_agreement_semif_bf16": agreement(reference),
            "mean_max_dp_semif_bf16": drift(reference),
        }
    )
    return {
        "p50_ms": statistics.median(ms),
        "p95_ms": ms[int(0.95 * (len(ms) - 1))],
        "mean_ms": statistics.mean(ms),
        "mean_family_balanced_accuracy": mean_family_balanced_accuracy(
            labelled, {i: options[i][argmax(probs[i])] for i in ids}
        ),
        **parity,
        "argmax_agreement_baseline": agreement(baseline),
        "mean_max_dp_baseline": drift(baseline),
        "kl_from_baseline": kl,
        **calibration(labelled, probs),
    }
