"""JevBench public tiers through ``SemIfLM``, next to JevBench's published SemIf and Jev runs.

JevBench (https://github.com/fstandhartinger/jevbench, MIT) scores typed decisions written
in TypeSafe's wire format, so each item goes to ``SemIfLM`` as the payload JevAdapter
would send. Three tiers are public (easy 48, standard 72, hard 111 of 220); the judge
tier and the held-out halves are not, so these numbers are not the leaderboard's. The
references are re-scored on the same 231 public items from JevBench's per-task outcomes.

Reported per system: accuracy per tier. For ours also JevBench v1.2's Calibration axis
(hard tier: top-label ECE and mean TVD to the gold distributions) and serial p50/p95
latency. Cost is left out: it is a hosting price, not a property of the readout.

Example::

    llama-server -m Qwen3VL-4B-Instruct-Q4_K_M.gguf -ngl 99 -c 16384 --port 8089
    python -m benchmarking.jevbench.run --api-base http://localhost:8089/v1 \\
        --our-model "Qwen3-VL-4B-Instruct Q4_K_M"
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from llm_schema_lite.dspy_integration import SemIfLM

JEVBENCH_COMMIT = "2fa63fa3226cb369795525ed011800f57dcbd894"
RAW = f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{JEVBENCH_COMMIT}/"
TIERS = {  # tier -> (path, sha256); the hashes are the ones in JevBench's datasets/manifest.json
    "easy": (
        "datasets/public/easy.jsonl",
        "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
    ),
    "standard": (
        "datasets/public/original.jsonl",
        "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
    ),
    "hard": (
        "datasets/public/hard.jsonl",
        "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
    ),
}
PER_TASK = (
    "results/v1.2/jevbench-v1.2-per-task.json",
    "c772b3b85809f483449ec2a6ae0272347371ed5f40f7faff8394ecf4ee22ae7c",
)
REFERENCES = {
    "semif-qwen3.5-4b": "SemIf (Qwen3.5-4B BF16, SemIf's own code)",
    "jev-1.13.0": "Jev 1.13.0 (TypeSafe API)",
}
RESULTS = Path(__file__).parent / "results"


def fetch(path: str, sha256: str) -> bytes:
    """Download one JevBench file at the pinned commit and verify it before use."""
    with urllib.request.urlopen(RAW + path) as response:  # noqa: S310 - fixed https URL
        data: bytes = response.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{path} does not match its pinned SHA-256")
    return data


def load_tasks() -> list[dict[str, Any]]:
    return [
        {**json.loads(line), "tier": tier}
        for tier, (path, sha) in TIERS.items()
        for line in fetch(path, sha).decode().splitlines()
        if line.strip()
    ]


def payload(task: dict[str, Any]) -> dict[str, Any]:
    """The Jev request JevAdapter would send: instructions become the question text."""
    q = task["question"]
    question = {
        "type": q["type"],
        "instructions": {"task": "", "question": q["instructions"]},
        "criteria": q.get("criteria"),
    }
    return {"state": task["state"], "questions": {"q": question}}


def distribution(task: dict[str, Any], answer: dict[str, Any]) -> dict[str, float]:
    """A Jev answer as probabilities over the task's exact labels."""
    if answer["type"] == "noul":
        return {"yes": answer["noul"], "no": 1 - answer["noul"]}
    return dict(answer["probabilities"])


def _argmax(probs: dict[str, float]) -> str:
    return max(sorted(probs), key=probs.__getitem__)  # JevBench: ties go to the smallest label


def _ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float:
    """Top-label expected calibration error over equal-width bins, as JevBench computes it."""
    binned: dict[int, list[tuple[float, bool]]] = {}
    for conf, ok in pairs:
        binned.setdefault(min(int(conf * bins), bins - 1), []).append((conf, ok))
    return sum(
        len(b) / len(pairs) * abs(sum(ok for _, ok in b) / len(b) - sum(c for c, _ in b) / len(b))
        for b in binned.values()
    )


def summarize(
    tasks: list[dict[str, Any]],
    ours: list[dict[str, Any]],
    references: dict[str, dict[str, list[Any]]],
) -> dict[str, Any]:
    """Accuracy per tier for every system on the same items; calibration and latency for ours."""
    mine = {r["id"]: r for r in ours}
    right = {
        t["id"]: mine[t["id"]].get("probs") is not None
        and _argmax(mine[t["id"]]["probs"]) == str(t["expected"])
        for t in tasks
    }
    tiers = list(dict.fromkeys(t["tier"] for t in tasks))

    def accuracy(ok: dict[str, bool]) -> dict[str, float]:
        return {
            tier: statistics.mean(ok[t["id"]] for t in tasks if t["tier"] == tier) for tier in tiers
        }

    hard = [t for t in tasks if t["tier"] == "hard" and mine[t["id"]].get("probs")]
    ece = _ece([(max(mine[t["id"]]["probs"].values()), right[t["id"]]) for t in hard])
    tvds = [
        0.5 * sum(abs(mine[t["id"]]["probs"].get(k, 0) - p) for k, p in gold.items())
        for t in hard
        if (gold := t["provenance"].get("gold_probs"))
    ]
    seconds = [r["seconds"] for r in ours]
    summary: dict[str, Any] = {
        "ours": {
            "accuracy": accuracy(right),
            "errors": sum(r.get("probs") is None for r in ours),
            "hard_ece": ece if hard else None,
            "hard_mean_tvd": statistics.mean(tvds) if tvds else None,
            "p50_s": statistics.median(seconds),
            "p95_s": statistics.quantiles(seconds, n=20, method="inclusive")[18],
        }
    }
    if hard:  # JevBench v1.2 Calibration axis
        axis = max(0.0, 100 * (1 - ece / 0.5))
        summary["ours"]["calibration"] = (
            (axis + 100 * (1 - statistics.mean(tvds))) / 2 if tvds else axis
        )
    for name, outcomes in references.items():
        theirs = {t["id"]: outcomes[t["id"]][0] == "c" for t in tasks}
        summary[name] = {
            "accuracy": accuracy(theirs),
            "both_right": sum(right[i] and theirs[i] for i in right),
            "only_ours_right": sum(right[i] and not theirs[i] for i in right),
            "only_theirs_right": sum(theirs[i] and not right[i] for i in right),
        }
    return summary


def score_task(lm: SemIfLM, task: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    result: dict[str, Any] = {"id": task["id"]}
    try:
        [text] = lm(messages=[{"role": "user", "content": json.dumps(payload(task))}])
        result["probs"] = distribution(task, json.loads(text)["q"])
    except ValueError as error:  # no option letter in top_logprobs
        result["error"] = str(error)
    result["seconds"] = time.perf_counter() - t0
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="openai/local", help="LiteLLM model id")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--our-model", required=True, help="What the server runs, for the record")
    parser.add_argument("--workers", type=int, default=1, help="1 keeps latency serial")
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--question-first", action="store_true")
    args = parser.parse_args(argv)

    tasks = load_tasks()
    per_task = json.loads(fetch(*PER_TASK))["systems"]
    references = {name: per_task[name]["public_tasks"] for name in REFERENCES}
    lm = SemIfLM(
        args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        top_logprobs=args.top_logprobs,
        question_first=args.question_first,
        cache=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    with ThreadPoolExecutor(args.workers) as pool:
        ours = list(pool.map(lambda t: score_task(lm, t), tasks))

    summary = {
        **summarize(tasks, ours, references),
        "our_model": args.our_model,
        "question_first": args.question_first,
        "workers": args.workers,
        "references": REFERENCES,
        "jevbench_commit": JEVBENCH_COMMIT,
        "date": datetime.date.today().isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    stem = args.our_model.lower().replace(" ", "-") + ("-qfirst" if args.question_first else "")
    out = RESULTS / f"jevbench-{stem}-{summary['date']}.json"
    out.write_text(json.dumps({"summary": summary, "rows": ours}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
