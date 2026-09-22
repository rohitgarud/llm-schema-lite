"""SemIf parity benchmark: does ``SemIfLM`` reproduce SemIf's published decisions?

Scores SemIf's 144-row authored workload through ``SemIfLM`` and compares the result with
SemIf's own row-level predictions for the same model, fetched at a pinned commit and
checked against its SHA-256 sums:

- **mean family balanced accuracy**, the metric in SemIf's README, next to the reported one
- **argmax agreement** and probability drift, row by row, with the reference predictions
- **prompt parity** (``--template-url``, llama.cpp only): the server-rendered prompt's hash
  against SemIf's recorded ``prompt_sha256``

References are SemIf's native BF16 runs, so a quantized GGUF is expected to drift a little.
The prompt hash shows whether the input is identical.

Example, with llama.cpp serving SemIf's browser-demo GGUF::

    llama-server -m Qwen3-0.6B-Q8_0.gguf --jinja -ngl 99 --port 8089
    python -m benchmarking.semif.parity --reference qwen3-0.6b \\
        --api-base http://localhost:8089/v1 --template-url http://localhost:8089
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from llm_schema_lite.dspy_integration import SemIfLM
from llm_schema_lite.dspy_integration.adapters.semif_lm import _messages

SEMIF_COMMIT = "1f2dea3e25379f9dfc98cb83c324f00ab5deda37"
RAW = f"https://raw.githubusercontent.com/TheoLeeCJ/SemIf/{SEMIF_COMMIT}/"
WORKLOAD = (
    "benchmarks/data/authored144.jsonl",
    "8162d1c73f925af64453f1ec05ef36d583b3815bf698e60f0d454bd11537e079",
)
# name -> (predictions path, sha256, reported mean family balanced accuracy, GGUF used)
REFERENCES = {
    "qwen3-0.6b": (
        "results/raw/browser-ladder-qwen3-0.6b.predictions.jsonl",
        "f0a626a7442879a14d5fbbe702638e047358644151e0b71552d81fef7ac515ac",
        0.44035251527511593,
        "Qwen/Qwen3-0.6B-GGUF@23749fef Qwen3-0.6B-Q8_0.gguf",
    ),
    "minicpm5-2b": (
        "results/raw/browser-ladder-minicpm5-2b.predictions.jsonl",
        "4d6426ae4ab77379574faa0be043719005e6d26c33509f49b86d5e48cdf320b8",
        0.6862540337772537,
        "openbmb/MiniCPM5-2B-GGUF@2079a22f MiniCPM5-2B-Q4_K_M.gguf",
    ),
    "qwen3.5-4b": (
        "results/raw/browser-ladder-qwen3.5-4b.predictions.jsonl",
        "47e44a97364c1c64e1d1f1002f562fef392a412a0a9a0709c875d841ee860bd9",
        0.8132381607613807,
        "bartowski/Qwen_Qwen3.5-4B-GGUF@4168f45a Qwen_Qwen3.5-4B-Q4_K_M.gguf",
    ),
}
RESULTS = Path(__file__).parent / "results"


def fetch_jsonl(path: str, sha256: str) -> list[dict[str, Any]]:
    """Download one SemIf file at the pinned commit and verify it before use."""
    with urllib.request.urlopen(RAW + path) as response:  # noqa: S310 - fixed https URL
        data = response.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{path} does not match its pinned SHA-256")
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def question(row: dict[str, Any]) -> dict[str, Any]:
    """A SemIf row as the Jev ``choice`` question JevAdapter would send (no task text)."""
    return {
        "type": "choice",
        "instructions": {"task": "", "question": row["question"]},
        "criteria": {o["id"]: o["description"] for o in row["options"]},
    }


def mean_family_balanced_accuracy(rows: list[dict[str, Any]], predicted: dict[str, str]) -> float:
    """SemIf's headline metric: balanced accuracy (mean per-label recall) per family, averaged."""
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    scores: list[float] = []
    for part in families.values():
        gold = {r["id"]: r["options"][r["label"]]["id"] for r in part}
        recalls = [
            sum(predicted.get(i) == label for i, g in gold.items() if g == label)
            / sum(g == label for g in gold.values())
            for label in set(gold.values())
        ]
        scores.append(sum(recalls) / len(recalls))
    return sum(scores) / len(scores)


def score_row(lm: SemIfLM, row: dict[str, Any], template_url: str | None) -> dict[str, Any]:
    q = question(row)
    payload = {"state": row["state"], "questions": {"q": q}}
    result: dict[str, Any] = {"id": row["id"]}
    try:
        [text] = lm(messages=[{"role": "user", "content": json.dumps(payload)}])
        result["probabilities"] = json.loads(text)["q"]["probabilities"]
    except ValueError as error:  # no option letter in top_logprobs
        result["error"] = str(error)
    if template_url:
        body = {
            "messages": _messages(row["state"], q, q["criteria"]),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(  # noqa: S310 - caller-configured local server
            f"{template_url}/apply-template",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:  # noqa: S310
            prompt = json.load(response)["prompt"]
        result["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    return result


def compare(
    rows: list[dict[str, Any]], ours: list[dict[str, Any]], reference: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare our scored rows with SemIf's reference predictions for the same model."""
    ref = {p["id"]: dict(zip(p["option_ids"], p["probabilities"], strict=True)) for p in reference}
    ref_hash = {p["id"]: p["prompt_sha256"] for p in reference}
    mine = {r["id"]: r for r in ours}

    def argmax(probs: dict[str, float]) -> str:
        return max(probs, key=probs.__getitem__)

    ok = [r["id"] for r in rows if "probabilities" in mine[r["id"]]]
    agree = sum(argmax(mine[i]["probabilities"]) == argmax(ref[i]) for i in ok)
    drift = [max(abs(mine[i]["probabilities"][k] - p) for k, p in ref[i].items()) for i in ok]
    hashed = [i for i in mine if "prompt_sha256" in mine[i]]
    return {
        "rows": len(rows),
        "scored": len(ok),
        "ours_mean_family_balanced_accuracy": mean_family_balanced_accuracy(
            rows, {i: argmax(mine[i]["probabilities"]) for i in ok}
        ),
        "reference_mean_family_balanced_accuracy": mean_family_balanced_accuracy(
            rows, {r["id"]: argmax(ref[r["id"]]) for r in rows}
        ),
        "argmax_agreement": agree / len(ok) if ok else None,
        "max_probability_drift_mean": sum(drift) / len(drift) if drift else None,
        "max_probability_drift_max": max(drift, default=None),
        "prompt_hash_matches": sum(mine[i]["prompt_sha256"] == ref_hash[i] for i in hashed)
        if hashed
        else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", choices=sorted(REFERENCES), required=True)
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--model", help="LiteLLM model id (default: openai/<reference>)")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--template-url", help="llama.cpp server root, to check prompt hashes")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--our-model", help="What the server runs (default: SemIf's demo GGUF)")
    args = parser.parse_args(argv)

    path, sha, reported, gguf = REFERENCES[args.reference]
    rows = fetch_jsonl(*WORKLOAD)
    ids = {r["id"] for r in rows}
    reference = [p for p in fetch_jsonl(path, sha) if p["id"] in ids]

    lm = SemIfLM(
        args.model or f"openai/{args.reference}",
        api_base=args.api_base,
        api_key=args.api_key,
        top_logprobs=args.top_logprobs,
        cache=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    with ThreadPoolExecutor(args.workers) as pool:
        ours = list(pool.map(lambda r: score_row(lm, r, args.template_url), rows))

    summary = {
        "reference": args.reference,
        "reported_mean_family_balanced_accuracy": reported,
        **compare(rows, ours, reference),
        "reference_model": "native BF16 (SemIf browser ladder)",
        "our_model": args.our_model or gguf,
        "semif_commit": SEMIF_COMMIT,
        "date": datetime.date.today().isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    stem = summary["our_model"].split()[-1].removesuffix(".gguf").lower()
    out = RESULTS / f"parity-{stem}-{summary['date']}.json"
    out.write_text(json.dumps({"summary": summary, "rows": ours}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
