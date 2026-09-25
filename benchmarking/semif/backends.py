"""SemIfLM against a serving backend, next to the in-process runtime ablation.

Scores the rows ``runtime.py`` scored (SemIf's authored144, or JevBench's public hard tier
with ``--workload hard``) through ``SemIfLM`` against an OpenAI-compatible server, and
compares the probabilities with the in-process ``semif`` run saved in ``results/``. Each
row is timed through ``SemIfLM``, one request at a time: what a DSPy program sees, litellm
and HTTP included.

Prompt caching is off (``cache_prompt: false`` on llama.cpp; start vLLM with
``--no-enable-prefix-caching``), or the repeated passes would time cache hits. Even so,
llama.cpp answers a prompt identical to the slot's last one about 90 ms sooner on
Qwen3.5-4B, so rows are never sent twice in a row. ``--cache-ram 0 --ctx-checkpoints 0``
removes that per-request cost altogether::

    llama-server -m Qwen3-0.6B-Q8_0.gguf --jinja -ngl 99 -c 8192 --port 8089 \\
        --cache-ram 0 --ctx-checkpoints 0
    python -m benchmarking.semif.backends --reference qwen3-0.6b --server llamacpp \\
        --api-base http://localhost:8089/v1 --label "llama.cpp Q8_0"
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import time
from pathlib import Path
from typing import Any

from benchmarking.semif.common import hard_rows, summarize
from benchmarking.semif.parity import REFERENCES, WORKLOAD, fetch_jsonl, question
from llm_schema_lite.dspy_integration import SemIfLM

RESULTS = Path(__file__).parent / "results"


def baseline_file(reference: str, workload: str) -> Path:
    """The newest in-process ladder run for this model and workload."""
    runs = sorted(RESULTS.glob(f"runtime-{reference}-{workload}-ladder-*.json"))
    if not runs:
        raise FileNotFoundError(f"Run runtime.py for {reference} on {workload} first")
    return runs[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", required=True, help="Model key used by runtime.py")
    parser.add_argument("--workload", choices=["authored144", "hard"], default="authored144")
    parser.add_argument("--server", choices=["llamacpp", "vllm"], required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", default="local", help="Model name the server expects")
    parser.add_argument("--label", required=True, help="e.g. 'llama.cpp Q8_0'")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    rows = hard_rows() if args.workload == "hard" else fetch_jsonl(*WORKLOAD)
    rows = rows[: args.limit]
    base_path = baseline_file(args.reference, args.workload)
    baseline = json.loads(base_path.read_text())["probabilities"]["semif"]
    reference = None
    if args.workload == "authored144" and args.reference in REFERENCES:
        path, sha, *_ = REFERENCES[args.reference]
        reference = {p["id"]: p["probabilities"] for p in fetch_jsonl(path, sha)}

    extra: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
    if args.server == "llamacpp":
        extra["cache_prompt"] = False
    lm = SemIfLM(
        f"openai/{args.model}",
        api_base=args.api_base,
        api_key="local",
        cache=False,
        extra_body=extra,
    )

    def through_lm(row: dict[str, Any]) -> list[float] | None:
        payload = {"state": row["state"], "questions": {"q": question(row)}}
        try:
            [text] = lm(messages=[{"role": "user", "content": json.dumps(payload)}])
        except ValueError:  # no option letter among the top logprobs
            return None
        probs = json.loads(text)["q"]["probabilities"]
        return [probs[o["id"]] for o in row["options"]]

    for row in rows[:4]:  # warm-up: server kernels, graphs, connection
        through_lm(row)
    probs: dict[str, list[float]] = {}
    missing: set[str] = set()
    lm_times = []
    for rep in range(args.repeats):
        for row in rows:
            start = time.perf_counter()
            out = through_lm(row)
            lm_times.append(time.perf_counter() - start)
            if out is None:
                missing.add(row["id"])
            else:
                probs.setdefault(row["id"], out)
        print(f"pass {rep + 1}/{args.repeats}", flush=True)

    scored = [r for r in rows if r["id"] in probs]
    summary = {
        "label": args.label,
        "server": args.server,
        "reference": args.reference,
        "workload": args.workload,
        "baseline": base_path.name,
        "rows": len(rows),
        "missing_letters": sorted(missing),
        "date": datetime.date.today().isoformat(),
        # latency and parity through SemIfLM; agreement/KL against in-process `semif`
        "semiflm": summarize(
            scored,
            probs,
            lm_times,
            {i: reference[i] for i in probs} if reference else None,
            {i: baseline[i] for i in probs},
        ),
    }
    RESULTS.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", args.label.lower()).strip("-")
    out_path = RESULTS / f"backend-{args.reference}-{args.workload}-{slug}-{summary['date']}.json"
    out_path.write_text(json.dumps({"summary": summary, "probabilities": probs}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
