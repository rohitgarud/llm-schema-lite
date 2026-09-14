"""Feature coverage of the JSONSchemaBench dataset (epfl-dlab/JSONSchemaBench).

Restored from ``0432dda^`` with two changes. The analysis helpers now live in this
package: they were in ``benchmarking/base.py``, which the lean audit deleted along with
everything else that had no caller. And the dataset is read from the local
``jsonschembench_dataset.json`` when it is present, so a run needs neither a network
round-trip nor the ``datasets`` extra; the Hugging Face download stays as the fallback.

``--limit`` samples the front of the dataset, which is what makes this runnable in
seconds instead of minutes -- the full set is ~100 MB of schemas and every one of them
is pushed through ``simplify_schema`` twice (once per analysis pass).

    uv run python -m benchmarking.jsonschemabench.coverage --limit 2000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .base import analyze_dataset_coverage, analyze_schema_feature_coverage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# benchmarking/jsonschemabench/coverage.py -> repo root
LOCAL_DATASET = Path(__file__).resolve().parents[2] / "jsonschembench_dataset.json"


def load_schemas(limit: int | None = None) -> tuple[list[dict[str, Any]], str]:
    """Return ``(schemas, source)``, preferring the local dataset over a HF download.

    Every record carries its schema as a JSON *string* under ``json_schema``; records
    that fail to parse are counted out rather than aborting the run, since a single bad
    row in a 100 MB corpus should not cost the whole measurement.
    """
    records: list[dict[str, Any]]
    if LOCAL_DATASET.exists():
        with open(LOCAL_DATASET, encoding="utf-8") as fh:
            records = json.load(fh)
        source = str(LOCAL_DATASET)
    else:
        from datasets import load_dataset  # imported lazily: only the HF path needs it

        dataset = load_dataset("epfl-dlab/JSONSchemaBench")
        records = [item for _, split_data in dataset.items() for item in split_data]
        source = "huggingface: epfl-dlab/JSONSchemaBench"

    schemas: list[dict[str, Any]] = []
    skipped = 0
    for item in records:
        try:
            schemas.append(json.loads(item["json_schema"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            skipped += 1
            continue
        if limit is not None and len(schemas) >= limit:
            break

    if skipped:
        logger.warning(f"skipped {skipped} unparseable schema record(s)")
    return schemas, source


def analyze_coverage_jsonschemabench(limit: int | None = None) -> None:
    """Run both analysis passes over the dataset."""
    schemas, source = load_schemas(limit)
    logger.info(f"Loaded {len(schemas)} schemas from {source}")
    analyze_dataset_coverage(schemas)
    analyze_schema_feature_coverage(schemas)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="analyze only the first N schemas (default: the whole dataset)",
    )
    args = parser.parse_args()

    logger.info("🚀 Starting JSONSchemaBench feature coverage analysis...")
    if not LOCAL_DATASET.exists():
        logger.info("Local dataset not found; downloading. This may take a few minutes...")
    logger.info("")

    analyze_coverage_jsonschemabench(args.limit)


if __name__ == "__main__":
    main()
