"""Feature coverage of the JSONSchemaBench dataset (epfl-dlab/JSONSchemaBench).

Restored from ``0432dda^`` with two changes. The analysis helpers now live in this
package: they were in ``benchmarking/base.py``, which the lean audit deleted along with
everything else that had no caller. And the dataset is read from a local JSON file when
one is present, so a run needs neither a network round-trip nor the ``datasets`` extra;
the Hugging Face download stays as the fallback.

Two local files are understood, newest first. ``jsonschemabench_by_config.json`` keeps
each record's ``config`` and ``split``, which is what makes a per-config sweep possible --
the flat ``jsonschembench_dataset.json`` does not, and its first few hundred records are
all ``Github_trivial``/``Github_easy``, so a ``--limit`` run against it measures one
corner of the benchmark rather than the benchmark.

    uv run python -m benchmarking.jsonschemabench.coverage --all-configs
    uv run python -m benchmarking.jsonschemabench.coverage --config Kubernetes
    uv run python -m benchmarking.jsonschemabench.coverage --limit 2000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .base import analyze_dataset_coverage, analyze_schema_feature_coverage, get_feature_statistics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# benchmarking/jsonschemabench/coverage.py -> repo root
ROOT = Path(__file__).resolve().parents[2]
LOCAL_BY_CONFIG = ROOT / "jsonschemabench_by_config.json"
LOCAL_DATASET = ROOT / "jsonschembench_dataset.json"


def load_records() -> tuple[list[dict[str, Any]], str]:
    """Return ``(records, source)``, preferring the config-labelled local file."""
    for path in (LOCAL_BY_CONFIG, LOCAL_DATASET):
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                return json.load(fh), str(path)

    from datasets import load_dataset  # imported lazily: only the HF path needs it

    dataset = load_dataset("epfl-dlab/JSONSchemaBench")
    records = [item for _, split_data in dataset.items() for item in split_data]
    return records, "huggingface: epfl-dlab/JSONSchemaBench"


def parse_records(
    records: list[dict[str, Any]], limit: int | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse each record's ``json_schema``, keeping its ``unique_id`` alongside.

    Records that fail to parse are counted out rather than aborting the run: a single bad
    row in a 100 MB corpus should not cost the whole measurement. The ids travel with the
    schemas because a skipped record breaks index alignment with ``records``, and the
    slow-schema table names schemas by id.
    """
    schemas: list[dict[str, Any]] = []
    ids: list[str] = []
    skipped = 0
    for item in records:
        try:
            schemas.append(json.loads(item["json_schema"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            skipped += 1
            continue
        ids.append(str(item.get("unique_id", len(ids))))
        if limit is not None and len(schemas) >= limit:
            break

    if skipped:
        logger.warning(f"skipped {skipped} unparseable schema record(s)")
    return schemas, ids


def parse_schemas(records: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    """The id-free view, for callers that only need the schemas."""
    return parse_records(records, limit)[0]


def load_schemas(limit: int | None = None) -> tuple[list[dict[str, Any]], str]:
    """Return ``(schemas, source)`` for the whole corpus."""
    records, source = load_records()
    return parse_schemas(records, limit), source


def analyze_coverage_jsonschemabench(
    limit: int | None = None, timeout_s: float | None = None
) -> None:
    """Run both analysis passes over the dataset."""
    schemas, source = load_schemas(limit)
    logger.info(f"Loaded {len(schemas)} schemas from {source}")
    analyze_dataset_coverage(schemas, timeout_s)
    analyze_schema_feature_coverage(schemas)


def sweep_configs(
    limit: int | None = None,
    out_path: Path | None = None,
    timeout_s: float | None = None,
) -> list[dict[str, Any]]:
    """Measure every config separately and print one comparison table.

    Each config is parsed, measured and dropped before the next one starts, so peak memory
    tracks the largest single config rather than the whole 98 MB corpus.
    """
    records, source = load_records()
    if not any("config" in r for r in records[:1]):
        raise SystemExit(
            f"{source} has no per-record `config`; re-fetch with "
            "`uv run python -m benchmarking.jsonschemabench.fetch_dataset`"
        )

    by_config: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_config.setdefault(record["config"], []).append(record)
    del records

    rows: list[dict[str, Any]] = []
    for config in sorted(by_config):
        schemas, ids = parse_records(by_config.pop(config), limit)
        logger.info(f"\n{'=' * 70}\n### {config} -- {len(schemas)} schemas\n{'=' * 70}")
        row: dict[str, Any] = {"config": config}
        row.update(analyze_dataset_coverage(schemas, timeout_s, ids))
        stats = get_feature_statistics(schemas)
        row["distinct_features"] = stats["total_unique_features"]
        row["feature_counts"] = stats["feature_counts"]
        rows.append(row)
        if out_path is not None:
            # Flushed per config rather than once at the end: a sweep killed partway
            # through used to leave nothing behind, losing every config it had finished.
            out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    _print_table(rows, source)
    if out_path is not None:
        logger.info(f"\nWrote {out_path}")
    return rows


def _print_table(rows: list[dict[str, Any]], source: str) -> None:
    """One line per config, plus a corpus-wide total."""

    def pct(value: float | None) -> str:
        return "  n/a" if value is None else f"{value:5.1f}"

    logger.info(f"\n\n📊 Per-config summary ({source})")
    logger.info("=" * 78)
    logger.info(
        f"{'config':18} {'schemas':>8} {'ingested':>9} {'median↓':>8} "
        f"{'mean↓':>7} {'min↓':>7} {'feats':>6} {'slow':>5} {'err':>5}"
    )
    logger.info("-" * 78)
    for row in rows:
        logger.info(
            f"{row['config']:18} {row['total_schemas']:8d} "
            f"{row['coverage_percentage']:8.1f}% {pct(row['median_token_reduction'])}% "
            f"{pct(row['mean_token_reduction'])}% {pct(row['min_token_reduction'])}% "
            f"{row['distinct_features']:6d} {row.get('timeouts', 0):5d} "
            f"{len(row.get('failures', [])):5d}"
        )
    logger.info("-" * 78)
    total = sum(r["total_schemas"] for r in rows)
    supported = sum(r["supported_schemas"] for r in rows)
    features: set[str] = set()
    for row in rows:
        features.update(row["feature_counts"])
    logger.info(
        f"{'TOTAL':18} {total:8d} {supported / total * 100 if total else 0:8.1f}% "
        f"{'':8} {'':7} {'':7} {len(features):6d} "
        f"{sum(r.get('timeouts', 0) for r in rows):5d} "
        f"{sum(len(r.get('failures', [])) for r in rows):5d}"
    )
    slow = [(r["config"], sid) for r in rows for sid in r.get("slow_ids", [])]
    if slow:
        logger.info("\nSlow schemas (exceeded the per-schema budget):")
        for config, schema_id in slow:
            logger.info(f"  {config:18} {schema_id}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="analyze only the first N schemas (per config when sweeping)",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="measure each JSONSchemaBench config separately and print a comparison table",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="measure a single config (e.g. Kubernetes, Github_hard)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the per-config rows to this JSON file",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="per-schema render budget in seconds; 0 disables it (default: 10)",
    )
    args = parser.parse_args()

    logger.info("🚀 Starting JSONSchemaBench feature coverage analysis...")
    if not (LOCAL_BY_CONFIG.exists() or LOCAL_DATASET.exists()):
        logger.info("Local dataset not found; downloading. This may take a few minutes...")
    logger.info("")

    if args.all_configs:
        sweep_configs(args.limit, args.out, args.timeout)
        return

    if args.config:
        records, source = load_records()
        selected = [r for r in records if r.get("config") == args.config]
        if not selected:
            raise SystemExit(f"no records for config {args.config!r} in {source}")
        schemas, ids = parse_records(selected, args.limit)
        logger.info(f"Loaded {len(schemas)} {args.config} schemas from {source}")
        analyze_dataset_coverage(schemas, args.timeout, ids)
        analyze_schema_feature_coverage(schemas)
        return

    analyze_coverage_jsonschemabench(args.limit, args.timeout)


if __name__ == "__main__":
    main()
