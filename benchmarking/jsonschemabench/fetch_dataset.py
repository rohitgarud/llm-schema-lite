#!/usr/bin/env -S uv run python
"""Download JSONSchemaBench into jsonschembench_dataset.json at the repo root.

Idempotent: skips if the file already exists. To re-download, delete the file first.

Each record keeps the config it came from, so a measurement can be sliced by difficulty
(`Github_trivial` .. `Github_ultra`) or by source (`Kubernetes`, `Snowplow`, ...). The
earlier version flattened every split into one unlabelled list, which made exactly that
slicing impossible.

    uv run python -m benchmarking.jsonschemabench.fetch_dataset
    uv run python -m benchmarking.jsonschemabench.fetch_dataset --configs Github_easy Kubernetes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from datasets import get_dataset_config_names, load_dataset
except ImportError as e:
    raise ImportError(
        "datasets not installed; install the benchmark extra: uv pip install -e '.[benchmark]'"
    ) from e

DATASET_NAME = "epfl-dlab/JSONSchemaBench"
# benchmarking/jsonschemabench/fetch_dataset.py -> repo root
DATASET_PATH = Path(__file__).resolve().parents[2] / "jsonschembench_dataset.json"

# "default" is the union of the others; taking it as well would double every schema.
SKIP_CONFIGS = frozenset({"default"})


def fetch(path: Path = DATASET_PATH, configs: list[str] | None = None, force: bool = False) -> Path:
    """Write the flat ``[{json_schema, unique_id, config, split}]`` list.

    ``json_schema`` stays a JSON *string*, which is the shape every reader here already
    expects (``coverage.load_schemas`` and ``format_jsonschembench_schema``).
    """
    if path.exists() and not force:
        print(f"Already present, skipping download: {path}")
        return path

    names = configs or [c for c in get_dataset_config_names(DATASET_NAME) if c not in SKIP_CONFIGS]
    records: list[dict[str, str]] = []
    for config in names:
        dataset = load_dataset(DATASET_NAME, config)
        before = len(records)
        for split_name, split in dataset.items():
            for row in split:
                records.append(
                    {
                        "json_schema": row["json_schema"],
                        "unique_id": row.get("unique_id", ""),
                        "config": config,
                        "split": split_name,
                    }
                )
        print(f"  {config:18} {len(records) - before:6} schemas")

    path.write_text(json.dumps(records))
    print(f"Wrote {len(records)} schemas from {len(names)} configs to {path}")
    return path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=None, help="only these configs")
    parser.add_argument("--force", action="store_true", help="re-download over an existing file")
    args = parser.parse_args()
    fetch(configs=args.configs, force=args.force)


if __name__ == "__main__":
    main()
