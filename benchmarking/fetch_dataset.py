#!/usr/bin/env -S uv run python
"""Download JSONSchemaBench into jsonschembench_dataset.json at the repo root.

Idempotent: skips if the file already exists. To re-download, delete the file first.
"""

import json
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError as e:
    raise ImportError(
        "datasets not installed; install the benchmark extra: uv pip install -e '.[benchmark]'"
    ) from e

DATASET_PATH = Path(__file__).resolve().parent.parent / "jsonschembench_dataset.json"


def fetch(path: Path = DATASET_PATH) -> Path:
    """Write the flat [{json_schema, unique_id}] list format_jsonschembench_schema.py reads."""
    if path.exists():
        print(f"Already present, skipping download: {path}")
        return path
    records = [
        {"json_schema": row["json_schema"], "unique_id": row.get("unique_id", "")}
        for split in load_dataset("epfl-dlab/JSONSchemaBench").values()
        for row in split
    ]
    path.write_text(json.dumps(records))
    print(f"Wrote {len(records)} schemas to {path}")
    return path


if __name__ == "__main__":
    fetch()
