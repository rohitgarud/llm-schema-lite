#!/usr/bin/env -S uv run python
"""Format a JSON Schema from jsonschembench_dataset.json with llm-schema-lite."""

import json
import sys
from pathlib import Path
from typing import Literal

from llm_schema_lite import simplify_schema

FormatType = Literal["jsonish", "typescript", "yaml"]


def main() -> None:
    dataset_path = Path(__file__).resolve().parent.parent / "jsonschembench_dataset.json"
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    with open(dataset_path) as f:
        records = json.load(f)

    # First record by default (JSON Patch schema)
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    format_type: FormatType = (
        sys.argv[2]
        if len(sys.argv) > 2 and sys.argv[2] in ("jsonish", "typescript", "yaml")
        else "jsonish"
    )

    record = records[index]
    schema = json.loads(record["json_schema"])
    unique_id = record.get("unique_id", "")

    print(f"# Schema index={index} unique_id={unique_id!r} format={format_type}\n")
    print(json.dumps(schema, indent=2))
    print("-" * 100)
    lite = simplify_schema(schema, format_type=format_type)
    print(lite.to_string())


if __name__ == "__main__":
    main()
