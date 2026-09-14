#!/usr/bin/env -S uv run python
"""Format a JSON Schema from jsonschembench_dataset.json with llm-schema-lite."""

import json
import sys
from pathlib import Path
from typing import Literal

from llm_schema_lite import simplify_schema

FormatType = Literal["jsonish", "typescript", "yaml"]


def main() -> None:
    # benchmarking/jsonschemabench/format_jsonschembench_schema.py -> repo root
    dataset_path = Path(__file__).resolve().parents[2] / "jsonschembench_dataset.json"
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    with open(dataset_path) as f:
        records = json.load(f)

    # First record by default (JSON Patch schema)
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    # Narrowed via an explicit membership test so the Literal assignment type-checks;
    # `sys.argv[2] in (...)` alone does not narrow a plain `str` for mypy.
    format_type: FormatType = "jsonish"
    if len(sys.argv) > 2:
        requested = sys.argv[2]
        if requested == "typescript":
            format_type = "typescript"
        elif requested == "yaml":
            format_type = "yaml"

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
