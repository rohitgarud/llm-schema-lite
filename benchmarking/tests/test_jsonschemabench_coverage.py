"""One runnable check for the restored JSONSchemaBench coverage tooling.

Deliberately tiny: the 1,307 lines of tests deleted alongside this package in `0432dda`
tested the analysis helpers exhaustively and nothing called them. What matters here is
that feature detection recurses, that the aggregation totals line up, and that the
loader survives a record it cannot parse -- the three things a coverage run depends on.
"""

from __future__ import annotations

import json

from benchmarking.jsonschemabench.base import analyze_schema_features, get_feature_statistics
from benchmarking.jsonschemabench.coverage import load_schemas


def test_analyze_schema_features_finds_nested_keywords() -> None:
    """Features are detected below the root, not just on it."""
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {
            "a": {"type": "string", "pattern": "^x", "minLength": 2},
            "b": {"type": "array", "items": {"type": "integer", "minimum": 1}},
            "c": {"oneOf": [{"const": "y"}, {"type": "null"}]},
        },
    }
    found = set(analyze_schema_features(schema))
    # Root-level
    assert {"type", "required", "properties"} <= found
    # One and two levels down -- the recursion is the point
    assert {"pattern", "minLength"} <= found
    assert {"items", "minimum"} <= found
    assert {"oneOf", "const"} <= found


def test_get_feature_statistics_totals_agree() -> None:
    """Counts, percentages and the per-schema map stay consistent."""
    dataset = [
        {"type": "object", "properties": {"a": {"type": "string"}}},
        {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}},
    ]
    stats = get_feature_statistics(dataset)

    assert stats["total_schemas"] == 2
    assert stats["processed_schemas"] == 2
    assert stats["success_rate"] == 100.0
    # "required" is in exactly one of the two schemas.
    assert stats["feature_counts"]["required"] == 1
    assert stats["feature_percentages"]["required"] == 50.0
    # "type" is in both.
    assert stats["feature_counts"]["type"] == 2
    assert stats["feature_percentages"]["type"] == 100.0
    # The map carries one entry per input schema, and the sort is descending.
    assert set(stats["schema_feature_map"]) == {0, 1}
    counts = [n for _, n in stats["features_by_frequency"]]
    assert counts == sorted(counts, reverse=True)


def test_load_schemas_skips_unparseable_records(tmp_path, monkeypatch) -> None:
    """A malformed record is counted out rather than aborting the load."""
    dataset = tmp_path / "jsonschembench_dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {"json_schema": json.dumps({"type": "object"})},
                {"json_schema": "{not json"},  # malformed
                {"no_schema_key": True},  # missing key
                {"json_schema": json.dumps({"type": "array"})},
            ]
        )
    )
    monkeypatch.setattr("benchmarking.jsonschemabench.coverage.LOCAL_DATASET", dataset)

    schemas, source = load_schemas()
    assert [s["type"] for s in schemas] == ["object", "array"]
    assert str(dataset) == source


def test_load_schemas_respects_limit(tmp_path, monkeypatch) -> None:
    """`--limit` is what makes a run finish in seconds; it must actually stop early."""
    dataset = tmp_path / "jsonschembench_dataset.json"
    dataset.write_text(json.dumps([{"json_schema": json.dumps({"type": "object"})}] * 10))
    monkeypatch.setattr("benchmarking.jsonschemabench.coverage.LOCAL_DATASET", dataset)

    schemas, _ = load_schemas(limit=3)
    assert len(schemas) == 3
