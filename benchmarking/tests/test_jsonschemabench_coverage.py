"""One runnable check for the restored JSONSchemaBench coverage tooling.

Deliberately tiny: the 1,307 lines of tests deleted alongside this package in `0432dda`
tested the analysis helpers exhaustively and nothing called them. What matters here is
that feature detection recurses, that the aggregation totals line up, that the loader
survives a record it cannot parse, and that the per-config sweep splits by config -- the
four things a coverage run depends on.
"""

from __future__ import annotations

import json

import pytest

from benchmarking.jsonschemabench.base import analyze_schema_features, get_feature_statistics
from benchmarking.jsonschemabench.coverage import load_schemas, sweep_configs


@pytest.fixture
def local_dataset(tmp_path, monkeypatch):
    """Install `records` as a local dataset file and hide the other one.

    `load_records` prefers the config-labelled file, and the real one is a 98 MB file in
    the repo root -- patching only `LOCAL_DATASET` would silently measure that instead.
    """

    def _install(records: list, by_config: bool = False):
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(records))
        missing = tmp_path / "missing.json"
        monkeypatch.setattr(
            "benchmarking.jsonschemabench.coverage.LOCAL_BY_CONFIG",
            path if by_config else missing,
        )
        monkeypatch.setattr(
            "benchmarking.jsonschemabench.coverage.LOCAL_DATASET",
            missing if by_config else path,
        )
        return path

    return _install


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


def test_analyze_schema_features_descends_into_definitions() -> None:
    """Keywords inside `definitions`/`$defs` are counted; they used to be invisible.

    The recursion walked a hand-written subset of the subschema slots that omitted both,
    so anything living in a definition was missed. Measured on `patternProperties` across
    JSONSchemaBench: 416 schemas (4.4%) against the 687 (7.2%) the fixed walk finds, with
    934 of the missed occurrences under `definitions`. Every other keyword undercounted
    the same way.
    """
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/definitions/Env"}},
        "definitions": {
            "Env": {"type": "object", "patternProperties": {"^[A-Z]+$": {"type": "string"}}}
        },
        "$defs": {"Tag": {"type": "string", "minLength": 3}},
        # A subschema slot the old allowlist also never descended into.
        "additionalProperties": {"type": "number", "multipleOf": 5},
    }
    found = set(analyze_schema_features(schema))
    assert "patternProperties" in found  # under `definitions`
    assert "minLength" in found  # under `$defs`
    assert "multipleOf" in found  # under `additionalProperties`


def test_per_schema_timeout_is_counted_apart_from_an_error(monkeypatch) -> None:
    """A schema that blows the budget is "slow", not "broken".

    Ten schemas in the corpus render for 20-25 minutes apiece, so the sweep caps each one.
    The distinction matters to the report: a timeout means "too slow to use at prompt
    time", an exception means "the library failed on this input". A real slow schema would
    make this test take minutes, so the render is stubbed out with a sleep.
    """
    import time

    from benchmarking.jsonschemabench import base

    monkeypatch.setattr(base, "simplify_schema", lambda *a, **kw: time.sleep(5))

    result = base.analyze_dataset_coverage([{"type": "object"}], timeout_s=0.2, ids=["o27039"])

    assert result["timeouts"] == 1
    assert result["slow_ids"] == ["o27039"]
    assert result["failures"] == []  # not an exception
    assert result["supported_schemas"] == 0
    assert result["coverage_percentage"] == 0.0


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


def test_load_schemas_skips_unparseable_records(local_dataset) -> None:
    """A malformed record is counted out rather than aborting the load."""
    path = local_dataset(
        [
            {"json_schema": json.dumps({"type": "object"})},
            {"json_schema": "{not json"},  # malformed
            {"no_schema_key": True},  # missing key
            {"json_schema": json.dumps({"type": "array"})},
        ]
    )

    schemas, source = load_schemas()
    assert [s["type"] for s in schemas] == ["object", "array"]
    assert str(path) == source


def test_load_schemas_respects_limit(local_dataset) -> None:
    """`--limit` is what makes a run finish in seconds; it must actually stop early."""
    local_dataset([{"json_schema": json.dumps({"type": "object"})}] * 10)

    schemas, _ = load_schemas(limit=3)
    assert len(schemas) == 3


def test_sweep_configs_measures_each_config_separately(local_dataset) -> None:
    """The whole point of the sweep: one row per config, not one row for the corpus."""
    local_dataset(
        [
            {"config": "Kubernetes", "json_schema": json.dumps({"type": "object"})},
            {"config": "Kubernetes", "json_schema": json.dumps({"type": "array"})},
            {"config": "Snowplow", "json_schema": json.dumps({"type": "string"})},
        ],
        by_config=True,
    )

    rows = sweep_configs()
    assert [r["config"] for r in rows] == ["Kubernetes", "Snowplow"]
    assert [r["total_schemas"] for r in rows] == [2, 1]
    assert all(r["coverage_percentage"] == 100.0 for r in rows)


def test_sweep_configs_rejects_an_unlabelled_corpus(local_dataset) -> None:
    """The flat dataset has no `config`; sweeping it would silently mislabel everything."""
    local_dataset([{"json_schema": json.dumps({"type": "object"})}])

    with pytest.raises(SystemExit, match="no per-record `config`"):
        sweep_configs()
