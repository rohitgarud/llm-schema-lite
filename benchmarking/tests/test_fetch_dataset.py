"""Tests for benchmarking/jsonschemabench/fetch_dataset.py.

The script moved into the `jsonschemabench` package and now records which config each
schema came from, so a measurement can be sliced by difficulty or source. Both tests
track those two changes: the import path, and the `config`/`split` keys on every record.
"""

import json

from ..jsonschemabench import fetch_dataset


def _fake_load_dataset(name, config):
    """Stand-in for `datasets.load_dataset`, which now takes a config as well as a name."""
    return {
        "train": [{"json_schema": '{"type": "object"}', "unique_id": f"{config}-a"}],
        "val": [{"json_schema": '{"type": "string"}'}],
    }


def test_fetch_writes_labelled_records_for_all_splits(tmp_path, monkeypatch):
    """Every record carries its config and split; a missing unique_id degrades to ""."""
    monkeypatch.setattr(fetch_dataset, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(fetch_dataset, "get_dataset_config_names", lambda name: ["Snowplow"])
    target = tmp_path / "d.json"

    result = fetch_dataset.fetch(target)

    assert result == target
    records = json.loads(target.read_text())
    assert records == [
        {
            "json_schema": '{"type": "object"}',
            "unique_id": "Snowplow-a",
            "config": "Snowplow",
            "split": "train",
        },
        {
            "json_schema": '{"type": "string"}',
            "unique_id": "",
            "config": "Snowplow",
            "split": "val",
        },
    ]
    assert json.loads(records[0]["json_schema"]) == {"type": "object"}


def test_fetch_skips_the_union_config(tmp_path, monkeypatch):
    """`default` is the union of the others; taking it too would double every schema."""
    monkeypatch.setattr(fetch_dataset, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(
        fetch_dataset, "get_dataset_config_names", lambda name: ["Snowplow", "default"]
    )
    target = tmp_path / "d.json"

    fetch_dataset.fetch(target)

    configs = {r["config"] for r in json.loads(target.read_text())}
    assert configs == {"Snowplow"}


def test_fetch_honours_an_explicit_config_list(tmp_path, monkeypatch):
    """An explicit `configs=` skips discovery entirely."""
    monkeypatch.setattr(fetch_dataset, "load_dataset", _fake_load_dataset)

    def _no_discovery(name):
        raise AssertionError("get_dataset_config_names should not run when configs= is given")

    monkeypatch.setattr(fetch_dataset, "get_dataset_config_names", _no_discovery)
    target = tmp_path / "d.json"

    fetch_dataset.fetch(target, configs=["Kubernetes"])

    configs = {r["config"] for r in json.loads(target.read_text())}
    assert configs == {"Kubernetes"}


def test_fetch_skips_when_file_exists(tmp_path, monkeypatch):
    """Idempotent: an existing file is left untouched and nothing is downloaded."""
    target = tmp_path / "d.json"
    target.write_text("[]")

    def _raise(*args, **kwargs):
        raise AssertionError("load_dataset should not be called when the file already exists")

    monkeypatch.setattr(fetch_dataset, "load_dataset", _raise)

    result = fetch_dataset.fetch(target)

    assert result == target
    assert target.read_text() == "[]"


def test_force_redownloads_over_an_existing_file(tmp_path, monkeypatch):
    """`--force` is the documented way to refresh a stale dataset."""
    monkeypatch.setattr(fetch_dataset, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(fetch_dataset, "get_dataset_config_names", lambda name: ["Snowplow"])
    target = tmp_path / "d.json"
    target.write_text("[]")

    fetch_dataset.fetch(target, force=True)

    assert json.loads(target.read_text()) != []
