"""Tests for benchmarking/fetch_dataset.py."""

import json

from .. import fetch_dataset


def test_fetch_writes_flat_records_for_all_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch_dataset,
        "load_dataset",
        lambda name: {
            "train": [{"json_schema": '{"type": "object"}', "unique_id": "a"}],
            "val": [{"json_schema": '{"type": "string"}'}],
        },
    )
    target = tmp_path / "d.json"

    result = fetch_dataset.fetch(target)

    assert result == target
    records = json.loads(target.read_text())
    assert records == [
        {"json_schema": '{"type": "object"}', "unique_id": "a"},
        {"json_schema": '{"type": "string"}', "unique_id": ""},
    ]
    assert json.loads(records[0]["json_schema"]) == {"type": "object"}


def test_fetch_skips_when_file_exists(tmp_path, monkeypatch):
    target = tmp_path / "d.json"
    target.write_text("[]")

    def _raise(name):
        raise AssertionError("load_dataset should not be called when the file already exists")

    monkeypatch.setattr(fetch_dataset, "load_dataset", _raise)

    result = fetch_dataset.fetch(target)

    assert result == target
    assert target.read_text() == "[]"
