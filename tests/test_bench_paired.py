"""Paired bootstrap over a per-case accuracy CSV."""

import csv

import pytest

from benchmarking.dspy_adapters.paired import compare, main

COLUMNS = ["adapter", "case_id", "matched", "total", "recall_matched", "recall_total", "invented"]


def _csv(tmp_path, recall_by_adapter):
    path = tmp_path / "accuracy-pii-m-2026-09-11.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for adapter, hits in recall_by_adapter.items():
            for i, hit in enumerate(hits):
                writer.writerow([adapter, f"pii-{i:04d}", hit, 2, hit, 1, 0])
    return path


def test_a_clear_win_excludes_zero_and_an_identical_adapter_is_a_tie(tmp_path):
    path = _csv(tmp_path, {"json": [0] * 20, "good": [1] * 20, "same": [0] * 20})
    got = {adapter: rest for adapter, *rest in compare(path, "json", resamples=200)}
    assert got["good"] == [1.0, 0.0, 1.0, 1.0]
    assert got["same"] == [0.0, 0.0, 0.0, 0.0]


def test_a_one_case_lead_on_twenty_is_a_tie(tmp_path, capsys):
    path = _csv(tmp_path, {"json": [0] * 20, "lucky": [1] + [0] * 19})
    assert main([str(path), "--resamples", "200"]) == 0
    assert "| tie |" in capsys.readouterr().out


def test_a_missing_baseline_names_the_adapters_present(tmp_path):
    with pytest.raises(SystemExit, match="good"):
        list(compare(_csv(tmp_path, {"good": [1]}), "json"))
