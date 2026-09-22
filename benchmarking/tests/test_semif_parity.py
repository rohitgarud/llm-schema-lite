"""Offline checks for the SemIf parity benchmark's metric and comparison."""

from __future__ import annotations

import pytest

pytest.importorskip("dspy")

from benchmarking.semif.parity import compare, mean_family_balanced_accuracy, question  # noqa: E402

OPTIONS = [{"id": "yes", "description": "Y"}, {"id": "no", "description": "N"}]
ROWS = [
    {"id": "a", "family": "f1", "label": 0, "options": OPTIONS, "question": "q?", "state": "s"},
    {"id": "b", "family": "f1", "label": 0, "options": OPTIONS, "question": "q?", "state": "s"},
    {"id": "c", "family": "f1", "label": 1, "options": OPTIONS, "question": "q?", "state": "s"},
    {"id": "d", "family": "f2", "label": 1, "options": OPTIONS, "question": "q?", "state": "s"},
]


def test_balanced_accuracy_averages_label_recall_then_families() -> None:
    # f1: recall(yes) = 1/2, recall(no) = 1 -> 0.75; f2: recall(no) = 1 -> 1.0
    predicted = {"a": "yes", "b": "no", "c": "no", "d": "no"}
    assert mean_family_balanced_accuracy(ROWS, predicted) == pytest.approx(0.875)


def test_question_maps_options_to_choice_criteria() -> None:
    assert question(ROWS[0]) == {
        "type": "choice",
        "instructions": {"task": "", "question": "q?"},
        "criteria": {"yes": "Y", "no": "N"},
    }


def test_compare_reports_agreement_drift_and_prompt_hashes() -> None:
    reference = [
        {
            "id": r["id"],
            "option_ids": ["yes", "no"],
            "probabilities": [0.9, 0.1],
            "prompt_sha256": "h",
        }
        for r in ROWS
    ]
    ours = [
        {"id": r["id"], "probabilities": {"yes": 0.8, "no": 0.2}, "prompt_sha256": "h"}
        for r in ROWS[:3]
    ] + [{"id": "d", "error": "No option letter"}]

    out = compare(ROWS, ours, reference)

    assert out["scored"] == 3
    assert out["argmax_agreement"] == 1.0
    assert out["max_probability_drift_max"] == pytest.approx(0.1)
    assert out["prompt_hash_matches"] == 3
