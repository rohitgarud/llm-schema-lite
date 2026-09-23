"""Offline checks for the JevBench runner's request mapping and scoring."""

from __future__ import annotations

import pytest

pytest.importorskip("dspy")

from benchmarking.jevbench.run import distribution, payload, summarize  # noqa: E402

NOUL = {
    "id": "n1",
    "tier": "hard",
    "state": "s",
    "labels": ["no", "yes"],
    "expected": "yes",
    "question": {"type": "noul", "instructions": "Is it?", "criteria": {"true": "Y", "false": "N"}},
    "provenance": {"gold_probs": {"no": 0.4, "yes": 0.6}},
}
SCORE = {
    "id": "s1",
    "tier": "standard",
    "state": "s",
    "labels": ["0", "1", "2"],
    "expected": 2,
    "question": {"type": "score", "instructions": "How bad?", "criteria": ["a", "b", "c"]},
    "provenance": {},
}


def test_payload_wraps_the_instructions_as_jev_adapter_would() -> None:
    assert payload(NOUL) == {
        "state": "s",
        "questions": {
            "q": {
                "type": "noul",
                "instructions": {"task": "", "question": "Is it?"},
                "criteria": {"true": "Y", "false": "N"},
            }
        },
    }


def test_distribution_maps_answers_onto_the_task_labels() -> None:
    assert distribution(NOUL, {"type": "noul", "noul": 0.7}) == {
        "yes": 0.7,
        "no": pytest.approx(0.3),
    }
    probs = {"0": 0.1, "1": 0.2, "2": 0.7}
    assert distribution(SCORE, {"type": "score", "probabilities": probs}) == probs


def test_summarize_scores_tiers_calibration_and_the_reference_on_the_same_items() -> None:
    ours = [
        {"id": "n1", "probs": {"no": 0.1, "yes": 0.9}, "seconds": 0.2},  # right, conf 0.9
        {"id": "s1", "probs": {"0": 0.5, "1": 0.3, "2": 0.2}, "seconds": 0.4},  # wrong
    ]
    reference = {"jev": {"n1": ["w", 0.1], "s1": ["c", 0.1]}}

    out = summarize([NOUL, SCORE], ours, reference)

    assert out["ours"]["accuracy"] == {"hard": 1.0, "standard": 0.0}
    assert out["jev"]["accuracy"] == {"hard": 0.0, "standard": 1.0}
    assert out["ours"]["hard_ece"] == pytest.approx(0.1)  # one item: |1.0 - 0.9|
    assert out["ours"]["hard_mean_tvd"] == pytest.approx(0.3)  # |0.9 - 0.6|
    assert out["ours"]["calibration"] == pytest.approx((80 + 70) / 2)  # JevBench v1.2 formula
    assert out["ours"]["p50_s"] == pytest.approx(0.3)
    assert out["jev"]["both_right"] == 0 and out["jev"]["only_ours_right"] == 1
