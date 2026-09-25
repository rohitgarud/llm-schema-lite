"""Offline checks for the in-process SemIf runtime ablation's helpers."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("dspy")
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from benchmarking.semif.common import calibration  # noqa: E402
from benchmarking.semif.runtime import SelectedHead, bucket  # noqa: E402


@pytest.mark.parametrize(
    ("n", "expected"), [(1, 16), (16, 16), (17, 32), (147, 160), (647, 704), (3997, 4096)]
)
def test_bucket_rounds_up_with_at_most_an_eighth_of_padding(n: int, expected: int) -> None:
    assert bucket(n) == expected
    assert n <= bucket(n) < n + 16 or bucket(n) <= n * 1.125


def test_selected_head_matches_the_full_head_on_its_rows() -> None:
    head = torch.nn.Linear(8, 50, bias=True)
    rows = [3, 7, 11, 42]
    hidden = torch.randn(2, 5, 8)
    torch.testing.assert_close(SelectedHead(head, rows)(hidden), head(hidden)[..., rows])


def test_calibration_reports_gold_nll_and_top_choice_ece() -> None:
    rows = [{"id": "a", "label": 0}, {"id": "b", "label": 1}]
    probs = {"a": [0.9, 0.1], "b": [0.9, 0.1]}  # one right, one wrong, both at 0.9

    out = calibration(rows, probs)

    assert out["gold_nll"] == pytest.approx((-math.log(0.9) - math.log(0.1)) / 2)
    assert out["ece"] == pytest.approx(abs(0.9 - 0.5))


def test_quantizing_leaves_global_matmul_precision_alone() -> None:
    pytest.importorskip("torchao")
    if not torch.cuda.is_available():
        pytest.skip("quantized() places the copy on the GPU")
    from benchmarking.semif.runtime import quantized

    before = torch.get_float32_matmul_precision()
    quantized(torch.nn.Sequential(torch.nn.Linear(64, 64, dtype=torch.bfloat16)), "int8")
    assert torch.get_float32_matmul_precision() == before
