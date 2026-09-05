"""Offline smoke test for the DSPy adapter benchmark package.

This module imports **only** from ``benchmarking.dspy_adapters`` (plus stdlib, pytest,
pydantic and dspy exception types). It never imports the shared dspy test helpers in this
directory, and it never imports the benchmark package's ``config`` or ``cli`` modules --
so "the collected test suite touches no environment variable and no network" is a
structural property of the import graph rather than a matter of discipline.

``DummyLM`` reports every token count as ``0``, so every live-arm assertion here is about
**structure and classification**, never about token magnitudes. The exact token
magnitudes come from the offline ``adapter.format()`` arm, which is deterministic.

Budget: this file must stay under ~2 s wall time, because ``.pre-commit-config.yaml``
runs a local ``pytest -x`` hook on every commit.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("dspy", minversion="3.3.1")

import dspy  # noqa: E402
import pydantic  # noqa: E402
from dspy.adapters.base import Adapter  # noqa: E402
from dspy.clients.base_lm import BaseLM  # noqa: E402
from dspy.utils.dummies import DummyLM  # noqa: E402
from dspy.utils.exceptions import AdapterParseError, LMInvalidRequestError  # noqa: E402

from benchmarking.dspy_adapters import (  # noqa: E402
    ADAPTERS,
    SIGNATURES,
    Outcome,
    PromptRow,
    TrialRow,
    classify,
    run_live_arm,
    run_offline_arm,
    run_repro_1871_offline,
    write_live_report,
    write_offline_report,
)
from benchmarking.dspy_adapters.fakes import RawTextLM  # noqa: E402
from benchmarking.dspy_adapters.report import (  # noqa: E402
    LIVE_CSV_HEADER,
    PROMPT_COST_CSV_HEADER,
)


class _Tiny(dspy.Signature):
    """Answer."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _seeded_lm_factory(
    answers: list[dict[str, Any]],
) -> Callable[[Adapter], BaseLM]:
    """Return an lm_factory building DummyLM(answers, adapter=<the adapter under test>).

    DummyLM renders its canned answer through the adapter given to its constructor, so
    seeding with the default ChatAdapter() would feed YAML-mode adapters
    ChatAdapter-shaped text and manufacture false parse_errors.
    """

    def factory(adapter: Adapter) -> BaseLM:
        return DummyLM(list(answers), adapter=adapter)

    return factory


def _validation_error() -> pydantic.ValidationError:
    """Build a real pydantic.ValidationError (it subclasses ValueError)."""

    class _M(pydantic.BaseModel):
        count: int

    try:
        _M(count="not-a-number")  # type: ignore[arg-type]
    except pydantic.ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


def _parse_error(lm_response: str) -> AdapterParseError:
    """Build a real AdapterParseError; it needs a real signature object."""
    return AdapterParseError(
        adapter_name="TestAdapter",
        signature=_Tiny,
        lm_response=lm_response,
    )


def test_benchmarking_package_imports() -> None:
    """The `benchmarking.dspy_adapters` import mechanism works under pytest."""
    assert len(ADAPTERS) == 9
    assert len(SIGNATURES) == 6


def test_every_adapter_cell_constructs() -> None:
    """All nine adapter factories build an adapter and carry a non-empty config_repr."""
    for cell_id, cell in ADAPTERS.items():
        adapter = cell.factory()
        assert isinstance(adapter, Adapter), cell_id
        assert isinstance(cell.config_repr, str) and cell.config_repr, cell_id


def test_every_signature_cell_builds() -> None:
    """All six signatures expose fields, and their input dicts match input_fields."""
    for cell_id, cell in SIGNATURES.items():
        assert cell.signature.input_fields, cell_id
        assert cell.signature.output_fields, cell_id
        assert set(cell.inputs) == set(cell.signature.input_fields), cell_id


def test_offline_arm_covers_full_matrix() -> None:
    """run_offline_arm() returns exactly 54 unique PromptRows (9 adapters x 6 sigs)."""
    rows = run_offline_arm()
    assert len(rows) == 54
    assert all(isinstance(row, PromptRow) for row in rows)
    assert len({(row.adapter, row.signature) for row in rows}) == 54


def test_offline_arm_reports_positive_tokens() -> None:
    """Every ok offline row has positive chars/tokens and exactly two messages."""
    for row in run_offline_arm():
        if row.outcome is Outcome.OK:
            assert row.prompt_tokens is not None and row.prompt_tokens > 0
            assert row.prompt_chars is not None and row.prompt_chars > 0
            assert row.n_messages == 2


def test_baml_recursive_is_format_error() -> None:
    """BAMLAdapter refuses recursive models at format() time, before any LM call."""
    (row,) = [
        r
        for r in run_offline_arm(adapter_ids=["baml"], signature_ids=["recursive"])
        if r.adapter == "baml" and r.signature == "recursive"
    ]
    assert row.outcome is Outcome.FORMAT_ERROR
    assert row.error_class == "ValueError"
    assert row.prompt_tokens is None


def test_classify_orders_validation_before_value_error() -> None:
    """pydantic.ValidationError subclasses ValueError and must be matched first."""
    outcome, error_class = classify(_validation_error(), 1)
    assert outcome is Outcome.VALIDATION_ERROR
    assert error_class == "ValidationError"


def test_classify_lm_error_wins_over_zero_calls() -> None:
    """An LMError raised inside forward() has lm_calls == 0 but is NOT a format_error."""
    outcome, error_class = classify(LMInvalidRequestError("boom", status=400), 0)
    assert outcome is Outcome.TRANSPORT_ERROR
    assert error_class == "LMInvalidRequestError"


def test_classify_empty_adapter_parse_error() -> None:
    """An empty lm_response is empty_response; real text is parse_error."""
    assert classify(_parse_error(""), 1)[0] is Outcome.EMPTY_RESPONSE
    assert classify(_parse_error("some text"), 1)[0] is Outcome.PARSE_ERROR


def test_live_runner_ok_under_dummy_lm_per_mode() -> None:
    """The real per-mode parse paths execute against an adapter-seeded DummyLM."""
    rows = run_live_arm(
        _seeded_lm_factory([{"answer": "blue", "confidence": "0.9"}]),
        adapter_ids=[
            "sola-json-sections",
            "sola-jsonish-sections",
            "sola-yaml-sections",
        ],
        signature_ids=["flat"],
        disable_cache=False,
    )
    assert len(rows) == 3
    assert all(isinstance(row, TrialRow) for row in rows)
    for row in rows:
        assert row.outcome is Outcome.OK, (row.adapter, row.error_class)
        assert row.lm_calls == 1


def test_chat_adapter_fallback_is_disabled() -> None:
    """ChatAdapter must not silently retry through JSONAdapter and report a fake ok."""
    rows = run_live_arm(
        lambda adapter: RawTextLM(
            ["garbage", '{"answer":"masked-success","confidence":0.9}'],
            adapter=adapter,
        ),
        adapter_ids=["chat"],
        signature_ids=["flat"],
        disable_cache=False,
    )
    (row,) = rows
    assert row.outcome is Outcome.PARSE_ERROR
    assert row.lm_calls == 1
    assert row.fallback_suspected is False


def test_repro_1871_capability_vector() -> None:
    """Which adapters choose to send response_format={"type": "json_object"}."""
    expected = {
        "chat": "none",
        "json": "json_object",
        "baml": "json_object",
        "sola-json-sections": "json_object",
        "sola-jsonish-sections": "json_object",
        "sola-yaml-sections": "none",
        "sola-jsonish-nojsonobject": "none",
    }
    rows = [r for r in run_repro_1871_offline() if r.part == "capability"]
    assert {r.adapter: r.response_format_sent for r in rows} == expected


def test_repro_1871_error_vector() -> None:
    """An LM Studio-shaped 400 on json_object is observable, not masked."""
    expected = {
        "chat": (Outcome.OK, 1),
        "json": (Outcome.TRANSPORT_ERROR, 0),
        "baml": (Outcome.TRANSPORT_ERROR, 0),
        "sola-json-sections": (Outcome.TRANSPORT_ERROR, 0),
        "sola-jsonish-sections": (Outcome.TRANSPORT_ERROR, 0),
        "sola-yaml-sections": (Outcome.OK, 1),
        "sola-jsonish-nojsonobject": (Outcome.OK, 1),
    }
    rows = [r for r in run_repro_1871_offline() if r.part == "error"]
    assert {r.adapter: (r.outcome, r.lm_calls) for r in rows} == expected
    for row in rows:
        if row.outcome is Outcome.TRANSPORT_ERROR:
            assert row.error_class == "LMInvalidRequestError"


def test_reports_write_parseable_csv_and_markdown(tmp_path: Path) -> None:
    """Both writers emit the exact CSV headers and a markdown metric-integrity block."""
    offline_rows = run_offline_arm(adapter_ids=["json"], signature_ids=["flat"])
    md_path, csv_path = write_offline_report(offline_rows, tmp_path)
    csv_lines = csv_path.read_text().strip().splitlines()
    assert csv_lines[0] == ",".join(PROMPT_COST_CSV_HEADER)
    assert len(csv_lines) - 1 == len(offline_rows)
    offline_md = md_path.read_text()
    assert "Metric integrity" in offline_md
    assert "response_format" in offline_md

    live_rows = run_live_arm(
        _seeded_lm_factory([{"answer": "blue", "confidence": "0.9"}]),
        adapter_ids=["json"],
        signature_ids=["flat"],
        disable_cache=False,
    )
    live_md_path, live_csv_path = write_live_report(live_rows, tmp_path)
    live_csv_lines = live_csv_path.read_text().strip().splitlines()
    assert live_csv_lines[0] == ",".join(LIVE_CSV_HEADER)
    assert len(live_csv_lines) - 1 == len(live_rows)
    live_md = live_md_path.read_text()
    assert "Metric integrity" in live_md
    assert "response_format" in live_md


def test_report_never_overwrites(tmp_path: Path) -> None:
    """A same-day rerun appends -2 rather than clobbering the earlier file."""
    rows = run_offline_arm(adapter_ids=["json"], signature_ids=["flat"])
    first_md, first_csv = write_offline_report(rows, tmp_path)
    second_md, second_csv = write_offline_report(rows, tmp_path)
    assert first_md != second_md
    assert second_md.stem.endswith("-2")
    assert second_csv.stem.endswith("-2")
    assert first_csv.exists() and second_csv.exists()
