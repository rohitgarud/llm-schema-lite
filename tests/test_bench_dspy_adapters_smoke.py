"""Offline smoke test for the DSPy adapter benchmark package.

This module imports **only** from ``benchmarking.dspy_adapters`` (plus stdlib, pytest,
pydantic and dspy exception types). It never imports the shared dspy test helpers in this
directory, and it never imports the benchmark package's ``config`` or ``cli`` modules --
so "the collected test suite constructs no ``dspy.LM``" is a structural property of the
import graph rather than a matter of discipline. It does import ``.encoding``, but the
seeding tests pass an injected ``dict`` as ``environ``, so nothing here reads or writes
the real ``os.environ``. Token counting is the one operation that can want a network:
``tiktoken`` fetches ``cl100k_base`` once on a cold cache. Nothing here requires that
fetch to succeed -- ``test_offline_arm_degrades_when_encoding_is_unavailable`` pins the
degraded contract.

``DummyLM`` reports every token count as ``0``, so every live-arm assertion here is about
**structure and classification**, never about token magnitudes. The exact token
magnitudes come from the offline ``adapter.format()`` arm, which is deterministic.

Budget: this file must stay under ~2 s wall time, because ``.pre-commit-config.yaml``
runs a local ``pytest -x`` hook on every commit.
"""

from __future__ import annotations

import csv
import dataclasses
import re
from collections.abc import Callable, Iterator
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
from benchmarking.dspy_adapters import encoding as encoding_module  # noqa: E402
from benchmarking.dspy_adapters import runner as runner_module  # noqa: E402
from benchmarking.dspy_adapters.fakes import (  # noqa: E402
    Issue1871LM,
    JsonObjectOnlyLM,
    RawTextLM,
)
from benchmarking.dspy_adapters.outcomes import (  # noqa: E402
    parse_success_rate,
    validation_success_rate,
)
from benchmarking.dspy_adapters.provenance import (  # noqa: E402
    PROG,
    REDACTED,
    invocation_command,
    redact_lm_kwargs,
    redact_url_userinfo,
)
from benchmarking.dspy_adapters.report import (  # noqa: E402
    LIVE_AGGREGATE_COLUMNS,
    LIVE_CSV_HEADER,
    PROMPT_COST_CSV_HEADER,
    RunMeta,
)


class _Tiny(dspy.Signature):
    """Answer."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


_RESULTS_DIR: Path = Path(runner_module.__file__).resolve().parent / "results"
"""The committed results directory, derived from an already-imported module's path.

Deliberately **not** `cli.DEFAULT_OUT_DIR`: this file's module contract forbids importing
`cli`. `runner` is already imported above, and `results/` is its package sibling.
"""


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


def _trial_row(outcome: Outcome, **overrides: Any) -> TrialRow:
    """Build a synthetic TrialRow carrying `outcome`, for the pure rate/render tests.

    Every other field gets a neutral placeholder; `overrides` replaces any of them. Used
    only by the rate tests, which are about the outcome taxonomy and nothing else.
    """
    fields: dict[str, Any] = {
        "adapter": "json",
        "adapter_config": "cfg",
        "signature": "flat",
        "trial": 1,
        "outcome": outcome,
        "error_class": "",
        "wall_s": 0.0,
        "lm_calls": 1,
        "fallback_suspected": False,
        "response_format_sent": "none",
        "total_tokens": None,
        "prompt_tokens_reported": None,
        "completion_tokens_reported": None,
    }
    fields.update(overrides)
    return TrialRow(**fields)


@pytest.fixture
def cold_encoding_memo() -> Iterator[None]:
    """Clear `runner._get_encoding`'s lru_cache around one test.

    Cleared *before* so a patched failure is actually observed rather than short-
    circuited by a warm memo, and *after* so the real encoding is restored for this
    file's other tests -- the memo is process-global and xdist workers run
    sequentially within a process.
    """
    runner_module._get_encoding.cache_clear()
    try:
        yield
    finally:
        runner_module._get_encoding.cache_clear()


def test_benchmarking_package_imports() -> None:
    """The `benchmarking.dspy_adapters` import mechanism works under pytest."""
    assert len(ADAPTERS) == 10
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
    """run_offline_arm() returns exactly 60 unique PromptRows (10 adapters x 6 sigs)."""
    rows = run_offline_arm()
    assert len(rows) == 60
    assert all(isinstance(row, PromptRow) for row in rows)
    assert len({(row.adapter, row.signature) for row in rows}) == 60


def test_offline_arm_reports_positive_tokens() -> None:
    """Every ok offline row has positive chars/tokens and exactly two messages."""
    if not runner_module.encoding_available():
        pytest.skip("cl100k_base unavailable: no tiktoken cache reachable offline")
    for row in run_offline_arm():
        if row.outcome is Outcome.OK:
            assert row.prompt_tokens is not None and row.prompt_tokens > 0
            assert row.prompt_chars is not None and row.prompt_chars > 0
            assert row.n_messages == 2


def test_offline_arm_degrades_when_encoding_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, cold_encoding_memo: None
) -> None:
    """A cold cache with no network degrades `prompt_tokens` to None, not to 0.

    `ConnectionError` is the builtin (an `OSError` subclass), which is exactly what
    `_get_encoding` catches -- no `requests` import needed to prove the contract.
    """

    def _raise(name: str) -> None:
        raise ConnectionError(f"no network for {name}")

    monkeypatch.setattr(runner_module.tiktoken, "get_encoding", _raise)

    (row,) = runner_module.run_offline_arm(adapter_ids=["json"], signature_ids=["flat"])

    assert row.outcome is Outcome.OK
    assert row.n_messages == 2
    assert row.prompt_chars is not None and row.prompt_chars > 0
    assert row.prompt_tokens is None
    assert runner_module.encoding_available() is False


def test_offline_report_renders_unavailable_tokens_as_dash(tmp_path: Path) -> None:
    """An unavailable token count renders as an empty CSV cell and a markdown em-dash."""
    row = PromptRow(
        adapter="json",
        adapter_config="cfg",
        signature="flat",
        n_messages=2,
        prompt_chars=123,
        prompt_tokens=None,
        outcome=Outcome.OK,
        error_class="",
    )
    md_path, csv_path = write_offline_report([row], tmp_path)

    (record,) = csv.DictReader(csv_path.read_text().splitlines())
    assert record["prompt_tokens"] == ""
    assert record["prompt_chars"] == "123"

    md_lines = md_path.read_text().splitlines()
    assert "| json | — | — | — | — | — | — |" in md_lines
    assert "| json | cfg | flat | 2 | 123 | — | ok |" in md_lines


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


def test_repro_1871_error_vector_reports_response_format_sent() -> None:
    """The error half reports what was sent, not a hardcoded "none" (AC-1)."""
    rows = [r for r in run_repro_1871_offline() if r.part == "error"]
    assert {r.adapter: r.response_format_sent for r in rows} == {
        "chat": "none",
        "json": "json_object",
        "baml": "json_object",
        "sola-json-sections": "json_object",
        "sola-jsonish-sections": "json_object",
        "sola-yaml-sections": "none",
        "sola-jsonish-nojsonobject": "none",
    }


def test_response_format_sent_needs_no_fake_only_attribute() -> None:
    """An LM that carries no recording attribute still reports response_format_sent."""
    adapter = dspy.JSONAdapter()
    lm = JsonObjectOnlyLM([{"answer": "blue", "confidence": "0.9"}], adapter=adapter)
    row = runner_module.run_one_trial(adapter, "json", "cfg", "flat", 1, lm)
    assert row.response_format_sent == "json_object"
    assert not hasattr(lm, "lm_kwargs_history")


def test_raw_text_lm_still_reports_a_response_format() -> None:
    """An LM with no recording attribute at all still reports response_format_sent."""
    rows = run_live_arm(
        lambda a: RawTextLM(["garbage"], adapter=a),
        adapter_ids=["chat"],
        signature_ids=["flat"],
        disable_cache=False,
    )
    (row,) = rows
    assert row.outcome is Outcome.PARSE_ERROR
    assert row.lm_calls == 1
    assert row.response_format_sent == "none"


def test_format_error_cell_reports_no_response_format() -> None:
    """A format_error cell (no LM call attempted) reports response_format_sent="none"."""
    rows = run_live_arm(
        _seeded_lm_factory([{"answer": "blue", "confidence": "0.9"}]),
        adapter_ids=["baml"],
        signature_ids=["recursive"],
        disable_cache=False,
    )
    (row,) = rows
    assert row.outcome is Outcome.FORMAT_ERROR
    assert row.lm_calls == 0
    assert row.response_format_sent == "none"


def test_response_format_recorder_records_the_last_attempt() -> None:
    """ResponseFormatRecorder tracks every attempt and reports the last one."""
    rec = runner_module.ResponseFormatRecorder()
    assert rec.last() == "none"
    rec.on_lm_start(
        call_id="c1", instance=None, inputs={"kwargs": {"response_format": {"type": "json_object"}}}
    )
    assert rec.last() == "json_object"
    rec.on_lm_start(call_id="c2", instance=None, inputs={"kwargs": {"response_format": None}})
    assert rec.last() == "none"
    assert len(rec.formats) == 2


def test_probe_1871_reports_the_rejected_format() -> None:
    """A rejected first call still reports the format it sent, not a hardcoded literal."""
    lm = Issue1871LM([{"answer": "blue"}, {"answer": "blue"}], adapter=dspy.ChatAdapter())
    row = runner_module.probe_1871_live(lm)
    assert row.outcome is Outcome.TRANSPORT_ERROR
    assert row.error_class == "LMInvalidRequestError"
    assert row.response_format_sent == "json_object"
    assert row.lm_calls == 0
    assert row.part == "live_probe"


def test_probe_1871_reports_json_schema_when_both_accepted() -> None:
    """When both calls succeed, the row reports the second call's format, not the first."""
    lm = JsonObjectOnlyLM([{"answer": "blue"}, {"answer": "blue"}], adapter=dspy.ChatAdapter())
    row = runner_module.probe_1871_live(lm)
    assert row.outcome is Outcome.OK
    assert row.response_format_sent == "json_schema"
    assert row.note.startswith("not_reproducible")
    assert row.lm_calls == 2


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


def test_rates_are_none_when_nothing_was_attempted() -> None:
    """The C1 regression test: an unattempted cell's rates are None, not 0.0 or 1.0."""
    fmt = [_trial_row(Outcome.FORMAT_ERROR)]
    assert parse_success_rate(fmt) is None
    assert validation_success_rate(fmt) is None

    transport = [_trial_row(Outcome.TRANSPORT_ERROR)]
    assert parse_success_rate(transport) is None
    assert validation_success_rate(transport) is None


def test_rates_exclude_unattempted_rows_from_the_denominator() -> None:
    """An unattempted row is dropped from the denominator, not counted as a failure."""
    rows = [_trial_row(Outcome.OK), _trial_row(Outcome.TRANSPORT_ERROR)]
    assert parse_success_rate(rows) == 1.0
    assert validation_success_rate(rows) == 1.0

    rows = [_trial_row(Outcome.OK), _trial_row(Outcome.PARSE_ERROR)]
    assert parse_success_rate(rows) == 0.5
    assert validation_success_rate(rows) == 0.5


def test_validation_error_counts_as_a_parse_success() -> None:
    """A reply that parsed but failed field validation is a parse success, not a full one."""
    rows = [_trial_row(Outcome.VALIDATION_ERROR)]
    assert parse_success_rate(rows) == 1.0
    assert validation_success_rate(rows) == 0.0


def test_other_error_is_not_a_parse_success() -> None:
    """Pins the positive-list numerator: the old `1 - bad/total` form returns 1.0 here."""
    rows = [_trial_row(Outcome.OTHER_ERROR)]
    assert parse_success_rate(rows) == 0.0
    assert validation_success_rate(rows) == 0.0


def test_empty_row_list_rates_are_none() -> None:
    """An empty input list has no attempted rows, so both rates are None."""
    assert parse_success_rate([]) is None
    assert validation_success_rate([]) is None


def test_live_aggregate_renders_rate_columns(tmp_path: Path) -> None:
    """AC-2: the live markdown aggregate table carries a parse-rate and validation-rate cell."""
    rows = [_trial_row(Outcome.OK), _trial_row(Outcome.PARSE_ERROR)]
    md_path, _ = write_live_report(rows, tmp_path)
    md_lines = md_path.read_text().splitlines()
    assert LIVE_AGGREGATE_COLUMNS in md_lines
    assert "parse rate" in LIVE_AGGREGATE_COLUMNS and "validation rate" in LIVE_AGGREGATE_COLUMNS
    assert any("| 0.50 | 0.50 |" in line for line in md_lines)


def test_live_aggregate_renders_unattempted_rates_as_dash(tmp_path: Path) -> None:
    """A cell that attempted nothing renders both rate cells as the undefined em dash."""
    rows = [_trial_row(Outcome.FORMAT_ERROR)]
    md_path, _ = write_live_report(rows, tmp_path)
    assert any("| — | — |" in line for line in md_path.read_text().splitlines())


def test_live_aggregate_separator_matches_header_width(tmp_path: Path) -> None:
    """AC-2 guard: the separator row's `---` count always tracks the header's column count."""
    expected = len(LIVE_AGGREGATE_COLUMNS.strip("|").split("|"))
    assert expected == 15
    rows = [_trial_row(Outcome.OK), _trial_row(Outcome.PARSE_ERROR)]
    md_path, _ = write_live_report(rows, tmp_path)
    md_lines = md_path.read_text().splitlines()
    header_index = md_lines.index(LIVE_AGGREGATE_COLUMNS)
    sep = md_lines[header_index + 1]
    assert sep.count("---") == expected


def test_seed_tiktoken_cache_finds_a_bundled_blob_without_network(tmp_path: Path) -> None:
    """A candidate holding the sha1-named blob is adopted, pinning both cache variables."""
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / encoding_module.cache_key()).write_bytes(b"not-a-real-bpe-table")
    empty = tmp_path / "empty"
    empty.mkdir()

    environ = {"TIKTOKEN_CACHE_DIR": str(empty)}
    result = encoding_module.seed_tiktoken_cache(environ=environ, candidates=[bundled])

    assert result == str(bundled)
    assert environ["TIKTOKEN_CACHE_DIR"] == str(bundled)
    assert environ["CUSTOM_TIKTOKEN_CACHE_DIR"] == str(bundled)


def test_seed_tiktoken_cache_leaves_a_working_cache_alone(tmp_path: Path) -> None:
    """A cache that already holds the blob is never overridden -- no-override, idempotent."""
    warm = tmp_path / "warm"
    other = tmp_path / "other"
    for directory in (warm, other):
        directory.mkdir()
        (directory / encoding_module.cache_key()).write_bytes(b"not-a-real-bpe-table")

    environ = {"TIKTOKEN_CACHE_DIR": str(warm)}

    assert encoding_module.seed_tiktoken_cache(environ=environ, candidates=[other]) == str(warm)
    assert environ == {"TIKTOKEN_CACHE_DIR": str(warm)}


def test_encoding_name_constants_agree() -> None:
    """`runner.py` keeps its own `cl100k_base` literal; this pins it to `encoding.py`'s."""
    assert runner_module.ENCODING_NAME == encoding_module.ENCODING_NAME


def test_invocation_command_drops_the_interpreter_path() -> None:
    """`argv[0]` (an absolute, machine-specific `__main__.py` path) is never in the output."""
    argv = ["/home/u/repo/benchmarking/dspy_adapters/__main__.py", "--offline"]
    assert invocation_command(argv) == "python -m benchmarking.dspy_adapters --offline"


def test_invocation_command_quotes_arguments() -> None:
    """An argument containing a comma-space is shell-quoted so it round-trips."""
    p = "/abs/__main__.py"
    assert invocation_command([p, "--adapters", "json, baml"]) == f"{PROG} --adapters 'json, baml'"


def test_invocation_command_with_no_arguments() -> None:
    """No trailing space when there are no arguments to append."""
    assert invocation_command(["/abs/__main__.py"]) == PROG


def test_redacts_credential_shaped_keys() -> None:
    """Every credential-shaped key is redacted; the key itself survives verbatim."""
    for key in (
        "api_key",
        "openai_api_key",
        "Authorization",
        "azure_ad_token",
        "hf-token",
        "client_secret",
        "X-Api-Key",
        "aws_session_token",
        "password",
    ):
        result = redact_lm_kwargs({key: "s3cr3t"})
        assert result[key] == REDACTED, key
        assert "s3cr3t" not in repr(result), key


def test_keeps_legitimate_lm_kwargs() -> None:
    """Non-credential kwargs, including the `max_tokens` floor, are never redacted."""
    kwargs = {
        "max_tokens": 900,
        "temperature": 0.0,
        "cache": False,
        "num_retries": 0,
        "seed": 7,
        "top_p": 0.9,
        "model": "openai/qwen3:8b",
    }
    assert redact_lm_kwargs(kwargs) == kwargs
    assert redact_lm_kwargs(kwargs)["max_tokens"] == 900


def test_redacts_nested_and_listed_secrets() -> None:
    """Redaction recurses into nested mappings and into list elements."""
    raw = {
        "extra_headers": {"Authorization": "Bearer s3cr3t"},
        "providers": [{"api_key": "k-live"}],
    }
    result = redact_lm_kwargs(raw)
    rendered = repr(result)
    assert "s3cr3t" not in rendered
    assert "k-live" not in rendered
    assert "extra_headers" in result and "providers" in result
    assert rendered.count(REDACTED) == 2


def test_redact_url_userinfo() -> None:
    """Userinfo is redacted; a userinfo-free URL and `None` pass through unchanged."""
    assert redact_url_userinfo("https://u:p@host/v1") == f"https://{REDACTED}@host/v1"
    assert redact_url_userinfo("http://localhost:11434/v1") == "http://localhost:11434/v1"
    assert redact_url_userinfo(None) is None


def test_redaction_is_idempotent() -> None:
    """Redacting an already-redacted mapping or URL is a no-op."""
    raw = {"api_key": "sk-live", "max_tokens": 900, "extra_headers": {"Authorization": "Bearer x"}}
    once = redact_lm_kwargs(raw)
    assert redact_lm_kwargs(once) == once
    assert redact_url_userinfo(redact_url_userinfo("https://u:p@host/v1")) == redact_url_userinfo(
        "https://u:p@host/v1"
    )


def test_live_report_contains_no_credential_value(tmp_path: Path) -> None:
    """AC-3, end to end: redaction lives in RunMeta.__post_init__, not in cli.py."""
    meta = dataclasses.replace(
        RunMeta.minimal("live"),
        model="openai/qwen3:8b",
        api_base="https://u:pw@host/v1",
        lm_kwargs={"api_key": "sk-live-SECRET", "max_tokens": 900},
    )
    md_path, _ = write_live_report([_trial_row(Outcome.OK)], tmp_path, meta=meta)
    md = md_path.read_text()
    assert "sk-live-SECRET" not in md
    assert "pw@" not in md
    assert REDACTED in md
    assert "max_tokens" in md
    assert "900" in md


def test_committed_results_are_provenance_clean() -> None:
    """AC-3, as committed: no machine path and no credential in results/.

    Content-based rather than a golden file, so it is machine-independent and survives
    every regeneration. It asserts the property the ticket cares about, not the bytes a
    particular run happened to produce.
    """
    if not _RESULTS_DIR.is_dir():
        pytest.skip("results/ absent (sdist)")
    paths = sorted(_RESULTS_DIR.glob("*.md")) + sorted(_RESULTS_DIR.glob("*.csv"))
    assert paths  # a glob finding nothing must fail, not pass vacuously
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text, path
        assert not re.search(r"^command: /", text, re.MULTILINE), path
        assert not re.search(r"(?i)\b(api_key|secret|authorization)\b\s*[:=]", text), path


def test_trials_get_independent_seeds() -> None:
    """Each trial of a cell must be an independent sample, not the same request N times.

    The live arm shares one dspy.LM across every cell, so a fixed `seed` would make all N
    trials of a cell byte-identical - the cell then reads 0/N or N/N with stddev 0.000,
    which looks like confidence but is an artefact of replaying one call. The seed must
    advance per trial and must not compound across cells.
    """
    from benchmarking.dspy_adapters import runner as runner_module

    class _SeededLM:
        def __init__(self) -> None:
            self.kwargs = {"seed": 7}

    lm = _SeededLM()
    seen: list[tuple[str, str, int, int]] = []

    def _spy(adapter, adapter_id, adapter_config, sig_id, trial, lm):  # noqa: ANN001
        seen.append((adapter_id, sig_id, trial, lm.kwargs["seed"]))
        return None

    original = runner_module.run_one_trial
    runner_module.run_one_trial = _spy
    try:
        runner_module.run_live_arm(
            lambda _adapter: lm,
            adapter_ids=["json", "baml"],
            signature_ids=["flat", "nested"],
            trials=3,
            disable_cache=False,
        )
    finally:
        runner_module.run_one_trial = original

    per_cell: dict[tuple[str, str], list[int]] = {}
    for adapter_id, sig_id, _trial, seed in seen:
        per_cell.setdefault((adapter_id, sig_id), []).append(seed)
    assert per_cell, "run_live_arm produced no cells"
    for cell, seeds in per_cell.items():
        assert seeds == [7, 8, 9], (cell, seeds)
