"""Runners for the two DSPy adapter benchmark arms, plus the #1871 reproduction.

Two measurement arms live here, and they are never merged (see `outcomes.py`):

- The offline **prompt-cost** arm (`run_offline_arm` / `measure_prompt`) renders
  `adapter.format(signature, [], inputs)` for every (adapter, signature) cell and counts
  messages/characters/`tiktoken` tokens. It is exact, deterministic, model-free, and
  sub-second.
- The live **outcomes** arm (`run_live_arm` / `run_one_trial`) drives real
  `dspy.Predict` calls against a caller-supplied LM and classifies what happens.

R6 is enforced structurally, not just by discipline: `run_offline_arm` (and everything it
calls) never constructs a `dspy.LM`, this module never imports `.config` or `.cli`, and it
takes nothing from `.encoding` but the `ENCODING_NAME` constant (that module reads the
environment only inside a function body) -- so "the collected test suite constructs no
`dspy.LM` and reads no environment variable" is a fact about the import graph, not a
promise about behaviour.
The one outbound call this module can still provoke is `tiktoken`'s one-time fetch of the
`cl100k_base` table on a cold cache; `_get_encoding` treats that fetch as optional and
returns `None` when it cannot be satisfied, so the arm never fails for want of a network.

Δ2 construction rule: `run_live_arm` takes an `lm_factory: Callable[[Adapter], BaseLM]`
rather than a single shared `lm`, because a `DummyLM`-family fake renders its canned answer
*through the adapter passed to its own constructor* -- a fake seeded with the wrong adapter
manufactures false `parse_error`s. The factory lets each cell build its own
adapter-seeded LM.

Anti-masking (R9) enforced here: `run_one_trial` isolates every call with
`dspy.context(lm=..., adapter=..., track_usage=True)`, never global
`dspy.settings.configure`; it records the raw LM-history-length delta as `lm_calls` and
flags `fallback_suspected` whenever `lm_calls >= 2` regardless of the classified outcome;
and `run_live_arm` disables dspy's cache once, up front, via `dspy.configure_cache(...)`
when `disable_cache` is `True` (the smoke test passes `False` so it never mutates global
dspy cache state).

Also here: `run_repro_1871_offline`, a synthetic, fully offline reproduction of DSPy issue
#1871 (ticket AC-4) built from the adapter-seeded fakes in `.fakes`, and `probe_1871_live`,
the single live probe row -- called only by `cli.py`, only when the live environment is
configured.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, NamedTuple

import dspy
import tiktoken
from dspy.adapters.base import Adapter
from dspy.clients.base_lm import BaseLM
from dspy.utils.callback import BaseCallback

from .accuracy import AccuracyRow, score
from .adapters import (
    ADAPTERS,
    LIVE_DEFAULT_ADAPTER_IDS,
    REPRO_1871_ADAPTER_IDS,
    REPRO_1871_ADAPTERS,
)
from .cases import Case
from .encoding import ENCODING_NAME
from .fakes import JSON_OBJECT_RESPONSE_FORMAT, Issue1871LM, JsonObjectOnlyLM
from .outcomes import Outcome, PromptRow, ReproRow, TrialRow, classify
from .signatures import SIGNATURE_IDS, SIGNATURES

_REPRO_1871_ANSWER: dict[str, Any] = {"answer": "blue", "confidence": 0.9}
"""Canned Flat-signature answer that satisfies every output mode used in the matrix."""


@functools.lru_cache(maxsize=1)
def _get_encoding() -> tiktoken.Encoding | None:
    """Return the memoised `cl100k_base` encoding, or `None` if it cannot be loaded.

    `tiktoken.get_encoding` downloads the BPE table on a cold cache. Offline that
    download raises, and this arm treats the token count as optional rather than
    failing: the caller writes `prompt_tokens=None`, which `report.py` already renders
    as `—` in both markdown tables and as an empty CSV cell. Both outcomes are memoised,
    so a failure is attempted once per process, not 54 times.

    Tests reset the memo with `_get_encoding.cache_clear()`.
    """
    try:
        return tiktoken.get_encoding(ENCODING_NAME)
    except OSError:
        # `requests.exceptions.RequestException` subclasses `OSError`, so this single
        # catch covers every offline failure tiktoken/load.py can raise (ProxyError,
        # ConnectionError, Timeout, SSLError, HTTPError, and the cache-write re-raise).
        # Do not widen it, and do not `import requests` to name them explicitly.
        return None


def encoding_available() -> bool:
    """Whether `cl100k_base` could be loaded. Memoised; never retries a failed load."""
    return _get_encoding() is not None


def normalize_response_format(value: Any) -> str:
    """Map a recorded `response_format` kwarg onto "none" | "json_object" | "json_schema".

    `None` (and any other falsy value, e.g. an absent key defaulting to `None`) -> "none".
    `{"type": "json_object"}` -> "json_object". A Pydantic model class, or
    `{"type": "json_schema", ...}`, -> "json_schema". Imitates
    `tests/dspy_helpers.py::recorded_response_format`'s callers; does not import it.
    """
    if not value:
        return "none"
    if isinstance(value, dict):
        return "json_object" if value.get("type") == "json_object" else "json_schema"
    return "json_schema"


class ResponseFormatRecorder(BaseCallback):  # type: ignore[misc]
    """Record the normalized `response_format` of every LM call *attempted* in one cell.

    Answers a pre-call question with a pre-call fact. `BaseLM.__call__` is decorated
    `@with_callbacks`, whose `sync_wrapper` runs the start handlers *before* `fn(...)`
    (`dspy/utils/callback.py:381`, `:390`), so this fires even for a call that goes on to
    raise -- which `lm.history` never records, because `BaseLM.update_history` appends only
    after `forward()` returns. `inputs["kwargs"]` carries the per-call kwargs the adapter
    passed, because `__call__`'s signature ends in `**kwargs` and
    `inspect.getcallargs` bundles them there (`callback.py:314`).

    One fresh instance per cell, created by the function that opens the cell's
    `dspy.context`. Never a module-level singleton: an object whose lifetime is the `with`
    block cannot leak, whereas a shared object with a reset can have its reset forgotten.
    That is also the whole thread-safety story -- `dspy.context` overrides live in a
    `contextvars.ContextVar` (`settings.py:47`, `:250-257`), so a future parallelised
    `run_live_arm` keeps per-worker isolation for free. No lock is needed and none is added.

    Stores only the normalized three-value string, never the raw kwargs: the recorder must
    never become a second place a credential can be observed, and `on_lm_start` must stay a
    single cheap append because `with_callbacks` swallows callback exceptions and only logs
    a warning (`callback.py:322-323`).
    """

    def __init__(self) -> None:
        """Start with an empty attempt list."""
        self.formats: list[str] = []

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        """Append this attempt's normalized `response_format`. Never raises."""
        kwargs = inputs.get("kwargs") or {}
        self.formats.append(normalize_response_format(kwargs.get("response_format")))

    def last(self) -> str:
        """The last attempted call's response_format; "none" when no call was attempted.

        Means exactly what `lm.history[-1]["kwargs"]` meant -- *the last thing actually
        sent* -- so no table, CSV column or README description changes meaning. The
        structured-output -> json_object downgrade issues two calls, which
        `fallback_suspected = lm_calls >= 2` already flags with a dagger; the attempt count
        is available here as a side effect and is deliberately not exposed.
        """
        return self.formats[-1] if self.formats else "none"


class _Observed(NamedTuple):
    """What one `_observe`d call produced, and what the harness measured around it."""

    result: Any  # the call's return value; None when it raised
    exc: BaseException | None
    outcome: Outcome
    error_class: str
    wall_s: float
    lm_calls: int
    response_format_sent: str
    tokens: tuple[int | None, int | None, int | None]  # total, prompt, completion


def _observe(lm: BaseLM, call: Callable[[], Any], /, **context: Any) -> _Observed:
    """Run `call` under `dspy.context(**context)` and classify it. Never raises.

    The one isolation contract every LM-backed cell shares (R9): a fresh
    `ResponseFormatRecorder` as the context's callback list (this harness sets no global
    callbacks, so replacing that list loses nothing), `lm_calls` as `lm`'s history-length
    delta across the call, wall time, `classify(exc, lm_calls)`, and the provider-reported
    usage of the calls this cell added (`_reported_tokens`).
    """
    recorder = ResponseFormatRecorder()
    n0 = len(lm.history)
    t0 = time.perf_counter()
    result: Any = None
    exc: BaseException | None = None
    try:
        with dspy.context(**context, callbacks=[recorder]):
            result = call()
    except BaseException as caught:
        exc = caught
    wall_s = time.perf_counter() - t0
    lm_calls = len(lm.history) - n0
    outcome, error_class = classify(exc, lm_calls)
    return _Observed(
        result,
        exc,
        outcome,
        error_class,
        wall_s,
        lm_calls,
        recorder.last(),
        _reported_tokens(lm, n0, lm_calls),
    )


def measure_prompt(adapter: Adapter, cell_id: str, sig_id: str) -> PromptRow:
    """Measure one offline prompt-cost cell via `adapter.format()`. Never constructs an LM.

    `adapter_config` is read from the `ADAPTERS` registry's `config_repr` for `cell_id`. On
    a `format()`-time failure, `classify(exc, lm_calls=0)` decides the `Outcome` and the
    three measured fields (`n_messages`, `prompt_chars`, `prompt_tokens`) are `None`.
    """
    adapter_config = ADAPTERS[cell_id].config_repr
    sig_cell = SIGNATURES[sig_id]

    try:
        messages = adapter.format(sig_cell.signature, [], sig_cell.inputs)
    except BaseException as exc:
        outcome, error_class = classify(exc, lm_calls=0)
        return PromptRow(
            adapter=cell_id,
            adapter_config=adapter_config,
            signature=sig_id,
            n_messages=None,
            prompt_chars=None,
            prompt_tokens=None,
            outcome=outcome,
            error_class=error_class,
        )

    n_messages = len(messages)
    prompt_chars = sum(len(str(message["content"])) for message in messages)
    concatenated_content = "".join(str(message["content"]) for message in messages)
    encoding = _get_encoding()
    prompt_tokens = None if encoding is None else len(encoding.encode(concatenated_content))
    return PromptRow(
        adapter=cell_id,
        adapter_config=adapter_config,
        signature=sig_id,
        n_messages=n_messages,
        prompt_chars=prompt_chars,
        prompt_tokens=prompt_tokens,
        outcome=Outcome.OK,
        error_class="",
    )


def run_offline_arm(
    adapter_ids: list[str] | None = None,
    signature_ids: list[str] | None = None,
) -> list[PromptRow]:
    """Measure every (adapter, signature) prompt-cost cell. Never constructs a `dspy.LM`.

    Defaults to all 9 adapters x all 6 signatures = 54 rows, in `ADAPTERS` x `SIGNATURES`
    insertion order. Deterministic and sub-second. Needs no network: token counting degrades
    to `prompt_tokens=None` when `cl100k_base` cannot be loaded offline.
    """
    ids = list(ADAPTERS) if adapter_ids is None else adapter_ids
    sig_ids = list(SIGNATURE_IDS) if signature_ids is None else signature_ids

    rows: list[PromptRow] = []
    for adapter_id in ids:
        for sig_id in sig_ids:
            adapter = ADAPTERS[adapter_id].factory()
            rows.append(measure_prompt(adapter, adapter_id, sig_id))
    return rows


def run_live_arm(
    lm_factory: Callable[[Adapter], BaseLM],
    adapter_ids: list[str] | None = None,
    signature_ids: list[str] | None = None,
    trials: int = 1,
    disable_cache: bool = True,
) -> list[TrialRow]:
    """Run live `dspy.Predict` trials for every (adapter, signature, trial) cell.

    `lm_factory` receives the adapter instance under test and returns the LM for that cell
    (Δ2 construction rule: a fake LM must be seeded with the adapter under test). Defaults
    to `LIVE_DEFAULT_ADAPTER_IDS` x all 6 signatures x 1 trial. When `disable_cache` is
    `True`, `dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)` is
    called once, before the first cell; the smoke test passes `False` so it never mutates
    global dspy cache state.
    """
    if disable_cache:
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    ids = list(LIVE_DEFAULT_ADAPTER_IDS) if adapter_ids is None else adapter_ids
    sig_ids = list(SIGNATURE_IDS) if signature_ids is None else signature_ids

    rows: list[TrialRow] = []
    # Trial isolation: the live path shares one dspy.LM across every cell, so a fixed
    # `seed` makes all N trials of a cell the *same* deterministic request - N trials then
    # carry one trial's worth of information, and a cell reads 0/N or N/N with a stddev of
    # 0.000 that looks like confidence but is an artefact. Offset the seed per trial so a
    # trial is an independent sample. Captured once: the shared LM is mutated in place, so
    # re-reading its seed each cell would compound the offset.
    base_seed: int | None = None
    for adapter_id in ids:
        cell = ADAPTERS[adapter_id]
        for sig_id in sig_ids:
            for trial in range(1, trials + 1):
                adapter = cell.factory()
                lm = lm_factory(adapter)
                kwargs = getattr(lm, "kwargs", None)
                if isinstance(kwargs, dict) and isinstance(kwargs.get("seed"), int):
                    if base_seed is None:
                        base_seed = kwargs["seed"]
                    lm.kwargs = {**kwargs, "seed": base_seed + trial - 1}
                rows.append(run_one_trial(adapter, adapter_id, cell.config_repr, sig_id, trial, lm))
    return rows


def _reported_tokens(
    lm: BaseLM, n0: int, lm_calls: int
) -> tuple[int | None, int | None, int | None]:
    """Sum the provider-reported usage of the calls this cell added to `lm.history`.

    All three are `None` when no call was made, and when every recorded usage dict is
    empty -- a cache hit reports nothing, and summing that to `0` would publish a
    measured zero for a figure that was never measured.
    """
    if lm_calls == 0:
        return None, None, None
    usages = [entry["usage"] for entry in lm.history[n0:]]
    if all(not usage for usage in usages):
        return None, None, None
    return (
        sum(usage.get("total_tokens") or 0 for usage in usages),
        sum(usage.get("prompt_tokens") or 0 for usage in usages),
        sum(usage.get("completion_tokens") or 0 for usage in usages),
    )


def run_one_trial(
    adapter: Adapter,
    adapter_id: str,
    adapter_config: str,
    sig_id: str,
    trial: int,
    lm: BaseLM,
) -> TrialRow:
    """Run one live `dspy.Predict` trial, isolated via `dspy.context`, and classify it.

    Isolation is always via `dspy.context(lm=lm, adapter=adapter, track_usage=True)`, never
    global `dspy.settings.configure` (R9). `lm_calls` is the LM-history-length delta across
    the call; `lm_calls >= 2` sets `fallback_suspected=True` regardless of the classified
    outcome. The three token fields are `None` when `lm_calls == 0` or when every recorded
    usage dict is empty (a cache hit). `response_format_sent` is observed from the LM call
    itself via a scoped `ResponseFormatRecorder`, so a request the endpoint rejected still
    reports what was sent; `"none"` means no LM call was attempted.
    """
    sig_cell = SIGNATURES[sig_id]
    obs = _observe(
        lm,
        lambda: dspy.Predict(sig_cell.signature)(**sig_cell.inputs),
        lm=lm,
        adapter=adapter,
        track_usage=True,
    )
    total_tokens, prompt_tokens_reported, completion_tokens_reported = obs.tokens
    return TrialRow(
        adapter=adapter_id,
        adapter_config=adapter_config,
        signature=sig_id,
        trial=trial,
        outcome=obs.outcome,
        error_class=obs.error_class,
        wall_s=obs.wall_s,
        lm_calls=obs.lm_calls,
        fallback_suspected=obs.lm_calls >= 2,
        response_format_sent=obs.response_format_sent,
        total_tokens=total_tokens,
        prompt_tokens_reported=prompt_tokens_reported,
        completion_tokens_reported=completion_tokens_reported,
    )


def run_accuracy_arm(
    lm_factory: Callable[[Adapter], BaseLM],
    cases: list[Case],
    adapter_ids: list[str] | None = None,
    disable_cache: bool = True,
) -> list[AccuracyRow]:
    """Score every (adapter, case) cell against ground truth via each case's own signature.

    Same Δ2 construction rule as `run_live_arm`: `lm_factory` receives the adapter under
    test. No `trials` parameter and no seed offsetting - the outcomes arm needs those
    because it re-sends one identical request N times, whereas here every case is a
    different request, so variation comes from the corpus and a fixed `seed` is a feature
    (it makes the run reproducible) rather than the collapse-to-one-sample defect it is
    there.
    """
    if disable_cache:
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    ids = list(LIVE_DEFAULT_ADAPTER_IDS) if adapter_ids is None else adapter_ids

    rows: list[AccuracyRow] = []
    for adapter_id in ids:
        cell = ADAPTERS[adapter_id]
        for case in cases:
            adapter = cell.factory()
            rows.append(
                run_one_case(adapter, adapter_id, cell.config_repr, case, lm_factory(adapter))
            )
    return rows


def run_one_case(
    adapter: Adapter,
    adapter_id: str,
    adapter_config: str,
    case: Case,
    lm: BaseLM,
) -> AccuracyRow:
    """Run one extraction and score the record it produced against the case's ground truth.

    A cell that raised passes `produced=None` to `score`, which charges it 0 against the
    full expected denominator rather than dropping it - excluding failures would let an
    adapter that answers half the corpus outrank one that answers all of it imperfectly.
    Isolation, callbacks and classification are the same contract as `run_one_trial`.
    """

    def extract() -> Any:
        prediction = dspy.Predict(case.signature)(**{case.input_field: case.text})
        return getattr(prediction, case.output_field)

    obs = _observe(lm, extract, lm=lm, adapter=adapter, track_usage=True)
    expected, produced = case.expected, obs.result
    if case.align is not None:
        expected, produced = case.align(expected, produced)
    field_score = score(expected, produced)

    return AccuracyRow(
        adapter=adapter_id,
        adapter_config=adapter_config,
        case_id=case.case_id,
        outcome=obs.outcome,
        error_class=obs.error_class,
        wall_s=obs.wall_s,
        lm_calls=obs.lm_calls,
        response_format_sent=obs.response_format_sent,
        matched=field_score.matched,
        total=field_score.total,
        wrong=field_score.wrong,
        missing=field_score.missing,
        spurious=field_score.spurious,
        total_tokens=obs.tokens[0],
        recall_matched=field_score.recall_matched,
        recall_total=field_score.recall_total,
        invented=field_score.invented,
    )


def run_repro_1871_offline() -> list[ReproRow]:
    """Synthetic, fully offline reproduction of DSPy issue #1871 (ticket AC-4).

    Part 1 (`part="capability"`) drives each of `REPRO_1871_ADAPTER_IDS` through an
    adapter-seeded `JsonObjectOnlyLM` for its `response_format_sent` (this fake never
    raises, so every row is `ok`). Part 2 (`part="error"`) drives the same seven ids through
    an adapter-seeded `Issue1871LM`, which rejects `response_format={"type": "json_object"}`
    the way LM Studio does. Both parts go through `run_one_trial`, so
    `outcome`/`error_class`/`lm_calls` come from the exact same `classify()` contract as the
    live arm. 14 rows, part 1 then part 2, each in the id order of
    `REPRO_1871_ADAPTER_IDS`. No env, no network, no `dspy.LM`.
    """
    rows: list[ReproRow] = []
    for part, lm_class in (("capability", JsonObjectOnlyLM), ("error", Issue1871LM)):
        for adapter_id in REPRO_1871_ADAPTER_IDS:
            cell = REPRO_1871_ADAPTERS[adapter_id]
            adapter = cell.factory()
            lm = lm_class([dict(_REPRO_1871_ANSWER)], adapter=adapter)
            trial_row = run_one_trial(adapter, adapter_id, cell.config_repr, "flat", 1, lm)
            rows.append(
                ReproRow(
                    part=part,
                    adapter=adapter_id,
                    adapter_config=cell.config_repr,
                    response_format_sent=trial_row.response_format_sent,
                    outcome=trial_row.outcome,
                    error_class=trial_row.error_class,
                    lm_calls=trial_row.lm_calls,
                    note="",
                )
            )
    return rows


def probe_1871_live(lm: BaseLM) -> ReproRow:
    """Send a trivial 1-field payload twice through a real endpoint and record one row.

    First with `response_format={"type": "json_object"}`, then with a
    `{"type": "json_schema", ...}`-shaped payload. Called only by `cli.py`, only when the
    live environment is configured; not re-exported from `__init__.py`.

    When both succeed, `note` is exactly
    `not_reproducible: endpoint accepted both {"type":"json_object"} and
    {"type":"json_schema"}` and `outcome=Outcome.OK`. When one is rejected, `note` carries
    the `LMError`'s class and message and `outcome`/`error_class` come from `classify`.
    `response_format_sent` is the format of the last call attempted, so a rejected second
    call reports `json_schema`, not the first call's `json_object`.
    """

    def send_both() -> None:
        lm(
            messages=[{"role": "user", "content": "Reply with a JSON object."}],
            response_format=JSON_OBJECT_RESPONSE_FORMAT,
        )
        lm(
            messages=[{"role": "user", "content": "Reply with a JSON object."}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "probe",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                },
            },
        )

    obs = _observe(lm, send_both)
    note = (
        'not_reproducible: endpoint accepted both {"type":"json_object"} and {"type":"json_schema"}'
        if obs.exc is None
        else f"{type(obs.exc).__name__}: {obs.exc}"
    )
    return ReproRow(
        part="live_probe",
        adapter="probe_1871_live",
        adapter_config="probe_1871_live",
        response_format_sent=obs.response_format_sent,
        outcome=obs.outcome,
        error_class=obs.error_class,
        lm_calls=obs.lm_calls,
        note=note,
    )
