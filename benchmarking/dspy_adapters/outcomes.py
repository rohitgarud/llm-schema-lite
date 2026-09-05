"""Outcome taxonomy and row types shared by the offline and live benchmark arms.

The `except`-chain ordering in `classify` below is a **contract** and must be reproduced
literally, for two verified reasons: (1) `pydantic.ValidationError` (== `pydantic_core
.ValidationError`) **subclasses `ValueError`**, so it must be tested *before* the
`ValueError` arm or every Pydantic validation failure would be misclassified as a bare
`ValueError`; (2) an `LMError` raised inside `LM.forward` happens *before*
`BaseLM.update_history` appends to the LM's call history, so `lm_calls == 0` even though a
network round-trip was attempted — exception **type** must therefore decide before
`lm_calls` is ever consulted, or the entire DSPy issue #1871 reproduction gets mislabelled
`format_error` instead of `transport_error`.

`PromptRow` (offline prompt-cost arm) and `TrialRow` (live outcomes arm) are deliberately
**not** merged into one row type or one table: the two arms are not on the same accounting
basis (one measures `adapter.format()` token cost with no LM involved, the other measures
real `dspy.Predict` calls against a live LM), so joining them would manufacture a
comparison that does not exist.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import pydantic
from dspy.utils.exceptions import AdapterParseError, LMError


class Outcome(str, enum.Enum):
    """Outcome of one benchmark cell. `str` mixin so `.value` serialises straight to CSV."""

    OK = "ok"
    FORMAT_ERROR = "format_error"
    EMPTY_RESPONSE = "empty_response"
    PARSE_ERROR = "parse_error"
    VALIDATION_ERROR = "validation_error"
    TRANSPORT_ERROR = "transport_error"
    OTHER_ERROR = "other_error"


def classify(exc: BaseException | None, lm_calls: int) -> tuple[Outcome, str]:
    """Map an exception plus the LM-call delta onto an Outcome and its leaf class name.

    Pure: no LM, no I/O. `exc is None` -> (Outcome.OK, ""). The branch ORDER below is the
    contract (see module docstring) and must not be reordered or collapsed.
    """
    if exc is None:
        return Outcome.OK, ""
    if isinstance(exc, pydantic.ValidationError):  # 1. MUST precede ValueError
        return Outcome.VALIDATION_ERROR, type(exc).__name__
    if isinstance(exc, LMError):  # 2. type wins over lm_calls
        return Outcome.TRANSPORT_ERROR, type(exc).__name__
    if isinstance(exc, AdapterParseError):  # 3. sibling of LMError
        empty = not (getattr(exc, "lm_response", "") or "").strip()
        return (Outcome.EMPTY_RESPONSE if empty else Outcome.PARSE_ERROR), type(exc).__name__
    if isinstance(exc, ValueError):  # 4. format-time vs post-LM
        return (Outcome.FORMAT_ERROR if lm_calls == 0 else Outcome.VALIDATION_ERROR), type(
            exc
        ).__name__
    return (  # 5. honest catch-all
        Outcome.FORMAT_ERROR if lm_calls == 0 else Outcome.OTHER_ERROR
    ), type(exc).__name__


@dataclass(frozen=True)
class PromptRow:
    """One offline prompt-cost cell: adapter x signature, measured via adapter.format()."""

    adapter: str
    adapter_config: str
    signature: str
    n_messages: int | None
    prompt_chars: int | None
    prompt_tokens: int | None
    outcome: Outcome
    error_class: str


@dataclass(frozen=True)
class TrialRow:
    """One live outcomes trial: adapter x signature x trial, measured via dspy.Predict."""

    adapter: str
    adapter_config: str
    signature: str
    trial: int
    outcome: Outcome
    error_class: str
    wall_s: float
    lm_calls: int
    fallback_suspected: bool
    response_format_sent: str  # "none" | "json_object" | "json_schema"
    total_tokens: int | None
    prompt_tokens_reported: int | None
    completion_tokens_reported: int | None


@dataclass(frozen=True)
class ReproRow:
    """One #1871 reproduction cell — offline synthetic, or the single live probe."""

    part: str  # "capability" | "error" | "live_probe"
    adapter: str
    adapter_config: str
    response_format_sent: str  # "none" | "json_object" | "json_schema"
    outcome: Outcome
    error_class: str
    lm_calls: int
    note: str  # "" for offline rows; the live probe's finding text


def parse_success_rate(rows: list[TrialRow]) -> float:
    """1 - (parse_error + empty_response) / total; 0.0 for an empty input list."""
    if not rows:
        return 0.0
    bad = sum(1 for row in rows if row.outcome in (Outcome.PARSE_ERROR, Outcome.EMPTY_RESPONSE))
    return 1 - (bad / len(rows))


def validation_success_rate(rows: list[TrialRow]) -> float:
    """ok / total; 0.0 for an empty input list."""
    if not rows:
        return 0.0
    ok = sum(1 for row in rows if row.outcome is Outcome.OK)
    return ok / len(rows)


class UnknownCellError(ValueError):
    """Raised when --adapters/--signatures names an id that is not in the registry.

    Carries the pre-rendered stderr message; cli.main turns it into exit code 2.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
