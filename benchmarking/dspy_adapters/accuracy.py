"""Field-level accuracy scoring for the accuracy arm.

The metric is the one the reference benchmarks report: **percentage of exact field matches
across all schema fields**, computed against ground truth rather than against the schema.
It is deliberately a different quantity from the outcomes arm's ``ok`` rate, and the two are
never combined into one score - a reply can be perfectly valid and entirely wrong.

Two denominator decisions carry the whole metric, and both follow the precedent set when the
parse/validation rates were fixed (ticket lsl-2026-09-05-015):

1. **A failure scores 0.0, it is not excluded.** :func:`score` accepts ``produced=None`` for
   a cell that raised, and returns ``matched=0`` against the full expected denominator.
   Dropping failures instead would let an adapter that answers half its cases outrank one
   that answers every case imperfectly - the exact "favourable fact about a cell that never
   got far enough to have it" defect that ticket fixed once already.
2. **The denominator is the EXPECTED fields, not the produced ones.** Scoring against
   produced fields would reward an adapter for emitting fewer of them, and a reply that
   omitted everything but ``name`` would score 1.00.

Extra fields the model invents are counted in :attr:`FieldScore.spurious` and reported, but
they do not inflate or deflate the ratio - a field that should not exist has no expected
value to match, and folding it into the denominator would double-penalise a reply that also
got a real field wrong.

Because a correct ``None`` is a match, field accuracy pays a reply for extracting nothing
wherever the gold is sparse (0.949 on the first 30 PII cases, see :func:`null_floor`). Two
counts split the expected fields by their gold value so that cannot hide:
:attr:`FieldScore.recall_matched` / :attr:`FieldScore.recall_total` are ``matched`` and
``total`` restricted to fields whose gold is not ``None`` - recall, whose all-null floor is 0
- and :attr:`FieldScore.invented` counts fields whose gold is ``None`` that the reply filled
anyway.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .outcomes import Outcome, outcome_counts

__all__ = [
    "FieldScore",
    "AccuracyRow",
    "flatten",
    "score",
    "null_floor",
    "normalize",
    "aggregate_accuracy",
    "worst_fields",
]

_INDEX_RE = re.compile(r"\[\d+\]")


@dataclass(frozen=True)
class FieldScore:
    """One cell's accuracy: how many expected fields the reply got exactly right."""

    matched: int
    total: int
    missing: tuple[str, ...] = ()
    wrong: tuple[str, ...] = ()
    spurious: tuple[str, ...] = ()
    recall_matched: int = 0  # `matched`, counting only fields whose gold is not None
    recall_total: int = 0  # `total`, counting only fields whose gold is not None
    invented: int = 0  # fields whose gold is None that the reply gave a value

    @property
    def ratio(self) -> float:
        """Matched fraction; an empty expected record scores 1.0, never a ZeroDivisionError."""
        return 1.0 if self.total == 0 else self.matched / self.total


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a model/dict/list tree to ``dotted.path -> scalar``.

    Lists are indexed (``contacts[0].email``) rather than compared as wholes, so a reply
    that gets two of three list items right scores 2/3 instead of 0/1. A ``None`` is a
    scalar leaf and is kept: "correctly absent" is a field the model can get right, and
    dropping it would silently shrink the denominator for exactly the optional fields this
    benchmark cares most about.
    """
    if isinstance(value, BaseModel):
        # JSON mode, because gold labels are JSON: a `date` field must flatten to the
        # "2024-04-22" the label holds, not to a `datetime.date` that never equals it.
        value = value.model_dump(mode="json")

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out

    if isinstance(value, (list, tuple)):
        out = {}
        for index, item in enumerate(value):
            out.update(flatten(item, f"{prefix}[{index}]"))
        return out

    return {prefix: value}


def normalize(value: Any) -> Any:
    """Reduce a leaf to its comparable form.

    Strings are stripped of surrounding whitespace - a model that pads a value has not got
    it wrong. Nothing else is coerced: case is significant, ``"36"`` is not ``36``, and
    ``None`` is not ``""``. Loosening any of those would let a wrong answer score as right,
    which is the one direction this metric must not fail in.
    """
    return value.strip() if isinstance(value, str) else value


def score(expected: Any, produced: Any | None) -> FieldScore:
    """Score ``produced`` against ``expected`` field by field.

    ``produced=None`` means the cell raised before yielding a value; it scores 0 against the
    full expected denominator rather than being skipped.

    A gold-``None`` field is ``invented`` when the reply puts a value at its path: a scalar
    (scored ``wrong``) or a whole subtree, which leaves the path ``missing`` and its children
    ``spurious`` - an entity list for a category the gold says is empty, or an object where
    the gold has none.
    """
    expected_flat = {k: normalize(v) for k, v in flatten(expected).items()}
    recall_total = sum(want is not None for want in expected_flat.values())

    if produced is None:
        return FieldScore(
            matched=0,
            total=len(expected_flat),
            missing=tuple(sorted(expected_flat)),
            recall_total=recall_total,
        )

    produced_flat = {k: normalize(v) for k, v in flatten(produced).items()}

    matched = recall_matched = 0
    missing: list[str] = []
    wrong: list[str] = []
    for path, want in expected_flat.items():
        if path not in produced_flat:
            missing.append(path)
        elif produced_flat[path] == want:
            matched += 1
            recall_matched += want is not None
        else:
            wrong.append(path)

    spurious = sorted(set(produced_flat) - set(expected_flat))
    invented = sum(
        want is None
        and (
            produced_flat.get(path) is not None
            or any(p.startswith((f"{path}.", f"{path}[")) for p in spurious)
        )
        for path, want in expected_flat.items()
    )
    return FieldScore(
        matched=matched,
        total=len(expected_flat),
        missing=tuple(missing),
        wrong=tuple(wrong),
        spurious=tuple(spurious),
        recall_matched=recall_matched,
        recall_total=recall_total,
        invented=invented,
    )


def null_floor(cases: Iterable[Any]) -> float:
    """Micro field accuracy of a reply with every output field ``None`` - "extracted nothing".

    A correct ``None`` is a match (see :func:`flatten`), so on a sparse corpus this is far
    above 0 - 0.949 on the first 30 PII cases - and a score only means something as a
    distance above it. Goes through each case's ``align`` exactly as a real reply does.
    """
    matched = total = 0
    for case in cases:
        model = case.signature.output_fields[case.output_field].annotation
        expected, produced = case.expected, dict.fromkeys(model.model_fields)
        if case.align is not None:
            expected, produced = case.align(expected, produced)
        result = score(expected, produced)
        matched += result.matched
        total += result.total
    return 1.0 if total == 0 else matched / total


@dataclass(frozen=True)
class AccuracyRow:
    """One accuracy cell: adapter x case, measured via `dspy.Predict(case.signature)`.

    Deliberately its own row type rather than a field bolted onto `TrialRow`: this arm is
    scored against ground truth, the outcomes arm against the schema, and the two are not
    on one accounting basis (`outcomes.py`'s module docstring, applied a third time).
    """

    adapter: str
    adapter_config: str
    case_id: str
    outcome: Outcome
    error_class: str
    wall_s: float
    lm_calls: int
    response_format_sent: str
    matched: int
    total: int
    wrong: tuple[str, ...]
    missing: tuple[str, ...]
    spurious: tuple[str, ...]
    total_tokens: int | None
    recall_matched: int
    recall_total: int
    invented: int

    @property
    def ratio(self) -> float:
        """This cell's field-match fraction; 1.0 for an empty expected record."""
        return 1.0 if self.total == 0 else self.matched / self.total

    @property
    def recall(self) -> float | None:
        """Matched fraction of the fields whose gold is not None; None when there are none."""
        return None if self.recall_total == 0 else self.recall_matched / self.recall_total

    @property
    def exact(self) -> bool:
        """Whether the whole record came back right - no wrong, missing or invented field."""
        return self.matched == self.total and not self.spurious


def aggregate_accuracy(rows: list[AccuracyRow]) -> list[dict[str, Any]]:
    """Collapse per-case rows into one record per adapter, in first-seen order.

    ``field_accuracy`` is **micro**-averaged - summed matches over summed expected fields,
    not the mean of per-case ratios. That is the quantity the reference benchmarks report,
    and it stops a case with few expected fields (an unemployed person with no contacts)
    from carrying the same weight as a fully-populated one. ``recall`` (over gold non-null
    fields) and ``invented_rate`` (over gold-``None`` fields) are micro-averaged the same way.
    """
    groups: dict[str, list[AccuracyRow]] = {}
    for row in rows:
        groups.setdefault(row.adapter, []).append(row)

    records: list[dict[str, Any]] = []
    for adapter, group in groups.items():
        matched = sum(row.matched for row in group)
        total = sum(row.total for row in group)
        recall_matched = sum(row.recall_matched for row in group)
        recall_total = sum(row.recall_total for row in group)
        invented = sum(row.invented for row in group)
        null_total = total - recall_total
        exact = sum(1 for row in group if row.exact)
        token_totals = [row.total_tokens for row in group if row.total_tokens is not None]
        formats = {row.response_format_sent for row in group}
        records.append(
            {
                "adapter": adapter,
                "adapter_config": group[0].adapter_config,
                "cases": len(group),
                **outcome_counts(group),
                "matched": matched,
                "total": total,
                "field_accuracy": (matched / total) if total else None,
                "exact_records": exact,
                "exact_rate": exact / len(group),
                "median_wall_s": statistics.median([row.wall_s for row in group]),
                "median_total_tokens": (statistics.median(token_totals) if token_totals else None),
                "response_format": next(iter(formats)) if len(formats) == 1 else "mixed",
                "recall_matched": recall_matched,
                "recall_total": recall_total,
                "recall": (recall_matched / recall_total) if recall_total else None,
                "invented": invented,
                "null_total": null_total,
                "invented_rate": (invented / null_total) if null_total else None,
            }
        )
    return records


def worst_fields(rows: list[AccuracyRow], limit: int = 8) -> list[tuple[str, int, int]]:
    """The field paths most often wrong or missing, as ``(path, wrong, missing)``.

    List indices are collapsed (``contacts[0].email`` -> ``contacts[].email``) so a path is
    counted as one field rather than fragmenting across positions. Rows from a cell that
    raised contribute their whole expected record to ``missing``, which is the honest
    reading: nothing came back, so every field is absent.
    """
    tally: dict[str, list[int]] = {}
    for row in rows:
        for path in row.wrong:
            tally.setdefault(_collapse_index(path), [0, 0])[0] += 1
        for path in row.missing:
            tally.setdefault(_collapse_index(path), [0, 0])[1] += 1
    ranked = sorted(tally.items(), key=lambda item: (-sum(item[1]), item[0]))
    return [(path, counts[0], counts[1]) for path, counts in ranked[:limit]]


def _collapse_index(path: str) -> str:
    """``contacts[0].email`` -> ``contacts[].email``; a path without an index is unchanged."""
    return _INDEX_RE.sub("[]", path)
