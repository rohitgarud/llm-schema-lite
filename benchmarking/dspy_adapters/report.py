"""CSV and markdown report writers for the DSPy adapter benchmark.

This module renders the two benchmark arms -- the offline prompt-cost arm
(`PromptRow`) and the live outcomes arm (`TrialRow`) -- into on-disk artifacts: one
machine-parseable CSV file per arm/run, one companion markdown file carrying
human-readable tables plus a provenance/caveat header block, and the per-cell
aggregation of live trials into one summary record per `(adapter, signature)` pair via
`aggregate_live`.

CSV files carry **no** header/provenance block of their own -- they must stay strictly
machine-parseable (the column-header line is always line 1, data rows follow). All
provenance (the `RunMeta` fields, the DSPy issue-1871 / research-C5 token-accounting
caveat, and the "these two tables are never joined" note) lives instead in the
same-named `.md` sibling, under a leading fenced provenance block and a mandatory
`## Metric integrity` section.

The offline (`prompt-cost-*`) and live (`live-*-*`) tables are **never** joined into one
table, one file, or one derived score -- see `outcomes.py`'s module docstring for why
(the two arms are not on the same accounting basis). `total_tokens` is the only
provider-reported column ever aggregated *across adapters*: because YAML mode and
`ChatAdapter` send no `response_format` while the JSON-family adapters send
`json_object`, `prompt_tokens_reported` / `completion_tokens_reported` sit on different
accounting bases per adapter and are shown only in the per-row "Provider accounting"
table -- never aggregated. `total_tokens` totals stay comparable (181 vs 180, per
research C5) and are the only reported-token figure `aggregate_live` reduces with
`statistics.median`.

`resolve_output_path` is the non-clobbering output-path resolver: a same-day rerun of
either writer appends `-2`, `-3`, ... before the file extension rather than silently
overwriting a prior run's results.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import statistics
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import astuple, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any

from .accuracy import AccuracyRow, aggregate_accuracy, worst_fields
from .outcomes import (
    Outcome,
    PromptRow,
    TrialRow,
    outcome_counts,
    parse_success_rate,
    validation_success_rate,
)
from .provenance import redact_lm_kwargs, redact_url_userinfo


@dataclass
class RunMeta:
    """Provenance + integrity facts stamped at the top of each .md file."""

    arm: str  # "prompt-cost" | "live"
    generated: str  # ISO-8601 timestamp
    command: str  # the argv used
    git_head: str  # `git rev-parse --short HEAD` + "-dirty", "unknown" on failure
    dspy_version: str
    llm_schema_lite_version: str
    encoding: str | None = None  # "cl100k_base"; offline arm only
    model: str | None = None  # live only
    api_base: str | None = None  # live only
    lm_kwargs: dict[str, Any] = field(
        default_factory=dict
    )  # live only, effective; pre-redacted by __post_init__
    supports_response_schema: bool | None = None  # live only
    supports_function_calling: bool | None = None  # live only
    integrity_overridden: bool = False  # live only

    def __post_init__(self) -> None:
        """Redact credential-shaped values at construction, so RunMeta is safe by construction.

        This -- not `cli.py` -- is the assembly boundary. `dataclasses.replace` re-invokes
        the constructor, so `cli.py`'s two `replace(...)` sites are covered without either
        of them knowing, and every present and future consumer of `lm_kwargs` inherits the
        guarantee. Both redactors are idempotent, so double application is harmless.

        Residual, stated not fixed: `RunMeta` is a mutable dataclass, so assigning
        `meta.lm_kwargs = {...}` after construction bypasses this. No `__setattr__` guard is
        added -- `__post_init__` covers every construction path in the package, and a
        setter guard would be machinery out of proportion to the risk.
        """
        self.lm_kwargs = redact_lm_kwargs(self.lm_kwargs)
        self.api_base = redact_url_userinfo(self.api_base)

    @classmethod
    def minimal(cls, arm: str) -> RunMeta:
        """Build a RunMeta with no subprocess call (git_head="unknown") for tests."""
        return cls(
            arm=arm,
            generated=dt.datetime.now(dt.timezone.utc).isoformat(),
            command="",
            git_head="unknown",
            dspy_version=_package_version("dspy"),
            llm_schema_lite_version=_package_version("llm-schema-lite"),
        )


def git_head(repo_root: Path | None = None) -> str:
    """Return `git rev-parse --short HEAD`, suffixed `-dirty`, or "unknown" on any failure.

    A results file exists to name the code that produced it, so a bare sha is a lie the
    moment the tree carries edits: the committed `2026-09-10` artefacts stamp `daf210b`
    while running an uncommitted YAML fix and a `sola-yaml-rescue` cell that commit does
    not define, and stamp `867819b` while running a `--corpus` flag added five hours
    later. Neither stamp can reproduce its own file. The suffix makes that visible.

    Untracked files are deliberately ignored: the arms write their own results into the
    tree, and a run must not be marked dirty by the file it is in the middle of writing.
    The residual gap that leaves -- a brand-new *source* file, untracked and imported --
    is not worth a second subprocess call to close.
    """
    root = Path(__file__).resolve().parents[2] if repo_root is None else repo_root
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        head = result.stdout.strip()
        if not head:
            return "unknown"
        modified = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return f"{head}-dirty" if modified.stdout.strip() else head
    except Exception:  # noqa: BLE001 -- swallow every failure into "unknown"
        return "unknown"


def _package_version(name: str) -> str:
    """Best-effort installed-package version lookup; "unknown" on any failure."""
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# One CSV column per row-dataclass field, in declaration order.
PROMPT_COST_CSV_HEADER: tuple[str, ...] = tuple(f.name for f in fields(PromptRow))
LIVE_CSV_HEADER: tuple[str, ...] = tuple(f.name for f in fields(TrialRow))

# Not AccuracyRow's fields: adds the `ratio`/`exact` properties, in its own column order.
ACCURACY_CSV_HEADER: tuple[str, ...] = (
    "adapter",
    "adapter_config",
    "case_id",
    "outcome",
    "error_class",
    "wall_s",
    "lm_calls",
    "response_format_sent",
    "matched",
    "total",
    "ratio",
    "exact",
    "wrong",
    "missing",
    "spurious",
    "total_tokens",
    "recall_matched",
    "recall_total",
    "recall",
    "invented",
    "replies",  # JSON list of the raw reply texts, one per LM call; read by --replay
)

ACCURACY_AGGREGATE_COLUMNS = (
    "| adapter | cases | field accuracy | exact records | ok | parse | validation | empty "
    "| transport | format | median wall_s | median total_tokens | response_format "
    "| recall (non-null gold) | invented (null gold) |"
)

ACCURACY_FIELDS_COLUMNS = "| field | wrong | missing |"

# Accuracy-arm-only addendum to the "## Metric integrity" section.
ACCURACY_EXTRA_CAVEAT = (
    "field accuracy is micro-averaged over the **expected** fields of every case: a cell "
    "that raised scores 0 against its full denominator rather than being excluded, and "
    "emitting fewer fields can never raise the score. Fields the model invents are "
    "reported under `spurious` and break `exact`, but do not enter the denominator. "
    "A correct `None` also counts as a match, so on a sparse corpus field accuracy pays a "
    "reply for extracting nothing. **recall (non-null gold)** is the extraction-quality "
    "headline: matches over only the fields whose gold value is not `None`, a cell that "
    "raised scoring 0 against them, so extracting nothing scores 0. **invented (null "
    "gold)** is the other half: of the fields whose gold is `None`, how many the reply "
    "filled anyway."
)

SYNTHETIC_GROUND_TRUTH_NOTE = (
    "Ground truth is the generated record the prose was rendered from, so the corpus "
    "measures schema-following under paraphrase — not real-world extraction."
)

OFFLINE_PIVOT_COLUMNS = "| adapter | flat | nested | list_of_model | enum | optional | recursive |"

OFFLINE_DETAIL_COLUMNS = (
    "| adapter | adapter config | signature | messages | chars | prompt tokens | outcome |"
)

LIVE_AGGREGATE_COLUMNS = (
    "| adapter | signature | trials | ok | parse | validation | empty | transport | format "
    "| parse rate | validation rate "
    "| median wall_s | stddev wall_s | median total_tokens | response_format |"
)

LIVE_DETAIL_COLUMNS = (
    "| adapter | signature | trial | outcome | error class | wall_s | lm_calls | response_format |"
)

PROVIDER_ACCOUNTING_HEADING = (
    "### Provider accounting — NOT comparable across the `response_format` groups"
)

PROVIDER_ACCOUNTING_COLUMNS = (
    "| adapter | signature | response_format | prompt_tokens_reported "
    "| completion_tokens_reported | total_tokens |"
)

METRIC_INTEGRITY_HEADING = "## Metric integrity"

RESPONSE_FORMAT_CAVEAT = (
    "prompt/completion token counts are NOT comparable across the `response_format` groups"
)

# Research C5: verbatim measurement quoted into every provenance block's "## Metric
# integrity" section (see Phase 8 of the implementation plan).
METRIC_INTEGRITY_C5 = (
    "Same prompt, same model, greedy: without `response_format` → "
    "`{'prompt_tokens': 21, 'completion_tokens': 160}`; with "
    '`response_format={"type": "json_object"}` → '
    "`{'prompt_tokens': 173, 'completion_tokens': 7}` "
    "(identical 649-char reasoning trace in both). Ollama re-prefills the thinking "
    "trace and bills it as *prompt* tokens under constrained decoding. Since YAML mode "
    "and `ChatAdapter` send **no** `response_format` while "
    "JSON/JSONISH/`JSONAdapter`/`BAMLAdapter` send `json_object`, the two groups are "
    "**not** on the same accounting basis. Total tokens stay comparable (181 vs 180)."
)

# Offline-arm-only addendum to the "## Metric integrity" section.
OFFLINE_EXTRA_CAVEAT = (
    "the offline count measures the textual prompt only — it excludes "
    "`_call_preprocess` tool/native-type handling and the `response_format` request "
    "kwarg, which is a request field and not a message. For this matrix that is "
    "correct: the thing under test is the text."
)

NEVER_JOINED_NOTE = (
    "The offline prompt-cost table and the live outcomes table are never joined into "
    "one table or one derived score."
)

# The six signature ids, in the order OFFLINE_PIVOT_COLUMNS declares them -- derived
# from the header string itself so the two can never drift apart.
_PIVOT_SIGNATURE_ORDER: tuple[str, ...] = tuple(
    part.strip() for part in OFFLINE_PIVOT_COLUMNS.strip("|").split("|")
)[1:]

# The per-outcome count columns of both aggregate tables, in column order.
_COUNT_COLUMNS: tuple[str, ...] = ("ok", "parse", "validation", "empty", "transport", "format")


def slugify_model(model: str) -> str:
    """Lowercase `model`, then replace "/" and ":" with "-" (e.g. "openai/qwen3:8b" ->
    "openai-qwen3-8b")."""
    return model.lower().replace("/", "-").replace(":", "-")


def resolve_output_path(out_dir: Path, stem: str, suffix: str) -> Path:
    """Return `out_dir/<stem><suffix>`, or the first free `<stem>-N<suffix>` (N >= 2).

    Never silently overwrites an existing file.
    """
    candidate = out_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = out_dir / f"{stem}-{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def _dash(value: Any, spec: str = "") -> str:
    """`value` formatted with `spec`, or the em dash every table uses for an undefined value."""
    return "—" if value is None else format(value, spec)


def _md_table(header: str, rows: Iterable[Sequence[str]]) -> list[str]:
    """A markdown table: `header`, a `---` separator exactly as wide, then one line per row."""
    sep = "|" + "|".join(["---"] * (header.count("|") - 1)) + "|"
    return [header, sep, *("| " + " | ".join(cells) + " |" for cells in rows)]


def _csv_cell(value: Any) -> Any:
    """`None` -> an empty CSV cell, an `Enum` -> its value; everything else passes through."""
    if value is None:
        return ""
    return value.value if isinstance(value, Enum) else value


def _write_report(
    out_dir: Path,
    stem: str,
    today: dt.date | None,
    header: tuple[str, ...],
    csv_rows: Iterable[Iterable[Any]],
    markdown: str,
) -> tuple[Path, Path]:
    """Write `<stem>-<YYYY-MM-DD>.{md,csv}` into `out_dir` (created if absent), never clobbering.

    `today=None` uses `dt.date.today()`. Returns `(md_path, csv_path)`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{stem}-{(today or dt.date.today()).isoformat()}"
    md_path = resolve_output_path(out_dir, stem, ".md")
    csv_path = resolve_output_path(out_dir, stem, ".csv")
    cells = [[_csv_cell(value) for value in row] for row in csv_rows]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(cells)
    md_path.write_text(markdown, encoding="utf-8")
    return md_path, csv_path


def _render_provenance_block(meta: RunMeta) -> list[str]:
    lines = ["```"]
    lines.append(f"arm: {meta.arm}")
    lines.append(f"generated: {meta.generated}")
    lines.append(f"command: {meta.command}")
    lines.append(f"git_head: {meta.git_head}")
    lines.append(f"dspy_version: {meta.dspy_version}")
    lines.append(f"llm_schema_lite_version: {meta.llm_schema_lite_version}")
    if meta.encoding is not None:
        lines.append(f"encoding: {meta.encoding}")
    if meta.model is not None:
        lines.append(f"model: {meta.model}")
    if meta.api_base is not None:
        lines.append(f"api_base: {meta.api_base}")
    if meta.lm_kwargs:
        lines.append(f"lm_kwargs: {meta.lm_kwargs}")
    if meta.supports_response_schema is not None:
        lines.append(f"supports_response_schema: {meta.supports_response_schema}")
    if meta.supports_function_calling is not None:
        lines.append(f"supports_function_calling: {meta.supports_function_calling}")
    if meta.integrity_overridden:
        lines.append(f"integrity_overridden: {meta.integrity_overridden}")
    lines.append("```")
    return lines


def _metric_integrity_section(extra: str | None) -> list[str]:
    lines = ["", METRIC_INTEGRITY_HEADING, "", METRIC_INTEGRITY_C5, "", RESPONSE_FORMAT_CAVEAT]
    if extra:
        lines.extend(["", extra])
    lines.extend(["", NEVER_JOINED_NOTE])
    return lines


def _render_offline_pivot(rows: list[PromptRow]) -> list[str]:
    # Only an `ok` cell has a token count to show; anything else is the em dash.
    ok_tokens: dict[str, dict[str, int | None]] = {}
    for row in rows:
        ok_tokens.setdefault(row.adapter, {})[row.signature] = (
            row.prompt_tokens if row.outcome is Outcome.OK else None
        )
    return _md_table(
        OFFLINE_PIVOT_COLUMNS,
        (
            [adapter, *(_dash(cells.get(sig)) for sig in _PIVOT_SIGNATURE_ORDER)]
            for adapter, cells in ok_tokens.items()
        ),
    )


def _render_offline_detail(rows: list[PromptRow]) -> list[str]:
    return _md_table(
        OFFLINE_DETAIL_COLUMNS,
        (
            [
                row.adapter,
                row.adapter_config,
                row.signature,
                _dash(row.n_messages),
                _dash(row.prompt_chars),
                _dash(row.prompt_tokens),
                row.outcome.value,
            ]
            for row in rows
        ),
    )


def aggregate_live(rows: list[TrialRow]) -> list[dict[str, Any]]:
    """Collapse per-trial rows into one aggregate record per `(adapter, signature)`.

    Each record carries counts per `Outcome`, the median/stddev of `wall_s`, the median
    of `total_tokens` (over the trials that reported one), and the single
    `response_format_sent` value observed for the cell (or "mixed" if it varied). Each
    record also carries `parse_success_rate` / `validation_success_rate` over *attempted*
    rows only, `None` when the cell attempted nothing.
    `statistics.stdev` needs >= 2 points, so a single-trial cell reports `0.0` rather
    than raising.
    """
    groups: dict[tuple[str, str], list[TrialRow]] = {}
    for row in rows:
        groups.setdefault((row.adapter, row.signature), []).append(row)

    records: list[dict[str, Any]] = []
    for (adapter, signature), group_rows in groups.items():
        wall_times = [row.wall_s for row in group_rows]
        token_totals = [row.total_tokens for row in group_rows if row.total_tokens is not None]
        response_formats = {row.response_format_sent for row in group_rows}
        response_format = next(iter(response_formats)) if len(response_formats) == 1 else "mixed"
        records.append(
            {
                "adapter": adapter,
                "signature": signature,
                "trials": len(group_rows),
                **outcome_counts(group_rows),
                "parse_success_rate": parse_success_rate(group_rows),  # float | None
                "validation_success_rate": validation_success_rate(group_rows),  # float | None
                "median_wall_s": statistics.median(wall_times) if wall_times else 0.0,
                "stddev_wall_s": (statistics.stdev(wall_times) if len(wall_times) >= 2 else 0.0),
                "median_total_tokens": (statistics.median(token_totals) if token_totals else None),
                "response_format": response_format,
            }
        )
    return records


def _render_live_aggregate(rows: list[TrialRow]) -> list[str]:
    return _md_table(
        LIVE_AGGREGATE_COLUMNS,
        (
            [
                record["adapter"],
                record["signature"],
                str(record["trials"]),
                *(str(record[key]) for key in _COUNT_COLUMNS),
                _dash(record["parse_success_rate"], ".2f"),
                _dash(record["validation_success_rate"], ".2f"),
                f"{record['median_wall_s']:.3f}",
                f"{record['stddev_wall_s']:.3f}",
                _dash(record["median_total_tokens"]),
                record["response_format"],
            ]
            for record in aggregate_live(rows)
        ),
    )


def _render_live_detail(rows: list[TrialRow]) -> list[str]:
    return _md_table(
        LIVE_DETAIL_COLUMNS,
        (
            [
                row.adapter,
                row.signature,
                str(row.trial),
                row.outcome.value,
                row.error_class,
                f"{row.wall_s:.3f}",
                f"{row.lm_calls}†" if row.fallback_suspected else str(row.lm_calls),
                row.response_format_sent,
            ]
            for row in rows
        ),
    )


def _render_provider_accounting(rows: list[TrialRow]) -> list[str]:
    return _md_table(
        PROVIDER_ACCOUNTING_COLUMNS,
        (
            [
                row.adapter,
                row.signature,
                row.response_format_sent,
                _dash(row.prompt_tokens_reported),
                _dash(row.completion_tokens_reported),
                str(row.total_tokens),
            ]
            for row in rows
            if row.total_tokens is not None
        ),
    )


def _render_offline_markdown(rows: list[PromptRow], meta: RunMeta) -> str:
    lines: list[str] = []
    lines.extend(_render_provenance_block(meta))
    lines.extend(_metric_integrity_section(OFFLINE_EXTRA_CAVEAT))
    lines.append("")
    lines.append("## Prompt cost — pivot (prompt tokens per adapter x signature)")
    lines.append("")
    lines.extend(_render_offline_pivot(rows))
    lines.append("")
    lines.append("## Prompt cost — detail")
    lines.append("")
    lines.extend(_render_offline_detail(rows))
    lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _render_live_markdown(rows: list[TrialRow], meta: RunMeta) -> str:
    lines: list[str] = []
    lines.extend(_render_provenance_block(meta))
    lines.extend(_metric_integrity_section(None))
    lines.append("")
    lines.append("## Live outcomes — aggregate")
    lines.append("")
    lines.extend(_render_live_aggregate(rows))
    lines.append("")
    lines.append("## Live outcomes — detail")
    lines.append("")
    lines.extend(_render_live_detail(rows))
    lines.append("")
    lines.append(PROVIDER_ACCOUNTING_HEADING)
    lines.append("")
    lines.extend(_render_provider_accounting(rows))
    lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def write_offline_report(
    rows: list[PromptRow],
    out_dir: Path,
    meta: RunMeta | None = None,
    today: dt.date | None = None,
) -> tuple[Path, Path]:
    """Write `results/prompt-cost-<YYYY-MM-DD>.{md,csv}` and return `(md_path, csv_path)`.

    No model name appears in the file name -- the offline arm never touches an LM.
    `out_dir` is created if absent. `meta=None` builds `RunMeta.minimal("prompt-cost")`;
    `today=None` uses `dt.date.today()`.
    """
    meta = meta or RunMeta.minimal("prompt-cost")
    return _write_report(
        out_dir,
        "prompt-cost",
        today,
        PROMPT_COST_CSV_HEADER,
        map(astuple, rows),
        _render_offline_markdown(rows, meta),
    )


def write_live_report(
    rows: list[TrialRow],
    out_dir: Path,
    meta: RunMeta | None = None,
    today: dt.date | None = None,
) -> tuple[Path, Path]:
    """Write `results/live-<model-slug>-<YYYY-MM-DD>.{md,csv}` and return the two paths.

    The slug comes from `meta.model` via `slugify_model`; `"unknown"` when
    `meta.model is None`. `out_dir` is created if absent. `meta=None` builds
    `RunMeta.minimal("live")`; `today=None` uses `dt.date.today()`.
    """
    meta = meta or RunMeta.minimal("live")
    model_slug = "unknown" if meta.model is None else slugify_model(meta.model)
    return _write_report(
        out_dir,
        f"live-{model_slug}",
        today,
        LIVE_CSV_HEADER,
        map(astuple, rows),
        _render_live_markdown(rows, meta),
    )


def _render_accuracy_aggregate(rows: list[AccuracyRow]) -> list[str]:
    return _md_table(
        ACCURACY_AGGREGATE_COLUMNS,
        (
            [
                record["adapter"],
                str(record["cases"]),
                _dash(record["field_accuracy"], ".3f"),
                f"{record['exact_records']}/{record['cases']} ({record['exact_rate']:.2f})",
                *(str(record[key]) for key in _COUNT_COLUMNS),
                f"{record['median_wall_s']:.3f}",
                _dash(record["median_total_tokens"]),
                record["response_format"],
                f"{_dash(record['recall'], '.3f')} "
                f"({record['recall_matched']}/{record['recall_total']})",
                f"{_dash(record['invented_rate'], '.3f')} "
                f"({record['invented']}/{record['null_total']})",
            ]
            for record in aggregate_accuracy(rows)
        ),
    )


def _render_accuracy_fields(rows: list[AccuracyRow]) -> list[str]:
    """Per-adapter "which fields did it get wrong" tables — the arm's diagnostic half.

    An aggregate accuracy number says an adapter lost; only this says where. Rendered per
    adapter because the interesting comparison is which *different* fields each one drops.
    """
    lines: list[str] = []
    by_adapter: dict[str, list[AccuracyRow]] = {}
    for row in rows:
        by_adapter.setdefault(row.adapter, []).append(row)
    for adapter, group in by_adapter.items():
        ranked = worst_fields(group)
        if not ranked:
            continue
        lines.extend(["", f"**{adapter}**", ""])
        lines.extend(
            _md_table(
                ACCURACY_FIELDS_COLUMNS,
                ([f"`{path}`", str(wrong), str(missing)] for path, wrong, missing in ranked),
            )
        )
    return lines


def _render_accuracy_markdown(
    rows: list[AccuracyRow],
    meta: RunMeta,
    corpus: str | None = None,
    null_floor: float | None = None,
) -> str:
    ground_truth = (
        SYNTHETIC_GROUND_TRUTH_NOTE
        if corpus is None
        else (
            f"Ground truth is the label shipped with the third-party `{corpus}` corpus "
            "(Hugging Face, revision in `lm_kwargs`), and the prompt is that benchmark's own "
            "signature, not one written by this package's authors."
        )
    )
    lines: list[str] = []
    lines.extend(_render_provenance_block(meta))
    lines.extend(_metric_integrity_section(f"{ACCURACY_EXTRA_CAVEAT} {ground_truth}"))
    lines.append("")
    lines.append("## Extraction accuracy — aggregate")
    lines.append("")
    lines.extend(_render_accuracy_aggregate(rows))
    if null_floor is not None:
        lines.append("")
        lines.append(
            f"**All-null floor: {null_floor:.3f}** — the field accuracy of a reply that "
            "extracts nothing (every output field `None`) on these same cases. A correct "
            "`None` counts as a match, so field accuracy is only a distance from this "
            "floor, and an adapter near it may simply have extracted nothing. Recall "
            "(non-null gold) has a floor of 0 by construction: compare adapters on that."
        )
    lines.append("")
    lines.append("## Most-missed fields")
    lines.extend(_render_accuracy_fields(rows))
    lines.append("")
    lines.append("Per-case detail is in the companion `.csv`; it is not duplicated here.")
    lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def write_accuracy_report(
    rows: list[AccuracyRow],
    out_dir: Path,
    meta: RunMeta | None = None,
    today: dt.date | None = None,
    corpus: str | None = None,
    null_floor: float | None = None,
) -> tuple[Path, Path]:
    """Write `results/accuracy[-<corpus>]-<model-slug>-<YYYY-MM-DD>.{md,csv}`; return both.

    `corpus=None` is the synthetic corpus, which keeps the stem its committed artefacts
    already have; a third-party corpus is named in the stem so two corpora run against one
    model on one day never collide.

    A third file stem, never merged into `live-*`: this arm scores against ground truth
    while the outcomes arm scores against the schema, and one file would invite exactly
    the join `outcomes.py` forbids.
    """
    meta = meta or RunMeta.minimal("accuracy")
    model_slug = "unknown" if meta.model is None else slugify_model(meta.model)
    prefix = "accuracy" if corpus is None else f"accuracy-{corpus}"
    csv_rows = (
        [
            row.adapter,
            row.adapter_config,
            row.case_id,
            row.outcome,
            row.error_class,
            row.wall_s,
            row.lm_calls,
            row.response_format_sent,
            row.matched,
            row.total,
            round(row.ratio, 4),
            row.exact,
            ";".join(row.wrong),
            ";".join(row.missing),
            ";".join(row.spurious),
            row.total_tokens,
            row.recall_matched,
            row.recall_total,
            None if row.recall is None else round(row.recall, 4),
            row.invented,
            json.dumps(list(row.replies), ensure_ascii=False),
        ]
        for row in rows
    )
    return _write_report(
        out_dir,
        f"{prefix}-{model_slug}",
        today,
        ACCURACY_CSV_HEADER,
        csv_rows,
        _render_accuracy_markdown(rows, meta, corpus, null_floor),
    )
