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
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .accuracy import AccuracyRow, aggregate_accuracy, worst_fields
from .outcomes import Outcome, PromptRow, TrialRow, parse_success_rate, validation_success_rate
from .provenance import redact_lm_kwargs, redact_url_userinfo


@dataclass
class RunMeta:
    """Provenance + integrity facts stamped at the top of each .md file."""

    arm: str  # "prompt-cost" | "live"
    generated: str  # ISO-8601 timestamp
    command: str  # the argv used
    git_head: str  # `git rev-parse --short HEAD`, "unknown" on failure
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


def git_head() -> str:
    """Return `git rev-parse --short HEAD` for the repo root, or "unknown" on any failure."""
    try:
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        head = result.stdout.strip()
        return head or "unknown"
    except Exception:  # noqa: BLE001 -- swallow every failure into "unknown"
        return "unknown"


def _package_version(name: str) -> str:
    """Best-effort installed-package version lookup; "unknown" on any failure."""
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


PROMPT_COST_CSV_HEADER: tuple[str, ...] = (
    "adapter",
    "adapter_config",
    "signature",
    "n_messages",
    "prompt_chars",
    "prompt_tokens",
    "outcome",
    "error_class",
)

LIVE_CSV_HEADER: tuple[str, ...] = (
    "adapter",
    "adapter_config",
    "signature",
    "trial",
    "outcome",
    "error_class",
    "wall_s",
    "lm_calls",
    "fallback_suspected",
    "response_format_sent",
    "total_tokens",
    "prompt_tokens_reported",
    "completion_tokens_reported",
)

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
)

ACCURACY_AGGREGATE_COLUMNS = (
    "| adapter | cases | field accuracy | exact records | ok | parse | validation | empty "
    "| transport | format | median wall_s | median total_tokens | response_format |"
)

ACCURACY_FIELDS_COLUMNS = "| field | wrong | missing |"

# Accuracy-arm-only addendum to the "## Metric integrity" section.
ACCURACY_EXTRA_CAVEAT = (
    "field accuracy is micro-averaged over the **expected** fields of every case: a cell "
    "that raised scores 0 against its full denominator rather than being excluded, and "
    "emitting fewer fields can never raise the score. Fields the model invents are "
    "reported under `spurious` and break `exact`, but do not enter the denominator."
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

# Derived from the header string itself so the two can never drift apart -- the same rule
# `_PIVOT_SIGNATURE_ORDER` (above) already applies to the offline pivot. This ticket is the
# drift event that guard was invented for.
_LIVE_AGGREGATE_COLUMN_COUNT: int = len(LIVE_AGGREGATE_COLUMNS.strip("|").split("|"))


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


def _csv_row(values: list[Any]) -> list[Any]:
    """Render `None` as an empty CSV cell; every other value passes through untouched."""
    return ["" if value is None else value for value in values]


def _write_csv_file(path: Path, header: tuple[str, ...], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


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
    by_adapter: dict[str, dict[str, PromptRow]] = {}
    order: list[str] = []
    for row in rows:
        if row.adapter not in by_adapter:
            by_adapter[row.adapter] = {}
            order.append(row.adapter)
        by_adapter[row.adapter][row.signature] = row
    sep = "|" + "|".join(["---"] * (len(_PIVOT_SIGNATURE_ORDER) + 1)) + "|"
    lines = [OFFLINE_PIVOT_COLUMNS, sep]
    for adapter in order:
        cells = [adapter]
        for sig in _PIVOT_SIGNATURE_ORDER:
            cell_row = by_adapter[adapter].get(sig)
            if (
                cell_row is None
                or cell_row.outcome is not Outcome.OK
                or cell_row.prompt_tokens is None
            ):
                cells.append("—")
            else:
                cells.append(str(cell_row.prompt_tokens))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _render_offline_detail(rows: list[PromptRow]) -> list[str]:
    sep = "|" + "|".join(["---"] * 7) + "|"
    lines = [OFFLINE_DETAIL_COLUMNS, sep]
    for row in rows:
        messages = "—" if row.n_messages is None else str(row.n_messages)
        chars = "—" if row.prompt_chars is None else str(row.prompt_chars)
        tokens = "—" if row.prompt_tokens is None else str(row.prompt_tokens)
        cells = [
            row.adapter,
            row.adapter_config,
            row.signature,
            messages,
            chars,
            tokens,
            row.outcome.value,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


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
    order: list[tuple[str, str]] = []
    for row in rows:
        key = (row.adapter, row.signature)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    records: list[dict[str, Any]] = []
    for key in order:
        group_rows = groups[key]
        counts = dict.fromkeys(Outcome, 0)
        for row in group_rows:
            counts[row.outcome] += 1
        wall_times = [row.wall_s for row in group_rows]
        token_totals = [row.total_tokens for row in group_rows if row.total_tokens is not None]
        response_formats = {row.response_format_sent for row in group_rows}
        response_format = next(iter(response_formats)) if len(response_formats) == 1 else "mixed"
        records.append(
            {
                "adapter": key[0],
                "signature": key[1],
                "trials": len(group_rows),
                "ok": counts[Outcome.OK],
                "parse": counts[Outcome.PARSE_ERROR],
                "validation": counts[Outcome.VALIDATION_ERROR],
                "empty": counts[Outcome.EMPTY_RESPONSE],
                "transport": counts[Outcome.TRANSPORT_ERROR],
                "format": counts[Outcome.FORMAT_ERROR],
                "other": counts[Outcome.OTHER_ERROR],
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
    sep = "|" + "|".join(["---"] * _LIVE_AGGREGATE_COLUMN_COUNT) + "|"
    lines = [LIVE_AGGREGATE_COLUMNS, sep]
    for record in aggregate_live(rows):
        median_tokens = record["median_total_tokens"]
        median_tokens_cell = "—" if median_tokens is None else str(median_tokens)
        parse_rate = record["parse_success_rate"]
        parse_rate_cell = "—" if parse_rate is None else f"{parse_rate:.2f}"
        validation_rate = record["validation_success_rate"]
        validation_rate_cell = "—" if validation_rate is None else f"{validation_rate:.2f}"
        cells = [
            record["adapter"],
            record["signature"],
            str(record["trials"]),
            str(record["ok"]),
            str(record["parse"]),
            str(record["validation"]),
            str(record["empty"]),
            str(record["transport"]),
            str(record["format"]),
            parse_rate_cell,
            validation_rate_cell,
            f"{record['median_wall_s']:.3f}",
            f"{record['stddev_wall_s']:.3f}",
            median_tokens_cell,
            record["response_format"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _render_live_detail(rows: list[TrialRow]) -> list[str]:
    sep = "|" + "|".join(["---"] * 8) + "|"
    lines = [LIVE_DETAIL_COLUMNS, sep]
    for row in rows:
        lm_calls_cell = f"{row.lm_calls}†" if row.fallback_suspected else str(row.lm_calls)
        cells = [
            row.adapter,
            row.signature,
            str(row.trial),
            row.outcome.value,
            row.error_class,
            f"{row.wall_s:.3f}",
            lm_calls_cell,
            row.response_format_sent,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _render_provider_accounting(rows: list[TrialRow]) -> list[str]:
    sep = "|" + "|".join(["---"] * 6) + "|"
    lines = [PROVIDER_ACCOUNTING_COLUMNS, sep]
    for row in rows:
        if row.total_tokens is None:
            continue
        prompt_reported = (
            "—" if row.prompt_tokens_reported is None else str(row.prompt_tokens_reported)
        )
        completion_reported = (
            "—" if row.completion_tokens_reported is None else str(row.completion_tokens_reported)
        )
        cells = [
            row.adapter,
            row.signature,
            row.response_format_sent,
            prompt_reported,
            completion_reported,
            str(row.total_tokens),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if meta is None:
        meta = RunMeta.minimal("prompt-cost")
    if today is None:
        today = dt.date.today()

    stem = f"prompt-cost-{today.isoformat()}"
    md_path = resolve_output_path(out_dir, stem, ".md")
    csv_path = resolve_output_path(out_dir, stem, ".csv")

    csv_rows = [
        _csv_row(
            [
                row.adapter,
                row.adapter_config,
                row.signature,
                row.n_messages,
                row.prompt_chars,
                row.prompt_tokens,
                row.outcome.value,
                row.error_class,
            ]
        )
        for row in rows
    ]
    _write_csv_file(csv_path, PROMPT_COST_CSV_HEADER, csv_rows)
    md_path.write_text(_render_offline_markdown(rows, meta), encoding="utf-8")
    return md_path, csv_path


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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if meta is None:
        meta = RunMeta.minimal("live")
    if today is None:
        today = dt.date.today()

    model_slug = "unknown" if meta.model is None else slugify_model(meta.model)
    stem = f"live-{model_slug}-{today.isoformat()}"
    md_path = resolve_output_path(out_dir, stem, ".md")
    csv_path = resolve_output_path(out_dir, stem, ".csv")

    csv_rows = [
        _csv_row(
            [
                row.adapter,
                row.adapter_config,
                row.signature,
                row.trial,
                row.outcome.value,
                row.error_class,
                row.wall_s,
                row.lm_calls,
                row.fallback_suspected,
                row.response_format_sent,
                row.total_tokens,
                row.prompt_tokens_reported,
                row.completion_tokens_reported,
            ]
        )
        for row in rows
    ]
    _write_csv_file(csv_path, LIVE_CSV_HEADER, csv_rows)
    md_path.write_text(_render_live_markdown(rows, meta), encoding="utf-8")
    return md_path, csv_path


def _render_accuracy_aggregate(rows: list[AccuracyRow]) -> list[str]:
    sep = "|" + "|".join(["---"] * len(ACCURACY_AGGREGATE_COLUMNS.strip("|").split("|"))) + "|"
    lines = [ACCURACY_AGGREGATE_COLUMNS, sep]
    for record in aggregate_accuracy(rows):
        accuracy = record["field_accuracy"]
        median_tokens = record["median_total_tokens"]
        cells = [
            record["adapter"],
            str(record["cases"]),
            "—" if accuracy is None else f"{accuracy:.3f}",
            f"{record['exact_records']}/{record['cases']} ({record['exact_rate']:.2f})",
            str(record["ok"]),
            str(record["parse"]),
            str(record["validation"]),
            str(record["empty"]),
            str(record["transport"]),
            str(record["format"]),
            f"{record['median_wall_s']:.3f}",
            "—" if median_tokens is None else str(median_tokens),
            record["response_format"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _render_accuracy_fields(rows: list[AccuracyRow]) -> list[str]:
    """Per-adapter "which fields did it get wrong" tables — the arm's diagnostic half.

    An aggregate accuracy number says an adapter lost; only this says where. Rendered per
    adapter because the interesting comparison is which *different* fields each one drops.
    """
    lines: list[str] = []
    by_adapter: dict[str, list[AccuracyRow]] = {}
    for row in rows:
        by_adapter.setdefault(row.adapter, []).append(row)
    sep = "|---|---|---|"
    for adapter, group in by_adapter.items():
        ranked = worst_fields(group)
        if not ranked:
            continue
        lines.extend(["", f"**{adapter}**", "", ACCURACY_FIELDS_COLUMNS, sep])
        lines.extend(f"| `{path}` | {wrong} | {missing} |" for path, wrong, missing in ranked)
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
            "`None` counts as a match, so read every score above as a distance from this."
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if meta is None:
        meta = RunMeta.minimal("accuracy")
    if today is None:
        today = dt.date.today()

    model_slug = "unknown" if meta.model is None else slugify_model(meta.model)
    prefix = "accuracy" if corpus is None else f"accuracy-{corpus}"
    stem = f"{prefix}-{model_slug}-{today.isoformat()}"
    md_path = resolve_output_path(out_dir, stem, ".md")
    csv_path = resolve_output_path(out_dir, stem, ".csv")

    csv_rows = [
        _csv_row(
            [
                row.adapter,
                row.adapter_config,
                row.case_id,
                row.outcome.value,
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
            ]
        )
        for row in rows
    ]
    _write_csv_file(csv_path, ACCURACY_CSV_HEADER, csv_rows)
    md_path.write_text(_render_accuracy_markdown(rows, meta, corpus, null_floor), encoding="utf-8")
    return md_path, csv_path
