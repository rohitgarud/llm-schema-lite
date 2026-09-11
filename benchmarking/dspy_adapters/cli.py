"""Command-line entry point for the DSPy adapter benchmark.

Exposes ``build_parser()`` (an :mod:`argparse` parser, not bare ``sys.argv`` handling --
there are eight independent flags with type conversion, comma-separated-list parsing and
mutually-adjusting defaults, which argparse is built for) and ``main()``, the function
``__main__.py`` calls.

This module is the **only** caller of :mod:`.config` in the whole package. ``config``,
``.runner`` and ``.report`` are imported **lazily, inside the branches that need them** --
never at module import time -- so that ``--list`` and (in the common case, with no
``LSL_BENCH_MODEL``/``LSL_BENCH_API_BASE`` set) ``--repro-1871`` never touch
``os.environ``. ``--offline`` is the one exception: it calls
``.encoding.seed_tiktoken_cache()``, which may set ``TIKTOKEN_CACHE_DIR`` and
``CUSTOM_TIKTOKEN_CACHE_DIR`` so the ``cl100k_base`` table can be loaded without a
network. cli.py itself still never calls ``os.environ`` directly. The one exception
is ``--repro-1871``'s optional live-probe row, which is genuinely conditional on the live
environment ("plus ``probe_1871_live()`` iff env is configured"); that check is a bare
presence check of the two required variable *names* (re-exported as constants by
``config``, never read as raw strings here) before ``config.load_config()`` is ever called,
so an unconfigured environment never triggers ``config``'s fail-loud path and never prints
its "not configured" banner.

Exit codes (see the table in the plan, Phase 9): ``0`` means the requested arms ran and
every requested results file was written -- a **recorded** cell failure (a ``format_error``
or ``transport_error`` row) is data, not a harness fault, and never changes this. ``1``
means the harness itself failed: the output directory could not be written, or the live arm
ran at least one cell and *every* row's ``error_class`` is exactly ``"LMTransportError"``
(the leaf class, not merely ``outcome == "transport_error"``). ``2`` means misconfiguration:
missing env, invalid ``LSL_BENCH_LM_KWARGS`` JSON, or an unknown ``--adapters``/
``--signatures`` id (``UnknownCellError``).

Dispatch: neither ``--offline`` nor ``--live`` runs **both** arms (env required for the live
half); ``--offline --live`` together is treated the same as neither, for the same reason.
``--list`` and ``--repro-1871`` short-circuit everything else and never write a file.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .adapters import ADAPTERS, LIVE_DEFAULT_ADAPTER_IDS
from .external import CORPORA
from .outcomes import ReproRow, UnknownCellError, resolve_ids
from .provenance import PROG, invocation_command
from .signatures import SIGNATURE_IDS, SIGNATURES

if TYPE_CHECKING:
    from .cases import Case
    from .config import BenchConfig
    from .report import RunMeta

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results"

_ADAPTER_ERROR = "Unknown adapter id {!r}. Valid adapter ids: {}"
_SIGNATURE_ERROR = "Unknown signature id {!r}. Valid ids are: {}"

REPRO_COLUMNS = (
    "| part | adapter | adapter_config | response_format_sent | outcome "
    "| error_class | lm_calls | note |"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the eight-flag argparse surface for `python -m benchmarking.dspy_adapters`."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "DSPy adapter benchmark: an offline prompt-cost arm, a live outcomes arm "
            "against a local Ollama endpoint, and a synthetic reproduction of DSPy "
            "issue #1871."
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Run the offline prompt-cost arm plus the synthetic #1871 reproduction. "
            "No dspy.LM and no network needed; token counts are reported as unavailable "
            "if the tiktoken cl100k_base table cannot be loaded offline. Sub-second."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the live outcomes arm only. Requires LSL_BENCH_MODEL/LSL_BENCH_API_BASE.",
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help=(
            "Run the live extraction-accuracy arm only: N labeled cases (see --corpus) "
            "scored field-by-field against ground truth. Requires the same live environment "
            "as --live. Never combined with the other arms into one score."
        ),
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "Re-score the replies an accuracy CSV recorded, with the current adapter code: "
            "no model and no env. Pass the --corpus/--cases/--cases-seed that run used. "
            "Valid only while the adapters' prompts are unchanged."
        ),
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=30,
        help="Accuracy-arm case count. Default: 30. Ignored by every other arm.",
    )
    parser.add_argument(
        "--cases-seed",
        type=int,
        default=0,
        help="Synthetic accuracy-corpus seed; recorded in the report. Default: 0.",
    )
    parser.add_argument(
        "--corpus",
        choices=("synthetic", *CORPORA),
        default="synthetic",
        help=(
            "Accuracy-arm corpus. 'synthetic' (default) is generated locally from "
            "--cases-seed; the others are third-party labeled sets fetched at a pinned "
            "revision (Hugging Face, or GitHub for patient-notes; first --cases rows; the HF "
            "ones need the `benchmark` extra). "
            "Ignored by every other arm."
        ),
    )
    parser.add_argument(
        "--adapters",
        type=str,
        default=None,
        help=(
            "Comma-separated adapter ids. Default: all 9 offline, the 6 *-sections ids "
            "live. Unknown id -> stderr listing valid ids, exit 2."
        ),
    )
    parser.add_argument(
        "--signatures",
        type=str,
        default=None,
        help="Comma-separated signature ids. Default: all 6. Unknown id -> exit 2.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Live-arm repetitions per cell. Ignored by the offline arm (deterministic).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory; created if absent. Default: <package dir>/results.",
    )
    parser.add_argument(
        "--repro-1871",
        action="store_true",
        help=(
            "Run only the #1871 work: the offline synthetic reproduction always, plus "
            "the live probe iff the live environment is configured."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print every adapter id and signature id with its config, then exit 0.",
    )
    return parser


def _render_repro_table(rows: list[ReproRow]) -> str:
    """Render #1871 reproduction rows as a markdown table (stdout only, never a file)."""
    from .report import _md_table

    cells = (
        [
            row.part,
            row.adapter,
            row.adapter_config,
            row.response_format_sent,
            row.outcome.value,
            row.error_class,
            str(row.lm_calls),
            row.note,
        ]
        for row in rows
    )
    return "\n".join(_md_table(REPRO_COLUMNS, cells))


def _write_results(write: Callable[[], tuple[Path, Path]], out: Path) -> bool:
    """Run one report writer and print both paths; on `OSError` say why and return False."""
    try:
        md_path, csv_path = write()
    except OSError as exc:
        print(f"error: could not write results to {out}: {exc}", file=sys.stderr)
        return False
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    return True


def _run_list() -> int:
    """Print `adapters:` then `signatures:` sections and exit 0. Reads no environment."""
    print("adapters:")
    for adapter_id, cell in ADAPTERS.items():
        print(f"  {adapter_id}  {cell.config_repr}")
    print("signatures:")
    for sig_id, sig_cell in SIGNATURES.items():
        input_keys = ", ".join(sig_cell.inputs)
        print(f"  {sig_id}  {sig_cell.signature.__name__}(inputs={input_keys})")
    return 0


def _maybe_live_probe() -> ReproRow | None:
    """Return the #1871 live-probe row iff the live env is configured, else None.

    Checks only whether the two required variable *names* (re-exported by `.config`) are
    present in `os.environ` -- never their values, and never via `config.load_config()`,
    which would print its fail-loud "not configured" banner. Only when both names are
    present does this call `config.load_config()` / `config.build_lm()` for real, so a
    genuine misconfiguration at that point (e.g. invalid `LSL_BENCH_LM_KWARGS` JSON) still
    exits 2 loudly, as it should.
    """
    from . import config as config_module
    from .runner import probe_1871_live

    if not (os.environ.get(config_module.ENV_MODEL) and os.environ.get(config_module.ENV_API_BASE)):
        return None

    cfg = config_module.load_config()
    lm = config_module.build_lm(cfg)
    return probe_1871_live(lm)


def _run_repro_1871() -> int:
    """Run the #1871 offline reproduction (always) plus the live probe (iff configured).

    Prints a markdown table to stdout and writes no file. `--adapters`/`--signatures`/
    `--trials`/`--out` are ignored -- the #1871 cell set is fixed.
    """
    from .runner import run_repro_1871_offline

    rows: list[ReproRow] = list(run_repro_1871_offline())
    probe_row = _maybe_live_probe()
    if probe_row is not None:
        rows.append(probe_row)
    print(_render_repro_table(rows))
    return 0


def _run_offline(args: argparse.Namespace) -> int:
    """Run the offline prompt-cost arm, write its report, then print the #1871 repro table."""
    from . import report as report_module
    from .encoding import ENCODING_NAME, seed_tiktoken_cache
    from .runner import encoding_available, run_offline_arm, run_repro_1871_offline

    seed_tiktoken_cache()

    adapter_ids = resolve_ids(args.adapters, ADAPTERS, ADAPTERS, _ADAPTER_ERROR)
    signature_ids = resolve_ids(args.signatures, SIGNATURES, SIGNATURE_IDS, _SIGNATURE_ERROR)

    rows = run_offline_arm(adapter_ids, signature_ids)
    available = encoding_available()
    if not available:
        print(
            f"warning: could not load the {ENCODING_NAME} tiktoken encoding offline; "
            "prompt_tokens is reported as unavailable in this report. Point "
            "TIKTOKEN_CACHE_DIR at a directory holding the cl100k_base blob, or run "
            "once with network access, for real counts.",
            file=sys.stderr,
        )
    meta = dataclasses.replace(
        report_module.RunMeta.minimal("prompt-cost"),
        command=invocation_command(),
        git_head=report_module.git_head(),
        encoding=ENCODING_NAME if available else f"{ENCODING_NAME} (unavailable)",
    )
    if not _write_results(
        lambda: report_module.write_offline_report(rows, args.out, meta=meta), args.out
    ):
        return 1
    print(_render_repro_table(run_repro_1871_offline()))
    return 0


def _finish_live_arm(
    arm: str,
    rows: Sequence[Any],
    cfg: BenchConfig,
    lm: object,
    out: Path,
    write: Callable[[RunMeta], tuple[Path, Path]],
    **provenance: object,
) -> int:
    """The shared tail of the two live arms: transport check, `RunMeta`, write, exit code.

    Exit 1 when the results could not be written, or when at least one cell ran and every
    row's `error_class` is exactly `"LMTransportError"`; otherwise 0. `provenance` is
    recorded in `lm_kwargs` after the effective LM kwargs.
    """
    from . import config as config_module
    from . import report as report_module

    harness_failed = bool(rows) and all(row.error_class == "LMTransportError" for row in rows)
    if harness_failed:
        print(f"error: no {arm} cell reached the endpoint; is {cfg.api_base} up?", file=sys.stderr)

    meta = dataclasses.replace(
        report_module.RunMeta.minimal(arm),
        command=invocation_command(),
        git_head=report_module.git_head(),
        model=cfg.model,
        api_base=cfg.api_base,
        lm_kwargs={**config_module.BASELINE_LM_KWARGS, **cfg.lm_kwargs, **provenance},
        supports_response_schema=getattr(lm, "supports_response_schema", None),
        supports_function_calling=getattr(lm, "supports_function_calling", None),
        integrity_overridden=cfg.integrity_overridden,
    )
    if not _write_results(lambda: write(meta), out):
        return 1
    return 1 if harness_failed else 0


def _run_live(args: argparse.Namespace) -> int:
    """Run the live outcomes arm against one shared `dspy.LM`, then write its report."""
    from . import config as config_module
    from . import report as report_module
    from .runner import run_live_arm

    adapter_ids = resolve_ids(args.adapters, ADAPTERS, LIVE_DEFAULT_ADAPTER_IDS, _ADAPTER_ERROR)
    signature_ids = resolve_ids(args.signatures, SIGNATURES, SIGNATURE_IDS, _SIGNATURE_ERROR)

    cfg = config_module.load_config()  # exits 2 itself when not configured
    lm = config_module.build_lm(cfg)

    rows = run_live_arm(
        lambda _adapter: lm,
        adapter_ids=adapter_ids,
        signature_ids=signature_ids,
        trials=args.trials,
    )
    return _finish_live_arm(
        "live",
        rows,
        cfg,
        lm,
        args.out,
        lambda meta: report_module.write_live_report(rows, args.out, meta=meta),
    )


def _run_accuracy(args: argparse.Namespace) -> int:
    """Run the live accuracy arm against one shared `dspy.LM`, then write its report."""
    from . import config as config_module
    from . import report as report_module
    from .accuracy import null_floor
    from .runner import run_accuracy_arm

    adapter_ids = resolve_ids(args.adapters, ADAPTERS, LIVE_DEFAULT_ADAPTER_IDS, _ADAPTER_ERROR)

    cfg = config_module.load_config()  # exits 2 itself when not configured
    lm = config_module.build_lm(cfg)

    cases, corpus_meta = _load_cases(args)
    rows = run_accuracy_arm(lambda _adapter: lm, cases, adapter_ids=adapter_ids)
    return _finish_live_arm(
        "accuracy",
        rows,
        cfg,
        lm,
        args.out,
        lambda meta: report_module.write_accuracy_report(
            rows,
            args.out,
            meta=meta,
            corpus=None if args.corpus == "synthetic" else args.corpus,
            null_floor=null_floor(cases),
        ),
        # Corpus identity belongs in provenance: two accuracy runs are only comparable if
        # they scored the same cases, and (n, seed) or (repo@revision, n) fixes that.
        **corpus_meta,
    )


def _load_cases(args: argparse.Namespace) -> tuple[list[Case], dict[str, object]]:
    """The accuracy corpus `--corpus`/`--cases`/`--cases-seed` select, and its provenance.

    Corpus identity belongs in provenance: two accuracy runs are only comparable if they
    scored the same cases, and (n, seed) or (repo@revision, n) fixes that.
    """
    from .cases import generate

    if args.corpus == "synthetic":
        return generate(args.cases, seed=args.cases_seed), {
            "cases": args.cases,
            "cases_seed": args.cases_seed,
        }
    from .external import load as load_corpus

    cases = load_corpus(args.corpus, args.cases)
    source = CORPORA[args.corpus]
    return cases, {"corpus": f"{source.repo}@{source.revision}", "cases": len(cases)}


def _run_replay(args: argparse.Namespace) -> int:
    """Re-score the replies an accuracy CSV recorded with today's code; no LM, no env.

    `--corpus`/`--cases`/`--cases-seed` must name the cases the recorded run scored. Exit 2
    when the CSV has no `replies` column (recorded before it existed) or names a case the
    selected corpus lacks.
    """
    import csv

    from . import report as report_module
    from .accuracy import null_floor
    from .runner import replay_accuracy

    with args.replay.open(encoding="utf-8", newline="") as handle:
        recorded = list(csv.DictReader(handle))
    cases, corpus_meta = _load_cases(args)
    if recorded and "replies" not in recorded[0]:
        print(f"error: {args.replay} has no replies column to replay", file=sys.stderr)
        return 2
    unknown = {rec["case_id"] for rec in recorded} - {case.case_id for case in cases}
    if unknown:
        print(
            f"error: {len(unknown)} recorded case ids are not in this corpus slice; pass the "
            "--corpus/--cases/--cases-seed the recorded run used",
            file=sys.stderr,
        )
        return 2
    rows = replay_accuracy(recorded, cases)
    meta = dataclasses.replace(
        report_module.RunMeta.minimal("accuracy"),
        command=invocation_command(),
        git_head=report_module.git_head(),
        model="replay",
        lm_kwargs={"replay": str(args.replay), **corpus_meta},
    )
    written = _write_results(
        lambda: report_module.write_accuracy_report(
            rows,
            args.out,
            meta=meta,
            corpus=None if args.corpus == "synthetic" else args.corpus,
            null_floor=null_floor(cases),
        ),
        args.out,
    )
    return 0 if written else 1


def main(argv: list[str] | None = None) -> int:
    """Parse argv, dispatch to the requested arm(s), and return the process exit code.

    `--list` and `--repro-1871` short-circuit everything else. Otherwise: neither
    `--offline` nor `--live` runs both arms; `--offline --live` together also runs both
    (same as neither); a single flag runs only that arm. Never raises to the caller --
    `UnknownCellError` (unknown `--adapters`/`--signatures` id) becomes exit code 2; a
    missing/invalid live configuration already exits 2 on its own, from inside `.config`.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.list:
            return _run_list()
        if args.repro_1871:
            return _run_repro_1871()
        # Explicit opt-in, and never part of the "neither flag runs both" default: the
        # accuracy arm costs N_adapters x N_cases live calls, so it must never start
        # because someone ran the benchmark with no flags at all.
        if args.replay:
            return _run_replay(args)
        if args.accuracy:
            return _run_accuracy(args)

        do_offline = args.offline or not args.live
        do_live = args.live or not args.offline

        codes = []
        if do_offline:
            codes.append(_run_offline(args))
        if do_live:
            codes.append(_run_live(args))
        return max(codes) if codes else 0
    except UnknownCellError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
