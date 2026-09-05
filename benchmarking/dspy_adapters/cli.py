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
from pathlib import Path

from .adapters import ADAPTERS, LIVE_DEFAULT_ADAPTER_IDS, resolve_adapter_ids
from .outcomes import ReproRow, UnknownCellError
from .signatures import SIGNATURES, resolve_signature_ids

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results"


def build_parser() -> argparse.ArgumentParser:
    """Build the eight-flag argparse surface for `python -m benchmarking.dspy_adapters`."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarking.dspy_adapters",
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
    header = (
        "| part | adapter | adapter_config | response_format_sent | outcome "
        "| error_class | lm_calls | note |"
    )
    sep = "|" + "|".join(["---"] * 8) + "|"
    lines = [header, sep]
    for row in rows:
        cells = [
            row.part,
            row.adapter,
            row.adapter_config,
            row.response_format_sent,
            row.outcome.value,
            row.error_class,
            str(row.lm_calls),
            row.note,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _run_list() -> int:
    """Print `adapters:` then `signatures:` sections and exit 0. Reads no environment."""
    print("adapters:")
    for adapter_id, cell in ADAPTERS.items():
        print(f"  {adapter_id}  {cell.config_repr}")
    print("signatures:")
    for sig_id, cell in SIGNATURES.items():
        input_keys = ", ".join(cell.inputs)
        print(f"  {sig_id}  {cell.signature.__name__}(inputs={input_keys})")
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

    adapter_ids = resolve_adapter_ids(args.adapters, tuple(ADAPTERS))
    signature_ids = resolve_signature_ids(args.signatures)

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
        command=" ".join(sys.argv),
        git_head=report_module.git_head(),
        encoding=ENCODING_NAME if available else f"{ENCODING_NAME} (unavailable)",
    )
    try:
        md_path, csv_path = report_module.write_offline_report(rows, args.out, meta=meta)
    except OSError as exc:
        print(f"error: could not write results to {args.out}: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    print(_render_repro_table(run_repro_1871_offline()))
    return 0


def _run_live(args: argparse.Namespace) -> int:
    """Run the live outcomes arm against one shared `dspy.LM`, then write its report."""
    from . import config as config_module
    from . import report as report_module
    from .runner import run_live_arm

    adapter_ids = resolve_adapter_ids(args.adapters, LIVE_DEFAULT_ADAPTER_IDS)
    signature_ids = resolve_signature_ids(args.signatures)

    cfg = config_module.load_config()  # exits 2 itself when not configured
    lm = config_module.build_lm(cfg)

    rows = run_live_arm(
        lambda _adapter: lm,
        adapter_ids=adapter_ids,
        signature_ids=signature_ids,
        trials=args.trials,
    )

    harness_failed = bool(rows) and all(row.error_class == "LMTransportError" for row in rows)
    if harness_failed:
        print(
            f"error: no live cell reached the endpoint; is {cfg.api_base} up?",
            file=sys.stderr,
        )

    meta = dataclasses.replace(
        report_module.RunMeta.minimal("live"),
        command=" ".join(sys.argv),
        git_head=report_module.git_head(),
        model=cfg.model,
        api_base=cfg.api_base,
        lm_kwargs={**config_module.BASELINE_LM_KWARGS, **cfg.lm_kwargs},
        supports_response_schema=getattr(lm, "supports_response_schema", None),
        supports_function_calling=getattr(lm, "supports_function_calling", None),
        integrity_overridden=cfg.integrity_overridden,
    )
    try:
        md_path, csv_path = report_module.write_live_report(rows, args.out, meta=meta)
    except OSError as exc:
        print(f"error: could not write results to {args.out}: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    return 1 if harness_failed else 0


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
