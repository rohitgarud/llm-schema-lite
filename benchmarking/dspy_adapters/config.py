"""Live-run configuration for the DSPy adapter benchmark.

This is the **only** module in the ``benchmarking.dspy_adapters`` package that touches
``os.environ``. It reads the four ``LSL_BENCH_*`` variables (``LSL_BENCH_MODEL``,
``LSL_BENCH_API_BASE``, ``LSL_BENCH_API_KEY``, ``LSL_BENCH_LM_KWARGS``), fails loudly with
a "not configured" stderr message and ``sys.exit(2)`` when the required variables are
missing, and builds the ``dspy.LM`` used by the live arm via :func:`build_lm`.

``cache=False`` and ``num_retries=0`` in :data:`BASELINE_LM_KWARGS` are
integrity-critical: they prevent the live arm from silently serving cached or
transparently-retried responses that would corrupt the ``lm_calls`` / fallback-detection
bookkeeping (see Anti-masking, R9). An override of either value via
``LSL_BENCH_LM_KWARGS`` is allowed, but it is never silent — :func:`load_config` sets
``integrity_overridden=True`` and prints a loud stderr warning when this happens.

``benchmarking/format_jsonschembench_schema.py:21-23`` is the in-repo precedent for the
"print to stderr, then ``sys.exit``" fail-loud pattern used here.

This module is never imported by ``__init__.py``, ``runner.py``, or the smoke test —
those must remain importable and runnable fully offline, with no dependency on
``os.environ``. ``encoding.py`` follows the same quarantine for the same reason: it is
the only other module here that writes ``os.environ``, it does so only inside
``seed_tiktoken_cache``, and only ``cli.py`` calls that, lazily inside ``_run_offline``
(``runner.py`` imports nothing from it but the ``ENCODING_NAME`` constant).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, NoReturn

import dspy

ENV_MODEL = "LSL_BENCH_MODEL"
ENV_API_BASE = "LSL_BENCH_API_BASE"
ENV_API_KEY = "LSL_BENCH_API_KEY"
ENV_LM_KWARGS = "LSL_BENCH_LM_KWARGS"

DEFAULT_API_KEY = "not-needed"

BASELINE_LM_KWARGS: dict[str, Any] = {
    "temperature": 0.0,
    "max_tokens": 900,
    "cache": False,
    "num_retries": 0,
    "seed": 7,
}
"""All verified end-to-end. max_tokens=900 is a MEASURED FLOOR, not a round number:
empty_response was observed at 300 because qwen3's thinking trace exhausts the budget.
Lowering it fills the live arm with empty_response rows."""

INTEGRITY_VALUES: dict[str, Any] = {"cache": False, "num_retries": 0}


@dataclass
class BenchConfig:
    """Resolved live-run configuration read from the four LSL_BENCH_* variables."""

    model: str
    api_base: str
    api_key: str
    lm_kwargs: dict[str, Any] = field(default_factory=dict)
    integrity_overridden: bool = False


NOT_CONFIGURED_BLOCK = """
  export LSL_BENCH_MODEL=openai/qwen3:8b
  export LSL_BENCH_API_BASE=http://localhost:11434/v1
  export LSL_BENCH_API_KEY=not-needed                # optional, default "not-needed"
  export LSL_BENCH_LM_KWARGS='{"max_tokens": 900}'   # optional, JSON object

The offline arm needs no endpoint:
  make bench-dspy BENCH_ARGS=--offline
"""


def fail_not_configured(missing: list[str]) -> NoReturn:
    """Print a fail-loud "not configured" message to stderr and exit with code 2."""
    verb = "is" if len(missing) == 1 else "are"
    print(f"error: {', '.join(missing)} {verb} required for a live run.", file=sys.stderr)
    print(NOT_CONFIGURED_BLOCK.rstrip("\n"), file=sys.stderr)
    sys.exit(2)


def load_config() -> BenchConfig:
    """Read the four LSL_BENCH_* variables and build a BenchConfig, or exit 2."""
    missing = []
    if not os.environ.get(ENV_MODEL):
        missing.append(ENV_MODEL)
    if not os.environ.get(ENV_API_BASE):
        missing.append(ENV_API_BASE)
    if missing:
        fail_not_configured(missing)

    model = os.environ[ENV_MODEL]
    api_base = os.environ[ENV_API_BASE]
    api_key = os.environ.get(ENV_API_KEY) or DEFAULT_API_KEY

    raw_kwargs = os.environ.get(ENV_LM_KWARGS)
    lm_kwargs: dict[str, Any] = {}
    if raw_kwargs:
        try:
            parsed = json.loads(raw_kwargs)
        except json.JSONDecodeError as e:
            print(f"error: {ENV_LM_KWARGS} is not valid JSON: {e}", file=sys.stderr)
            sys.exit(2)
        if not isinstance(parsed, dict):
            print(f"error: {ENV_LM_KWARGS} is not valid JSON: not an object", file=sys.stderr)
            sys.exit(2)
        lm_kwargs = parsed

    integrity_overridden = False
    for key, expected in INTEGRITY_VALUES.items():
        if key in lm_kwargs and lm_kwargs[key] != expected:
            integrity_overridden = True
            print(
                f"warning: {ENV_LM_KWARGS} overrides integrity-critical setting "
                f"{key}={lm_kwargs[key]!r} (expected {expected!r}); "
                "results may not be trustworthy.",
                file=sys.stderr,
            )

    return BenchConfig(
        model=model,
        api_base=api_base,
        api_key=api_key,
        lm_kwargs=lm_kwargs,
        integrity_overridden=integrity_overridden,
    )


def build_lm(cfg: BenchConfig) -> dspy.LM:
    """Build the dspy.LM used by the live arm, merging cfg.lm_kwargs last."""
    return dspy.LM(
        cfg.model,
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        **{**BASELINE_LM_KWARGS, **cfg.lm_kwargs},
    )
