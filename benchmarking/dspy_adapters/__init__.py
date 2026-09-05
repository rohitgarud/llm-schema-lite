"""DSPy adapter benchmark harness — package surface.

This package benchmarks nine DSPy adapters across six ``dspy.Signature`` fixtures in
two arms that never share a row or a file:

- **Offline ``prompt-cost`` arm** — ``adapter.format(sig, [], inputs)`` plus ``tiktoken``,
  producing ``PromptRow`` records. Exact, deterministic, model-free, sub-second, and
  requires no ``dspy.LM``. It requires no network either, given a reachable ``tiktoken``
  cache; without one it reports ``prompt_tokens`` as unavailable instead of failing.
- **Live ``outcomes`` arm** — real ``dspy.Predict`` calls against a local LM, producing
  ``TrialRow`` records.

The two arms are kept structurally separate: distinct dataclasses, distinct CSV files,
distinct markdown reports, with no joined table anywhere.

This module deliberately imports **no** ``config`` and **no** ``cli``, so importing this
package never touches ``os.environ`` or the network.
"""

from __future__ import annotations

from .adapters import ADAPTERS
from .outcomes import Outcome, PromptRow, TrialRow, classify
from .report import write_live_report, write_offline_report
from .runner import run_live_arm, run_offline_arm, run_repro_1871_offline
from .signatures import SIGNATURES

__all__ = [
    "ADAPTERS",
    "SIGNATURES",
    "Outcome",
    "classify",
    "PromptRow",
    "TrialRow",
    "run_offline_arm",
    "run_live_arm",
    "run_repro_1871_offline",
    "write_offline_report",
    "write_live_report",
]
