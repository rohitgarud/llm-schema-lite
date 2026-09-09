"""DSPy adapter benchmark harness — package surface.

This package benchmarks nine DSPy adapters across six ``dspy.Signature`` fixtures in
two arms that never share a row or a file:

- **Offline ``prompt-cost`` arm** — ``adapter.format(sig, [], inputs)`` plus ``tiktoken``,
  producing ``PromptRow`` records. Exact, deterministic, model-free, sub-second, and
  requires no ``dspy.LM``. It requires no network either, given a reachable ``tiktoken``
  cache; without one it reports ``prompt_tokens`` as unavailable instead of failing.
- **Live ``outcomes`` arm** — real ``dspy.Predict`` calls against a local LM, producing
  ``TrialRow`` records. Scores *validity*: did the reply parse and satisfy the schema.
- **Live ``accuracy`` arm** — the same live calls over synthetic labeled cases, scored
  field-by-field against ground truth, producing ``AccuracyRow`` records. Scores
  *correctness*, which validity cannot: a reply can be perfectly shaped and entirely wrong.

The arms are kept structurally separate: distinct dataclasses, distinct CSV files,
distinct markdown reports, with no joined table anywhere.

This module deliberately imports **no** ``config`` and **no** ``cli``, so importing this
package never touches ``os.environ`` or the network.
"""

from __future__ import annotations

from .accuracy import AccuracyRow, FieldScore, score
from .adapters import ADAPTERS
from .cases import Case, generate
from .outcomes import Outcome, PromptRow, TrialRow, classify
from .report import write_accuracy_report, write_live_report, write_offline_report
from .runner import run_accuracy_arm, run_live_arm, run_offline_arm, run_repro_1871_offline
from .signatures import SIGNATURES

__all__ = [
    "ADAPTERS",
    "SIGNATURES",
    "Outcome",
    "classify",
    "PromptRow",
    "TrialRow",
    "AccuracyRow",
    "FieldScore",
    "Case",
    "score",
    "generate",
    "run_offline_arm",
    "run_live_arm",
    "run_accuracy_arm",
    "run_repro_1871_offline",
    "write_offline_report",
    "write_live_report",
    "write_accuracy_report",
]
