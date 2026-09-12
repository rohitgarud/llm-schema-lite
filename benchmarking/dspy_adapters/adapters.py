"""Adapter registry for the DSPy adapter benchmark.

This module is pure data: it builds zero-arg factories for each adapter under
test and never touches an LM, the environment, or I/O. ``ChatAdapter`` is
always constructed with ``use_json_adapter_fallback=False`` -- measured, the
default ``True`` turns a real ``AdapterParseError`` into a fake ``ok`` result
at ``lm_calls=2``, which would silently mask parsing failures. ``BAMLAdapter``
is importable only from ``dspy.adapters.baml_adapter``; ``dspy.BAMLAdapter``
does not exist. ``OutputMode`` is a plain ``enum.Enum`` (not a ``str`` mixin),
so every factory below passes the enum member itself, never a bare string.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dspy.adapters import ChatAdapter, JSONAdapter
from dspy.adapters.baml_adapter import BAMLAdapter  # C1: the only working import path
from dspy.adapters.base import Adapter

from llm_schema_lite import ParseConfig
from llm_schema_lite.dspy_integration import (
    OutputMode,
    PromptLayout,
    StructuredOutputAdapter,
)


@dataclass(frozen=True)
class AdapterCell:
    """One adapter row of the matrix: a zero-arg factory and its config string."""

    factory: Callable[[], Adapter]
    config_repr: str


def _sola(
    mode: OutputMode, layout: PromptLayout, rescue: bool = False, partial: bool = False
) -> AdapterCell:
    """A `StructuredOutputAdapter` cell; `rescue`/`partial` pick its `parse_config`.

    `parse_config=None` is the constructor's own default, so a cell with neither flag
    builds exactly what omitting the argument builds. `partial` is `rescue` with
    `ParseConfig(partial=True)`: leaf salvage only runs once the structural repairs have
    already failed, so a partial cell is its rescue twin with one more thing armed.

    Each call builds a fresh `ParseConfig`, so no two cells share one instance.
    """
    extra = ""
    if partial:
        extra = ", parse_config=ParseConfig(partial=True)"
    elif rescue:
        extra = ", parse_config=ParseConfig()"

    def build() -> Adapter:
        config: ParseConfig | None = None
        if partial:
            config = ParseConfig(partial=True)
        elif rescue:
            config = ParseConfig()
        return StructuredOutputAdapter(
            output_mode=mode,
            prompt_layout=layout,
            parse_config=config,
        )

    return AdapterCell(
        factory=build,
        config_repr=(
            f"StructuredOutputAdapter(output_mode={mode.name}, prompt_layout={layout.name}{extra})"
        ),
    )


ADAPTERS: dict[str, AdapterCell] = {
    "chat": AdapterCell(
        factory=lambda: ChatAdapter(use_json_adapter_fallback=False),
        config_repr="ChatAdapter(use_json_adapter_fallback=False)",
    ),
    "json": AdapterCell(factory=lambda: JSONAdapter(), config_repr="JSONAdapter()"),
    "baml": AdapterCell(factory=lambda: BAMLAdapter(), config_repr="BAMLAdapter()"),
    "sola-json-sections": _sola(OutputMode.JSON, PromptLayout.SECTIONS),
    "sola-jsonish-sections": _sola(OutputMode.JSONISH, PromptLayout.SECTIONS),
    "sola-yaml-sections": _sola(OutputMode.YAML, PromptLayout.SECTIONS),
    "sola-json-block": _sola(OutputMode.JSON, PromptLayout.JSON_BLOCK),
    "sola-jsonish-block": _sola(OutputMode.JSONISH, PromptLayout.JSON_BLOCK),
    "sola-json-rescue": _sola(OutputMode.JSON, PromptLayout.SECTIONS, rescue=True),
    "sola-jsonish-rescue": _sola(OutputMode.JSONISH, PromptLayout.SECTIONS, rescue=True),
    "sola-yaml-rescue": _sola(OutputMode.YAML, PromptLayout.SECTIONS, rescue=True),
    "sola-json-partial": _sola(OutputMode.JSON, PromptLayout.SECTIONS, partial=True),
    "sola-jsonish-partial": _sola(OutputMode.JSONISH, PromptLayout.SECTIONS, partial=True),
    "sola-yaml-partial": _sola(OutputMode.YAML, PromptLayout.SECTIONS, partial=True),
    "sola-yaml-block": _sola(OutputMode.YAML, PromptLayout.JSON_BLOCK),
}


LIVE_DEFAULT_ADAPTER_IDS: tuple[str, ...] = (
    "chat",
    "json",
    "baml",
    "sola-json-sections",
    "sola-jsonish-sections",
    "sola-yaml-sections",
    "sola-jsonish-rescue",
    "sola-yaml-rescue",
)
"""The live arms default to the six *-sections ids plus the two parse-time rescue cells.

`sola-jsonish-rescue` differs from `sola-jsonish-sections` in exactly one thing --
`parse_config=ParseConfig()`, which arms the coercion rescue and the structural repair
(copied nested markers stripped, list-wrapped objects unwrapped, hoisted fields re-nested,
all-null objects nulled, lone values wrapped in lists, all-null list items dropped, and a
missing output key restored) --
so the pair isolates what parse-time repair is worth on a given model. Its PROMPT is
byte-identical to `sola-jsonish-sections`, which the offline arm makes visible: the two
rows must agree on every token count, or the cell is measuring more than it claims.
Measured on qwen3.5:0.8b, 30 accuracy cases: 0.745 -> 0.929 field accuracy, 24/30 ->
30/30 parsed; inert on granite3.1-moe:1b and llama3.2:1b.

`sola-yaml-rescue` is the same controlled pair for YAML. What it rescues is the same
all-null list item, not YAML's typed scalars (`postcode: 95014` loads as an int): the
coercion rescue declines nested-model fields, so that failure passes through. Measured
on qwen3.5:0.8b over 40 accuracy cases, against `sola-yaml-sections` on the same run:
0.714 -> 0.848 field accuracy, 16/40 -> 22/40 exact records, 11 -> 5 validation errors.

`sola-json-rescue` is the JSON twin and is opt-in (`--adapters`), not a default: it was
added after the `2026-09-10` accuracy artefacts, which it would otherwise leave
incomplete. The repair it isolates is the list unwrap -- JSON mode is the only mode seen
sending a whole record as a one-item list. Replaying qwen3.5:0.8b's financial-NER
completions with and without it: 9/30 -> 26/30 parsed, 0.209 -> 0.606 field accuracy.
"""

REPRO_1871_ADAPTER_IDS: tuple[str, ...] = (
    "chat",
    "json",
    "baml",
    "sola-json-sections",
    "sola-jsonish-sections",
    "sola-yaml-sections",
    "sola-jsonish-nojsonobject",
)
"""The seven ids of the #1871 cell set, in the exact order of its tables."""


REPRO_1871_ADAPTERS: dict[str, AdapterCell] = {
    key: ADAPTERS[key] for key in REPRO_1871_ADAPTER_IDS[:-1]
}
REPRO_1871_ADAPTERS["sola-jsonish-nojsonobject"] = AdapterCell(
    factory=lambda: StructuredOutputAdapter(
        output_mode=OutputMode.JSONISH,
        use_json_object_response_format=False,
    ),
    config_repr=(
        "StructuredOutputAdapter(output_mode=JSONISH, use_json_object_response_format=False)"
    ),
)
"""``ADAPTERS`` restricted to the first six #1871 ids, plus a repro-only tenth entry.

The ``sola-jsonish-nojsonobject`` id exists ONLY here -- never in ``ADAPTERS``.
"""
