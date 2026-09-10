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
    """One adapter row of the matrix: its id, a zero-arg factory, and its config string."""

    id: str
    factory: Callable[[], Adapter]
    config_repr: str


ADAPTERS: dict[str, AdapterCell] = {
    "chat": AdapterCell(
        id="chat",
        factory=lambda: ChatAdapter(use_json_adapter_fallback=False),
        config_repr="ChatAdapter(use_json_adapter_fallback=False)",
    ),
    "json": AdapterCell(
        id="json",
        factory=lambda: JSONAdapter(),
        config_repr="JSONAdapter()",
    ),
    "baml": AdapterCell(
        id="baml",
        factory=lambda: BAMLAdapter(),
        config_repr="BAMLAdapter()",
    ),
    "sola-json-sections": AdapterCell(
        id="sola-json-sections",
        factory=lambda: StructuredOutputAdapter(
            output_mode=OutputMode.JSON,
            prompt_layout=PromptLayout.SECTIONS,
        ),
        config_repr="StructuredOutputAdapter(output_mode=JSON, prompt_layout=SECTIONS)",
    ),
    "sola-jsonish-sections": AdapterCell(
        id="sola-jsonish-sections",
        factory=lambda: StructuredOutputAdapter(
            output_mode=OutputMode.JSONISH,
            prompt_layout=PromptLayout.SECTIONS,
        ),
        config_repr="StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS)",
    ),
    "sola-yaml-sections": AdapterCell(
        id="sola-yaml-sections",
        factory=lambda: StructuredOutputAdapter(
            output_mode=OutputMode.YAML,
            prompt_layout=PromptLayout.SECTIONS,
        ),
        config_repr="StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS)",
    ),
    "sola-json-block": AdapterCell(
        id="sola-json-block",
        factory=lambda: StructuredOutputAdapter(
            output_mode=OutputMode.JSON,
            prompt_layout=PromptLayout.JSON_BLOCK,
        ),
        config_repr="StructuredOutputAdapter(output_mode=JSON, prompt_layout=JSON_BLOCK)",
    ),
    "sola-jsonish-block": AdapterCell(
        id="sola-jsonish-block",
        factory=lambda: StructuredOutputAdapter(
            output_mode=OutputMode.JSONISH,
            prompt_layout=PromptLayout.JSON_BLOCK,
        ),
        config_repr="StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=JSON_BLOCK)",
    ),
    "sola-jsonish-rescue": AdapterCell(
        id="sola-jsonish-rescue",
        factory=lambda: StructuredOutputAdapter(
            output_mode=OutputMode.JSONISH,
            prompt_layout=PromptLayout.SECTIONS,
            parse_config=ParseConfig(),
        ),
        config_repr=(
            "StructuredOutputAdapter(output_mode=JSONISH, prompt_layout=SECTIONS, "
            "parse_config=ParseConfig())"
        ),
    ),
    "sola-yaml-rescue": AdapterCell(
        id="sola-yaml-rescue",
        factory=lambda: StructuredOutputAdapter(
            output_mode=OutputMode.YAML,
            prompt_layout=PromptLayout.SECTIONS,
            parse_config=ParseConfig(),
        ),
        config_repr=(
            "StructuredOutputAdapter(output_mode=YAML, prompt_layout=SECTIONS, "
            "parse_config=ParseConfig())"
        ),
    ),
    "sola-yaml-block": AdapterCell(
        id="sola-yaml-block",
        factory=lambda: StructuredOutputAdapter(
            output_mode=OutputMode.YAML,
            prompt_layout=PromptLayout.JSON_BLOCK,
        ),
        config_repr="StructuredOutputAdapter(output_mode=YAML, prompt_layout=JSON_BLOCK)",
    ),
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
`parse_config=ParseConfig()`, which arms the coercion and all-null-list-item rescues --
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
    id="sola-jsonish-nojsonobject",
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


def resolve_adapter_ids(raw: str | None, default: tuple[str, ...]) -> list[str]:
    """Parse a comma-separated ``--adapters`` value against ``ADAPTERS``.

    Returns ``list(default)`` when ``raw`` is ``None``. Each id is stripped of
    surrounding whitespace. Raises ``UnknownCellError`` naming the offending id
    and listing every valid id when an id is not a key of ``ADAPTERS``.
    """
    from .outcomes import UnknownCellError

    if raw is None:
        return list(default)

    ids = [item.strip() for item in raw.split(",")]
    for adapter_id in ids:
        if adapter_id not in ADAPTERS:
            valid = ", ".join(ADAPTERS)
            raise UnknownCellError(f"Unknown adapter id {adapter_id!r}. Valid adapter ids: {valid}")
    return ids
