"""Shared fixtures and assertion helpers for the DSPy integration test suite.

Tier 1 (exact snapshot): assert full string equality only when the literal lives in
    this repo's own source, is short, and is free of the defects in research Q19.
Tier 2 (anchor): for whole prompts and message lists, assert only mode-discriminating
    literals and our own clauses; never assert on DSPy-owned text such as "Your input
    fields are:" or "In adhering to this structure, your objective is:".
Tier 3 (xfail): for known-wrong behaviour, assert the INTENDED behaviour under
    @pytest.mark.xfail(reason=...) naming the owning ticket, or "no owning ticket -
    research Q19.N" when there is none; never snapshot a defect.

NOTE (design D13): the Pydantic models below (`Address`, `Person`) are deliberately
NOT imported from `tests/conftest.py`. conftest's `Address` has five fields with regex
patterns and a different `country` default; every Tier-1 string asserted by the
test_dspy_*.py modules was produced by the three-field models defined here. Importing
conftest's models would silently invalidate every Tier-1 snapshot. Do not "tidy up"
this duplication.
"""

from __future__ import annotations

import asyncio
import enum
from typing import Any, Literal

import pytest

pytest.importorskip("dspy", minversion="3.3.1")

import dspy  # noqa: E402
from dspy.utils.dummies import DummyLM  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from llm_schema_lite.dspy_integration import (  # noqa: E402
    OutputMode,
    PromptLayout,
    StructuredOutputAdapter,
)


class Address(BaseModel):
    """Three-field address used to produce every Tier-1 captured string."""

    street: str
    city: str
    country: str = "US"


class Person(BaseModel):
    """Person nested under Extract.person; matches the captured fixture."""

    name: str = Field(description="Full name")
    age: int
    address: Address | None = None


class Extract(dspy.Signature):
    """Extract a person."""

    text: str = dspy.InputField()
    meta: Address = dspy.InputField()
    person: Person = dspy.OutputField()
    score: float = dspy.OutputField()


class QA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class QAOptional(dspy.Signature):
    """Answer the question, optionally noting a caveat."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    note: str | None = dspy.OutputField()


class Unordered(dspy.Signature):
    """Declares zeta before alpha to probe YAML key-order preservation (Q19.2)."""

    q: str = dspy.InputField()
    zeta: str = dspy.OutputField()
    alpha: str = dspy.OutputField()


class AliasedAddress(BaseModel):
    """Aliased address used to prove input values honour by_alias (lsl-2026-09-04-007)."""

    street_name: str = Field(alias="streetName")
    city: str

    model_config = ConfigDict(populate_by_name=True)


class Colour(str, enum.Enum):
    """Two-member string enum used by the Choices signature."""

    RED = "red"
    BLUE = "blue"


class AliasIn(dspy.Signature):
    """Probe alias-honouring, indented Pydantic input rendering (lsl-2026-09-04-007)."""

    text: str = dspy.InputField()
    addr: AliasedAddress = dspy.InputField()


class ListOut(dspy.Signature):
    """Extract a list of people."""

    text: str = dspy.InputField()
    items: list[Person] = dspy.OutputField()


class Choices(dspy.Signature):
    """Probe Optional/Literal/Enum output rendering (lsl-2026-09-04-007)."""

    text: str = dspy.InputField()
    colour: Colour = dspy.OutputField()
    tier: Literal["a", "b"] = dspy.OutputField()
    note: str | None = dspy.OutputField()


class HistoryIn(dspy.Signature):
    """Probe dspy.History carve-out and input rendering (lsl-2026-09-04-007)."""

    history: dspy.History = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ImageIn(dspy.Signature):
    """Probe dspy.Image custom-type marker survival (lsl-2026-09-04-007)."""

    img: dspy.Image = dspy.InputField()
    caption: str = dspy.OutputField()


JSON_OUTPUT_HEADER = "Outputs will be a JSON object with the following fields."
YAML_OUTPUT_HEADER = "Outputs will be in YAML format with the following fields."
DSPY_OWNED_TEXT = (
    "Your input fields are:",
    "In adhering to this structure, your objective is:",
)

REQUIRED_LEGEND = {
    "jsonish": "// Fields marked with * are required",
    "yaml": "# Fields marked with * are required",
}


def make_adapter(
    mode: OutputMode, layout: PromptLayout = PromptLayout.SECTIONS
) -> StructuredOutputAdapter:
    """Return a StructuredOutputAdapter configured for the given output mode and layout."""
    return StructuredOutputAdapter(output_mode=mode, prompt_layout=layout)


class SchemaCapableDummyLM(DummyLM):  # type: ignore[misc]
    """DummyLM subclass that advertises response_format/schema support.

    Plain DummyLM reports `supported_params == set()` and
    `supports_response_schema is False`, so it can never reach the adapter's
    response_format branches. This subclass overrides both and records the
    lm_kwargs each call was invoked with, so tests can assert on `response_format`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize like DummyLM, then seed the recorded-kwargs list."""
        super().__init__(*args, **kwargs)
        self.lm_kwargs_history: list[dict[str, Any]] = []

    @property
    def supported_params(self) -> set[str]:
        """Return {"response_format"} instead of DummyLM's empty set."""
        return {"response_format"}

    @property
    def supports_response_schema(self) -> bool:
        """Return True instead of DummyLM's False."""
        return True

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to DummyLM.forward, recording lm_kwargs into history first."""
        self.lm_kwargs_history.append(dict(kwargs))
        return super().forward(*args, **kwargs)


def call_sync_and_async(
    adapter: StructuredOutputAdapter,
    lm: DummyLM,
    signature: type[dspy.Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run adapter(...) and asyncio.run(adapter.acall(...)) and return both results."""
    sync_result = adapter(lm, {}, signature, demos, inputs)
    async_result = asyncio.run(adapter.acall(lm, {}, signature, demos, inputs))
    return sync_result, async_result


def assert_mode_header(text: str, mode: OutputMode) -> None:
    """Assert `text` contains the one header literal that matches `mode` and no other."""
    expected = YAML_OUTPUT_HEADER if mode is OutputMode.YAML else JSON_OUTPUT_HEADER
    other = JSON_OUTPUT_HEADER if mode is OutputMode.YAML else YAML_OUTPUT_HEADER
    assert expected in text, f"Expected {mode} header {expected!r} missing. Snippet: {text[:200]!r}"
    assert (
        other not in text
    ), f"Unexpected header {other!r} present for {mode}. Snippet: {text[:200]!r}"


def assert_has_field_marker(text: str, name: str) -> None:
    """Assert `text` contains the `[[ ## name ## ]]` field marker."""
    marker = f"[[ ## {name} ## ]]"
    assert marker in text, f"Marker {marker!r} missing. Snippet: {text[:200]!r}"


def assert_note_clause(text: str, field: str) -> None:
    """Assert `text` has a `# note:` clause immediately after `field`'s placeholder."""
    placeholder = "{" + field + "}"
    idx = text.find(placeholder)
    assert idx != -1, f"Placeholder {placeholder!r} missing. Snippet: {text[:200]!r}"
    tail = text[idx + len(placeholder) :]
    assert tail.startswith("        # note:"), (
        f"Field {field!r} placeholder is not followed by a '# note:' clause. "
        f"Tail snippet: {tail[:200]!r}"
    )


def assert_no_dspy_owned_text(text: str) -> None:
    """Assert `text` contains no DSPy-owned system-message boilerplate."""
    for owned in DSPY_OWNED_TEXT:
        assert (
            owned not in text
        ), f"DSPy-owned text {owned!r} leaked into our output. Snippet: {text[:200]!r}"


def assert_message_roles(messages: list[dict[str, Any]], expected: list[str]) -> None:
    """Assert `messages` has role sequence exactly `expected`, else show the actual list."""
    actual = [m["role"] for m in messages]
    assert actual == expected, f"Role sequence mismatch: expected={expected} actual={actual}"
