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
from dspy.utils.exceptions import LMError  # noqa: E402
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


class Typed(dspy.Signature):
    """Typed outputs with an explicit default and a Literal, for the ParseConfig tests.

    QAOptional cannot serve them: its only optional field is `note: str | None`, which
    almost any scalar satisfies, so no input makes parse_value fail; and it has no
    field with an explicit `default=`, so it can only reach apply_output_field_defaults
    case 3 (annotation allows None). `count` supplies a genuinely-invalidatable value
    AND case 2 (explicit default); `tier` supplies the Literal case the coercion
    rescue acts on.
    """

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    count: int = dspy.OutputField(default=0)
    tier: Literal["a", "b"] = dspy.OutputField(default="a")


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


class ImageListIn(dspy.Signature):
    """Probe list[dspy.Image] input carve-out (lsl-2026-09-05-007, AC-3)."""

    images: list[dspy.Image] = dspy.InputField()
    caption: str = dspy.OutputField()


class OptImageIn(dspy.Signature):
    """Probe Optional[dspy.Image] input carve-out (lsl-2026-09-05-007, ticket Risk)."""

    image: dspy.Image | None = dspy.InputField()
    caption: str = dspy.OutputField()


def _echo(text: str) -> str:
    """Echo the input.

    A named function rather than a lambda so ruff stays quiet and the description
    dspy.Tool derives from it stays stable across runs.
    """
    return text


ECHO_TOOL = dspy.Tool(_echo, name="echo", desc="Echo the input.")

TOOL_INPUTS = {"question": "?", "tools": [ECHO_TOOL]}

# A ToolCalls payload the adapter's own parser accepts, so a tool-signature call
# completes in exactly one LM call instead of falling back. See plan correction C2.
TOOL_ANSWER = {"tool_calls": {"tool_calls": [{"name": "echo", "args": {"text": "hi"}}]}}


class ToolCallSig(dspy.Signature):
    """Tool input plus ToolCalls output.

    The list[Tool] INPUT is mandatory, not decoration: Adapter._call_preprocess
    (dspy/adapters/base.py:100-105) raises ValueError for a ToolCalls output field with
    no Tool input.
    """

    question: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField()
    tool_calls: dspy.ToolCalls = dspy.OutputField()


class ToolInputSig(dspy.Signature):
    """Tool input with NO ToolCalls output.

    The only fixture that discriminates the BROAD tool predicate (_has_tool_fields,
    JSONish) from the NARROW one (_has_tool_calls_output, JSON mode) — design D5.
    """

    question: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField()
    answer: str = dspy.OutputField()


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
    mode: OutputMode, layout: PromptLayout = PromptLayout.SECTIONS, **kwargs: Any
) -> StructuredOutputAdapter:
    """Return a StructuredOutputAdapter configured for the given output mode and layout.

    Extra keyword arguments are forwarded verbatim to the constructor (for example
    use_json_object_response_format=False or parallel_tool_calls=True), which keeps the
    existing two-positional call sites working untouched.
    """
    return StructuredOutputAdapter(output_mode=mode, prompt_layout=layout, **kwargs)


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


class JsonObjectOnlyDummyLM(SchemaCapableDummyLM):  # type: ignore[misc]
    """Advertises response_format but cannot do structured-output schemas.

    This is LM Studio's shape, and the reason JSON mode must still fall back to
    {"type": "json_object"} rather than sending a Pydantic model.
    """

    @property
    def supports_response_schema(self) -> bool:
        """Return False, overriding SchemaCapableDummyLM's True."""
        return False


class SchemaCapableRaisingLM(SchemaCapableDummyLM):  # type: ignore[misc]
    """Schema-capable LM that records lm_kwargs, then raises LMError (design D8).

    Recording happens *before* the raise; otherwise the "exactly one LM call"
    assertion has nothing to count.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Record lm_kwargs into lm_kwargs_history, then raise LMError."""
        self.lm_kwargs_history.append(dict(kwargs))
        raise LMError("boom")


class FunctionCallingDummyLM(SchemaCapableDummyLM):  # type: ignore[misc]
    """Schema-capable LM that also advertises native function-calling support.

    DummyLM.supports_function_calling is False (dspy/clients/base_lm.py:266-269), which
    makes parallel_tool_calls unobservable; without this double the pass-through test
    would pass vacuously.
    """

    @property
    def supports_function_calling(self) -> bool:
        """Return True, overriding DummyLM's False."""
        return True


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


def recorded_calls(lm: DummyLM) -> list[dict[str, Any]]:
    """Return the lm_kwargs recorded for every call this LM received, oldest first.

    SchemaCapableDummyLM and its subclasses record into lm_kwargs_history; a plain
    DummyLM does not, so its BaseLM history is read instead. One lookup serves both.
    """
    history = getattr(lm, "lm_kwargs_history", None)
    if history is not None:
        return history
    return [entry["kwargs"] for entry in lm.history]


def recorded_response_format(lm: DummyLM, index: int = 0) -> Any:
    """Return the response_format recorded for call `index`, or None if never set."""
    return recorded_calls(lm)[index].get("response_format")


def response_format_projection(value: Any) -> Any:
    """Project a recorded response_format into something comparable with `==`.

    A class produced by pydantic.create_model is never `==` another instance of itself,
    so a pydantic class projects to (class name, sorted field names). None and dict
    values pass through unchanged.
    """
    if value is None or isinstance(value, dict):
        return value
    return (value.__name__, sorted(value.model_fields))


def assert_mode_header(text: str, mode: OutputMode) -> None:
    """Assert `text` contains the one header literal that matches `mode` and no other."""
    expected = YAML_OUTPUT_HEADER if mode is OutputMode.YAML else JSON_OUTPUT_HEADER
    other = JSON_OUTPUT_HEADER if mode is OutputMode.YAML else YAML_OUTPUT_HEADER
    assert expected in text, f"Expected {mode} header {expected!r} missing. Snippet: {text[:200]!r}"
    assert other not in text, (
        f"Unexpected header {other!r} present for {mode}. Snippet: {text[:200]!r}"
    )


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
        assert owned not in text, (
            f"DSPy-owned text {owned!r} leaked into our output. Snippet: {text[:200]!r}"
        )


def assert_message_roles(messages: list[dict[str, Any]], expected: list[str]) -> None:
    """Assert `messages` has role sequence exactly `expected`, else show the actual list."""
    actual = [m["role"] for m in messages]
    assert actual == expected, f"Role sequence mismatch: expected={expected} actual={actual}"
