"""Signature fixtures for the DSPy adapter benchmark.

The six signature fixtures and their Pydantic models defined in this module are pure
data — no LM, no env, no I/O; they are shaped after `tests/dspy_helpers.py` but
deliberately not imported from it (the package must be self-contained and the
dependency edge points `tests/` -> `benchmarking` only); `flat` deliberately carries a
second output field so the matrix has one multi-field coercion target outside the
Pydantic cases.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Literal

import dspy
import pydantic


@dataclass(frozen=True)
class SignatureCell:
    """One signature row of the matrix: the dspy.Signature class and its input dict."""

    signature: type[dspy.Signature]
    inputs: dict[str, Any]


# NOTE: the fixture models below deliberately carry NO class docstring. Pydantic emits a
# class docstring as the JSON-schema "description", which inflates every adapter's prompt
# and would silently change the committed prompt-cost numbers.
class Address(pydantic.BaseModel):
    street: str
    city: str


class Person(pydantic.BaseModel):
    name: str
    age: int
    address: Address | None = None


class Colour(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Node(pydantic.BaseModel):
    value: str
    children: list[Node] = pydantic.Field(default_factory=list)


Node.model_rebuild()  # required — resolves the forward reference


class Flat(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    confidence: float = dspy.OutputField()


class Nested(dspy.Signature):
    """Extract the person."""

    text: str = dspy.InputField()
    person: Person = dspy.OutputField()


class ListOfModel(dspy.Signature):
    """Extract all people."""

    text: str = dspy.InputField()
    people: list[Person] = dspy.OutputField()


class EnumSig(dspy.Signature):
    """Classify."""

    text: str = dspy.InputField()
    colour: Colour = dspy.OutputField()
    tier: Literal["a", "b"] = dspy.OutputField()


class OptionalSig(dspy.Signature):
    """Answer with an optional note."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    note: str | None = dspy.OutputField()


class Recursive(dspy.Signature):
    """Build a tree."""

    text: str = dspy.InputField()
    tree: Node = dspy.OutputField()


SIGNATURES: dict[str, SignatureCell] = {
    "flat": SignatureCell(Flat, {"question": "What colour is the sky?"}),
    "nested": SignatureCell(Nested, {"text": "Ada Lovelace, 36, 12 Baker St, London"}),
    "list_of_model": SignatureCell(ListOfModel, {"text": "Ada 36; Alan 41"}),
    "enum": SignatureCell(EnumSig, {"text": "The sky at noon"}),
    "optional": SignatureCell(OptionalSig, {"question": "What is 2+2?"}),
    "recursive": SignatureCell(Recursive, {"text": "root with two leaves a and b"}),
}

SIGNATURE_IDS: tuple[str, ...] = tuple(SIGNATURES)
