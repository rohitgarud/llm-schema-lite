"""Signature fixtures for the DSPy adapter benchmark.

The seven signature fixtures and their Pydantic models defined in this module are pure
data — no LM, no env, no I/O; they are shaped after `tests/dspy_helpers.py` but
deliberately not imported from it (the package must be self-contained and the
dependency edge points `tests/` -> `benchmarking` only); `flat` deliberately carries a
second output field so the matrix has one multi-field coercion target outside the
Pydantic cases.

`ref_heavy` is the odd one out and exists for a reason the others do not cover: it is the
only fixture whose render emits a back-reference (`object // defined above: PostalAddress`).
That marker is measured over the corpus in `docs/JSONSchemaBench_feature_report.md` §4 --
but purely as *token* arithmetic. Whether a small model can still populate a field whose
body has been replaced by a pointer to an earlier one is an accuracy question, and no
other cell here asks it. `recursive` is not a substitute: its placeholder means "stop",
this one means "look up".
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


class PostalAddress(pydantic.BaseModel):
    # Five described fields, so the rendered body clears `backreference_min_chars` (200)
    # and a repeat is named rather than inlined. Trim it and the fixture stops testing
    # the thing it is here for.
    street: str = pydantic.Field(description="Street name and house or building number")
    city: str = pydantic.Field(description="City or town")
    region: str | None = pydantic.Field(default=None, description="State, province or region")
    postcode: str | None = pydantic.Field(default=None, description="Postal or ZIP code")
    country: str = pydantic.Field(description="ISO 3166-1 alpha-2 country code")


class Party(pydantic.BaseModel):
    name: str
    address: PostalAddress


class Shipment(pydantic.BaseModel):
    # PostalAddress is reachable five times over three shapes -- directly, through Party,
    # and through a list -- and Party itself twice, so both a leaf def and a def that
    # *contains* one get named on repeat.
    reference: str
    shipper: Party
    consignee: Party
    origin: PostalAddress
    destination: PostalAddress
    waypoints: list[PostalAddress] = pydantic.Field(default_factory=list)


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


class RefHeavy(dspy.Signature):
    """Extract the shipment."""

    text: str = dspy.InputField()
    shipment: Shipment = dspy.OutputField()


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
    "ref_heavy": SignatureCell(
        RefHeavy,
        {
            "text": (
                "Shipment REF-4417. Shipper Analytical Engines Ltd, 12 Baker St, London "
                "NW1 6XE, GB. Consignee Bell Labs, 600 Mountain Ave, Murray Hill, NJ 07974, "
                "US. Collected from 5 Rue Lafayette, Paris 75009, FR and delivering to "
                "1 Infinite Loop, Cupertino, CA 95014, US, routed via 88 Koenigsallee, "
                "Duesseldorf 40212, DE."
            )
        },
    ),
}

SIGNATURE_IDS: tuple[str, ...] = tuple(SIGNATURES)
