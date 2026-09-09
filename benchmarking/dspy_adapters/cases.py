"""Labeled extraction cases for the accuracy arm.

**Why this exists.** The live outcomes arm answers *validity* - did the reply parse and
satisfy the schema. It cannot answer *accuracy* - are the values right. A model can emit
perfectly-shaped JSON with entirely wrong values and the outcomes arm records it as ``ok``.
Every published comparison this arm is modelled on (the DSPy ``BAMLAdapter`` PR #8614, and
thedataquarry/structured-outputs) reports field-level accuracy against ground truth, so
that is the quantity this module supplies the labels for.

**Ground truth is the source instance.** A case is generated as a ``PersonRecord``, then
rendered to prose. The extractor sees only the prose; the instance it is scored against is
the one the prose was rendered from, so labels are exact and free - no dataset download, no
annotation, no licence question.

**The honest weakness.** The renderer writes the text the extractor reads, so difficulty is
bounded by how adversarial the rendering is. A literal rendering would let every adapter
score ~100% and the arm would discriminate nothing. :func:`render` therefore varies phrasing
per case (``_PHRASINGS``), reorders the optional trailer, and omits optional fields outright
- and when it omits one, the expected value is ``None``, so the case scores a *correct*
absence rather than a guess. It is still synthetic prose, not clinical notes; it measures
schema-following under paraphrase, not real-world extraction.

Determinism: every case is a pure function of ``seed``. ``generate(n, seed=S)`` returns the
same n cases on any machine, and the seed is recorded in the report provenance so a run is
reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import dspy
from pydantic import BaseModel, Field

__all__ = [
    "Address",
    "Contact",
    "Employment",
    "PersonRecord",
    "ExtractPerson",
    "Case",
    "generate",
    "render",
]


class Address(BaseModel):
    """A postal address; ``postcode`` is genuinely optional in the rendered prose."""

    street: str
    city: str
    postcode: str | None = None


class Employment(BaseModel):
    """Current employment. The whole object is omitted for unemployed records."""

    company: str
    role: str
    years: int = Field(ge=0, le=50)


class Contact(BaseModel):
    """One contact method; ``phone`` is often absent."""

    email: str
    phone: str | None = None


class PersonRecord(BaseModel):
    """The extraction target: nested models, an optional object, and two lists.

    Deliberately shaped like the reference benchmarks' targets - nesting, optionality and
    repetition are where schema-following actually breaks down, and a flat model would
    make every adapter look identical.
    """

    name: str
    age: int = Field(ge=0, le=120)
    address: Address
    employment: Employment | None = None
    contacts: list[Contact] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ExtractPerson(dspy.Signature):
    """Extract the person record described by the text."""

    text: str = dspy.InputField()
    record: PersonRecord = dspy.OutputField()


@dataclass(frozen=True)
class Case:
    """One labeled case: the prose the model sees, and the record it must recover."""

    case_id: str
    text: str
    expected: PersonRecord


_FIRST = ["Ada", "Grace", "Alan", "Katherine", "Linus", "Barbara", "Edsger", "Radia"]
_LAST = ["Lovelace", "Hopper", "Turing", "Johnson", "Torvalds", "Liskov", "Dijkstra", "Perlman"]
_STREET = ["12 Baker St", "5 Rue Lafayette", "88 Königsallee", "1 Infinite Loop", "7 Nassau St"]
_CITY = ["London", "Paris", "Düsseldorf", "Cupertino", "Princeton", "Kraków", "Osaka"]
_POSTCODE = ["NW1 6XE", "75009", "40212", "95014", "08540"]
_COMPANY = ["Analytical Engines Ltd", "Naval Systems", "Bell Labs", "Xerox PARC", "Sun Micro"]
_ROLE = ["staff engineer", "research lead", "compiler architect", "network designer"]
_DOMAIN = ["example.com", "mail.test", "corp.example", "post.example"]
_TAG = ["mathematics", "cryptography", "compilers", "networking", "distributed-systems"]

# Each phrasing must be lossless for the fields it carries: the extractor cannot be scored
# on a value the prose never stated.
_PHRASINGS = [
    "{name}, aged {age}, lives at {street} in {city}.",
    "{name} is {age} years old and can be found at {street}, {city}.",
    "Record for {name} ({age}): residence {street}, {city}.",
    "Our subject {name} — {age} this year — resides at {street} in {city}.",
]


def _email(first: str, last: str, domain: str) -> str:
    return f"{first.lower()}.{last.lower()}@{domain}"


def _phone(rng: random.Random) -> str:
    return f"+{rng.randint(1, 99)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}"


def generate(n: int, seed: int = 0) -> list[Case]:
    """Return ``n`` labeled cases, deterministically derived from ``seed``.

    Each case gets its own ``Random(seed + index)`` so that ``generate(50, seed=S)`` is a
    strict prefix-extension of ``generate(10, seed=S)`` - growing the case set never
    reshuffles the cases already measured, which keeps two runs at different ``n``
    comparable on their common prefix.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return [_one(index, random.Random(seed + index)) for index in range(n)]


def _one(index: int, rng: random.Random) -> Case:
    """Build one case and its prose from an already-seeded RNG."""
    first, last = rng.choice(_FIRST), rng.choice(_LAST)
    name = f"{first} {last}"

    # Optionality is decided here, once, and drives BOTH the record and the prose - so an
    # absent field is a correct `None`, never an unstated value the model must invent.
    has_postcode = rng.random() < 0.6
    has_employment = rng.random() < 0.7
    n_contacts = rng.choice([0, 1, 1, 2])
    n_tags = rng.choice([0, 1, 2, 2, 3])

    address = Address(
        street=rng.choice(_STREET),
        city=rng.choice(_CITY),
        postcode=rng.choice(_POSTCODE) if has_postcode else None,
    )
    employment = (
        Employment(
            company=rng.choice(_COMPANY),
            role=rng.choice(_ROLE),
            years=rng.randint(1, 30),
        )
        if has_employment
        else None
    )
    contacts = [
        Contact(
            email=_email(first, f"{last}{i}" if i else last, rng.choice(_DOMAIN)),
            phone=_phone(rng) if rng.random() < 0.5 else None,
        )
        for i in range(n_contacts)
    ]
    tags = rng.sample(_TAG, n_tags)

    record = PersonRecord(
        name=name,
        age=rng.randint(18, 90),
        address=address,
        employment=employment,
        contacts=contacts,
        tags=tags,
    )
    return Case(case_id=f"case-{index:04d}", text=render(record, rng), expected=record)


def render(record: PersonRecord, rng: random.Random) -> str:
    """Render ``record`` to prose that states every non-``None`` field exactly once.

    The invariant that makes the arm fair: a field is scored only if the prose states it.
    Anything the record leaves ``None`` is simply absent here, so the correct extraction is
    ``None`` and a model that invents a value is penalised for inventing it.
    """
    parts = [
        rng.choice(_PHRASINGS).format(
            name=record.name,
            age=record.age,
            street=record.address.street,
            city=record.address.city,
        )
    ]
    if record.address.postcode is not None:
        parts.append(f"The postcode is {record.address.postcode}.")
    if record.employment is not None:
        job = record.employment
        parts.append(f"They work at {job.company} as a {job.role}, {job.years} years so far.")
    for contact in record.contacts:
        if contact.phone is not None:
            parts.append(f"Contact: {contact.email}, phone {contact.phone}.")
        else:
            parts.append(f"Contact: {contact.email}.")
    if record.tags:
        parts.append(f"Interests: {', '.join(record.tags)}.")

    # Shuffle only the trailer: the opening phrasing carries name/age/street/city together
    # and reads as a sentence, while the trailing facts are order-independent. Shuffling
    # them stops a model from learning a fixed slot order instead of reading the text.
    head, trailer = parts[0], parts[1:]
    rng.shuffle(trailer)
    return " ".join([head, *trailer])
