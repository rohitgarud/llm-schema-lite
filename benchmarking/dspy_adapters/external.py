"""Third-party labeled corpora for the accuracy arm, fetched at a pinned revision.

**Why this exists.** ``cases.py`` is a corpus this package's authors wrote, rendered by a
renderer they also wrote, so an adapter can end up fitted to that renderer's habits without
the arm ever noticing. These are the four structured-output tasks run by
https://github.com/thedataquarry/structured-outputs (MIT): three Cleanlab benchmarks
(https://github.com/cleanlab/structured-output-benchmark) and clinical notes written from
synthetic FHIR records. That repo's signature instructions and field descriptions are
copied verbatim, so the task wording is theirs, not ours; the insurance-claims schema is
Cleanlab's (Apache-2.0), in thedataquarry's relaxed form.

**Schema deviations** exist only where upstream's own gold could not otherwise match, and
each is marked at its field. Everywhere: ``X | None`` for ``Optional[X]``, which renders the
same JSON schema. For patient-notes: the record is flat like its gold (upstream nests the
patient under ``patient``; its gold does not), ``primaryLanguage`` is a free string (upstream
misspells it ``primarLanguage`` and allows only English/Spanish, while gold holds values like
"English (United States)"), immunization ``status`` admits the gold's ``"unknown"``, the
never-labelled ``substances`` is dropped, and the patient email's description no longer
reads "Phone number of the patient". Every adapter sees the same schema, so none of this
favours one of them.

**The data is never vendored.** None of the sources states a licence, so rows are fetched at
run time - from Hugging Face (``datasets``, the ``benchmark`` extra), or for patient-notes
from thedataquarry's repo, where its notes and gold live - and only scores and field paths
reach ``results/``.

**Scoring is this arm's own** (:func:`.accuracy.score`), not the upstream evaluators, so the
numbers are not comparable with thedataquarry's tables. A correct ``null`` counts as a
matched field - upstream PII scoring does the same, which is why an all-null PII reply
already scores most of its fields - and ``insured_objects`` is compared by position, where
Cleanlab's evaluator searches for the best pairing. patient-notes adopts upstream's
leniencies (:func:`align_patient_record`) but compares every practitioner and immunization
field, where upstream compares the first practitioner and only counts immunizations.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal
from urllib.request import urlopen

import dspy
from pydantic import BaseModel, Field

from .cases import Case

__all__ = [
    "CORPORA",
    "Corpus",
    "align_entity_lists",
    "align_patient_record",
    "cases_from_rows",
    "load",
]


# --- pii: thedataquarry src/pii/dspy/{schema,extract}.py -------------------------------


class PII(BaseModel):
    ACCOUNTNAME: str | None = None
    ACCOUNTNUMBER: str | None = None
    AGE: str | None = None
    AMOUNT: str | None = None
    BIC: str | None = None
    BITCOINADDRESS: str | None = None
    BUILDINGNUMBER: str | None = None
    CITY: str | None = None
    COMPANYNAME: str | None = None
    COUNTY: str | None = None
    CREDITCARDCVV: str | None = None
    CREDITCARDISSUER: str | None = None
    CREDITCARDNUMBER: str | None = None
    CURRENCY: str | None = None
    CURRENCYCODE: str | None = None
    CURRENCYNAME: str | None = None
    CURRENCYSYMBOL: str | None = None
    DATE: str | None = None
    DOB: str | None = None
    EMAIL: str | None = None
    ETHEREUMADDRESS: str | None = None
    EYECOLOR: str | None = None
    FIRSTNAME: str | None = None
    GENDER: str | None = None
    HEIGHT: str | None = None
    IBAN: str | None = None
    IP: str | None = None
    IPV4: str | None = None
    IPV6: str | None = None
    JOBAREA: str | None = None
    JOBTITLE: str | None = None
    JOBTYPE: str | None = None
    LASTNAME: str | None = None
    LITECOINADDRESS: str | None = None
    MAC: str | None = None
    MASKEDNUMBER: str | None = None
    MIDDLENAME: str | None = None
    NEARBYGPSCOORDINATE: str | None = None
    ORDINALDIRECTION: str | None = None
    PASSWORD: str | None = None
    PHONEIMEI: str | None = None
    PHONENUMBER: str | None = None
    PIN: str | None = None
    PREFIX: str | None = None
    SECONDARYADDRESS: str | None = None
    SEX: str | None = None
    SSN: str | None = None
    STATE: str | None = None
    STREET: str | None = None
    TIME: str | None = None
    URL: str | None = None
    USERAGENT: str | None = None
    USERNAME: str | None = None
    VEHICLEVIN: str | None = None
    VEHICLEVRM: str | None = None
    ZIPCODE: str | None = None


class PIIInfo(dspy.Signature):
    """
    Extract PII from the following text.
    - If you are unsure about a field, leave it as null.
    """

    text: str = dspy.InputField()
    pii: PII = dspy.OutputField()


# --- financial-ner: thedataquarry src/financial_ner/dspy/extract.py -----------------------


class Entities(BaseModel):
    Company: list[str] | None = Field(
        None,
        description="Official or unofficial name of a registered company or a brand.",
    )
    Date: list[str] | None = Field(
        None,
        description=(
            'Specific time period, whether explicitly mentioned (e.g., "year ended March 2020")'
            ' or implicitly referred to (e.g., "last month"), in the past, present, or future.'
        ),
    )
    Location: list[str] | None = Field(
        None,
        description=(
            "Represents geographical locations, such as political regions, countries, states,"
            " cities, roads, or any other location, even when used as adjectives."
        ),
    )
    Money: list[str] | None = Field(
        None,
        description="Monetary value expressed in any world currency, including digital currencies.",
    )
    Person: list[str] | None = Field(
        None,
        description="Name of an individual.",
    )
    Product: list[str] | None = Field(
        None,
        description=(
            "Any physical object or service manufactured or provided by a company to consumers,"
            " excluding references to businesses or sectors within the financial context."
        ),
    )
    Quantity: list[str] | None = Field(
        None,
        description=(
            "Any numeric value that is not categorized as Money, such as percentages, numbers,"
            " measurements (e.g., weight, length), or other similar quantities. Note that unit"
            " of measurements are also part of the entity."
        ),
    )


class FinancialEntitiesInfo(dspy.Signature):
    """
    Identify and extract entities from the following financial news text into the following categories:
    - Extract all relevant entities as a list of strings, preserving the wording from the text
    - Use None if no entities are found in that category
    - Only extract entities that are explicitly mentioned in the text itself, do not make inferences or reason about what entities might be implied based on URLs, domain names, or other indirect references
    - Extract individual items rather than compound or ranged entities (e.g., if a range or compound entity is mentioned, extract each individual item separately)
    """  # noqa: E501

    text: str = dspy.InputField()
    entities: Entities = dspy.OutputField()


# --- insurance-claims: thedataquarry src/insurance_claims/dspy/{schema,extract}.py --------
# Copyright Cleanlab.ai
# SPDX-License-Identifier: Apache-2.0
# Modified by thedataquarry from its original version to be less strict, so as to allow for
# partial extractions. Original: https://github.com/cleanlab/structured-output-benchmark


class ClaimHeader(BaseModel):
    claim_id: str | None = Field(
        ..., description="Claim ID in format CLM-XXXXXX, where X is a digit"
    )
    report_date: date | None = Field(
        ..., description="Date claim was reported in YYYY-MM-DD format"
    )
    incident_date: date | None = Field(
        ..., description="Date incident occurred in YYYY-MM-DD format"
    )
    reported_by: str | None = Field(
        ..., min_length=1, description="Full name of person reporting claim"
    )
    channel: Literal["Email", "Phone", "Portal", "In-Person"] | None = Field(
        ..., description="Channel used to report claim"
    )


class PolicyDetails(BaseModel):
    policy_number: str | None = Field(
        ..., description="Policy number in format POL-XXXXXXXXX, where X is a digit"
    )
    policyholder_name: str | None = Field(
        ..., min_length=1, description="Full legal name on policy"
    )
    coverage_type: Literal["Property", "Auto", "Liability", "Health", "Travel", "Other"] | None = (
        Field(..., description="Type of insurance coverage")
    )
    effective_date: date | None = Field(
        ..., description="Policy effective start date in YYYY-MM-DD format"
    )
    expiration_date: date | None = Field(
        ..., description="Policy expiration end date in YYYY-MM-DD format"
    )


class InsuredObject(BaseModel):
    object_id: str | None = Field(
        ...,
        description=(
            "Unique identifier for insured object. For vehicles, use VIN format (e.g.,"
            " VIN12345678901234567). For buildings, use PROP-XXXXXX format. For liability, use"
            " LIAB-XXXXXX format. For other objects, use OBJ-XXXXXX format, where X is a digit"
        ),
    )
    object_type: Literal["Vehicle", "Building", "Person", "Other"] = Field(
        ..., description="Type of insured object"
    )
    make_model: str | None = Field(
        None,
        description=(
            "Make and model for vehicles (use standardtized manufacturer names and models),"
            " or building type for property"
        ),
    )
    year: int | None = Field(None, description="Year for vehicles or year built for buildings")
    location_address: str | None = Field(
        None,
        description="Full street address where object is located or originated from",
    )
    estimated_value: int | None = Field(
        None, description="Estimated monetary value in USD without currency symbol"
    )


class IncidentDescription(BaseModel):
    incident_type: Literal[
        "rear_end_collision",
        "side_impact_collision",
        "head_on_collision",
        "parking_lot_collision",
        "house_fire",
        "kitchen_fire",
        "electrical_fire",
        "burst_pipe_flood",
        "storm_damage",
        "roof_leak",
        "slip_and_fall",
        "property_injury",
        "product_liability",
        "theft_burglary",
        "vandalism",
    ] = Field(..., description="Specific standardized incident type")

    location_type: Literal[
        "intersection",
        "highway",
        "parking_lot",
        "driveway",
        "residential_street",
        "residence_interior",
        "residence_exterior",
        "commercial_property",
        "public_property",
    ] = Field(..., description="Standardized location type where incident occurred")

    estimated_damage_amount: int | None = Field(
        None, description="Estimated damage in USD without currency symbol"
    )
    police_report_number: str | None = Field(None, description="Police report number if applicable")


class InsuranceClaim(BaseModel):
    header: ClaimHeader = Field(..., description="Basic claim information")
    policy_details: PolicyDetails | None = Field(
        None, description="Policy information if available"
    )
    insured_objects: list[InsuredObject] | None = Field(
        None, description="List of insured objects involved, if applicable"
    )
    incident_description: IncidentDescription | None = Field(
        ..., description="Structured incident details"
    )


class InsuranceClaimInfo(dspy.Signature):
    """
    Extract the insurance claim information from the following text.
    - If you are unsure about a field, leave it as null.
    """

    claim_text: str = dspy.InputField()
    claim: InsuranceClaim = dspy.OutputField()


# --- patient-notes: thedataquarry src/patient_notes/dspy/extract.py -----------------------


class PersonNameAndTitle(BaseModel):
    family: str | None = Field(default=None, description="Surname of the patient")
    given: list[str] | None = Field(
        default=None,
        description="Given name(s) of the patient",
    )
    prefix: str | None = Field(default=None, description="Title of the patient")


class Address(BaseModel):
    line: str | None
    city: str | None
    state: str | None
    postalCode: str | None
    country: Literal["US"] | None


class Practitioner(BaseModel):
    name: PersonNameAndTitle | None
    phone: str | None = Field(default=None, description="Phone number of the healthcare provider")
    email: str | None = Field(default=None, description="Email address of the healthcare provider")
    address: Address | None = Field(default=None, description="Address of the healthcare provider")


class Immunization(BaseModel):
    traits: list[str] | None = Field(
        description="Text describing the name and traits of the immunization",
    )
    # Deviation: upstream allows only "completed"; 274 of the 437 gold immunizations are
    # "unknown". Upstream's `substances` field is dropped - no gold record ever has it.
    status: Literal["completed", "unknown"] | None
    occurrenceDate: str | None = Field(
        description="ISO-8601 format for datetime including timezone"
    )


class Substance(BaseModel):
    category: Literal["environment", "food", "medication", "other"]
    name: str | None
    manifestation: str | None = Field(
        description="Text describing the manifestation of the allergy or intolerance",
    )


class Allergy(BaseModel):
    substance: list[Substance] | None = Field(
        description="The substance that the patient is allergic to"
    )


class PatientRecord(BaseModel):
    """Deviation: flat, like the gold. Upstream nests these under ``patient`` and puts
    ``allergy`` in a list; its gold does neither. ``record_id`` is dropped - upstream sets
    it after the call, it is never extracted."""

    name: PersonNameAndTitle | None
    age: int | None
    gender: Literal["male", "female"] | None
    birthDate: str | None = Field(description="Date of birth of the patient in ISO-8601 format")
    address: Address | None = Field(description="Residence address of the patient")
    phone: str | None = Field(default=None, description="Phone number of the patient")
    # Deviation: upstream's description here is a copy of phone's, "Phone number of the patient".
    email: str | None = Field(default=None, description="Email address of the patient")
    maritalStatus: Literal["Married", "Divorced", "Widowed", "NeverMarried"] | None
    # Deviation: upstream `primarLanguage: Literal["English", "Spanish"] | None`.
    primaryLanguage: str | None
    allergy: Allergy | None
    practitioner: list[Practitioner] | None
    immunization: list[Immunization] | None


class PatientRecordInfo(dspy.Signature):
    """
    Extract patient, practitioner, and immunization information from the given note.
    - Do not infer any information that is not explicitly mentioned in the text.
    - If you are unsure about any field, leave it as None.
    """

    # Deviation: upstream annotates this `PatientNote` (id + text) but passes the text alone.
    note: str = dspy.InputField()
    patient_record: PatientRecord = dspy.OutputField()


# --- scoring and loading ----------------------------------------------------------------


def align_entity_lists(expected: Any, produced: Any | None) -> tuple[Any, Any | None]:
    """Give each NER category set semantics without keying field paths by entity text.

    Upstream scores a category as ``|gold & produced| / |gold|``, blind to order and
    duplicates. :func:`.accuracy.score` indexes lists, so the produced list is rewritten
    into the gold list's order: a found entity lands on its gold index, an unfound one
    leaves ``None`` there (a wrong field), and anything invented is appended past the gold
    length, where it surfaces as a spurious path. Upstream also treats ``[]`` and ``None``
    alike, so both become ``None``.

    Stricter than upstream in one place: entities produced for a category whose gold is
    ``None`` cost that category's one field, where upstream scores it 0/0.
    """
    gold = {
        key: list(dict.fromkeys(s.strip() for s in value or [])) for key, value in expected.items()
    }
    aligned_expected = {key: items or None for key, items in gold.items()}
    if produced is None:
        return aligned_expected, None

    if isinstance(produced, BaseModel):
        produced = produced.model_dump(mode="json")
    aligned: dict[str, Any] = {}
    for key, want in gold.items():
        got = list(dict.fromkeys(s.strip() for s in produced.get(key) or []))
        found, wanted = set(got), set(want)
        aligned[key] = [w if w in found else None for w in want] + [
            g for g in got if g not in wanted
        ] or None
    return aligned_expected, aligned


_US_STATES = dict(
    item.split(":")
    for item in (
        "AL:Alabama AK:Alaska AZ:Arizona AR:Arkansas CA:California CO:Colorado CT:Connecticut"
        " DE:Delaware FL:Florida GA:Georgia HI:Hawaii ID:Idaho IL:Illinois IN:Indiana IA:Iowa"
        " KS:Kansas KY:Kentucky LA:Louisiana ME:Maine MD:Maryland MA:Massachusetts"
        " MI:Michigan MN:Minnesota MS:Mississippi MO:Missouri MT:Montana NE:Nebraska"
        " NV:Nevada NH:New_Hampshire NJ:New_Jersey NM:New_Mexico NY:New_York"
        " NC:North_Carolina ND:North_Dakota OH:Ohio OK:Oklahoma OR:Oregon PA:Pennsylvania"
        " RI:Rhode_Island SC:South_Carolina SD:South_Dakota TN:Tennessee TX:Texas UT:Utah"
        " VT:Vermont VA:Virginia WA:Washington WV:West_Virginia WI:Wisconsin WY:Wyoming"
    ).split()
)


def _iso_date(text: str) -> str:
    """Upstream's ``normalize_date``: keep ``YYYY-MM-DD``, parse the written-out forms."""
    text = text.split("T")[0].strip()
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text


def _canon_patient(value: Any, key: str = "") -> Any:
    """One canonical form for a patient record, gold or reply, per upstream's evaluator."""
    if isinstance(value, dict):
        out = {
            k: _canon_patient(v, k)
            for k, v in value.items()
            # Never asked for: `record_id` is set by upstream's caller, `encounter_period`
            # has no field in upstream's schema.
            if k not in ("record_id", "encounter_period")
        }
        # {"substance": null} and a null allergy both mean "no known allergies".
        return None if list(out) == ["substance"] and out["substance"] is None else out
    if isinstance(value, list):
        if key == "line" and len(value) == 1:  # upstream: ["9 Main St"] == "9 Main St"
            return _canon_patient(value[0], key)
        return [_canon_patient(item, key) for item in value] or None  # [] == null upstream
    if isinstance(value, str):
        text = value.strip()
        if key == "state":
            return _US_STATES.get(text.upper(), text).replace("_", " ").lower()
        if key == "birthDate":
            return _iso_date(text).lower()
        return text.lower()  # upstream compares every string case-insensitively
    return value


def align_patient_record(expected: Any, produced: Any | None) -> tuple[Any, Any | None]:
    """Put gold and reply in upstream's canonical form before :func:`.accuracy.score`.

    Upstream's evaluator is lenient in four ways, all kept: strings compare
    case-insensitively, a state abbreviation equals its name, a written-out birth date
    ("November 12, 1988", which is how the notes state it) equals its ISO form, and an empty
    list, a null and an empty allergy are all "nothing".
    """
    if isinstance(produced, BaseModel):
        produced = produced.model_dump(mode="json")
    return _canon_patient(expected), None if produced is None else _canon_patient(produced)


@dataclass(frozen=True)
class Corpus:
    """Where one corpus lives, and how its rows become :class:`.cases.Case` objects."""

    repo: str
    revision: str
    text_column: str
    signature: type[dspy.Signature]
    output_field: str
    align: Callable[[Any, Any | None], tuple[Any, Any | None]] | None = None
    # None = a Hugging Face dataset with a Python-literal `ground_truth` column.
    fetch: Callable[[Corpus], list[dict[str, Any]]] | None = None


def _fetch_patient_notes(corpus: Corpus) -> list[dict[str, Any]]:
    """Notes and gold from thedataquarry's repo at the pinned commit; they are not on HF."""
    base = (
        f"https://raw.githubusercontent.com/{corpus.repo}/{corpus.revision}/src/patient_notes/data"
    )
    notes, gold = (
        json.loads(urlopen(f"{base}/{name}.json", timeout=60).read()) for name in ("note", "gold")
    )
    if [row["record_id"] for row in notes] != [row["record_id"] for row in gold]:
        raise ValueError("patient-notes: note.json and gold.json disagree on record order")
    return [{"note": n["note"], "ground_truth": g} for n, g in zip(notes, gold, strict=True)]


# Revisions pinned 2026-09-10. Moving one changes the corpus, so it belongs in a commit that
# also regenerates every `accuracy-<corpus>-*` artefact.
CORPORA: dict[str, Corpus] = {
    "pii": Corpus(
        repo="Cleanlab/pii-extraction",
        revision="ca07a3a51edbc5ca99d13e452c6c1c49fd83ff9f",
        text_column="text",
        signature=PIIInfo,
        output_field="pii",
    ),
    "financial-ner": Corpus(
        repo="Cleanlab/fire-financial-ner-extraction",
        revision="d354ba26e96b07d216db1c22888a7f7e8be52fed",
        text_column="text",
        signature=FinancialEntitiesInfo,
        output_field="entities",
        align=align_entity_lists,
    ),
    "insurance-claims": Corpus(
        repo="Cleanlab/insurance-claims-extraction",
        revision="e86f533a4d9a855d5ae6c8bbeaed4f511f7b49eb",
        text_column="claim_text",
        signature=InsuranceClaimInfo,
        output_field="claim",
    ),
    "patient-notes": Corpus(
        repo="thedataquarry/structured-outputs",
        revision="47924a59d8b5b4eb19426c3e0defe4e33ff4f861",
        text_column="note",
        signature=PatientRecordInfo,
        output_field="patient_record",
        align=align_patient_record,
        fetch=_fetch_patient_notes,
    ),
}


def cases_from_rows(name: str, rows: Iterable[Mapping[str, Any]]) -> list[Case]:
    """Turn raw rows into cases; HF ``ground_truth`` is a Python-literal string, not JSON."""
    corpus = CORPORA[name]
    return [
        Case(
            case_id=f"{name}-{index:04d}",
            text=row[corpus.text_column],
            expected=(
                row["ground_truth"]
                if isinstance(row["ground_truth"], dict)
                else ast.literal_eval(row["ground_truth"])
            ),
            signature=corpus.signature,
            input_field=corpus.text_column,
            output_field=corpus.output_field,
            align=corpus.align,
        )
        for index, row in enumerate(rows)
    ]


def load(name: str, n: int) -> list[Case]:
    """Fetch the first ``n`` rows of corpus ``name`` at its pinned revision."""
    corpus = CORPORA[name]
    if corpus.fetch is not None:
        return cases_from_rows(name, corpus.fetch(corpus)[:n])
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("--corpus needs the `benchmark` extra (`datasets`)") from exc

    rows = load_dataset(corpus.repo, split="train", revision=corpus.revision)
    return cases_from_rows(name, rows.select(range(min(n, len(rows)))))
