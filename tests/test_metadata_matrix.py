"""The `include_metadata` / `include_descriptions` / `include_constraints` flag matrix.

Covers the full 3-formatter x 8-flag-combination interaction (24 cases, design §8.3),
plus the `additionalProperties: false` structural exemption (design §5.2) and the
Phase 3C.3 drift-pickup regression for the TypeScript no-`type` fallback branch.
"""

from __future__ import annotations

import itertools

import pytest
from pydantic import BaseModel

from llm_schema_lite import FormatterConfig, simplify_schema
from tests.formatter_helpers import extract_comment_slots, render_all_formatters

# Per-formatter description markers. Spellings differ per formatter; YAML and TypeScript
# deliberately omit "A postal address." because their $ref renderers never read the
# referenced definition's own description (design non-goal, not a bug). JSONish's former
# omission of "Role of the user" and "Home address" WAS a bug and is fixed by
# lsl-2026-09-05-005.
DESCRIPTION_MARKERS = {
    "jsonish": [
        "Full name",
        "Age in years",
        "Role of the user",
        "Tags",
        "Home address",
        "Optional nickname",
        "Street line",
        "ZIP",
        "A patient record.",
        "A postal address.",
    ],
    "yaml": [
        "Full name",
        "Age in years",
        "Role of the user",
        "Tags",
        "Home address",
        "Optional nickname",
        "Street line",
        "ZIP",
        "A patient record.",
    ],
    "typescript": [
        "Full name",
        "Age in years",
        "Role of the user",
        "Tags",
        "Home address",
        "Optional nickname",
        "Street line",
        "ZIP",
        "A patient record.",
    ],
}
CONSTRAINT_MARKERS = {
    "jsonish": ["PATTERN:", "<= 50 chars", "0 to 130", "<= 5 items", "(default=", "one of:"],
    # The lower-case "pattern:" entry is gone: pattern/format are now owned by the YAML
    # type token and the trailing comment no longer restates them (lsl-2026-09-05-006).
    "yaml": ["PATTERN:", "<= 50 chars", "0 to 130", "<= 5 items", "(default=", "one of:"],
    "typescript": ["≤50 chars", "0-130", "<= 5 items", "(defaults to", "pattern:"],
}
STRUCTURAL_MARKERS = {
    "jsonish": ["name*", "age*", "role", "tags", "address*", "nickname", "OR null", "[]", "{"],
    "yaml": ["name*", "age*", "role", "tags", "address*", "nickname", "OR null", "list["],
    "typescript": [
        "name*",
        "age*",
        "role",
        "tags",
        "address*",
        "nickname",
        "| null",
        "Array<",
        "interface Schema",
    ],
}

# The closed-world marker can surface as its own slot ("no additional properties"),
# folded into a schema-level "Schema-level constraints:" slot, or prefixed per-object
# ("Root: no additional properties") -- so callers must check by substring, never by
# strict slot equality.
_CLOSED_WORLD_SUBSTRINGS = ("no additional properties", "Schema-level constraints:")

CLOSED_SHAPES = {
    "bare_closed": {"type": "object", "additionalProperties": False},
    "with_props_closed": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "value": {"type": "integer"}},
        "required": ["name", "value"],
        "additionalProperties": False,
    },
    "nested_closed": {
        "type": "object",
        "properties": {
            "inner": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "additionalProperties": False,
            }
        },
        "additionalProperties": False,
    },
}


@pytest.mark.parametrize("fmt", ["jsonish", "yaml", "typescript"])
@pytest.mark.parametrize("im,desc,cons", list(itertools.product((True, False), repeat=3)))
def test_metadata_matrix(
    fmt: str, im: bool, desc: bool, cons: bool, patient_model: type[BaseModel]
) -> None:
    """Every (formatter, include_metadata, include_descriptions, include_constraints) case.

    Contract (design §8.3):
    - every description marker present iff `im and desc`
    - every constraint marker present iff `im and cons`
    - every structural marker present in all 24 cases
    - no metadata comment slot survives when `include_metadata` is False; the only
      comment slot permitted there relates to the closed-world marker.
    """

    config = FormatterConfig(
        include_metadata=im, include_descriptions=desc, include_constraints=cons
    )
    rendered = render_all_formatters(patient_model, config)[fmt]

    expect_desc = im and desc
    for marker in DESCRIPTION_MARKERS[fmt]:
        assert (marker in rendered) == expect_desc, (
            f"description marker {marker!r} presence mismatch for fmt={fmt} im={im} "
            f"desc={desc} cons={cons}: expected={expect_desc}. Output:\n{rendered}"
        )

    expect_cons = im and cons
    for marker in CONSTRAINT_MARKERS[fmt]:
        assert (marker in rendered) == expect_cons, (
            f"constraint marker {marker!r} presence mismatch for fmt={fmt} im={im} "
            f"desc={desc} cons={cons}: expected={expect_cons}. Output:\n{rendered}"
        )

    for marker in STRUCTURAL_MARKERS[fmt]:
        assert marker in rendered, (
            f"structural marker {marker!r} missing for fmt={fmt} im={im} desc={desc} "
            f"cons={cons}. Output:\n{rendered}"
        )

    if not im:
        slots = extract_comment_slots(rendered, fmt)
        for slot in slots:
            assert any(sub in slot for sub in _CLOSED_WORLD_SUBSTRINGS), (
                f"unexpected metadata comment slot {slot!r} survived include_metadata=False "
                f"for fmt={fmt} desc={desc} cons={cons}. Output:\n{rendered}"
            )


@pytest.mark.parametrize("fmt", ["jsonish", "yaml", "typescript"])
@pytest.mark.parametrize("shape_name", list(CLOSED_SHAPES.keys()))
def test_closed_world_marker_survives_include_metadata_false(shape_name: str, fmt: str) -> None:
    """`additionalProperties: false` is structural, not metadata (design §5.2).

    It must render even at `include_metadata=False`, across bare, populated, and
    nested-object closed shapes, in all three formatters.
    """

    schema = CLOSED_SHAPES[shape_name]
    config = FormatterConfig(include_metadata=False)
    rendered = simplify_schema(schema, config=config, format_type=fmt).to_string()

    detail = f"closed-world marker missing for shape={shape_name} fmt={fmt}"
    assert "no additional properties" in rendered, f"{detail}. Output:\n{rendered}"


def test_typescript_no_type_closed_schema_keeps_closed_world_marker() -> None:
    """typescript_formatter.py:586 -- the bare `else` fallback must not drop the marker.

    A schema with `additionalProperties: false` and none of type/oneOf/anyOf/allOf takes
    the `interface Schema {}` fallback branch, whose schema-level guard the design doc
    omitted from its list of four. Assert TypeScript only: JSONish/YAML behaviour on this
    no-`type` shape is outside the locked site list and is not exercised here.
    """

    schema = {"additionalProperties": False}
    out = simplify_schema(
        schema, config=FormatterConfig(include_metadata=False), format_type="typescript"
    ).to_string()
    assert "no additional properties" in out
