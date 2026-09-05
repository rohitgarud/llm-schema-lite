"""The `include_metadata` / `include_descriptions` / `include_constraints` flag matrix.

Covers the full 3-formatter x 8-flag-combination interaction (24 cases, design §8.3),
plus the `additionalProperties: false` structural exemption (design §5.2) and the
Phase 3C.3 drift-pickup regression for the TypeScript no-`type` fallback branch.
"""

from __future__ import annotations

import itertools
from typing import Any

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
    "jsonish": ["PATTERN:", "<= 50 chars", "0 to 130", "<= 5 items", "(default="],
    # The lower-case "pattern:" entry is gone: pattern/format are now owned by the YAML
    # type token and the trailing comment no longer restates them (lsl-2026-09-05-006).
    "yaml": ["PATTERN:", "<= 50 chars", "0 to 130", "<= 5 items", "(default="],
    "typescript": ["<= 50 chars", "0 to 130", "<= 5 items", "(defaults to", "pattern:"],
}
STRUCTURAL_MARKERS = {
    "jsonish": [
        "name*",
        "age*",
        "role",
        "tags",
        "address*",
        "nickname",
        "OR null",
        "[]",
        "{",
        "one of:",
    ],
    "yaml": [
        "name*",
        "age*",
        "role",
        "tags",
        "address*",
        "nickname",
        "OR null",
        "list[",
        "one of:",
    ],
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
        '"admin" | "user"',
    ],
}

# The structural allow-list covers every comment slot that survives
# `include_metadata=False`: the closed-world marker, which can surface as its own slot
# ("no additional properties"), folded into a schema-level "Schema-level constraints:"
# slot, or prefixed per-object ("Root: no additional properties"); and an enum's value
# set ("one of: ..."), which is structural (`STRUCTURAL_KEYWORDS` in config.py) and
# therefore also survives. Callers must check by substring, never by strict slot
# equality.
_STRUCTURAL_SLOT_SUBSTRINGS = (
    "no additional properties",
    "Schema-level constraints:",
    "one of:",
)

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
      comment slot permitted there is structural (the closed-world marker or an enum
      value set).
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
            assert any(sub in slot for sub in _STRUCTURAL_SLOT_SUBSTRINGS), (
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


_TUPLE_ELEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pair": {
            "type": "array",
            "prefixItems": [
                {"type": "string", "minLength": 1, "maxLength": 5},
                {"type": "integer"},
            ],
            "minItems": 2,
            "maxItems": 2,
        }
    },
    "required": ["pair"],
}
_ARRAY_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items_field": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 5},
        }
    },
    "required": ["items_field"],
}
_MAPPING_VALUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mapping_field": {
            "type": "object",
            "additionalProperties": {"type": "string", "minLength": 1, "maxLength": 5},
        }
    },
    "required": ["mapping_field"],
}
_NESTED_POSITION_SCHEMAS = {
    "tuple_element": _TUPLE_ELEMENT_SCHEMA,
    "array_item": _ARRAY_ITEM_SCHEMA,
    "mapping_value": _MAPPING_VALUE_SCHEMA,
}


@pytest.mark.parametrize("fmt", ["jsonish", "yaml", "typescript"])
@pytest.mark.parametrize("position", list(_NESTED_POSITION_SCHEMAS.keys()))
@pytest.mark.parametrize("cons", [True, False])
def test_nested_positions_honour_include_constraints(fmt: str, position: str, cons: bool) -> None:
    """The length constraint at a tuple element / array item / mapping value must obey
    `include_constraints` in every formatter (AC1 / research §B — the JSONish tuple
    element was the sole leak at HEAD).
    """
    schema = _NESTED_POSITION_SCHEMAS[position]
    config = FormatterConfig(include_constraints=cons)
    rendered = simplify_schema(schema, config=config, format_type=fmt).to_string()

    assert (
        "1-5 chars" in rendered
    ) is cons, f"position={position} fmt={fmt} cons={cons}. Output:\n{rendered}"


_BOTH_BOUNDS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "minLength": 1, "maxLength": 5},
        "score": {"type": "integer", "minimum": 1, "maximum": 10},
    },
    "required": ["code", "score"],
}
_ONE_BOUND_EXPECTATIONS = {
    "minLength": ("<= 5 chars", "1-5 chars"),
    "maxLength": (">= 1 chars", "1-5 chars"),
    "minimum": ("<= 10", "1 to 10"),
    "maximum": (">= 1", "1 to 10"),
}


@pytest.mark.parametrize("fmt", ["jsonish", "yaml", "typescript"])
@pytest.mark.parametrize("disabled", list(_ONE_BOUND_EXPECTATIONS.keys()))
def test_disabling_one_bound_hides_only_that_bound(fmt: str, disabled: str) -> None:
    """AC2: disabling exactly one of minLength/maxLength (or minimum/maximum) must hide
    only that bound and render the one-sided form -- the TypeScript `or`-over-the-pair
    defect this ticket closes.
    """
    surviving, two_sided = _ONE_BOUND_EXPECTATIONS[disabled]
    config = FormatterConfig(metadata_inclusion={disabled: False})
    rendered = simplify_schema(_BOTH_BOUNDS_SCHEMA, config=config, format_type=fmt).to_string()

    assert surviving in rendered, f"disabled={disabled} fmt={fmt}. Output:\n{rendered}"
    assert two_sided not in rendered, f"disabled={disabled} fmt={fmt}. Output:\n{rendered}"


_GLYPH_SWEEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "len_both": {"type": "string", "minLength": 1, "maxLength": 5},
        "len_min_only": {"type": "string", "minLength": 1},
        "len_max_only": {"type": "string", "maxLength": 5},
        "num_both": {"type": "integer", "minimum": 1, "maximum": 10},
        "num_min_only": {"type": "integer", "minimum": 1},
        "num_max_only": {"type": "integer", "maximum": 10},
        "counted": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4,
        },
        "pair": {
            "type": "array",
            "prefixItems": [
                {"type": "string", "maxLength": 5},
                {"type": "integer"},
            ],
            "minItems": 2,
            "maxItems": 2,
        },
        "cond_min": {
            "type": "object",
            "if": {"properties": {"age": {"minimum": 3}}},
            "then": {"required": ["guardian"]},
        },
        "cond_max": {
            "type": "object",
            "if": {"properties": {"age": {"maximum": 9}}},
            "then": {"required": ["guardian"]},
        },
    },
}


@pytest.mark.parametrize("fmt", ["jsonish", "yaml", "typescript"])
@pytest.mark.parametrize("im,desc,cons", list(itertools.product((True, False), repeat=3)))
def test_rendered_output_never_uses_unicode_comparison_glyphs(
    fmt: str, im: bool, desc: bool, cons: bool
) -> None:
    """AC3 / D3 global invariant: no formatter output ever contains the Unicode
    comparison glyphs, across every flag combination and every constraint family
    (two-sided/one-sided length, two-sided/one-sided numeric range, item counts, a
    tuple element, and the `_describe_condition` if/then sweep).
    """
    config = FormatterConfig(
        include_metadata=im, include_descriptions=desc, include_constraints=cons
    )
    rendered = simplify_schema(_GLYPH_SWEEP_SCHEMA, config=config, format_type=fmt).to_string()

    assert "≥" not in rendered, f"fmt={fmt} im={im} desc={desc} cons={cons}: {rendered}"
    assert "≤" not in rendered, f"fmt={fmt} im={im} desc={desc} cons={cons}: {rendered}"
