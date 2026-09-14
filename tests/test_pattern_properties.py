"""``patternProperties`` renders structurally in every mode (classifier rule 7b).

Before this, a ``patternProperties``-only node classified as a plain OBJECT, and each mode
lost the keys in its own way: JSONish emitted a bare ``{}``; YAML and TypeScript emitted
``object`` plus a comment that leaked JSONish's ``//`` marker and, on the property path, a
raw Python dict repr (``patternProperties: {'^[A-Z]+$': {'type': 'string'}}``).

It is its own ``ContainerShape`` kind rather than a "mapping" because a mapping carries ONE
value schema for all keys while ``patternProperties`` carries one per pattern -- reusing
"mapping" would have kept only the first. Decision C1 is untouched: see
``test_c1_properties_still_win`` below.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_schema_lite import simplify_schema

FORMATS = ("jsonish", "yaml", "typescript")

TWO_PATTERNS: dict[str, Any] = {
    "type": "object",
    "patternProperties": {"^S_": {"type": "string"}, "^N_": {"type": "integer"}},
}


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_pattern_survives(fmt: str) -> None:
    """Both regexes AND both value types reach the output -- the reason for a new kind."""
    out = simplify_schema(TWO_PATTERNS, format_type=fmt).to_string()
    assert "^S_" in out, out
    assert "^N_" in out, out
    assert "string" in out, out
    assert "int" in out or "number" in out, out


@pytest.mark.parametrize("fmt", FORMATS)
def test_nested_pattern_object_is_not_empty(fmt: str) -> None:
    """A patternProperties node under a property used to render as ``{}`` / ``object``."""
    out = simplify_schema(
        {
            "type": "object",
            "properties": {
                "env": {"type": "object", "patternProperties": {"^[A-Z]+$": {"type": "string"}}}
            },
        },
        format_type=fmt,
    ).to_string()
    assert "^[A-Z]+$" in out, out
    assert "env: {}" not in out, out
    assert "env: object" not in out, out


@pytest.mark.parametrize("fmt", FORMATS)
def test_no_raw_dict_repr_leaks(fmt: str) -> None:
    """The metadata lambda used to interpolate the raw dict; guard the exact leak shape."""
    out = simplify_schema(TWO_PATTERNS, format_type=fmt).to_string()
    assert "{'" not in out, out
    assert "'type':" not in out, out


@pytest.mark.parametrize("fmt", FORMATS)
def test_c1_properties_still_win(fmt: str) -> None:
    """Decision C1: a node declaring ``properties`` stays an object, patterns or not."""
    out = simplify_schema(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "patternProperties": {"^x_": {"type": "integer"}},
        },
        format_type=fmt,
    ).to_string()
    assert "name" in out, out
    # Rendered as an object, so the patterns stay in a comment rather than as key slots.
    assert "<^x_>" not in out, out


@pytest.mark.parametrize("fmt", FORMATS)
def test_closed_pattern_mapping_keeps_its_note(fmt: str) -> None:
    """``additionalProperties: false`` still means closed -- a pattern mapping is not open."""
    out = simplify_schema(
        {
            "type": "object",
            "patternProperties": {"^a": {"type": "string"}},
            "additionalProperties": False,
        },
        format_type=fmt,
    ).to_string()
    assert "^a" in out, out
    assert "no additional properties" in out, out
