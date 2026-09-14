"""`contains` and `multipleOf` render once, identically, in all three formatters.

Both keywords previously had more than one emitter. `contains` reached YAML and
TypeScript through `process_contains` AND `METADATA_MAP`, printing two or three times on
one line and leaking a raw Python dict repr (`contains: {'const': 'z'}`) when the schema
carried a `const`; JSONish rendered it not at all. `multipleOf` reached YAML and
TypeScript only, as `# multipleOf: 5`.

Both now come from a single owner -- `array_constraint_tokens` and `multiple_of_token` --
so the guard that matters is *one* occurrence, spelled the *same* way, in *every* mode.
"""

from __future__ import annotations

import pytest

from llm_schema_lite import simplify_schema
from llm_schema_lite.formatters.config import FormatterConfig

FORMATS = ("jsonish", "yaml", "typescript")


def _render(schema: dict, fmt: str, **kwargs) -> str:
    return " ".join(simplify_schema(schema, format_type=fmt, **kwargs).to_string().split())


@pytest.mark.parametrize("fmt", FORMATS)
def test_contains_renders_once_and_is_not_a_dict_repr(fmt: str) -> None:
    """A `const` target used to fall through `_format_contains` to `str(dict)`."""
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "array", "items": {"type": "string"}, "contains": {"const": "z"}}
        },
        "required": ["a"],
    }
    out = _render(schema, fmt)

    assert out.count("contains") == 1, out
    # JSONish dict values pass through `json.dumps`, so its quotes arrive escaped.
    assert ('contains "z"' in out) or ('contains \\"z\\"' in out), out
    # The old leak, in either quoting style.
    assert "{'const'" not in out
    assert '{"const"' not in out


@pytest.mark.parametrize("fmt", FORMATS)
def test_contains_type_target_renders_once(fmt: str) -> None:
    """The `type` arm was already reachable; it must not regress to multiple emitters."""
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "array", "items": {"type": "string"}, "contains": {"type": "integer"}}
        },
    }
    out = _render(schema, fmt)

    assert out.count("contains") == 1, out
    assert "contains integer" in out


@pytest.mark.parametrize("fmt", FORMATS)
def test_multiple_of_renders_once_in_every_mode(fmt: str) -> None:
    """JSONish dropped it entirely; YAML/TS spelled it differently."""
    schema = {"type": "object", "properties": {"n": {"type": "integer", "multipleOf": 5}}}
    out = _render(schema, fmt)

    assert out.count("multiple of 5") == 1, out
    assert "multipleOf" not in out  # the old METADATA_MAP spelling


@pytest.mark.parametrize("fmt", FORMATS)
def test_multiple_of_shares_one_group_with_the_range(fmt: str) -> None:
    """`float (0 to 10, multiple of 0.5)`, not two adjacent parenthesised groups."""
    schema = {
        "type": "object",
        "properties": {"m": {"type": "number", "minimum": 0, "maximum": 10, "multipleOf": 0.5}},
    }
    out = _render(schema, fmt)

    assert "(0 to 10, multiple of 0.5)" in out, out
    assert ") (" not in out


@pytest.mark.parametrize("fmt", FORMATS)
def test_both_are_metadata_and_obey_the_gate(fmt: str) -> None:
    """Neither is structural, so `include_metadata=False` must remove both."""
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "array", "items": {"type": "string"}, "contains": {"const": "z"}},
            "n": {"type": "integer", "multipleOf": 5},
        },
    }
    out = _render(schema, fmt, config=FormatterConfig(include_metadata=False))

    assert "contains" not in out
    assert "multiple of" not in out
