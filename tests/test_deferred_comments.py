"""Tests for the deferred-comment mechanism and its literal/type helpers."""

from __future__ import annotations

import re
from typing import Any

from llm_schema_lite.formatters.base import BaseFormatter, format_literal_value, infer_json_type
from llm_schema_lite.formatters.jsonish_formatter import JSONishFormatter

TOKEN_RE = re.compile(r"^⟪lsl[0-9a-f]{8}\.\d+⟫$")


def _formatter() -> JSONishFormatter:
    """Build a concrete ``BaseFormatter`` instance (the base class is abstract)."""
    return JSONishFormatter({"type": "object", "properties": {}})


# ============================================================================
# format_literal_value
# ============================================================================


def test_format_literal_value_bool_before_int() -> None:
    """Bools render as JSON booleans, never as their int values."""
    assert format_literal_value(True) == "true"
    assert format_literal_value(False) == "false"
    assert format_literal_value(True) != "1"
    assert format_literal_value(False) != "0"


def test_format_literal_value_none_is_null() -> None:
    """``None`` renders as the JSON null literal."""
    assert format_literal_value(None) == "null"


def test_format_literal_value_string_is_json_quoted() -> None:
    """Strings are JSON-quoted, with inner quotes escaped and non-ASCII preserved."""
    assert format_literal_value("US") == '"US"'
    assert format_literal_value('say "hi"') == '"say \\"hi\\""'
    assert format_literal_value("café") == '"café"'


def test_format_literal_value_numbers_bare() -> None:
    """Numbers render bare, without quotes."""
    assert format_literal_value(7) == "7"
    assert format_literal_value(1.5) == "1.5"


# ============================================================================
# infer_json_type
# ============================================================================


def test_infer_json_type_homogeneous() -> None:
    """A homogeneous value list infers its shared JSON Schema type name."""
    assert infer_json_type(["a", "b"]) == "string"
    assert infer_json_type([1, 2]) == "integer"
    assert infer_json_type([True, False]) == "boolean"
    assert infer_json_type([1.0, 2.0]) == "number"


def test_infer_json_type_ignores_none() -> None:
    """``None`` entries do not participate in inference."""
    assert infer_json_type(["a", None]) == "string"
    assert infer_json_type([None]) is None
    assert infer_json_type([]) is None


def test_infer_json_type_heterogeneous_returns_none() -> None:
    """A mixed-type value list has no single inferable type."""
    assert infer_json_type(["a", 1, True, None]) is None


# ============================================================================
# The deferred-comment slot table
# ============================================================================


def test_defer_comment_is_content_keyed() -> None:
    """The same body reuses its slot rather than allocating a second one."""
    formatter = _formatter()
    first = formatter.defer_comment('one of: "a", "b"')
    second = formatter.defer_comment('one of: "a", "b"')
    assert first == second
    assert formatter._deferred_bodies == ['one of: "a", "b"']


def test_defer_comment_token_shape() -> None:
    """The minted token is an opaque, nonce-bearing marker."""
    formatter = _formatter()
    token = formatter.defer_comment('one of: "a"')
    assert TOKEN_RE.match(token)


def test_nonce_differs_per_instance() -> None:
    """Two formatters over the same schema mint different nonces."""
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    first = JSONishFormatter(schema)
    second = JSONishFormatter(schema)
    assert first._deferred_nonce != second._deferred_nonce


def test_carries_deferred_comment() -> None:
    """Only strings holding *this* instance's marker are recognised."""
    formatter = _formatter()
    other = _formatter()
    token = formatter.defer_comment('one of: "a"')
    assert formatter.carries_deferred_comment(f"string{token}") is True
    assert formatter.carries_deferred_comment("string") is False
    assert formatter.carries_deferred_comment({"a": 1}) is False
    assert formatter.carries_deferred_comment(["a"]) is False
    assert formatter.carries_deferred_comment(other.defer_comment("other body")) is False


def test_append_deferred_comment_folds_extra() -> None:
    """``extra`` is folded into the referenced slot body, not into the representation."""
    formatter = _formatter()
    representation = "string" + formatter.defer_comment('one of: "a", "b"')
    updated = formatter.append_deferred_comment(representation, "(default=null)")
    tokens = formatter._deferred_pattern.findall(updated)
    assert len(tokens) == 1
    assert formatter._deferred_bodies[int(tokens[0])] == 'one of: "a", "b"; (default=null)'


def test_append_deferred_comment_noop_without_marker() -> None:
    """Empty extras and marker-free representations are returned unchanged."""
    formatter = _formatter()
    representation = "string" + formatter.defer_comment('one of: "a"')
    assert formatter.append_deferred_comment(representation, "") == representation
    assert formatter.append_deferred_comment("string", "(default=null)") == "string"


# ============================================================================
# enum_type_token / build_enum_comment / base _get_title_description_default_value
# ============================================================================


def test_enum_type_token_from_type_key() -> None:
    """A single ``type`` key maps through ``TYPE_MAP``."""
    formatter = _formatter()
    assert formatter.enum_type_token({"enum": ["a"], "type": "string"}) == "string"
    assert formatter.enum_type_token({"enum": [1], "type": "integer"}) == "int"
    assert formatter.enum_type_token({"enum": [True], "type": "boolean"}) == "bool"


def test_enum_type_token_from_type_list_nullable() -> None:
    """A ``type`` list resolves to its non-null element."""
    formatter = _formatter()
    assert formatter.enum_type_token({"enum": ["a"], "type": ["string", "null"]}) == "string"
    assert formatter.enum_type_token({"enum": [1], "type": ["integer"]}) == "int"


def test_enum_type_token_inferred_when_no_type() -> None:
    """A missing ``type`` key falls back to inference from the enum values."""
    formatter = _formatter()
    assert formatter.enum_type_token({"enum": ["a", "b"]}) == "string"
    assert formatter.enum_type_token({"enum": [1, 2]}) == "int"


def test_enum_type_token_heterogeneous_is_any() -> None:
    """A heterogeneous or empty value list with no ``type`` key renders as ``any``."""
    formatter = _formatter()
    assert formatter.enum_type_token({"enum": ["a", 1, True, None]}) == "any"
    assert formatter.enum_type_token({"enum": []}) == "any"


def test_build_enum_comment_plain() -> None:
    """A plain enum with no descriptions/aliases joins its literals with ``', '``."""
    formatter = _formatter()
    assert (
        formatter.build_enum_comment({"enum": ["US", "CA"]}, ["US", "CA"]) == 'one of: "US", "CA"'
    )
    assert formatter.build_enum_comment({"enum": [1, 2]}, [1, 2]) == "one of: 1, 2"


def test_build_enum_comment_with_descriptions_and_aliases() -> None:
    """Per-value descriptions and aliases fold into the comment body."""
    formatter = _formatter()
    node = {
        "enum": ["low", "critical"],
        "x-enum-descriptions": {
            "low": "Non-urgent, can wait",
            "critical": "Urgent, blocking issue",
        },
        "x-enum-aliases": {"critical": ["urgent", "blocker"]},
    }
    result = formatter.build_enum_comment(node, ["low", "critical"])
    assert result == (
        'one of: "low" (Non-urgent, can wait), '
        '"critical" (Urgent, blocking issue; aliases: urgent, blocker)'
    )


def test_base_get_title_description_default_value_returns_empty() -> None:
    """The base default returns four empty strings regardless of the subclass override."""
    assert BaseFormatter._get_title_description_default_value(_formatter(), {"any": "value"}) == (
        "",
        "",
        "",
        "",
    )
