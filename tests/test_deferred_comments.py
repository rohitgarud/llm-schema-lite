"""Tests for the deferred-comment mechanism and its literal/type helpers."""

from __future__ import annotations

import re
from typing import Any

from llm_schema_lite.formatters.base import BaseFormatter, format_literal_value, infer_json_type
from llm_schema_lite.formatters.jsonish_formatter import JSONishFormatter
from llm_schema_lite.formatters.yaml_formatter import YAMLFormatter

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


def test_identity_and_deferred_tags_are_disjoint() -> None:
    """An identity token and a defer_comment token can never cross-match.

    Pins the disjointness invariant recorded next to ``IDENTITY_TAG`` in
    ``formatters/base.py``: a defer_comment token's character after ``lsl`` is a hex
    digit, never ``i``, so ``lslid`` can never prefix one — by character class, not luck.
    """
    formatter = _formatter()
    minted = formatter._mint_identity("k")
    assert formatter._deferred_pattern.search(minted) is None
    assert formatter._identity_pattern.search(formatter.defer_comment('one of: "a"')) is None


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


# ============================================================================
# comment_lines / _continuation_indent / resolved_deferred_text
# (lsl-2026-09-05-006)
# ============================================================================


def _yaml_formatter() -> YAMLFormatter:
    """Build a concrete ``YAMLFormatter`` instance for hoist/helper unit tests."""
    return YAMLFormatter({"type": "object", "properties": {}})


def test_comment_lines_blank_line_has_no_dangling_space() -> None:
    """A blank physical line becomes a bare prefix, with no trailing space."""
    assert _yaml_formatter().comment_lines("a\n\nb") == ["# a", "#", "# b"]


def test_comment_lines_empty_body_returns_bare_prefix() -> None:
    """``comment_lines("")`` is ``[prefix]`` -- never an empty list."""
    assert _yaml_formatter().comment_lines("") == ["#"]


def test_comment_lines_applies_indent_to_every_line() -> None:
    """``indent`` is prepended before the prefix on every emitted line."""
    assert _yaml_formatter().comment_lines("a\nb", "  ") == ["  # a", "  # b"]


def test_comment_lines_uses_the_formatter_s_own_prefix() -> None:
    """JSONish renders ``//``, not ``#`` -- the prefix is read off ``self``."""
    assert _formatter().comment_lines("a\nb") == ["// a", "// b"]


def test_continuation_indent_top_level_is_empty() -> None:
    """A line starting in column 0 needs no continuation indent."""
    assert BaseFormatter._continuation_indent("ml: string") == ""


def test_continuation_indent_copies_leading_whitespace() -> None:
    """A nested block line's own leading run is copied verbatim."""
    assert BaseFormatter._continuation_indent("  z: string") == "  "
    assert BaseFormatter._continuation_indent("\t x: 1") == "\t "


def test_continuation_indent_replaces_dash_run_with_spaces() -> None:
    """A ``"- "`` line start aligns to the item's content column, not to the dash."""
    assert BaseFormatter._continuation_indent("- a: int") == "  "
    assert BaseFormatter._continuation_indent("  - a: int") == "    "


def test_continuation_indent_does_not_treat_bare_dash_key_as_sequence() -> None:
    """A mapping key beginning with ``-`` but no following space is not a sequence marker."""
    assert BaseFormatter._continuation_indent("-key: v") == ""


def test_resolved_deferred_text_returns_bodies_verbatim_for_multiline_body() -> None:
    """The query result keeps a multi-line body intact -- no comment-line rewriting."""
    formatter = _yaml_formatter()
    token = formatter.defer_comment("l1\nl2")
    assert formatter.resolved_deferred_text(f"ml: string{token}") == "ml: string\nl1\nl2"


def test_resolved_deferred_text_never_leaks_a_marker_token() -> None:
    """Every marker is excised from the query result."""
    formatter = _yaml_formatter()
    token = formatter.defer_comment("body")
    assert "⟪" not in formatter.resolved_deferred_text(f"k: v{token}")


def test_resolved_deferred_text_drops_duplicate_bodies() -> None:
    """Two markers referencing the same body contribute that body once."""
    formatter = _yaml_formatter()
    token = formatter.defer_comment("same")
    assert formatter.resolved_deferred_text(f"a{token} b{token}") == "a b\nsame"


def test_resolved_deferred_text_without_markers_is_the_input() -> None:
    """No marker means no body list and no newline joiner."""
    assert _yaml_formatter().resolved_deferred_text("k: v") == "k: v"


def test_resolved_deferred_text_non_string_input_stringifies() -> None:
    """``object`` typing: a non-str ``representation`` returns ``str(representation)``."""
    assert _yaml_formatter().resolved_deferred_text(5) == "5"


# ============================================================================
# _hoist_deferred_line (lsl-2026-09-05-006)
# ============================================================================


def test_hoist_deferred_line_absorbs_no_pre_existing_hash_comment() -> None:
    """Step-6 removal, pinned deliberately (design A7): ``# old  # new``, not a split value.

    A future reader who dislikes the double comment must change this test and read the
    removal note in ``_hoist_deferred_line`` rather than silently restoring the branch.
    """
    formatter = _yaml_formatter()
    token = formatter.defer_comment("new")
    assert formatter._hoist_deferred_line(f"k: v{token}  # old") == "k: v  # old  # new"


def test_hoist_deferred_line_preserves_hash_glued_pattern_value() -> None:
    """The ticket's headline case: a ``#`` glued to a preceding character stays intact."""
    formatter = _yaml_formatter()
    token = formatter.defer_comment("pattern: ^#[0-9a-f]{6}$")
    line = f"tag: 'string (PATTERN: ^#[0-9a-f]{{6}}$)'{token}"
    assert formatter._hoist_deferred_line(line) == (
        "tag: 'string (PATTERN: ^#[0-9a-f]{6}$)'  # pattern: ^#[0-9a-f]{6}$"
    )


def test_hoist_deferred_line_preserves_whitespace_preceded_hash_in_pattern() -> None:
    """Research Q28's ``^a #b$`` counter-example: whitespace before ``#`` also stays intact.

    This is the case option 1B ("only a whitespace-preceded prefix is a marker") would NOT
    have fixed, and is why the design deletes the absorb branch outright.
    """
    formatter = _yaml_formatter()
    token = formatter.defer_comment("odd")
    line = f"tag: 'string (PATTERN: ^a #b$)'{token}"
    assert formatter._hoist_deferred_line(line) == "tag: 'string (PATTERN: ^a #b$)'  # odd"


def test_hoist_deferred_line_jsonish_absorbs_no_pre_existing_slash_comment() -> None:
    """JSONish analogue of the ``# old  # new`` pin, including the trailing-comma rule."""
    formatter = _formatter()
    token = formatter.defer_comment("new")
    assert (
        formatter._hoist_deferred_line(f"  url: string{token}, // old")
        == "  url: string, // old // new"
    )


def test_hoist_deferred_line_multiline_body_top_level() -> None:
    """A body with ``"\\n"`` becomes one comment line per physical line, at column 0."""
    formatter = _yaml_formatter()
    token = formatter.defer_comment("l1\nl2\n\nl4")
    assert formatter._hoist_deferred_line(f"ml: string{token}") == (
        "ml: string  # l1\n# l2\n#\n# l4"
    )


def test_hoist_deferred_line_multiline_body_nested_block_indent() -> None:
    """Continuation lines inherit the dumped line's own leading-space indent."""
    formatter = _yaml_formatter()
    token = formatter.defer_comment("l1\nl2\n\nl4")
    assert formatter._hoist_deferred_line(f"  z: string{token}") == (
        "  z: string  # l1\n  # l2\n  #\n  # l4"
    )


def test_hoist_deferred_line_multiline_body_sequence_item_indent() -> None:
    """A ``"- "`` line start aligns continuations to the item's content column."""
    formatter = _yaml_formatter()
    token = formatter.defer_comment("l1\nl2\n\nl4")
    assert formatter._hoist_deferred_line(f"- a: int{token}") == (
        "- a: int  # l1\n  # l2\n  #\n  # l4"
    )


def test_hoist_deferred_line_multiline_body_nested_sequence_item_indent() -> None:
    """A ``"  - "`` line start compounds the leading-run and dash-run rules."""
    formatter = _yaml_formatter()
    token = formatter.defer_comment("l1\nl2\n\nl4")
    assert formatter._hoist_deferred_line(f"  - a: int{token}") == (
        "  - a: int  # l1\n    # l2\n    #\n    # l4"
    )


def test_hoist_deferred_line_single_line_body_is_byte_identical() -> None:
    """The single-line path is unchanged: no newline is ever introduced."""
    formatter = _yaml_formatter()
    token = formatter.defer_comment("one line")
    assert formatter._hoist_deferred_line(f"k: v{token}") == "k: v  # one line"


def test_hoist_deferred_line_without_marker_returns_input_unchanged() -> None:
    """The fast path is untouched."""
    assert _yaml_formatter()._hoist_deferred_line("k: 'a #b'  # plain") == "k: 'a #b'  # plain"
