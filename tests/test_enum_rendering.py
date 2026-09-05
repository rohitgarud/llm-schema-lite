"""Tests pinning the new enum/Literal/const rendering contract (lsl-2026-09-04-004).

These tests pin the OUTPUT CONTRACT in
``thoughts/tasks/lsl-2026-09-04-004-enum-and-literal-rendering-consistency/2026-09-04-plan.md``
§C, written *before* the implementation that satisfies it (Phase 3, per AGENTS.md TDD).

They are DELIBERATELY RED when this file lands: the JSONish half goes green in Phase 4
of that plan, the YAML half in Phase 6. Do not "fix" a failure here by weakening an
assertion -- fix it by landing the implementation in the phase that owns it.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from llm_schema_lite import simplify_schema
from llm_schema_lite.formatters.jsonish_formatter import JSONishFormatter
from llm_schema_lite.formatters.yaml_formatter import YAMLFormatter
from tests.conftest import (
    BoolLiterals,
    IntEnumModel,
    IntLiterals,
    IssueClassification,
    LiteralSingle,
    LiteralUnion,
    MixedTypeLiterals,
    ModelWithPriorityMetadata,
    Role,
    SingleConstInt,
)


def _line_with(rendered: str, needle: str) -> str:
    """Return the single rendered line containing `needle` (stripped)."""
    matches = [line.strip() for line in rendered.split("\n") if needle in line]
    assert len(matches) == 1, f"Expected exactly one line containing {needle!r}, got {matches!r}"
    return matches[0]


# ============================================================================
# JSONish tests (17)
# ============================================================================


def test_jsonish_enum_and_literal_render_identically() -> None:
    """AC 1: an enum-backed field and an equal-valued Literal render identically."""

    class Country(str, Enum):
        US = "US"
        CA = "CA"

    class WithEnum(BaseModel):
        country: Country

    class WithLiteral(BaseModel):
        country: Literal["US", "CA"]

    enum_result = simplify_schema(WithEnum, format_type="jsonish").to_string()
    literal_result = simplify_schema(WithLiteral, format_type="jsonish").to_string()

    expected = 'country*: string // one of: "US", "CA"'
    assert _line_with(enum_result, "country*") == expected
    assert _line_with(literal_result, "country*") == expected


def test_jsonish_int_enum_and_int_literal_render_as_int_one_of() -> None:
    """Contract rows 3 & 4: an int enum and an int Literal both render as `int // one of:`."""

    class P(int, Enum):
        A = 1
        B = 2

    class WithEnum(BaseModel):
        x: P

    class WithLiteral(BaseModel):
        x: Literal[1, 2]

    enum_result = simplify_schema(WithEnum, format_type="jsonish").to_string()
    literal_result = simplify_schema(WithLiteral, format_type="jsonish").to_string()

    expected = "x*: int // one of: 1, 2"
    assert _line_with(enum_result, "x*") == expected
    assert _line_with(literal_result, "x*") == expected


def test_jsonish_const_string_renders_as_one_of() -> None:
    """Contract row 6: a single-value string Literal (const) renders as `one of: "X"`."""

    class WithConst(BaseModel):
        kind: Literal["X"]

    result = simplify_schema(WithConst, format_type="jsonish").to_string()
    assert _line_with(result, "kind*") == 'kind*: string // one of: "X"'


def test_jsonish_const_int_renders_as_one_of() -> None:
    """Contract row 7: a single-value int Literal (const) renders as `one of: 7`."""

    class WithConst(BaseModel):
        version: Literal[7]

    result = simplify_schema(WithConst, format_type="jsonish").to_string()
    assert _line_with(result, "version*") == "version*: int // one of: 7"


def test_jsonish_optional_enum_renders_null_outside_comment() -> None:
    """Contract row 8: `OR null` stays outside the `one of:` value list."""

    class WithOptionalRole(BaseModel):
        role: Role | None = Role.USER

    result = simplify_schema(WithOptionalRole, format_type="jsonish").to_string()
    line = _line_with(result, "role:")

    # `Role`'s class docstring ("User role enum.") is the $defs description and is folded
    # into the same comment, exactly as contract row 15a folds `Country`'s docstring.
    expected = (
        'role: string OR null // one of: "admin", "user", "guest"; '
        "User role enum.; (default='user')"
    )
    assert line == expected
    assert line.index(" OR null") < line.index("//")
    assert "null" not in line.split("one of:", 1)[1].split(";")[0]


def test_jsonish_enum_output_has_no_escaped_quotes() -> None:
    """B1 regression guard: quoted enum values must not survive as an escaped `\\"`."""

    result = simplify_schema(LiteralUnion, format_type="jsonish").to_string()
    assert '"draft"' in result
    assert '\\"draft\\"' not in result


@pytest.mark.parametrize(
    "model",
    [
        LiteralUnion,
        IntLiterals,
        BoolLiterals,
        MixedTypeLiterals,
        IssueClassification,
        ModelWithPriorityMetadata,
        IntEnumModel,
    ],
)
def test_jsonish_enum_output_has_no_options_or_pipe_join(model: type[BaseModel]) -> None:
    """Ticket AC 3: no `OPTIONS` keyword and no `"| "` pipe join anywhere in output."""

    result = simplify_schema(model, format_type="jsonish").to_string()
    assert "OPTIONS" not in result
    assert "| " not in result


def test_jsonish_enum_class_name_not_emitted() -> None:
    """Ticket AC 3 / D4: the enum's Pydantic class name/title must not leak into output."""

    class WithRole(BaseModel):
        role: Role

    result = simplify_schema(WithRole, format_type="jsonish").to_string()
    assert "Role:" not in result

    result2 = simplify_schema(ModelWithPriorityMetadata, format_type="jsonish").to_string()
    assert "PriorityWithMetadata" not in result2


def test_jsonish_container_of_enum_hoists_comment() -> None:
    """Contract rows 11, 12, 12a: containers of an enum hoist the comment onto their line."""

    class WithRoleList(BaseModel):
        roles: list[Role]

    class WithRoleDict(BaseModel):
        by_role: dict[str, Role]

    # `Role`'s class docstring is the $defs description and is folded into the same
    # comment, exactly as contract row 15a folds `Country`'s docstring.
    list_result = simplify_schema(WithRoleList, format_type="jsonish").to_string()
    expected_list_line = 'roles*: string [] // one of: "admin", "user", "guest"; User role enum.'
    assert _line_with(list_result, "roles*") == expected_list_line

    dict_result = simplify_schema(WithRoleDict, format_type="jsonish").to_string()
    expected_inner_line = '<string>: string // one of: "admin", "user", "guest"; User role enum.'
    assert _line_with(dict_result, "<string>:") == expected_inner_line


def test_jsonish_two_fragments_on_one_line_joined_with_semicolon() -> None:
    """Contract row 16 / §C.2 rule 2: two enum fragments on one line join with `"; "`."""

    schema = {
        "type": "object",
        "properties": {
            "tup": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "prefixItems": [
                    {"enum": ["x"], "type": "string"},
                    {"enum": ["p", "q"], "type": "string"},
                ],
            }
        },
        "required": ["tup"],
    }
    result = JSONishFormatter(schema).transform_schema()
    expected = 'tup*: [string, string] // one of: "x"; one of: "p", "q"'
    assert _line_with(result, "tup*") == expected

    dup_schema = {
        "type": "object",
        "properties": {
            "tup": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "prefixItems": [
                    {"enum": ["x"], "type": "string"},
                    {"enum": ["x"], "type": "string"},
                ],
            }
        },
        "required": ["tup"],
    }
    dup_result = JSONishFormatter(dup_schema).transform_schema()
    expected_dup = 'tup*: [string, string] // one of: "x"'
    assert _line_with(dup_result, "tup*") == expected_dup


def test_jsonish_mixed_type_enum_renders_as_any() -> None:
    """Contract row 13 / D2-2b + D3: a heterogeneous, untyped enum renders as `any`."""

    schema = {
        "type": "object",
        "properties": {"untyped": {"enum": ["a", 1, True, None]}},
        "required": ["untyped"],
    }
    result = JSONishFormatter(schema).transform_schema()
    expected = 'untyped*: any // one of: "a", 1, true, null'
    assert _line_with(result, "untyped*") == expected


def test_jsonish_enum_value_with_quote_comma_and_newline_stays_one_line() -> None:
    """Contract row 14 / D3: pathological enum values stay on exactly one physical line."""

    class Weird(str, Enum):
        QUOTE = 'say "hi"'
        COMMA = "a,b"
        NL = "a\nb"

    class WithWeird(BaseModel):
        w: Weird

    result = simplify_schema(WithWeird, format_type="jsonish").to_string()
    one_of_lines = [ln for ln in result.split("\n") if "one of:" in ln]
    assert len(one_of_lines) == 1
    line = one_of_lines[0]
    assert '"say \\"hi\\""' in line
    assert '"a,b"' in line
    assert '"a\\nb"' in line


def test_jsonish_enriched_enum_single_line_no_literal_newline() -> None:
    """Contract row 10 / D7-7c: descriptions+aliases fold into one `one of:` line."""

    result = simplify_schema(ModelWithPriorityMetadata, format_type="jsonish").to_string()
    line = _line_with(result, "priority*")

    assert "one of:" in line
    assert '"low" (Non-urgent, can wait)' in line
    assert "aliases: urgent, blocker" in line
    assert "\\n" not in result


@pytest.mark.parametrize(
    "model",
    [
        LiteralUnion,
        IntLiterals,
        BoolLiterals,
        MixedTypeLiterals,
        IssueClassification,
        ModelWithPriorityMetadata,
        IntEnumModel,
        LiteralSingle,
        SingleConstInt,
    ],
)
def test_jsonish_no_deferred_marker_leaks(model: type[BaseModel]) -> None:
    """No deferred-comment marker (`⟪`/`⟫`) may survive into rendered output."""

    result = simplify_schema(model, format_type="jsonish").to_string()
    assert "⟪" not in result
    assert "⟫" not in result


def test_jsonish_marker_collision_with_user_text_is_harmless() -> None:
    """A description with token-shaped text must survive verbatim, not corrupt output."""

    class WithCollision(BaseModel):
        x: Literal["a", "b"] = Field(..., description="uses ⟪lsl0⟫ literally")

    result = simplify_schema(WithCollision, format_type="jsonish").to_string()
    assert result.count("uses ⟪lsl0⟫ literally") == 1
    assert result.count('one of: "a", "b"') == 1


def test_jsonish_thirty_value_enum_stays_on_one_line() -> None:
    """Contract row 17: a 30-value enum stays on one line, with no truncation."""

    values = tuple(f"v{i}" for i in range(30))

    class WithThirty(BaseModel):
        v: Literal[values]

    result = simplify_schema(WithThirty, format_type="jsonish").to_string()
    line = _line_with(result, "one of:")

    for i in range(30):
        assert f'"v{i}"' in line
    assert "..." not in line
    assert "options" not in line.lower()


@pytest.mark.parametrize("model", [LiteralUnion, LiteralSingle, MixedTypeLiterals])
def test_jsonish_enum_line_ends_cleanly_with_quote(model: type[BaseModel]) -> None:
    """Positive counterpart of the P5 STRAY_DELIMITER exemption.

    A hoisted `// one of: ...` line that ends in a quoted value ends in a real, authored
    quote -- never a leaked JSON-escaped `\\"`.
    """

    result = simplify_schema(model, format_type="jsonish").to_string()
    one_of_lines = [ln.rstrip() for ln in result.split("\n") if "// one of: " in ln]
    assert one_of_lines

    for line in one_of_lines:
        assert not line.endswith('\\"')
        assert not line.endswith('\\",')
        if '"' in line:
            assert line.endswith('"') or line.endswith('",')


# ============================================================================
# YAML tests (15)
# ============================================================================


def test_yaml_enum_and_literal_render_identically() -> None:
    """AC 1: an enum-backed field and an equal-valued Literal render identically in YAML."""

    class Country(str, Enum):
        US = "US"
        CA = "CA"

    class WithEnum(BaseModel):
        country: Country

    class WithLiteral(BaseModel):
        country: Literal["US", "CA"]

    enum_result = simplify_schema(WithEnum, format_type="yaml").to_string()
    literal_result = simplify_schema(WithLiteral, format_type="yaml").to_string()

    expected = 'country*: string  # one of: "US", "CA"'
    assert _line_with(enum_result, "country*") == expected
    assert _line_with(literal_result, "country*") == expected


def test_yaml_int_enum_and_int_literal_render_as_int_one_of() -> None:
    """Contract rows 3 & 4 (YAML): int enum and int Literal render as `int  # one of:`."""

    class P(int, Enum):
        A = 1
        B = 2

    class WithEnum(BaseModel):
        x: P

    class WithLiteral(BaseModel):
        x: Literal[1, 2]

    enum_result = simplify_schema(WithEnum, format_type="yaml").to_string()
    literal_result = simplify_schema(WithLiteral, format_type="yaml").to_string()

    expected = "x*: int  # one of: 1, 2"
    assert _line_with(enum_result, "x*") == expected
    assert _line_with(literal_result, "x*") == expected


def test_yaml_const_string_renders_as_one_of() -> None:
    """Contract row 6 (YAML): a single-value string Literal (const) renders as one of."""

    class WithConst(BaseModel):
        kind: Literal["X"]

    result = simplify_schema(WithConst, format_type="yaml").to_string()
    assert _line_with(result, "kind*") == 'kind*: string  # one of: "X"'


def test_yaml_const_int_renders_as_one_of() -> None:
    """Contract row 7 (YAML): a single-value int Literal (const) renders as one of."""

    class WithConst(BaseModel):
        version: Literal[7]

    result = simplify_schema(WithConst, format_type="yaml").to_string()
    assert _line_with(result, "version*") == "version*: int  # one of: 7"


def test_yaml_optional_enum_renders_null_outside_comment() -> None:
    """Contract row 8 (YAML): `OR null` stays outside the `one of:` value list."""

    class WithOptionalRole(BaseModel):
        role: Role | None = Role.USER

    result = simplify_schema(WithOptionalRole, format_type="yaml").to_string()
    line = _line_with(result, "role:")

    # ``tests/conftest.py``'s real ``Role`` carries the class docstring "User role enum.",
    # which is the ``$defs`` description and is folded into the same comment (contract row
    # 15a). The default is stated ONCE, as ``(default='user')``: lsl-2026-09-04-006 (D5)
    # widened ``add_metadata``'s ``exclude`` to drop METADATA_MAP's duplicate
    # ``(defaults to user)`` restatement, which was an AC3 duplication defect.
    expected = (
        'role: string OR null  # one of: "admin", "user", "guest"; '
        "User role enum.; (default='user')"
    )
    assert line == expected
    assert "null" not in line.split("one of:", 1)[1].split(";")[0]


def test_yaml_enum_line_is_bare_plain_scalar() -> None:
    """AC-4: a YAML enum line is a bare plain scalar with a real trailing comment."""

    result = simplify_schema(LiteralUnion, format_type="yaml").to_string()
    line = _line_with(result, "status*")

    expected = 'status*: string  # one of: "draft", "published", "archived"'
    assert line == expected
    assert "  # one of: " in line
    _, _, value_part = line.partition(": ")
    assert not value_part.startswith(("'", '"'))
    assert "OPTIONS" not in result

    class Country(str, Enum):
        US = "US"
        CA = "CA"

    class WithCountry(BaseModel):
        country: Country

    country_result = simplify_schema(WithCountry, format_type="yaml").to_string()
    country_line = _line_with(country_result, "country*")
    assert country_line == 'country*: string  # one of: "US", "CA"'
    _, _, country_value_part = country_line.partition(": ")
    assert not country_value_part.startswith(("'", '"'))

    # Row 17: a 30-value enum stays on one line -- no PyYAML folding.
    values = tuple(f"v{i}" for i in range(30))

    class WithThirty(BaseModel):
        v: Literal[values]

    thirty_result = simplify_schema(WithThirty, format_type="yaml").to_string()
    thirty_line = _line_with(thirty_result, "one of:")
    for i in range(30):
        assert f'"v{i}"' in thirty_line


@pytest.mark.parametrize(
    "model",
    [
        LiteralUnion,
        IntLiterals,
        BoolLiterals,
        MixedTypeLiterals,
        IssueClassification,
        ModelWithPriorityMetadata,
        IntEnumModel,
    ],
)
def test_yaml_enum_output_has_no_options_or_pipe_join(model: type[BaseModel]) -> None:
    """Ticket AC 3 (YAML): no `OPTIONS` keyword and no `"| "` pipe join anywhere in output."""

    result = simplify_schema(model, format_type="yaml").to_string()
    assert "OPTIONS" not in result
    assert "| " not in result


def test_yaml_enum_class_name_not_emitted() -> None:
    """Ticket AC 3 / D4 (YAML): the enum's Pydantic class name/title must not leak."""

    class WithRole(BaseModel):
        role: Role

    result = simplify_schema(WithRole, format_type="yaml").to_string()
    assert "Role:" not in result

    result2 = simplify_schema(ModelWithPriorityMetadata, format_type="yaml").to_string()
    assert "PriorityWithMetadata" not in result2


def test_yaml_container_of_enum_hoists_comment() -> None:
    """Contract rows 11, 12 (YAML): containers of an enum hoist the comment onto their line."""

    class WithRoleList(BaseModel):
        roles: list[Role]

    class WithRoleDict(BaseModel):
        by_role: dict[str, Role]

    list_result = simplify_schema(WithRoleList, format_type="yaml").to_string()
    # The trailing "; User role enum." is ``Role``'s class docstring, folded in per
    # contract row 15a.
    expected_list_line = 'roles*: list[string]  # one of: "admin", "user", "guest"; User role enum.'
    assert _line_with(list_result, "roles*") == expected_list_line

    dict_result = simplify_schema(WithRoleDict, format_type="yaml").to_string()
    expected_dict_line = (
        'by_role*: dict[string, string]  # one of: "admin", "user", "guest"; User role enum.'
    )
    assert _line_with(dict_result, "by_role*") == expected_dict_line


def test_yaml_two_fragments_on_one_line_joined_with_semicolon() -> None:
    """Contract row 16 (YAML): two enum fragments on one line join with `"; "`."""

    schema = {
        "type": "object",
        "properties": {
            "tup": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "prefixItems": [
                    {"enum": ["x"], "type": "string"},
                    {"enum": ["p", "q"], "type": "string"},
                ],
            }
        },
        "required": ["tup"],
    }
    result = YAMLFormatter(schema).transform_schema()
    expected = 'tup*: tuple[string, string]  # one of: "x"; one of: "p", "q"'
    assert _line_with(result, "tup*") == expected


def test_yaml_mixed_type_enum_renders_as_any() -> None:
    """Contract row 13 (YAML): a heterogeneous, untyped enum renders as `any`."""

    schema = {
        "type": "object",
        "properties": {"untyped": {"enum": ["a", 1, True, None]}},
        "required": ["untyped"],
    }
    result = YAMLFormatter(schema).transform_schema()
    expected = 'untyped*: any  # one of: "a", 1, true, null'
    assert _line_with(result, "untyped*") == expected


def test_yaml_enum_value_with_quote_comma_and_newline_stays_one_line() -> None:
    """Contract row 14 (YAML): pathological enum values stay on exactly one physical line."""

    class Weird(str, Enum):
        QUOTE = 'say "hi"'
        COMMA = "a,b"
        NL = "a\nb"

    class WithWeird(BaseModel):
        w: Weird

    result = simplify_schema(WithWeird, format_type="yaml").to_string()
    one_of_lines = [ln for ln in result.split("\n") if "one of:" in ln]
    assert len(one_of_lines) == 1
    line = one_of_lines[0]
    assert '"say \\"hi\\""' in line
    assert '"a,b"' in line
    assert '"a\\nb"' in line


def test_yaml_enriched_enum_single_line_no_literal_newline() -> None:
    """Contract row 10 (YAML): descriptions+aliases fold into one `one of:` line."""

    result = simplify_schema(ModelWithPriorityMetadata, format_type="yaml").to_string()
    line = _line_with(result, "priority*")

    assert "one of:" in line
    assert '"low" (Non-urgent, can wait)' in line
    assert "aliases: urgent, blocker" in line
    assert "\\n" not in result


@pytest.mark.parametrize(
    "model",
    [
        LiteralUnion,
        IntLiterals,
        BoolLiterals,
        MixedTypeLiterals,
        IssueClassification,
        ModelWithPriorityMetadata,
        IntEnumModel,
        LiteralSingle,
        SingleConstInt,
    ],
)
def test_yaml_no_deferred_marker_leaks(model: type[BaseModel]) -> None:
    """No deferred-comment marker (`⟪`/`⟫`) may survive into rendered YAML output."""

    result = simplify_schema(model, format_type="yaml").to_string()
    assert "⟪" not in result
    assert "⟫" not in result


def test_yaml_marker_collision_with_user_text_is_harmless() -> None:
    """A description with token-shaped text must survive verbatim in YAML output."""

    class WithCollision(BaseModel):
        x: Literal["a", "b"] = Field(..., description="uses ⟪lsl0⟫ literally")

    result = simplify_schema(WithCollision, format_type="yaml").to_string()
    assert result.count("uses ⟪lsl0⟫ literally") == 1
    assert result.count('one of: "a", "b"') == 1
