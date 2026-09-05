"""Tests for JSONish formatter."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

import pytest
from pydantic import BaseModel, Field, TypeAdapter

from llm_schema_lite import FormatterConfig, simplify_schema
from llm_schema_lite.formatters.base import DEFERRED_CLOSE, DEFERRED_OPEN
from llm_schema_lite.formatters.jsonish_formatter import JSONishFormatter
from llm_schema_lite.schema_normalization import auto_title_for_key
from tests.conftest import (
    ADDITIONAL_ITEMS_SCHEMA,
    EMPTY_SCHEMA,
    PREFIX_ITEMS_SCHEMA,
    Address,
    ComplexTypes,
    ConstrainedFormatterModel,
    ModelWithAlias,
    ModelWithPriorityMetadata,
    Order,
    OrderedFieldsModel,
    PersonWithAddress,
    Product,
    RequiredOptionalModel,
    Root,
    SimpleFormatterModel,
    UnionTypes,
    WithFieldDescriptions,
    WithTitleDescription,
)
from tests.formatter_helpers import (
    assert_required_optional_consistent,
    assert_schema_info_comment_presence,
    parse_jsonish_root_fields,
)

# Alias for backwards compatibility in tests
SimpleModel = SimpleFormatterModel


def test_jsonish_formatter_produces_valid_output():
    """Test that JSONish formatter produces valid output with required fields marked."""
    schema = SimpleModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert_schema_info_comment_presence(result, include_metadata=True, schema=schema)

    # Verify asterisk notation comment is present (schema has required fields)
    assert "Fields marked with * are required" in result
    assert "//" in result  # JSONish comment prefix


def test_jsonish_formatter_without_metadata():
    """Test JSONish formatter without metadata."""
    schema = SimpleModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert_schema_info_comment_presence(result, include_metadata=False, schema=schema)

    # Verify title comment is not included when metadata is off
    assert "//Title:" not in result


def test_jsonish_formatter_with_nested_defs():
    """Test JSONish formatter with nested $defs."""
    schema = PersonWithAddress.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Should contain required fields
    assert "name*:" in result
    assert "address*:" in result
    assert_required_optional_consistent(result, schema)

    # Nested Address required fields are now expanded onto their own lines; the
    # python-dict-repr alternative this test used to tolerate is the bug (ticket 001).
    assert "street*:" in result
    assert "city*:" in result


def test_jsonish_formatter_key_order_preserved():
    """Test that JSONish formatter preserves key order (dict order)."""
    schema = OrderedFieldsModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Find positions of each field in the output
    first_pos = result.find("first*:")
    second_pos = result.find("second*:")
    third_pos = result.find("third*:")

    # Verify all fields are present
    assert first_pos != -1, f"Expected 'first*:' in output. Snippet: {result[:200]!r}"
    assert second_pos != -1, f"Expected 'second*:' in output. Snippet: {result[:200]!r}"
    assert third_pos != -1, f"Expected 'third*:' in output. Snippet: {result[:200]!r}"

    # Check that fields appear in order
    assert first_pos < second_pos < third_pos


def test_jsonish_formatter_caching():
    """Test that formatter caching works correctly."""
    schema = SimpleModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)

    # First call
    result1 = formatter.transform_schema()
    # Second call should use cache
    result2 = formatter.transform_schema()

    # The cache is stored in simplified_schema
    assert hasattr(formatter, "simplified_schema")
    assert formatter.simplified_schema is not None

    # The first call returns the pre-normalized string, while subsequent calls
    # return the cached normalized string. Compare after normalizing whitespace.
    def _normalize_spaces(s: str) -> str:
        return re.sub(r"[ ]{2,}", " ", s)

    assert _normalize_spaces(result1) == _normalize_spaces(result2)
    assert result2 == formatter.simplified_schema
    assert_required_optional_consistent(result2, schema)


def test_jsonish_formatter_with_constraints():
    """Test that JSONish formatter includes field constraints."""
    schema = ConstrainedFormatterModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Verify constraints are included (ConstrainedFormatterModel has name, age, score, tags)
    assert_required_optional_consistent(result, schema)

    assert "name*:" in result
    assert (
        "(1-100 chars)" in result
    ), f"Expected string length constraint '(1-100 chars)' in output. Snippet: {result[:300]!r}"

    assert "age*:" in result
    assert (
        "(0 to 150)" in result
    ), f"Expected integer range constraint '(0 to 150)' in output. Snippet: {result[:300]!r}"

    assert "score*:" in result
    assert re.search(r"\(0(\.0)? to 100(\.0)?\)", result), (
        "Expected numeric range constraint '(0.0 to 100.0)' (or int form) in output. "
        f"Snippet: {result[:350]!r}"
    )


def test_jsonish_formatter_with_optional_union():
    """Test that JSONish formatter handles optional fields (union with None)."""
    schema = RequiredOptionalModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)

    # Explicit regression for union-with-null optional
    assert "required_field*:" in result
    assert "optional_field:" in result
    assert "optional_field*:" not in result


def test_required_optional_parsing_matches_schema():
    """Root-field parsing should match schema required/properties exactly."""
    schema = RequiredOptionalModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    fields = parse_jsonish_root_fields(result)
    field_names = {name for name, _ in fields}
    required_in_output = {name for name, is_req in fields if is_req}

    required_in_schema = set(schema.get("required", []) or [])
    properties_in_schema = set((schema.get("properties", {}) or {}).keys())

    missing_props = properties_in_schema - field_names
    assert not missing_props, f"Missing properties in output: {sorted(missing_props)}"

    assert required_in_output == required_in_schema, (
        f"Required fields in output should match schema. "
        f"schema={sorted(required_in_schema)} output={sorted(required_in_output)}"
    )


@pytest.mark.parametrize(
    "model_cls",
    [
        SimpleFormatterModel,
        RequiredOptionalModel,
        OrderedFieldsModel,
        ConstrainedFormatterModel,
    ],
)
def test_jsonish_lists_all_root_properties_for_models(model_cls):
    """Contract: output lists all schema root properties as field lines."""
    schema = model_cls.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    fields = parse_jsonish_root_fields(result)
    field_names = {name for name, _ in fields}
    expected_props = set((schema.get("properties", {}) or {}).keys())

    missing = expected_props - field_names
    assert not missing, (
        f"Expected all properties to appear as root fields; missing={sorted(missing)}. "
        f"Parsed fields={sorted(field_names)}"
    )


@pytest.mark.parametrize(
    "model_cls",
    [
        SimpleFormatterModel,
        RequiredOptionalModel,
        OrderedFieldsModel,
        ConstrainedFormatterModel,
    ],
)
def test_jsonish_required_optional_consistent_for_models(model_cls):
    """Contract: required/optional marking matches schema required list."""
    schema = model_cls.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()
    assert_required_optional_consistent(result, schema)


def test_jsonish_format_scaffolding_regression():
    """Regression: basic scaffolding is stable for a simple model."""
    schema = SimpleFormatterModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # First non-comment non-empty line should be the opening brace.
    lines = [
        ln.strip() for ln in result.splitlines() if ln.strip() and not ln.strip().startswith("//")
    ]
    assert lines, "Expected non-empty output"
    assert lines[0] == "{", f"Expected opening '{{' line, got: {lines[0]!r}"
    assert lines[-1] == "}", f"Expected closing '}}' line, got: {lines[-1]!r}"


def test_empty_object_schema():
    """Edge: empty object schema should not crash and should render braces."""
    formatter = JSONishFormatter(EMPTY_SCHEMA, include_metadata=False)
    result = formatter.transform_schema()

    assert "{" in result
    assert "}" in result
    assert parse_jsonish_root_fields(result) == []


def test_single_required_field():
    """Edge: single required property should be marked with '*'."""
    schema = {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert "id*:" in result
    assert_required_optional_consistent(result, schema)


def test_all_optional_no_asterisks():
    """Edge: no required list => no required markers in output."""
    schema = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert not re.search(r"[a-zA-Z_][a-zA-Z0-9_]*\*:", result), (
        "Did not expect any required markers ('*:') when schema has no required fields. "
        f"Snippet: {result[:250]!r}"
    )


# ============================================================================
# Enum and Literal Tests
# ============================================================================


def test_jsonish_formatter_with_string_enum():
    """Test JSONish formatter with string enum (Role)."""
    from pydantic import BaseModel

    from tests.conftest import Role

    class RoleModel(BaseModel):
        """Model with role enum."""

        role: Role

    schema = RoleModel.model_json_schema()
    formatter = JSONishFormatter(schema)
    result = formatter.transform_schema()

    # Should contain enum values
    assert "role*:" in result
    # Check for enum representation (OPTIONS keyword or pipe-separated values)
    assert "admin" in result
    assert "user" in result
    assert "guest" in result


def test_jsonish_formatter_with_int_enum():
    """Test JSONish formatter with integer enum."""
    from tests.conftest import IntEnumModel

    schema = IntEnumModel.model_json_schema()
    formatter = JSONishFormatter(schema)
    result = formatter.transform_schema()

    # Should contain priority field
    assert "priority*:" in result
    # Check for enum values in output
    assert "1" in result or "LOW" in result


def test_jsonish_formatter_enum_with_descriptions_and_aliases():
    """Test JSONish formatter shows OPTIONS with descriptions and aliases."""
    from llm_schema_lite import simplify_schema
    from tests.conftest import ModelWithPriorityMetadata

    result = simplify_schema(ModelWithPriorityMetadata, format_type="jsonish").to_string()
    assert "one of:" in result
    assert "(Non-urgent, can wait)" in result
    assert "aliases: urgent, blocker" in result
    assert "Non-urgent, can wait" in result
    assert "Urgent, blocking issue" in result
    assert "aliases:" in result
    assert "urgent" in result
    assert "blocker" in result
    line = _line_with(result, "priority*:")
    assert (
        'one of: "low" (Non-urgent, can wait), "medium" (Normal priority), '
        '"high" (Needs attention soon), "critical" (Urgent, blocking issue; '
        "aliases: urgent, blocker)"
    ) in line


def test_jsonish_formatter_enum_without_metadata_unchanged():
    """Test enum without _descriptions/_aliases has no description comment."""
    from pydantic import BaseModel

    from llm_schema_lite import simplify_schema
    from tests.conftest import Role

    class M(BaseModel):
        role: Role

    result = simplify_schema(M, format_type="jsonish").to_string()
    assert "string // one of:" in result
    assert "(" not in _line_with(result, "role")


def test_jsonish_formatter_with_literal_single():
    """Test JSONish formatter with single literal value."""
    from tests.conftest import LiteralSingle

    schema = LiteralSingle.model_json_schema()
    formatter = JSONishFormatter(schema)
    result = formatter.transform_schema()

    # Should contain api_version field
    assert "api_version*:" in result
    # Single literal should be rendered as the const value
    assert "v1" in result


def test_jsonish_formatter_with_literal_union():
    """Test JSONish formatter with union of literals."""
    from tests.conftest import LiteralUnion

    schema = LiteralUnion.model_json_schema()
    formatter = JSONishFormatter(schema)
    result = formatter.transform_schema()

    # Should contain status field
    assert "status*:" in result
    # Should use the new "string // one of:" format for multiple literals
    assert "string // one of:" in result
    # Should contain all literal values (quoted strings)
    assert "draft" in result
    assert "published" in result
    assert "archived" in result


def test_jsonish_formatter_with_int_literals():
    """Test JSONish formatter with integer literals."""
    from tests.conftest import IntLiterals

    schema = IntLiterals.model_json_schema()
    formatter = JSONishFormatter(schema)
    result = formatter.transform_schema()

    # Should contain priority field
    assert "priority*:" in result
    # Should use the new "int // one of:" format for multiple integer literals
    assert "int // one of:" in result
    # Should contain all integer values (unquoted)
    assert "1" in result
    assert "2" in result
    assert "3" in result
    assert "4" in result
    assert "5" in result
    # Verify unquoted format (no quotes around numbers)
    assert "int // one of: 1, 2, 3, 4, 5" in result


def test_jsonish_formatter_with_bool_literals():
    """Test JSONish formatter with boolean literals."""
    from tests.conftest import BoolLiterals

    schema = BoolLiterals.model_json_schema()
    formatter = JSONishFormatter(schema)
    result = formatter.transform_schema()

    # Should contain flag field
    assert "flag*:" in result
    # Should use the new "bool // one of:" format for boolean literals
    assert "bool // one of: true, false" in result
    # Should contain lowercase, unquoted boolean values
    assert "true" in result
    assert "false" in result


def test_jsonish_formatter_with_mixed_type_literals():
    """Test JSONish formatter with mixed type literals (string, int, bool)."""
    from tests.conftest import MixedTypeLiterals

    schema = MixedTypeLiterals.model_json_schema()
    formatter = JSONishFormatter(schema)
    result = formatter.transform_schema()

    # Should contain all three fields
    assert "status*:" in result
    assert "level*:" in result
    assert "enabled*:" in result

    # String literals should be quoted
    assert "active" in result
    assert "inactive" in result

    # Integer literals should be unquoted
    assert "1" in result and "2" in result and "3" in result

    # Boolean literals should be lowercase and unquoted
    assert "true" in result
    assert "false" in result


def test_jsonish_formatter_with_single_const_int():
    """Test JSONish formatter with single integer const."""
    from tests.conftest import SingleConstInt

    schema = SingleConstInt.model_json_schema()
    formatter = JSONishFormatter(schema)
    result = formatter.transform_schema()

    # Should contain version field
    assert "version*:" in result
    # Single integer literal should be rendered as unquoted number
    assert "version*: int // one of: 1" in result


def test_jsonish_formatter_with_issue_classification():
    """Test JSONish formatter with IssueClassification model (integration test)."""
    from tests.conftest import IssueClassification

    schema = IssueClassification.model_json_schema()
    formatter = JSONishFormatter(schema)
    result = formatter.transform_schema()

    # Should contain both fields
    assert "category*:" in result
    assert "priority*:" in result

    # String literals should be quoted with the new "string // one of:" format
    assert "string // one of:" in result
    assert "bug" in result
    assert "feature" in result
    assert "question" in result

    # Integer literals should be unquoted with OPTIONS format
    assert "1" in result
    assert "2" in result
    assert "3" in result
    assert "4" in result
    assert "5" in result


# ============================================================================
# Array and Collection Tests
# ============================================================================


def test_jsonish_formatter_with_array_of_strings():
    """Test JSONish formatter with simple array of strings."""
    from tests.conftest import ArrayOfStrings

    schema = ArrayOfStrings.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain items field
    assert "items*:" in result
    # Should indicate it's an array of strings
    assert "string" in result
    assert "[]" in result or "[" in result


def test_jsonish_formatter_with_array_constraints():
    """Test JSONish formatter with array min/max items constraints."""
    from tests.conftest import ArrayMinMaxItems

    schema = ArrayMinMaxItems.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Should contain tags field
    assert "tags*:" in result
    # Should include array length constraints
    # The formatter should show constraints like (1-5 items) or similar
    assert ("1" in result and "5" in result) or "items" in result.lower()


def test_jsonish_formatter_with_unique_items():
    """Test JSONish formatter with unique items constraint."""
    from tests.conftest import ArrayUniqueItems

    schema = ArrayUniqueItems.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain unique_tags field
    assert "unique_tags*:" in result
    # May show uniqueItems constraint if formatter supports it
    # At minimum should render as an array


def test_jsonish_formatter_with_array_of_refs():
    """Test JSONish formatter with array of referenced objects."""
    from tests.conftest import ArrayOfRefsModel

    schema = ArrayOfRefsModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain addresses and products fields
    assert "addresses*:" in result
    assert "products*:" in result
    assert "users:" in result  # optional field
    # Nested refs expand inline in compact array-of-object form, with no Python repr
    assert "addresses*: [{" in result
    assert "{'" not in result


# ============================================================================
# Nested and Complex Structure Tests
# ============================================================================


def test_jsonish_nested_optional_ref_renders_as_block() -> None:
    """AC1: a nested optional $ref model expands as a block, not a Python dict repr."""
    from llm_schema_lite import simplify_schema

    class Country(str, Enum):
        US = "US"
        CA = "CA"

    class Address(BaseModel):
        street: str = Field(..., description="Street")
        city: str = Field(..., description="City")
        country: Country = Field(..., description="Country code")

    class Addr(BaseModel):
        street: str = Field(..., description="Street")
        city: str = Field(..., description="City")

    class Patient(BaseModel):
        name: str = Field(..., description="Patient name")
        address: Address | None = Field(None, description="Home address")

    class P1(BaseModel):
        name: str = Field(..., description="Patient name")
        address: Addr | None = Field(None, description="Home address")

    result = simplify_schema(Patient, format_type="jsonish").to_string()
    assert result == "\n".join(
        [
            "//Title: Patient",
            "// Fields marked with * are required",
            "{",
            "  name*: string // Patient name,",
            "  address: {",
            "    street*: string // Street,",
            "    city*: string // City,",
            '    country*: string // one of: "US", "CA"',
            "  } OR null  // Home address (default=null)",
            "}",
        ]
    )
    assert "{'" not in result

    result_p1 = simplify_schema(P1, format_type="jsonish").to_string()
    assert result_p1 == "\n".join(
        [
            "//Title: P1",
            "// Fields marked with * are required",
            "{",
            "  name*: string // Patient name,",
            "  address: {",
            "    street*: string // Street,",
            "    city*: string // City",
            "  } OR null  // Home address (default=null)",
            "}",
        ]
    )


def test_jsonish_nested_list_ref_renders_as_compact_brackets() -> None:
    """AC2: a nested list[$ref] field renders as compact array-of-object brackets."""
    from llm_schema_lite import simplify_schema

    class Country(str, Enum):
        US = "US"
        CA = "CA"

    class Address(BaseModel):
        street: str = Field(..., description="Street")
        city: str = Field(..., description="City")
        country: Country = Field(..., description="Country code")

    class Addr(BaseModel):
        street: str = Field(..., description="Street")
        city: str = Field(..., description="City")

    class PatientList(BaseModel):
        name: str = Field(..., description="Patient name")
        addresses: list[Address] = Field(..., description="Addresses")

    class P2(BaseModel):
        name: str = Field(..., description="Patient name")
        addresses: list[Addr] = Field(..., description="Addresses")

    result = simplify_schema(PatientList, format_type="jsonish").to_string()
    assert result == "\n".join(
        [
            "//Title: PatientList",
            "// Fields marked with * are required",
            "{",
            "  name*: string // Patient name,",
            "  addresses*: [{",
            "    street*: string // Street,",
            "    city*: string // City,",
            '    country*: string // one of: "US", "CA"',
            "  }]  // Addresses",
            "}",
        ]
    )

    result_p2 = simplify_schema(P2, format_type="jsonish").to_string()
    assert result_p2 == "\n".join(
        [
            "//Title: P2",
            "// Fields marked with * are required",
            "{",
            "  name*: string // Patient name,",
            "  addresses*: [{",
            "    street*: string // Street,",
            "    city*: string // City",
            "  }]  // Addresses",
            "}",
        ]
    )


def test_jsonish_nested_optional_list_ref_renders_as_compact_brackets() -> None:
    """An optional list[$ref] field renders as compact brackets with an ``OR null`` tail."""
    from llm_schema_lite import simplify_schema

    class Country(str, Enum):
        US = "US"
        CA = "CA"

    class Address(BaseModel):
        street: str = Field(..., description="Street")
        city: str = Field(..., description="City")
        country: Country = Field(..., description="Country code")

    class Addr(BaseModel):
        street: str = Field(..., description="Street")
        city: str = Field(..., description="City")

    class PatientOptList(BaseModel):
        name: str = Field(..., description="Patient name")
        addresses: list[Address] | None = Field(None, description="Addresses")

    class P3(BaseModel):
        name: str = Field(..., description="Patient name")
        addresses: list[Addr] | None = Field(None, description="Addresses")

    result = simplify_schema(PatientOptList, format_type="jsonish").to_string()
    assert result == "\n".join(
        [
            "//Title: PatientOptList",
            "// Fields marked with * are required",
            "{",
            "  name*: string // Patient name,",
            "  addresses: [{",
            "    street*: string // Street,",
            "    city*: string // City,",
            '    country*: string // one of: "US", "CA"',
            "  }] OR null  // Addresses (default=null)",
            "}",
        ]
    )

    result_p3 = simplify_schema(P3, format_type="jsonish").to_string()
    assert result_p3 == "\n".join(
        [
            "//Title: P3",
            "// Fields marked with * are required",
            "{",
            "  name*: string // Patient name,",
            "  addresses: [{",
            "    street*: string // Street,",
            "    city*: string // City",
            "  }] OR null  // Addresses (default=null)",
            "}",
        ]
    )


def test_jsonish_nested_required_ref_renders_as_block() -> None:
    """A direct required $ref field (design v2 A-7) renders as an expanded block."""
    from llm_schema_lite import simplify_schema

    class Country(str, Enum):
        US = "US"
        CA = "CA"

    class Address(BaseModel):
        street: str = Field(..., description="Street")
        city: str = Field(..., description="City")
        country: Country = Field(..., description="Country code")

    class Addr(BaseModel):
        street: str = Field(..., description="Street")
        city: str = Field(..., description="City")

    class PatientReq(BaseModel):
        name: str = Field(..., description="Patient name")
        address: Address = Field(..., description="Home address")

    class P4(BaseModel):
        name: str = Field(..., description="Patient name")
        address: Addr = Field(..., description="Home address")

    result = simplify_schema(PatientReq, format_type="jsonish").to_string()
    assert result == "\n".join(
        [
            "//Title: PatientReq",
            "// Fields marked with * are required",
            "{",
            "  name*: string // Patient name,",
            "  address*: {",
            "    street*: string // Street,",
            "    city*: string // City,",
            '    country*: string // one of: "US", "CA"',
            "  }",
            "}",
        ]
    )

    result_p4 = simplify_schema(P4, format_type="jsonish").to_string()
    assert result_p4 == "\n".join(
        [
            "//Title: P4",
            "// Fields marked with * are required",
            "{",
            "  name*: string // Patient name,",
            "  address*: {",
            "    street*: string // Street,",
            "    city*: string // City",
            "  }",
            "}",
        ]
    )


def test_jsonish_nested_ref_two_sibling_fields_both_inline() -> None:
    """AC3: two sibling fields sharing a $ref type both render inline, not just the first."""
    from llm_schema_lite import simplify_schema

    class Country(str, Enum):
        US = "US"
        CA = "CA"

    class Address(BaseModel):
        street: str = Field(..., description="Street")
        city: str = Field(..., description="City")
        country: Country = Field(..., description="Country code")

    class TwoSiblings(BaseModel):
        home: Address = Field(..., description="Home")
        work: Address = Field(..., description="Work")

    result = simplify_schema(TwoSiblings, format_type="jsonish").to_string()
    assert result == "\n".join(
        [
            "//Title: TwoSiblings",
            "// Fields marked with * are required",
            "{",
            "  home*: {",
            "    street*: string // Street,",
            "    city*: string // City,",
            '    country*: string // one of: "US", "CA"',
            "  },",
            "  work*: {",
            "    street*: string // Street,",
            "    city*: string // City,",
            '    country*: string // one of: "US", "CA"',
            "  }",
            "}",
        ]
    )


def test_jsonish_nested_ref_defs_docstring_becomes_opening_line_comment() -> None:
    """A $defs model docstring attaches as a comment on the block's opening line."""
    from llm_schema_lite import simplify_schema

    class AddressD(BaseModel):
        """An address."""

        street: str = Field(..., description="Street")
        city: str = Field(..., description="City")

    class WithDoc(BaseModel):
        name: str = Field(..., description="Patient name")
        address: AddressD | None = Field(None, description="Home address")

    class WithDocList(BaseModel):
        addresses: list[AddressD] = Field(..., description="Addresses")

    result = simplify_schema(WithDoc, format_type="jsonish").to_string()
    assert result == "\n".join(
        [
            "//Title: WithDoc",
            "// Fields marked with * are required",
            "{",
            "  name*: string // Patient name,",
            "  address: { // An address.",
            "    street*: string // Street,",
            "    city*: string // City",
            "  } OR null  // Home address (default=null)",
            "}",
        ]
    )

    result_list = simplify_schema(WithDocList, format_type="jsonish").to_string()
    assert result_list == "\n".join(
        [
            "//Title: WithDocList",
            "// Fields marked with * are required",
            "{",
            "  addresses*: [{ // An address.",
            "    street*: string // Street,",
            "    city*: string // City",
            "  }]  // Addresses",
            "}",
        ]
    )


def test_jsonish_nested_ref_deep_nesting_four_levels() -> None:
    """R2 / design v2 4.10: the fixpoint collapse loop pins four levels of nesting."""
    deep = {
        "type": "object",
        "properties": {"l1": {"$ref": "#/$defs/L1"}},
        "required": ["l1"],
        "$defs": {
            "L1": {
                "type": "object",
                "properties": {"l2": {"type": "array", "items": {"$ref": "#/$defs/L2"}}},
                "required": ["l2"],
            },
            "L2": {
                "type": "object",
                "properties": {"l3": {"$ref": "#/$defs/L3"}},
                "required": ["l3"],
            },
            "L3": {
                "type": "object",
                "properties": {"l4": {"type": "array", "items": {"$ref": "#/$defs/L4"}}},
                "required": ["l4"],
            },
            "L4": {
                "type": "object",
                "properties": {"v": {"type": "string"}},
                "required": ["v"],
            },
        },
    }
    result = JSONishFormatter(deep).transform_schema()
    assert result == "\n".join(
        [
            "// Fields marked with * are required",
            "{",
            "  l1*: {",
            "    l2*: [{",
            "      l3*: {",
            "        l4*: [{",
            "          v*: string",
            "        }]",
            "      }",
            "    }]",
            "  }",
            "}",
        ]
    )


def test_jsonish_empty_model_optional_renders_once() -> None:
    """R1: an empty nested model wrapped as optional renders once, not duplicated."""
    from llm_schema_lite import simplify_schema

    class Empty(BaseModel):
        pass

    class WithEmpty(BaseModel):
        empt: Empty | None = None

    result = simplify_schema(WithEmpty, format_type="jsonish").to_string()
    assert result == "\n".join(
        [
            "//Title: WithEmpty",
            "{",
            "  empt: {} OR null  // (default=null)",
            "}",
        ]
    )
    assert result.count("empt: {} OR null") == 1


def test_jsonish_formatter_with_deep_nesting():
    """Test JSONish formatter with deeply nested structures (3+ levels)."""
    from tests.conftest import DeepNested

    schema = DeepNested.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain all levels
    assert "id*:" in result
    assert "level_b*:" in result
    # Should handle nested object expansions
    assert "{" in result


def test_jsonish_formatter_with_union_heavy():
    """Test JSONish formatter with multiple union types."""
    from tests.conftest import UnionHeavy

    schema = UnionHeavy.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain all union fields
    assert "id*:" in result
    assert "value*:" in result
    assert "status*:" in result
    assert "data*:" in result
    # Should show union representation (OR keyword or anyOf)
    assert "OR" in result or "anyOf" in result.lower() or "|" in result


def test_jsonish_formatter_with_complex_types():
    """Test JSONish formatter with ComplexTypes model."""
    from tests.conftest import ComplexTypes

    schema = ComplexTypes.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain all primitive fields
    assert "string_field*:" in result
    assert "int_field*:" in result
    assert "float_field*:" in result
    assert "bool_field:" in result  # has default, might be optional
    # Should contain arrays
    assert "string_list:" in result
    assert "int_list*:" in result
    # Should contain nested objects
    assert "address:" in result
    # Should contain optional fields
    assert "optional_str:" in result


# ============================================================================
# Default Values and Metadata Tests
# ============================================================================


def test_jsonish_formatter_with_defaults():
    """Test JSONish formatter with default values."""
    from tests.conftest import ObjectWithDefaults

    schema = ObjectWithDefaults.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Fields with defaults should not be marked as required
    assert "name:" in result
    assert "name*:" not in result
    assert "count:" in result
    assert "count*:" not in result
    # Default values should appear in output when metadata is included
    assert "default" in result


def test_jsonish_formatter_with_field_descriptions():
    """Test JSONish formatter includes field descriptions when metadata enabled."""
    from tests.conftest import WithFieldDescriptions

    schema = WithFieldDescriptions.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Should contain field names
    assert "name*:" in result
    assert "email*:" in result
    assert "age*:" in result
    # Should contain descriptions as comments
    assert "full name" in result.lower() or "user's full name" in result.lower()


def test_jsonish_formatter_with_examples():
    """Test JSONish formatter with example values."""
    schema = {
        "type": "object",
        "properties": {
            "email": {"type": "string", "examples": ["user@example.com", "admin@example.com"]}
        },
        "required": ["email"],
    }
    # Enable examples in metadata_inclusion to test example display
    config = FormatterConfig(include_metadata=True, metadata_inclusion={"examples": True})
    formatter = JSONishFormatter(schema, config=config)
    result = formatter.transform_schema()

    # Should include examples when metadata is on and examples are enabled
    assert "email*:" in result
    assert "EXAMPLE" in result or "example" in result.lower()


# ============================================================================
# String Format and Pattern Tests
# ============================================================================


def test_jsonish_formatter_with_email_format():
    """Test JSONish formatter with email format constraint."""
    from tests.conftest import StringFormatEmail

    schema = StringFormatEmail.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Should contain email field
    assert "email*:" in result
    # Should include format info
    assert "email" in result.lower()


def test_jsonish_formatter_with_uri_format():
    """Test JSONish formatter with URI format constraint."""
    from tests.conftest import StringFormatUri

    schema = StringFormatUri.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Should contain website field
    assert "website*:" in result
    # Should include format or type info
    assert "string" in result


def test_jsonish_formatter_with_pattern():
    """Test JSONish formatter with pattern constraint."""
    from tests.conftest import StringPattern

    schema = StringPattern.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Should contain code field
    assert "code*:" in result
    # Should include pattern when metadata is on
    assert "PATTERN" in result or "pattern" in result.lower()


def test_jsonish_formatter_with_multiple_patterns():
    """Test JSONish formatter with multiple pattern constraints."""
    from tests.conftest import PatternConstraints

    schema = PatternConstraints.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Should contain all pattern-constrained fields
    assert "phone*:" in result
    assert "zip_code*:" in result
    assert "username*:" in result


# ============================================================================
# Numeric Constraint Tests
# ============================================================================


def test_jsonish_formatter_with_exclusive_min_max():
    """Test JSONish formatter with exclusive minimum/maximum (gt/lt).

    Note: Current implementation does not render exclusiveMinimum/exclusiveMaximum
    constraints (from Pydantic gt/lt). This test verifies the model renders without
    crashing. TODO: Consider adding support for exclusive constraints.
    """
    from tests.conftest import ExclusiveMinMax

    schema = ExclusiveMinMax.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain value and count fields
    assert "value*:" in result
    assert "count*:" in result
    # Fields should render as float and int types
    assert "float" in result
    assert "int" in result
    # Note: Exclusive constraints (exclusiveMinimum/exclusiveMaximum) are not
    # currently rendered in the output, only inclusive (minimum/maximum) are.


# ============================================================================
# Dict and Additional Properties Tests
# ============================================================================


def test_jsonish_formatter_with_dict_fields():
    """Test JSONish formatter with dict/mapping fields."""
    from tests.conftest import DictOnlyModel

    schema = DictOnlyModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain metadata and config fields
    assert "metadata*:" in result
    assert "config:" in result  # optional
    # Should show object or dict notation
    assert "object" in result or "{" in result


def test_jsonish_formatter_with_additional_props_false():
    """Test JSONish formatter with additionalProperties: false."""
    from tests.conftest import ObjectAdditionalPropsFalse

    schema = ObjectAdditionalPropsFalse.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain name and value fields
    assert "name*:" in result
    assert "value*:" in result
    # The formatter should explicitly show additionalProperties constraint
    assert "{" in result
    # Check that "no additional properties" constraint appears
    assert "no additional properties" in result or "//no additional properties" in result


def test_jsonish_formatter_array_of_objects_not_duplicated():
    """Regression: array items should render as JSONish, not Python str."""
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product_name": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "price": {"type": "number"},
                    },
                    "required": ["product_name", "quantity", "price"],
                },
            }
        },
        "required": ["items"],
    }
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert result.count("items*:") == 1
    assert "items*: [" in result
    assert "{" in result
    assert "[{'" not in result


def test_jsonish_formatter_additional_props_with_object_schema():
    """Regression: additionalProperties object should show structure details."""
    schema = {
        "type": "object",
        "properties": {"result": {"type": "object"}},
        "required": ["result"],
        "additionalProperties": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    }
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Root level has fixed properties, so it shows the comment-style additional properties
    assert "additional:" in result
    assert "value* (required): string" in result


def test_jsonish_formatter_empty_schema_renders_as_any():
    """Test that empty schema {} renders as 'any', not 'string'."""
    schema = {
        "type": "object",
        "properties": {
            "value": {}  # Empty schema
        },
    }
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert "value: any" in result


def test_jsonish_formatter_object_with_complex_additional_props_shows_placeholder():
    """Test that object with only complex additionalProperties shows placeholder key."""
    schema = {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {"value": {}},
                    "required": ["value"],
                },
            }
        },
    }
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should show placeholder key with structure
    assert "<string>" in result
    assert "value*: any" in result


def test_jsonish_formatter_simple_additional_props_still_work():
    """Test that simple additionalProperties still render as comments."""
    schema = {
        "type": "object",
        "properties": {"config": {"type": "object", "additionalProperties": {"type": "string"}}},
    }
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Simple additionalProperties now render structurally, not as a comment (AC-1).
    assert "<string>: string" in result
    assert "additional:" not in result


def test_jsonish_formatter_additional_props_false_still_works():
    """Test that additionalProperties: false still works correctly."""
    schema = {"type": "object", "additionalProperties": False}
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should show "no additional properties" comment
    assert "no additional properties" in result


def test_jsonish_formatter_root_additional_props_prefix_only_once():
    """Regression: root prefix should only apply to the first occurrence."""
    schema = {
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            }
        },
        "required": ["config"],
        "additionalProperties": False,
    }
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert result.count("Root:") <= 1


# ============================================================================
# Datetime and Special Type Tests
# ============================================================================


def test_jsonish_formatter_with_datetime():
    """Test JSONish formatter with datetime fields."""
    from tests.conftest import EventWithDate

    schema = EventWithDate.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain name and date fields
    assert "name*:" in result
    assert "event_date*:" in result
    assert "created_at:" in result  # has default
    # Should show datetime as string type
    assert "string" in result


# ============================================================================
# Composition and Inheritance Tests
# ============================================================================


def test_jsonish_formatter_with_composition():
    """Test JSONish formatter with model composition (allOf-like)."""
    from tests.conftest import AllOfLike

    schema = AllOfLike.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain fields from both base classes
    assert "field_a*:" in result
    assert "count_a*:" in result
    assert "field_b*:" in result
    assert "count_b*:" in result
    assert "own_field*:" in result


# ============================================================================
# Advanced Schema Feature Tests
# ============================================================================


def test_jsonish_formatter_with_anyof_union():
    """Test JSONish formatter with anyOf at property level."""
    from tests.conftest import UnionTypes

    schema = UnionTypes.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain id field which can be int or string
    assert "id*:" in result
    # Should show union/anyOf representation
    assert "OR" in result or "int" in result or "string" in result


def test_jsonish_formatter_with_top_level_anyof():
    """Test JSONish formatter with anyOf at schema root."""
    from tests.conftest import ANY_OF_SCHEMA

    formatter = JSONishFormatter(ANY_OF_SCHEMA, include_metadata=False)
    result = formatter.transform_schema()

    # Should handle top-level anyOf
    assert "id" in result
    # Should show OR representation
    assert "OR" in result or "anyOf" in result.lower()


def test_jsonish_formatter_with_top_level_oneof():
    """Test JSONish formatter with oneOf at schema root."""
    from tests.conftest import ONE_OF_SCHEMA

    formatter = JSONishFormatter(ONE_OF_SCHEMA, include_metadata=False)
    result = formatter.transform_schema()

    # Should handle top-level oneOf
    assert "type" in result
    # Should show exclusive choice representation
    assert "ONE OF" in result or "OR" in result


def test_jsonish_formatter_with_top_level_allof():
    """Test JSONish formatter with allOf at schema root."""
    from tests.conftest import ALL_OF_SCHEMA

    formatter = JSONishFormatter(ALL_OF_SCHEMA, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain merged fields from allOf branches
    assert "name" in result
    assert "age" in result


def test_jsonish_formatter_with_const():
    """Test JSONish formatter with const keyword."""
    from tests.conftest import CONST_SCHEMA

    formatter = JSONishFormatter(CONST_SCHEMA)
    result = formatter.transform_schema()

    # Should contain api_version field
    assert "api_version*:" in result
    # Should show const value
    assert "v1.0" in result


def test_jsonish_formatter_with_dependencies():
    """Test JSONish formatter with schema dependencies."""
    from tests.conftest import DEPENDENCY_SCHEMA

    formatter = JSONishFormatter(DEPENDENCY_SCHEMA, include_metadata=True)
    result = formatter.transform_schema()

    # Should contain all fields
    assert "name:" in result
    assert "credit_card:" in result
    assert "billing_address:" in result
    # Should show dependency information when metadata is on
    assert "DEPENDS" in result or result  # At minimum should not crash


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_jsonish_formatter_handles_empty_properties():
    """Edge: schema with properties: {} should handle gracefully."""
    schema = {"type": "object", "properties": {}}
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should render empty object
    assert "{" in result
    assert "}" in result


def test_jsonish_formatter_with_null_type():
    """Edge: field with type: null should handle gracefully."""
    schema = {"type": "object", "properties": {"nullable_field": {"type": "null"}}}
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain the field
    assert "nullable_field:" in result
    # Should show null type
    assert "null" in result


def test_jsonish_formatter_with_type_array():
    """Edge: field with type as array [string, null] should handle gracefully."""
    schema = {
        "type": "object",
        "properties": {"flexible": {"type": ["string", "null"]}},
        "required": ["flexible"],
    }
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain the field
    assert "flexible*:" in result
    # Should show string type (and possibly null)
    assert "string" in result


def test_jsonish_formatter_preserves_property_order():
    """Contract: property order should be preserved from schema dict order."""
    schema = {
        "type": "object",
        "properties": {
            "zulu": {"type": "string"},
            "alpha": {"type": "integer"},
            "mike": {"type": "boolean"},
        },
        "required": ["zulu", "alpha", "mike"],
    }
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Find positions of each field
    zulu_pos = result.find("zulu*:")
    alpha_pos = result.find("alpha*:")
    mike_pos = result.find("mike*:")

    # Verify order is preserved (zulu, alpha, mike)
    assert zulu_pos < alpha_pos < mike_pos, "Field order should be preserved from schema"


def test_jsonish_formatter_with_schema_title():
    """Test that schema-level title appears in output when metadata is on."""
    from tests.conftest import WithTitleDescription

    schema = WithTitleDescription.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Should contain schema title in comment
    assert "//Title:" in result or "User Profile" in result


def test_no_auto_generated_property_titles_jsonish():
    """AC-1: JSONish drops auto-generated per-field titles from the output."""
    for model in (Address, SimpleFormatterModel, ModelWithAlias):
        schema = model.model_json_schema()
        formatter = JSONishFormatter(schema, include_metadata=True)
        result = formatter.transform_schema()

        for key in schema.get("properties", {}):
            assert f"{formatter.comment_prefix} {auto_title_for_key(key)}:" not in result


def test_docstring_model_omits_title_header():
    """AC-3: a docstring-derived identifier-shaped root title is dropped, but a title
    with no description, or a user-supplied non-identifier title, survives."""

    class LocalNoDescriptionModel(BaseModel):
        name: str

    schema_with_docstring = SimpleFormatterModel.model_json_schema()
    formatter = JSONishFormatter(schema_with_docstring, include_metadata=True)
    result = formatter.transform_schema()
    assert "//Title: SimpleFormatterModel" not in result

    schema_no_description = LocalNoDescriptionModel.model_json_schema()
    formatter = JSONishFormatter(schema_no_description, include_metadata=True)
    result = formatter.transform_schema()
    assert "//Title: LocalNoDescriptionModel" in result

    schema_user_title = WithTitleDescription.model_json_schema()
    formatter = JSONishFormatter(schema_user_title, include_metadata=True)
    result = formatter.transform_schema()
    assert "//Title: User Profile" in result


def test_user_supplied_title_and_description_render_once_jsonish():
    """AC-2: a user-supplied field title and description each render exactly once."""
    schema = WithFieldDescriptions.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert result.count("Full Name") == 1
    assert result.count("The user's full name") == 1


def test_metadata_inclusion_title_false_suppresses_title_jsonish():
    """The ``title`` metadata_inclusion lever suppresses field titles, not descriptions."""
    schema = WithFieldDescriptions.model_json_schema()
    config = FormatterConfig(include_metadata=True, metadata_inclusion={"title": False})
    formatter = JSONishFormatter(schema, config=config)
    result = formatter.transform_schema()

    assert "Full Name" not in result
    assert "The user's full name" in result


def test_jsonish_formatter_with_full_featured_model():
    """Comprehensive test with FullFeaturedModel (kitchen sink)."""
    from tests.conftest import FullFeaturedModel

    schema = FullFeaturedModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain various field types
    assert "name*:" in result
    assert "age*:" in result
    assert "score*:" in result
    assert "role:" in result
    assert "identifier*:" in result
    # Should not crash with this complex model
    assert "{" in result
    assert "}" in result


def test_jsonish_formatter_consistent_asterisk_usage():
    """Regression: asterisks should only appear on required fields."""
    schema = {
        "type": "object",
        "properties": {
            "required_one": {"type": "string"},
            "required_two": {"type": "integer"},
            "optional_one": {"type": "string"},
            "optional_two": {"type": "boolean"},
        },
        "required": ["required_one", "required_two"],
    }
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Required fields should have asterisk
    assert "required_one*:" in result
    assert "required_two*:" in result
    # Optional fields should NOT have asterisk
    assert "optional_one:" in result
    assert "optional_one*:" not in result
    assert "optional_two:" in result
    assert "optional_two*:" not in result


# --- lsl-2026-09-04-002: stray trailing quote after comments -------------------------------
#
# A `"` in JSONish `_remove_quotes` output is BAD iff it is an unescaped JSON string
# delimiter that survived to end-of-line -- i.e. preceded by an even number of backslashes
# (possibly zero). This predicate expresses a property of `_remove_quotes` output, not of a
# whole rendered schema: header/notes/links comment lines and `_apply_pending_postfix` text
# never pass through `_remove_quotes` and may legitimately end in a bare `"`. It is therefore
# only applied here to fixtures whose model-level title/description/notes/links contain no
# `"`; field-level quotes are exercised directly via `_QuoteDescriptions` below.
#
# The deferred enum/const comment hoisted by lsl-2026-09-04-004 is in the same
# category: it is substituted after _remove_quotes (see hoist_deferred_comments), so
# a value list ending in a quoted string legitimately ends the line in a bare ".
# Lines carrying that comment are therefore skipped below.
STRAY_DELIMITER = re.compile(r'(?:^|[^\\])(?:\\\\)*"\s*,?\s*$')


class _QuoteDescriptions(BaseModel):
    """Pathological field descriptions exercising quote and comment edge cases.

    The docstring is deliberately quote-free (see STRAY_DELIMITER's scoping note above).
    """

    q: str = Field(..., description='say "hi" now')
    e: str = Field(..., description='ends with quote"')
    m: str = Field(..., description="see http://x // note")
    p: str = Field(..., description=r"win path C:\dir\ ")
    plain: str


def _line_with(rendered: str, needle: str) -> str:
    """Return the single rendered line containing `needle` (stripped)."""
    matches = [line.strip() for line in rendered.split("\n") if needle in line]
    assert len(matches) == 1, f"Expected exactly one line containing {needle!r}, got {matches!r}"
    return matches[0]


@pytest.mark.parametrize(
    "model",
    [
        WithFieldDescriptions,
        PersonWithAddress,
        Product,
        ComplexTypes,
        ModelWithPriorityMetadata,
    ],
)
def test_jsonish_no_line_ends_with_stray_delimiter(model: type[BaseModel]) -> None:
    """AC-1: no _remove_quotes output line ends with an unescaped string delimiter."""
    result = JSONishFormatter(model.model_json_schema()).transform_schema()

    for line in result.split("\n"):
        if "// one of: " in line:
            continue  # hoisted enum comment: exempt, see the scoping note above
        assert STRAY_DELIMITER.search(line) is None, f"Stray delimiter: {line!r}"


def test_jsonish_nested_object_lines_have_no_stray_delimiter() -> None:
    """AC-1 on the process_ref recursion.

    Nested lines live inside one physical line, so the per-line scan in
    test_jsonish_no_line_ends_with_stray_delimiter cannot see them.
    """
    result = JSONishFormatter(PersonWithAddress.model_json_schema()).transform_schema()

    for token in ['Street:"', 'Street:\\"', 'City:"', 'City:\\"', '\\",\\n']:
        assert token not in result, f"Stray-quote token {token!r} present in nested render."


def test_jsonish_description_with_quote_preserved_once() -> None:
    """AC-2: a description containing a literal quote is preserved exactly once."""
    result = JSONishFormatter(_QuoteDescriptions.model_json_schema()).transform_schema()
    line = _line_with(result, "q*:")

    assert line.count('"') == 2, f"Expected exactly 2 quote characters in {line!r}"
    assert "hi" in line
    assert STRAY_DELIMITER.search(line) is None


def test_jsonish_description_ending_in_quote_not_truncated() -> None:
    """AC-2: an escaped trailing content quote is not mistaken for a stray delimiter."""
    result = JSONishFormatter(_QuoteDescriptions.model_json_schema()).transform_schema()
    line = _line_with(result, "e*:")

    assert "ends with quote" in line
    assert line.count('"') == 1, f"Expected exactly 1 quote character in {line!r}"
    assert STRAY_DELIMITER.search(line) is None


def test_jsonish_description_with_double_slash_preserved() -> None:
    """Guard against reintroducing any `//`-position heuristic."""
    result = JSONishFormatter(_QuoteDescriptions.model_json_schema()).transform_schema()
    line = _line_with(result, "m*:")

    assert "see http://x // note" in line


def test_jsonish_property_without_comment_is_unchanged() -> None:
    """Pin the already-working clean path (a field with no description)."""
    result = JSONishFormatter(_QuoteDescriptions.model_json_schema()).transform_schema()
    line = _line_with(result, "plain*:")

    assert line.split("//")[0].strip() == "plain*: string"
    assert '"' not in line


@pytest.mark.parametrize(
    ("input_line", "expected_line"),
    [
        # S1 - key + string value, the reported bug
        ('  "name*": "string // Name: Full name",', "  name*: string // Name: Full name,"),
        # S2 - key + open object
        ('  "address*": {', "  address*: {"),
        # S2b - key + open array
        ('  "items*": [', "  items*: ["),
        # S3 - key + empty container
        ('  "metadata": {},', "  metadata: {},"),
        # S4 - close container (L1 structural skip)
        ("  },", "  },"),
        # S5 - injected comment-only line (L3 guard)
        ("  // Root: additional: string", "  // Root: additional: string"),
        # S6 - escaped quote inside the value
        ('  "q*": "string // Q: say \\"hi\\" now",', '  q*: string // Q: say \\"hi\\" now,'),
        # S7 - literal backslash immediately before a delimiter
        ('  "p*": "string // P: path\\\\",', "  p*: string // P: path\\\\,"),
        # S8 - value containing : and //
        (
            '  "m*": "string // M: see http://x // note",',
            "  m*: string // M: see http://x // note,",
        ),
        # S9 - bare string value, no trailing comma (last property)
        ('  "plain*": "string"', "  plain*: string"),
        # S10 - value ending in a content quote
        ('  "e*": "string // E: ends with quote\\""', '  e*: string // E: ends with quote\\"'),
        # S11 - nested dict-repr carrying \n (ticket 001's shape)
        ('  "address*": "{\\n street*: string \\n}"', "  address*: {\\n street*: string \\n}"),
    ],
)
def test_remove_quotes_line_shapes(input_line: str, expected_line: str) -> None:
    """Contract test pinning every line shape in the design's worked table.

    The only new test allowed to know _remove_quotes' internal line representation --
    it makes the method safe for tickets 001/003/004/005/014/015 to edit around.
    """
    formatter = JSONishFormatter({"type": "object", "properties": {}})

    assert formatter._remove_quotes(input_line) == expected_line


def test_remove_quotes_does_not_unescape() -> None:
    """Pin the hard rule: _remove_quotes must not unescape anything.

    Ticket lsl-2026-09-04-001's nested dict-repr `\\n` marker would otherwise become
    ambiguous with a user's literal `\\n` text.
    """
    formatter = JSONishFormatter({"type": "object", "properties": {}})

    result = formatter._remove_quotes('  "p*": "a\\\\b and \\"c\\" end",')

    assert result == '  p*: a\\\\b and \\"c\\" end,'


def test_collapse_array_object_brackets_handles_braces_in_descriptions() -> None:
    """R6: brace characters inside a description must not desync the bracket scan."""
    formatter = JSONishFormatter({"type": "object", "properties": {}})

    # (a) a description containing an unbalanced open brace
    src_open = "\n".join(
        ["{", "  xs*: [", "    {", "      k*: string // open { brace", "    }", "  ]", "}"]
    )
    out_open = "\n".join(["{", "  xs*: [{", "    k*: string // open { brace", "  }]", "}"])
    assert formatter._collapse_array_object_brackets(src_open) == out_open

    # (b) a description containing an unbalanced close brace
    src_close = "\n".join(
        ["{", "  xs*: [", "    {", "      k*: string // close } brace", "    }", "  ]", "}"]
    )
    out_close = "\n".join(["{", "  xs*: [{", "    k*: string // close } brace", "  }]", "}"])
    assert formatter._collapse_array_object_brackets(src_close) == out_close

    # (c) four-level nesting, proving the fixpoint loop collapses every level
    src_deep = "\n".join(
        [
            "{",
            "  l1*: {",
            "    l2*: [",
            "      {",
            "        l3*: {",
            "          l4*: [",
            "            {",
            "              v*: string",
            "            }",
            "          ]",
            "        }",
            "      }",
            "    ]",
            "  }",
            "}",
        ]
    )
    out_deep = "\n".join(
        [
            "{",
            "  l1*: {",
            "    l2*: [{",
            "      l3*: {",
            "        l4*: [{",
            "          v*: string",
            "        }]",
            "      }",
            "    }]",
            "  }",
            "}",
        ]
    )
    assert formatter._collapse_array_object_brackets(src_deep) == out_deep

    # Pin _delimiter_balance directly for the line shapes exercised above.
    assert formatter._delimiter_balance("  xs*: [") == 1
    assert formatter._delimiter_balance("    {") == 1
    assert formatter._delimiter_balance("      k*: string // open { brace") == 0
    assert formatter._delimiter_balance("  }],") == -2
    assert formatter._delimiter_balance("  empt: {},") == 0
    assert formatter._delimiter_balance("  a: string") == 0


def test_apply_pending_postfix_balanced_line_not_duplicated() -> None:
    """R7 / D5: a balanced `{}` line gets its pending postfix once, not duplicated."""
    formatter = JSONishFormatter({"type": "object", "properties": {}})
    formatter.pending_postfix = {"ao": "// note"}
    result = formatter._apply_pending_postfix("{\n  ao: {},\n  z: string\n}")

    assert result == "{\n  ao: {} // note,\n  z: string\n}"
    assert result.count("ao: {}") == 1


def test_raw_array_schema_no_empty_comment_marker() -> None:
    """Arrays with no title/description/range/default/example must not emit a bare `//`.

    ``JSONishFormatter.process_types`` used to assign ``comment`` unconditionally on the
    array branches, so a plain array (no metadata at all) rendered a trailing bare `//`
    or `//,` marker with nothing after it.
    """
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}},
            "empty": {"type": "array"},
        },
    }
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert "tags: string []" in result
    assert "empty: []" in result
    for line in result.splitlines():
        if "tags: string []" in line or line.strip().startswith("empty: []"):
            assert "//" not in line


def test_no_empty_comment_marker_across_all_models_jsonish(all_pydantic_models) -> None:
    """No rendered JSONish line should end in a bare `//` comment marker.

    Sweeps every registered Pydantic model. This test is expected to start GREEN at
    HEAD, because arrays on real Pydantic models always carry an auto-generated title
    today -- it becomes load-bearing once ticket 003's Phase 3 drops those auto titles,
    at which point a regression here would mean the JSONish array branches emitted a
    marker with no metadata behind it. Starting green is intentional (see plan Phase 2).
    """
    # A bare marker is `//` with nothing (or only a trailing comma) after it.
    bare_marker_pattern = re.compile(r"//\s*,?\s*$")

    for _name, model in all_pydantic_models:
        schema = model.model_json_schema()
        formatter = JSONishFormatter(schema, include_metadata=True)
        result = formatter.transform_schema()

        for line in result.splitlines():
            assert not bare_marker_pattern.search(
                line
            ), f"{_name}: bare comment marker in line: {line!r}"


def test_no_dict_repr_across_all_models_jsonish(all_pydantic_models) -> None:
    """No registered model may render a Python dict repr in JSONish output (AC1)."""
    for name, model in all_pydantic_models:
        result = JSONishFormatter(
            model.model_json_schema(), include_metadata=True
        ).transform_schema()
        assert "{'" not in result, f"{name}: python dict repr in JSONish output"


def test_order_jsonish_token_count_decreases() -> None:
    """Dropping auto-generated titles shrinks the JSONish token count for ``Order``.

    HEAD (pre-ticket-003) measured 482 tokens; the fix is expected to land at 423.
    The assertion is strict-lower-than-HEAD rather than pinned to the exact number so
    unrelated future formatting tweaks do not spuriously fail this test.
    """
    pytest.importorskip("tiktoken")
    from llm_schema_lite import simplify_schema

    assert simplify_schema(Order).token_count() < 482


# ============================================================================
# Container types (dict / tuple / set / Any) -- lsl-2026-09-04-015
# ============================================================================


# NOTE (Phase 5a golden deviation): the design v2 5.2 AFTER block was captured against a
# ``Root``/``Inner``/``Strict`` trio that carried **no docstrings**, so it shows ``//Title: Root``
# and bare ``inner*: {`` / ``strict*: {`` lines. The Phase 4 conftest fixture is specified *with*
# docstrings; pydantic turns each into ``description``, which makes ``normalize_schema_titles``
# drop the now-redundant root ``title`` and adds an inline ``//`` comment to the two nested
# models. Those three header/comment lines below therefore differ from the design block. They are
# a fixture artefact, not a container-rendering change: they render identically before and after
# this ticket. Every container token in this block is the design's verbatim.
ROOT_JSONISH_DEFAULT = "\n".join(
    [
        "// Kitchen-sink fixture for lsl-2026-09-04-015 "
        "(dict/tuple/set/Any container rendering).",
        "",
        "Fields verbatim from the approved design (2026-09-04-design-discussion-v2.md 5.1).",
        "// Fields marked with * are required",
        "{",
        "  extra*: {",
        "    <string>: int",
        "  },",
        "  dict_of_models*: {",
        "    <string>: {",
        "      a*: int,",
        "      b*: string",
        "    }",
        "  },",
        "  by_color*: {",
        "    <string>: int",
        "  },",
        "  pair*: [int, string],",
        "  var_tuple*: int [],",
        "  tags*: string [] (unique),",
        "  anything*: any,",
        "  described*: any // free form,",
        "  opt_any: any OR null // (default=null),",
        "  any_list*: any [],",
        "  opt_extra: {",
        "    <string>: int",
        "  } OR null  // (default=null),",
        "  inner*: { // Nested model exercising dict/tuple fields one level below Root.",
        "    d*: {",
        "      <string>: int",
        "    },",
        "    t*: [int, string]",
        "  },",
        "  strict*: { // Minimal `extra: forbid` model nested inside "
        "the container-types Root fixture.",
        "    s*: string,",
        "  //no additional properties",
        "  }",
        "}",
    ]
)


def test_jsonish_formatter_root_fixture_default() -> None:
    """Whole-string golden for the container-types Root fixture (design v2 5.2 AFTER)."""
    result = JSONishFormatter(Root.model_json_schema()).transform_schema()

    assert result == ROOT_JSONISH_DEFAULT


def test_jsonish_formatter_root_fixture_include_metadata_false() -> None:
    """Structure survives with metadata off; only the header/comment words disappear (R1)."""
    result = JSONishFormatter(Root.model_json_schema(), include_metadata=False).transform_schema()

    assert "tags*: string []" in result
    assert "opt_any: any OR null" in result
    assert "//Title:" not in result
    assert "Fields marked with" not in result


def test_jsonish_formatter_root_fixture_metadata_inclusion_variant() -> None:
    """Gating array constraints removes the words but never the container structure (R1)."""
    config = FormatterConfig(
        metadata_inclusion={"uniqueItems": False, "minItems": False, "maxItems": False}
    )
    result = JSONishFormatter(Root.model_json_schema(), config=config).transform_schema()

    assert "tags*: string []" in result
    assert "(unique)" not in result
    assert "[int, string]" in result


def test_jsonish_formatter_root_fixture_no_forbidden_substrings() -> None:
    """AC-1: no legacy container vocabulary survives anywhere in the output."""
    result = JSONishFormatter(Root.model_json_schema()).transform_schema()

    assert "additional:" not in result
    assert "2-2 items" not in result
    # PA-5: the two-word phrase only -- the bare word ``unique`` is the new token.
    assert "unique items" not in result
    # design 10 risk 2: the ``// Root:`` prefix must not migrate onto nested models.
    assert "Root:" not in result


class _AnyVariants(BaseModel):
    plain: Any
    described: Any = Field(..., description="free form")
    titled: Any = Field(..., title="Custom Title")


def test_jsonish_formatter_any_never_renders_as_string() -> None:
    """AC-2: an ``Any`` field never degrades to ``string``, however it is annotated."""
    result = JSONishFormatter(_AnyVariants.model_json_schema()).transform_schema()

    for field in ("plain*:", "described*:", "titled*:"):
        line = _line_with(result, field)
        assert "string" not in line, f"{field} rendered as string: {line!r}"


def test_jsonish_formatter_root_fixture_metadata_off_structure_survives() -> None:
    """R1: turning metadata off strips words, never mapping/tuple structure."""
    result = JSONishFormatter(Root.model_json_schema(), include_metadata=False).transform_schema()

    assert "<string>: int" in result
    assert "[int, string]" in result
    assert "unique" not in result.lower()


def test_jsonish_formatter_draft7_tuple_items_no_crash() -> None:
    """Crash pin: a draft-7 list-valued ``items`` no longer raises ``AttributeError``."""
    result = JSONishFormatter(ADDITIONAL_ITEMS_SCHEMA).transform_schema()

    assert "[string, int]" in result


def test_jsonish_formatter_prefix_items_tuple_with_variadic_tail() -> None:
    """``prefixItems`` plus a homogeneous ``items`` tail renders as a variadic tuple."""
    result = JSONishFormatter(PREFIX_ITEMS_SCHEMA).transform_schema()

    assert "[string, int, bool, ...string]" in result


# Recursive golden models: the placeholder embeds the class name, so these names
# must match the measured goldens exactly, and they must carry no docstring (a
# docstring becomes the model description and is rendered as a comment).
class ListNode(BaseModel):
    label: str
    kids: list[ListNode] = Field(default_factory=list)


ListNode.model_rebuild()


# A recursive ``$ref`` two array levels deep: the C4 key-less placeholder case.
class LLNode(BaseModel):
    name: str
    grid: list[list[LLNode]] = Field(default=[])


LLNode.model_rebuild()


# Recursive optional model exercising the recursion/postfix merge.
class Tree(BaseModel):
    value: str
    left: Tree | None = None
    right: Tree | None = None


Tree.model_rebuild()


_MAPPING_ANYOF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "m": {
            "type": "object",
            "additionalProperties": {"anyOf": [{"$ref": "#/$defs/N"}, {"type": "null"}]},
        }
    },
    "$defs": {
        "N": {
            "type": "object",
            "properties": {
                "m": {
                    "type": "object",
                    "additionalProperties": {"anyOf": [{"$ref": "#/$defs/N"}, {"type": "null"}]},
                }
            },
        }
    },
}


def test_jsonish_recursive_list_golden_default_depth() -> None:
    """AC2: the default config renders two expansions then a recursion placeholder."""
    result = simplify_schema(ListNode, format_type="jsonish").to_string()

    assert result == (
        "//Title: ListNode\n"
        "// Fields marked with * are required\n"
        "{\n"
        "  label*: string,\n"
        "  kids: [{\n"
        "    label*: string,\n"
        "    kids: object [] // recursive: ListNode\n"
        "  }]\n"
        "}"
    )


def test_jsonish_placeholder_does_not_swallow_array_brackets() -> None:
    """C4: a key-less placeholder uses the block form so ``[] []`` survives it."""
    result = JSONishFormatter(
        LLNode.model_json_schema(), config=FormatterConfig(max_recursion_depth=1)
    ).transform_schema()

    line = _line_with(result, "grid:")
    assert line == "grid: object /* recursive: LLNode */ [] [] // (default=[])"

    head, sep, tail = line.partition("*/")
    assert sep == "*/", f"key-less placeholder must use the block form, got {line!r}"
    assert "[]" not in head, f"comment swallowed the array brackets: {line!r}"
    assert "default" not in head, f"comment swallowed the default note: {line!r}"
    assert tail.strip() == "[] [] // (default=[])"


def test_jsonish_placeholder_does_not_swallow_or_null() -> None:
    """C4: ``OR null`` after a key-less placeholder stays outside the comment."""
    result = JSONishFormatter(
        _MAPPING_ANYOF_SCHEMA, config=FormatterConfig(max_recursion_depth=1)
    ).transform_schema()

    line = _line_with(result, "recursive: N")
    assert line == "<string>: object /* recursive: N */ OR null"

    head, sep, tail = line.partition("*/")
    assert sep == "*/", f"key-less placeholder must use the block form, got {line!r}"
    assert "OR null" not in head, f"comment swallowed the union tail: {line!r}"
    assert tail.strip() == "OR null"


def test_jsonish_recursive_optional_postfix_merge() -> None:
    """A field that already has a postfix gains one merged comment, not two."""
    result = simplify_schema(
        Tree, config=FormatterConfig(max_recursion_depth=1), format_type="jsonish"
    ).to_string()

    expected = "object OR null // (default=null), recursive: Tree"
    left_line = _line_with(result, "left:").rstrip(",")
    right_line = _line_with(result, "right:").rstrip(",")

    assert left_line == f"left: {expected}"
    assert right_line == f"right: {expected}"
    assert left_line.count("//") == 1, f"duplicated comment marker: {left_line!r}"
    assert left_line.count("OR null") == 1, f"duplicated union tail: {left_line!r}"


def test_jsonish_formatter_does_not_mutate_caller_config() -> None:
    """JSONishFormatter must not poison a `FormatterConfig` a sibling format reuses.

    Mirrors `test_yaml_formatter.py::test_yaml_formatter_does_not_mutate_caller_config`.
    HEAD wrote `config.union_separator` in place inside the constructor's `elif` branch,
    so a `TypeScriptFormatter` handed the same object afterwards emitted
    `number OR string` instead of `number | string`.
    """
    from llm_schema_lite.formatters.typescript_formatter import TypeScriptFormatter

    # AC 1: the caller's config survives a JSONish render structurally unchanged.
    config = FormatterConfig()
    assert config.union_separator == " | "

    JSONishFormatter(UnionTypes.model_json_schema(), config=config).transform_schema()

    assert config == FormatterConfig(), "JSONishFormatter mutated the caller's config"

    # AC 2: the same object still renders TypeScript with the package default separator.
    ts_result = TypeScriptFormatter(
        UnionTypes.model_json_schema(), config=config
    ).transform_schema()

    assert "number | string" in ts_result
    assert "number OR string" not in ts_result

    # Ticket risk: an explicitly chosen separator is honoured and reads back unchanged.
    # Assert BY VALUE ONLY. `with_format_default_separator` returns this very object in
    # the passthrough branch (behaviour table row 3), so `is not` would be wrong here.
    custom_config = FormatterConfig(union_separator=" or ")
    jsonish_result = JSONishFormatter(
        UnionTypes.model_json_schema(), config=custom_config
    ).transform_schema()

    assert "id*: int or string" in jsonish_result
    assert custom_config.union_separator == " or "


# ============================================================================
# lsl-2026-09-05-001 — per-occurrence identity for pending comments
# All fixture models below are LOCAL to this module by design (design v2 AS6).
# tests/conftest.py is deliberately NOT edited.
# ============================================================================


class _CollisionLeaf(BaseModel):
    x: int


class _CollisionInner(BaseModel):
    kids: list[_CollisionLeaf] = Field(..., description="Inner kids", max_length=3)


class _CollisionOuter(BaseModel):
    inner: _CollisionInner
    kids: list[_CollisionLeaf] = Field(..., description="Outer kids", max_length=9)


def test_jsonish_sibling_field_collision_keeps_own_metadata() -> None:
    """AC1: a nested field keeps its own description/constraint, not the outer field's."""
    result = simplify_schema(_CollisionOuter, format_type="jsonish").to_string()

    assert result.count("Inner kids (<= 3 items)") == 1
    assert result.count("Outer kids (<= 9 items)") == 1

    lines = result.split("\n")
    inner_line = next(line for line in lines if "Inner kids" in line)
    outer_line = next(line for line in lines if "Outer kids" in line)
    # The nested closer is indented one level deeper than the root closer.
    assert inner_line == "    }]  // Inner kids (<= 3 items)"
    assert outer_line == "  }]  // Outer kids (<= 9 items)"


class _RecNode(BaseModel):
    name: str
    children: list[_RecNode] = []


_RecNode.model_rebuild()


def test_jsonish_recursive_marker_on_truncated_level_only() -> None:
    """AC2: ``recursive: T`` marks only the truncated level, matching YAML/TypeScript."""
    result = simplify_schema(_RecNode, format_type="jsonish").to_string()

    assert result.count("recursive: _RecNode") == 1
    assert _line_with(result, "recursive: _RecNode") == (
        "children: object [] // (default=[]), recursive: _RecNode"
    )

    closers = [line for line in result.split("\n") if line.strip().startswith("}]")]
    assert len(closers) == 1
    assert closers[0].strip() == "}]  // (default=[])"


class _G1(BaseModel):
    kids: list[_CollisionLeaf] = Field(..., description="G1 kids", max_length=3)


class _G2(BaseModel):
    kids: list[_CollisionLeaf] = Field(..., description="G2 kids", max_length=5)


class _SiblingCollision(BaseModel):
    a: _G1
    b: _G2


def test_jsonish_same_depth_siblings_do_not_collide() -> None:
    """Two same-depth blocks each declaring ``kids`` keep their own metadata."""
    result = simplify_schema(_SiblingCollision, format_type="jsonish").to_string()

    assert result.count("G1 kids (<= 3 items)") == 1
    assert result.count("G2 kids (<= 5 items)") == 1
    assert _line_with(result, "G1 kids") == "}]  // G1 kids (<= 3 items)"
    assert _line_with(result, "G2 kids") == "}]  // G2 kids (<= 5 items)"


class _H1(BaseModel):
    kids: list[_CollisionLeaf] | None = Field(default=None, description="H1 kids optional")


class _HOuter(BaseModel):
    inner: _H1
    kids: list[_CollisionLeaf] = Field(..., description="HOuter kids required", max_length=9)


def test_jsonish_required_optional_shadowing_fixed() -> None:
    """An optional nested ``kids`` no longer shadows the required root ``kids*``."""
    result = simplify_schema(_HOuter, format_type="jsonish").to_string()

    assert result.count("H1 kids optional") == 1
    assert result.count("HOuter kids required (<= 9 items)") == 1
    assert _line_with(result, "H1 kids optional") == (
        "}] OR null  // H1 kids optional (default=null)"
    )
    assert _line_with(result, "HOuter kids required") == (
        "}]  // HOuter kids required (<= 9 items)"
    )


class _AddrA(BaseModel):
    """Address flavour A."""

    line1: str


class _AddrB(BaseModel):
    """Address flavour B."""

    line1: str


class _InnerP(BaseModel):
    addr: _AddrA | None = None


class _OuterP(BaseModel):
    inner: _InnerP
    addr: _AddrB | None = None


def test_jsonish_pending_prefix_collision_fixed() -> None:
    """A nested block opener keeps its own ``$defs`` docstring, not the root's."""
    result = simplify_schema(_OuterP, format_type="jsonish").to_string()

    assert result.count("// Address flavour A.") == 1
    assert result.count("// Address flavour B.") == 1
    assert _line_with(result, "Address flavour A.") == "addr: { // Address flavour A."
    assert _line_with(result, "Address flavour B.") == "addr: { // Address flavour B."


class _IdInner(BaseModel):
    """Id inner doc."""

    v: str


class _IdOuter(BaseModel):
    id: _IdInner
    identifier: str = Field(..., description="the identifier")


def test_jsonish_prefix_key_no_false_positive_on_substring_name() -> None:
    """Deleting the ``*:`` alternative must not loosen the ``:`` requirement."""
    result = simplify_schema(_IdOuter, format_type="jsonish").to_string()

    assert _line_with(result, "Id inner doc.") == "id*: { // Id inner doc."
    assert _line_with(result, "the identifier") == "identifier*: string // the identifier"


class _Same(BaseModel):
    """Shared def docstring."""

    kids: list[int] = Field(..., description="Same kids", max_length=3)


class _TwoOcc(BaseModel):
    p: _Same
    q: _Same


def test_jsonish_shared_ref_cache_both_occurrences_correct() -> None:
    """One def reached twice: BOTH occurrences keep the comment (not a residual collision)."""
    result = simplify_schema(_TwoOcc, format_type="jsonish").to_string()

    assert result.count("// Shared def docstring.") == 2
    assert result.count("// Same kids") == 2
    assert result.count("int [] (<= 3 items)") == 2


class _J1(BaseModel):
    kids: list[_CollisionLeaf] = Field(..., description="J1 kids", max_length=3)


class _JOuter(BaseModel):
    m: dict[str, _J1]
    kids: list[_CollisionLeaf] = Field(..., description="JOuter kids", max_length=7)


def test_jsonish_mapping_value_collision_fixed() -> None:
    """A ``dict[str, Model]`` value block keeps its own field metadata."""
    result = simplify_schema(_JOuter, format_type="jsonish").to_string()

    assert result.count("J1 kids (<= 3 items)") == 1
    assert result.count("JOuter kids (<= 7 items)") == 1
    assert _line_with(result, "J1 kids") == "}]  // J1 kids (<= 3 items)"
    assert _line_with(result, "JOuter kids") == "}]  // JOuter kids (<= 7 items)"


class _TreeFixture(BaseModel):
    value: int
    left: _TreeFixture | None = None
    right: _TreeFixture | None = None


_TreeFixture.model_rebuild()


def test_jsonish_multipath_recursion_marks_only_truncated_levels() -> None:
    """A type reached by two paths marks only its four truncated level-1 lines."""
    result = simplify_schema(_TreeFixture, format_type="jsonish").to_string()

    assert result.count("recursive: _TreeFixture") == 4

    level0_closers = [line for line in result.split("\n") if line.strip().startswith("} OR null")]
    assert len(level0_closers) == 2
    for closer in level0_closers:
        assert "recursive:" not in closer
        assert "(default=null)" in closer


def test_jsonish_additional_properties_sentinel_name_renders_correctly() -> None:
    """R6: a real property named like the sentinel is no longer swallowed by it."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "__additional_properties__": {"type": "string"},
            "ok": {"type": "string"},
        },
        "required": ["ok"],
    }
    result = JSONishFormatter(schema).transform_schema()

    assert "__additional_properties__: string" in result
    assert result.count("string") == 2


def test_jsonish_pending_maps_fully_applied_and_no_token_leak_corpus(
    all_pydantic_models: list[tuple[str, type[BaseModel]]],
) -> None:
    """Corpus invariant: no identity token escapes, and no pending comment is orphaned.

    This is the standing replacement for a runtime warning. It is what would have caught
    the pending-map defects this change fixes.
    """
    leaks: list[str] = []
    orphans: list[str] = []

    for name, model in all_pydantic_models:
        for depth in (1, 2, 3):
            formatter = JSONishFormatter(
                model.model_json_schema(),
                config=FormatterConfig(max_recursion_depth=depth),
            )
            rendered = formatter.transform_schema()

            if DEFERRED_OPEN in rendered or DEFERRED_CLOSE in rendered:
                leaks.append(f"{name}@depth{depth}")

            for map_name in ("pending_postfix", "pending_prefix"):
                for key, value in getattr(formatter, map_name).items():
                    body = value.removeprefix("//").strip()
                    if body and body not in rendered:
                        orphans.append(f"{name}@depth{depth} {map_name}[{key!r}] -> {body!r}")

    assert leaks == [], f"identity/deferred tokens leaked into output: {leaks}"
    assert orphans == [], f"pending entries never applied to any line: {orphans}"


class Item(BaseModel):
    name: str = Field(description="Item name")
    qty: int = Field(default=1, description="Quantity")


class Node(BaseModel):
    """A tree node."""

    value: str = Field(description="Node value")
    children: list[Node] = Field(default_factory=list, description="Child nodes")


Node.model_rebuild()


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (
            list[Item],
            "[{\n  name*: string // Item name,\n  qty: int // Quantity (default=1)\n}]",
        ),
        (
            Item | None,
            "{\n  name*: string // Item name,\n  qty: int // Quantity (default=1)\n} OR null",
        ),
        (
            list[Item] | None,
            "[{\n  name*: string // Item name,\n  qty: int // Quantity (default=1)\n}] OR null",
        ),
        (
            Node,
            "//Title: Node\n"
            "// A tree node.\n"
            "// Fields marked with * are required\n"
            "{\n"
            "  value*: string // Node value,\n"
            "  children: [{ // A tree node.\n"
            "    value*: string // Node value,\n"
            "    children: object [] // Child nodes, recursive: Node\n"
            "  }]  // Child nodes\n"
            "}",
        ),
        (list[list[int]], "int [] []"),
        (tuple[int, str], "[int, string]"),
        (
            dict[str, Item],
            "{\n"
            "  <string>: {\n"
            "    name*: string // Item name,\n"
            "    qty: int // Quantity (default=1)\n"
            "  }\n"
            "}",
        ),
        (
            Item | str | None,
            "{\n  name*: string // Item name,\n  qty: int // Quantity (default=1)\n}"
            " OR string OR null",
        ),
    ],
)
def test_root_shape_golden(annotation: Any, expected: str) -> None:
    """Whole-string golden for a root-level shape, byte-for-byte against a measured run."""
    result = simplify_schema(
        TypeAdapter(annotation).json_schema(), format_type="jsonish"
    ).to_string()

    assert result == expected


@pytest.mark.parametrize(
    "annotation",
    [
        list[Item],
        Item | None,
        list[Item] | None,
        dict[str, Item],
        Node,
    ],
)
def test_root_render_matches_nested_rendering(annotation: Any) -> None:
    """Root-parity companion (decision A-2): every substantive root line also appears nested.

    Part (a) -- whole-string equality against the pinned golden -- lives in
    ``test_root_shape_golden``. This is part (b): every non-comment, non-blank line of the
    root render, whitespace-stripped and with a trailing ``,`` stripped, occurs as a substring
    of the render of the same shape embedded as a nested property ``f``.
    """
    root_render = simplify_schema(
        TypeAdapter(annotation).json_schema(), format_type="jsonish"
    ).to_string()

    root_schema = TypeAdapter(annotation).json_schema()
    defs = root_schema.pop("$defs", None)
    nested = {"type": "object", "properties": {"f": root_schema}}
    if defs is not None:
        nested["$defs"] = defs
    nested_render = simplify_schema(nested, format_type="jsonish").to_string()

    for line in root_render.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        normalized = stripped.rstrip(",")
        assert normalized in nested_render, f"{normalized!r} not found in nested render"


def test_list_of_optional_differs_from_optional_list() -> None:
    """AC: ``list[Item | None]`` and ``list[Item] | None`` must not collapse to one rendering.

    Reference values (design v2 §4.9):
      - ``list[Item | None]`` ->
        ``"[{\\n  name*: string // Item name,\\n  qty: int // Quantity (default=1)\\n}]"``
        -- JSONish still drops the *inner* ``OR null`` here, a pre-existing nested-inside-array
        gap (design v2 FU-5), out of scope for this ticket.
      - ``list[Item] | None`` ->
        ``"[{\\n  name*: string // Item name,\\n  qty: int // Quantity (default=1)\\n}]"
        " OR null"``
    """
    render_a = simplify_schema(
        TypeAdapter(list[Item | None]).json_schema(), format_type="jsonish"
    ).to_string()
    render_b = simplify_schema(
        TypeAdapter(list[Item] | None).json_schema(), format_type="jsonish"
    ).to_string()

    assert render_a != render_b


def test_every_titled_root_is_decorated(
    all_pydantic_models: list[tuple[str, type[BaseModel]]],
) -> None:
    """The invariant that makes root cause #4 unrepeatable -- every root return path
    reaches ``root_decorations()``.
    """
    from llm_schema_lite.schema_normalization import normalize_schema_titles

    exercised = 0
    for name, model in all_pydantic_models:
        # Effective title, i.e. after the pre-existing auto-title-stripping normalization
        # every formatter applies -- see `formatter_helpers.assert_schema_info_comment_presence`
        # for the same computation. `model_json_schema()` always sets a raw root `title`
        # (Pydantic defaults it to the class name), so a *raw* truthy-title filter would
        # select every model in the corpus, including ones whose title is legitimately
        # dropped before rendering by unrelated, already-landed behaviour.
        schema = normalize_schema_titles(model.model_json_schema())
        title = schema.get("title")
        if not schema.get("properties") or not title:
            continue
        exercised += 1
        result = simplify_schema(model, format_type="jsonish").to_string()
        assert result.startswith(
            f"//Title: {title}"
        ), f"{name}: expected render to start with '//Title: {title}', got {result[:80]!r}"

    assert exercised > 0, "guard exercised zero models -- fixture or filter is broken"


@pytest.mark.xfail(
    strict=True,
    reason="FU-2: the render path mutates the input schema and pops `title` before "
    "get_schema_info_comment reads it. Pre-existing, orthogonal, out of scope.",
)
def test_root_array_title_is_rendered() -> None:
    schema = {
        "type": "array",
        "title": "Bag",
        "description": "A bag of items.",
        "items": {"type": "string"},
    }
    result = simplify_schema(schema, format_type="jsonish").to_string()
    assert result.startswith("//Title: Bag")
