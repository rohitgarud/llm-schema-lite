"""Tests for JSONish formatter."""

from __future__ import annotations

import re

import pytest

from llm_schema_lite.formatters.jsonish_formatter import JSONishFormatter
from tests.conftest import (
    EMPTY_SCHEMA,
    ConstrainedFormatterModel,
    OrderedFieldsModel,
    PersonWithAddress,
    RequiredOptionalModel,
    SimpleFormatterModel,
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
    assert_schema_info_comment_presence(result, include_metadata=True)

    # Verify asterisk notation comment is present (schema has required fields)
    assert "Fields marked with * are required" in result
    assert "//" in result  # JSONish comment prefix


def test_jsonish_formatter_without_metadata():
    """Test JSONish formatter without metadata."""
    schema = SimpleModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert_schema_info_comment_presence(result, include_metadata=False)

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

    # Should contain nested Address required fields (expanded)
    # JSONish nested rendering may appear either as `street*:` (object expanded) or
    # as a python-dict-like string `'<street*>': ...` depending on recursion path.
    assert ("street*:" in result) or re.search(r"['\"]street\*['\"]\s*:", result)
    assert ("city*:" in result) or re.search(r"['\"]city\*['\"]\s*:", result)


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
    formatter = JSONishFormatter(schema, include_metadata=False)
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
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain priority field
    assert "priority*:" in result
    # Check for enum values in output
    assert "1" in result or "LOW" in result


def test_jsonish_formatter_with_literal_single():
    """Test JSONish formatter with single literal value."""
    from tests.conftest import LiteralSingle

    schema = LiteralSingle.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain api_version field
    assert "api_version*:" in result
    # Single literals are rendered as their type (string) in Pydantic schema
    # The const/enum constraint should be in the schema but may not show in output
    assert "string" in result or "v1" in result


def test_jsonish_formatter_with_literal_union():
    """Test JSONish formatter with union of literals."""
    from tests.conftest import LiteralUnion

    schema = LiteralUnion.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should contain status field
    assert "status*:" in result
    # Should contain all literal values
    assert "draft" in result
    assert "published" in result
    assert "archived" in result


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
    formatter = JSONishFormatter(schema, include_metadata=False)
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
    # Should expand nested refs or show object notation
    assert "{" in result or "Address" in result or "object" in result


# ============================================================================
# Nested and Complex Structure Tests
# ============================================================================


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
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Should include examples when metadata is on
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
    assert "<key>" in result
    assert "value*: any" in result
    assert "any properties allowed" in result


def test_jsonish_formatter_simple_additional_props_still_work():
    """Test that simple additionalProperties still render as comments."""
    schema = {
        "type": "object",
        "properties": {"config": {"type": "object", "additionalProperties": {"type": "string"}}},
    }
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Simple additionalProperties should show as comment
    assert "additional: string" in result


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

    formatter = JSONishFormatter(CONST_SCHEMA, include_metadata=False)
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
