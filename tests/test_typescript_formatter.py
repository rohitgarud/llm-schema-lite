"""Tests for TypeScript formatter."""

from __future__ import annotations

import re

import pytest

from llm_schema_lite.formatters.typescript_formatter import TypeScriptFormatter
from tests.conftest import (
    ALL_OF_SCHEMA,
    ANY_OF_SCHEMA,
    CONST_SCHEMA,
    DEPENDENCY_SCHEMA,
    EMPTY_SCHEMA,
    ONE_OF_SCHEMA,
    AllOfLike,
    ArrayMinMaxItems,
    ArrayOfRefsModel,
    ArrayOfStrings,
    ArrayUniqueItems,
    ConstrainedFormatterModel,
    DeepNested,
    DictOnlyModel,
    EventWithDate,
    ExclusiveMinMax,
    FullFeaturedModel,
    IntEnumModel,
    LiteralSingle,
    LiteralUnion,
    ObjectAdditionalPropsFalse,
    ObjectRequiredOnly,
    ObjectWithDefaults,
    OrderedFieldsModel,
    PatternConstraints,
    PersonWithAddress,
    RequiredOptionalModel,
    Role,
    SimpleFormatterModel,
    StringFormatEmail,
    StringFormatUri,
    StringPattern,
    UnionHeavy,
    UnionTypes,
    WithFieldDescriptions,
    WithTitleDescription,
)
from tests.formatter_helpers import (
    assert_required_optional_consistent,
    assert_required_optional_fields_match_schema,
    assert_schema_title_comment_consistent,
    parse_typescript_interface_fields,
)

# Alias for backwards compatibility in tests
SimpleModel = SimpleFormatterModel


def test_typescript_formatter_produces_valid_output():
    """Test that TypeScript formatter produces valid output with required fields marked."""
    schema = SimpleModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Verify interface structure
    assert "interface Schema {" in result
    fields = parse_typescript_interface_fields(result, interface_name="Schema")
    assert_required_optional_fields_match_schema(fields, schema)
    assert_required_optional_consistent(result, schema)
    assert_schema_title_comment_consistent(
        result, schema, include_metadata=True, comment_prefix=formatter.comment_prefix
    )

    # Verify TypeScript types
    assert "string" in result
    assert "number" in result
    # Verify comment notation
    assert "//" in result
    assert "Fields marked with * are required" in result


def test_typescript_formatter_without_metadata():
    """Test TypeScript formatter without metadata."""
    schema = SimpleModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert "interface Schema {" in result
    fields = parse_typescript_interface_fields(result, interface_name="Schema")
    assert_required_optional_fields_match_schema(fields, schema)
    assert_required_optional_consistent(result, schema)
    assert_schema_title_comment_consistent(
        result, schema, include_metadata=False, comment_prefix=formatter.comment_prefix
    )

    # Description metadata should not appear as comment
    assert "Name field" not in result


def test_typescript_formatter_with_nested_defs():
    """Test TypeScript formatter with nested $defs."""
    schema = PersonWithAddress.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert "interface Address {" in result, "Expected nested Address interface to be emitted"

    # Address interface fields should match the nested $defs schema.
    defs = schema.get("$defs", schema.get("definitions", {})) or {}
    address_schema = defs.get("Address")
    assert isinstance(address_schema, dict), "Expected $defs.Address schema"
    address_fields = parse_typescript_interface_fields(result, interface_name="Address")
    assert_required_optional_fields_match_schema(address_fields, address_schema)

    # Main interface fields should match the root schema.
    main_fields = parse_typescript_interface_fields(result, interface_name="Schema")
    assert_required_optional_fields_match_schema(main_fields, schema)
    assert_required_optional_consistent(result, schema)


def test_typescript_formatter_key_order_preserved():
    """Test that TypeScript formatter preserves key order (dict order)."""
    schema = OrderedFieldsModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    fields = parse_typescript_interface_fields(result, interface_name="Schema")
    names = [n for n, _ in fields]
    assert names == ["first", "second", "third"], f"Unexpected field order: {names}"


def test_typescript_formatter_caching():
    """Test that formatter caching works correctly."""
    schema = SimpleModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)

    result1 = formatter.transform_schema()
    result2 = formatter.transform_schema()

    assert result1 == result2
    assert hasattr(formatter, "_processed_data")
    assert formatter._processed_data is not None


def test_typescript_formatter_with_constraints():
    """Test that TypeScript formatter includes field constraints when metadata enabled."""
    schema = ConstrainedFormatterModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # ConstrainedFormatterModel has name, age, score, tags
    fields = parse_typescript_interface_fields(result, interface_name="Schema")
    assert_required_optional_fields_match_schema(fields, schema)
    assert_required_optional_consistent(result, schema)

    # Constraints appear inline in the type when include_metadata=True
    assert "name*: string (1-100 chars)" in result
    assert "age*: number (0-150)" in result
    assert re.search(
        r"score\*:\s*number\s*\(0(\.0)?-100(\.0)?\)", result
    ), f"Expected score range constraint in output. Snippet: {result[:350]!r}"


def test_typescript_formatter_with_optional_union():
    """Test that TypeScript formatter handles optional fields (union with None)."""
    schema = RequiredOptionalModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    fields = parse_typescript_interface_fields(result, interface_name="Schema")
    assert_required_optional_fields_match_schema(fields, schema)
    assert_required_optional_consistent(result, schema)

    assert "required_field*:" in result
    assert "optional_field:" in result
    assert "optional_field*:" not in result


@pytest.mark.parametrize(
    "model_cls",
    [
        SimpleFormatterModel,
        RequiredOptionalModel,
        OrderedFieldsModel,
        ConstrainedFormatterModel,
    ],
)
def test_typescript_contract_root_fields_match_schema(model_cls):
    """Contract: root interface fields match schema required/properties."""
    schema = model_cls.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    fields = parse_typescript_interface_fields(result, interface_name="Schema")
    assert_required_optional_fields_match_schema(fields, schema)


# ============================================================================
# Parsing and Contract Tests
# ============================================================================


def test_typescript_required_optional_parsing_matches_schema():
    """Test that parsed TypeScript interface fields match schema required/properties."""
    schema = RequiredOptionalModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    fields = parse_typescript_interface_fields(result, interface_name="Schema")
    assert_required_optional_fields_match_schema(fields, schema)


@pytest.mark.parametrize(
    "model_cls",
    [
        SimpleFormatterModel,
        RequiredOptionalModel,
        OrderedFieldsModel,
        ConstrainedFormatterModel,
    ],
)
def test_typescript_lists_all_root_properties_for_models(model_cls):
    """Contract: TypeScript output lists all root properties from schema."""
    schema = model_cls.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    fields = parse_typescript_interface_fields(result, interface_name="Schema")
    field_names = {name for name, _ in fields}
    properties_in_schema = set((schema.get("properties", {}) or {}).keys())

    missing_props = properties_in_schema - field_names
    assert not missing_props, f"Missing properties in output: {sorted(missing_props)}"


@pytest.mark.parametrize(
    "model_cls",
    [
        SimpleFormatterModel,
        RequiredOptionalModel,
        OrderedFieldsModel,
        ConstrainedFormatterModel,
    ],
)
def test_typescript_required_optional_consistent_for_models(model_cls):
    """Contract: required fields have '*' and optional do not."""
    schema = model_cls.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)


# ============================================================================
# Edge Cases and Scaffolding
# ============================================================================


def test_typescript_empty_object_schema():
    """Test TypeScript formatter with empty object schema."""
    schema = EMPTY_SCHEMA
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should produce valid TypeScript interface
    assert "interface Schema {" in result
    assert "}" in result


def test_typescript_single_required_field():
    """Test TypeScript formatter with single required field."""
    schema = ObjectRequiredOnly.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # All fields in ObjectRequiredOnly are required
    assert "name*:" in result
    assert "age*:" in result


def test_typescript_all_optional_no_asterisks():
    """Test TypeScript formatter with all optional fields has no asterisks."""
    schema = ObjectWithDefaults.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # ObjectWithDefaults has defaults and no required fields
    required = set(schema.get("required", []))
    if not required:
        # No asterisks should appear in the field names
        fields = parse_typescript_interface_fields(result, interface_name="Schema")
        for name, is_req in fields:
            assert not is_req, f"Field '{name}' should not be marked as required"


def test_typescript_empty_properties():
    """Test TypeScript formatter with schema that has empty properties dict."""
    schema = {"type": "object", "properties": {}}
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should produce valid interface
    assert "interface Schema {" in result
    assert "}" in result


def test_typescript_schema_title_when_metadata_on():
    """Test TypeScript formatter includes schema title comment when metadata is on."""
    schema = WithTitleDescription.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_schema_title_comment_consistent(
        result, schema, include_metadata=True, comment_prefix=formatter.comment_prefix
    )
    # Title should appear as a comment
    if schema.get("title"):
        assert f"// {schema['title']}" in result or f"// Title: {schema['title']}" in result


def test_typescript_format_scaffolding():
    """Test that TypeScript output has expected top-level structure."""
    schema = SimpleFormatterModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should have interface declaration
    assert "interface Schema {" in result
    # Should have closing brace
    assert result.rstrip().endswith("}")


# ============================================================================
# Enum and Literal Types
# ============================================================================


def test_typescript_string_enum():
    """Test TypeScript formatter with string enum (Role)."""
    # Build model with Role enum
    from pydantic import BaseModel

    class ModelWithRole(BaseModel):
        role: Role

    schema = ModelWithRole.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "role*:" in result


def test_typescript_int_enum():
    """Test TypeScript formatter with integer enum."""
    schema = IntEnumModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "priority*:" in result


def test_typescript_literal_single():
    """Test TypeScript formatter with single literal value."""
    schema = LiteralSingle.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "api_version*:" in result
    # Single literal should be rendered as the literal value (quoted string)
    assert "v1" in result


def test_typescript_literal_union():
    """Test TypeScript formatter with union of literals."""
    schema = LiteralUnion.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "status*:" in result
    # Should use pipe-separated union format for literals
    assert "draft" in result
    assert "published" in result
    assert "archived" in result
    assert "|" in result


def test_typescript_int_literals():
    """Test TypeScript formatter with integer literals."""
    from tests.conftest import IntLiterals

    schema = IntLiterals.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "priority*:" in result
    # Should use pipe-separated union format with unquoted numbers
    assert "1 | 2 | 3 | 4 | 5" in result


def test_typescript_bool_literals():
    """Test TypeScript formatter with boolean literals."""
    from tests.conftest import BoolLiterals

    schema = BoolLiterals.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "flag*:" in result
    # Should use pipe-separated union with lowercase, unquoted booleans
    assert "true | false" in result


def test_typescript_mixed_type_literals():
    """Test TypeScript formatter with mixed type literals (string, int, bool)."""
    from tests.conftest import MixedTypeLiterals

    schema = MixedTypeLiterals.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Should contain all three fields
    assert "status*:" in result
    assert "level*:" in result
    assert "enabled*:" in result

    # String literals should be quoted
    assert "active" in result
    assert "inactive" in result

    # Integer literals should be unquoted
    assert "1 | 2 | 3" in result

    # Boolean literals should be lowercase and unquoted
    assert "true | false" in result


def test_typescript_single_const_int():
    """Test TypeScript formatter with single integer const."""
    from tests.conftest import SingleConstInt

    schema = SingleConstInt.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "version*:" in result
    # Single integer literal should be rendered as unquoted number
    assert "version*: 1" in result


def test_typescript_issue_classification():
    """Test TypeScript formatter with IssueClassification model (integration test)."""
    from tests.conftest import IssueClassification

    schema = IssueClassification.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Should contain both fields
    assert "category*:" in result
    assert "priority*:" in result

    # String literals should be quoted with pipe unions
    assert "bug" in result
    assert "feature" in result
    assert "question" in result

    # Integer literals should be unquoted with pipe unions
    assert "1 | 2 | 3 | 4 | 5" in result


# ============================================================================
# Array Types
# ============================================================================


def test_typescript_array_of_strings():
    """Test TypeScript formatter with array of strings."""
    schema = ArrayOfStrings.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "items*:" in result
    # TypeScript should use Array or []
    assert "Array" in result or "string[]" in result


def test_typescript_array_constraints():
    """Test TypeScript formatter with array min/max items constraints."""
    schema = ArrayMinMaxItems.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "tags*:" in result


def test_typescript_unique_items():
    """Test TypeScript formatter with unique items constraint."""
    schema = ArrayUniqueItems.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # unique_tags field should be present
    fields = parse_typescript_interface_fields(result, interface_name="Schema")
    field_names = {name for name, _ in fields}
    assert "unique_tags" in field_names


def test_typescript_array_of_refs():
    """Test TypeScript formatter with array of referenced objects."""
    schema = ArrayOfRefsModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Should have array fields
    assert "addresses*:" in result
    assert "products*:" in result


# ============================================================================
# Nested and Complex Types
# ============================================================================


def test_typescript_deep_nesting():
    """Test TypeScript formatter with deep nesting (A -> B -> C)."""
    schema = DeepNested.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Root fields should be present
    assert "id*:" in result
    assert "level_b*:" in result


def test_typescript_union_heavy():
    """Test TypeScript formatter with multiple union types."""
    schema = UnionHeavy.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Fields should be present
    fields = parse_typescript_interface_fields(result, interface_name="Schema")
    field_names = {name for name, _ in fields}
    assert "id" in field_names
    assert "value" in field_names


def test_typescript_complex_types():
    """Test TypeScript formatter with complex field types."""
    schema = UnionTypes.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Should have union type fields
    assert "id*:" in result
    assert "status*:" in result


# ============================================================================
# Defaults and Metadata
# ============================================================================


def test_typescript_defaults():
    """Test TypeScript formatter with default values."""
    schema = ObjectWithDefaults.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Fields with defaults are typically not required
    assert_required_optional_consistent(result, schema)
    # When metadata is on, defaults may appear as comments
    # At minimum, fields should be present
    fields = parse_typescript_interface_fields(result, interface_name="Schema")
    field_names = {name for name, _ in fields}
    assert "name" in field_names
    assert "count" in field_names


def test_typescript_field_descriptions():
    """Test TypeScript formatter with field descriptions."""
    schema = WithFieldDescriptions.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # When metadata is on, descriptions may appear as comments
    # At minimum, fields should be present
    assert "name*:" in result
    assert "email*:" in result
    assert "age*:" in result


def test_typescript_examples():
    """Test TypeScript formatter with examples in schema."""
    from tests.conftest import DEPRECATED_EXAMPLES_SCHEMA

    schema = DEPRECATED_EXAMPLES_SCHEMA
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Should produce valid output
    assert "interface Schema {" in result


# ============================================================================
# String Format and Pattern
# ============================================================================


def test_typescript_string_format_email():
    """Test TypeScript formatter with email format."""
    schema = StringFormatEmail.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "email*:" in result


def test_typescript_string_format_uri():
    """Test TypeScript formatter with URI format."""
    schema = StringFormatUri.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "website*:" in result


def test_typescript_string_pattern():
    """Test TypeScript formatter with pattern constraint."""
    schema = StringPattern.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "code*:" in result


def test_typescript_multiple_patterns():
    """Test TypeScript formatter with multiple pattern constraints."""
    schema = PatternConstraints.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # All fields have patterns
    assert "phone*:" in result
    assert "zip_code*:" in result
    assert "username*:" in result


# ============================================================================
# Numeric Constraints
# ============================================================================


def test_typescript_exclusive_min_max():
    """Test TypeScript formatter with exclusive minimum/maximum (gt/lt)."""
    schema = ExclusiveMinMax.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Fields should be present; formatter may not render exclusive bounds
    assert "value*:" in result
    assert "count*:" in result


# ============================================================================
# Dict and Additional Properties
# ============================================================================


def test_typescript_dict_fields():
    """Test TypeScript formatter with dict fields."""
    schema = DictOnlyModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "metadata*:" in result


def test_typescript_additional_properties_false():
    """Test TypeScript formatter with additionalProperties: false."""
    schema = ObjectAdditionalPropsFalse.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Fields should be present
    assert "name*:" in result
    assert "value*:" in result
    # Check that additionalProperties constraint appears (as comment)
    assert "no additional properties" in result or "//no additional properties" in result


# ============================================================================
# Datetime
# ============================================================================


def test_typescript_datetime_fields():
    """Test TypeScript formatter with datetime fields."""
    schema = EventWithDate.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Datetime fields should be present
    assert "name*:" in result
    assert "event_date*:" in result


# ============================================================================
# Composition (allOf)
# ============================================================================


def test_typescript_allof_like():
    """Test TypeScript formatter with allOf-like composition."""
    schema = AllOfLike.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # AllOfLike inherits from BaseA and BaseB
    fields = parse_typescript_interface_fields(result, interface_name="Schema")
    field_names = {name for name, _ in fields}
    # Should have fields from both bases
    assert "field_a" in field_names or "own_field" in field_names


# ============================================================================
# Advanced (anyOf/oneOf/allOf, const, dependencies)
# ============================================================================


def test_typescript_anyof_property_level():
    """Test TypeScript formatter with anyOf at property level."""
    schema = UnionTypes.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Union types generate anyOf in schema
    assert "id*:" in result


def test_typescript_top_level_anyof():
    """Test TypeScript formatter with anyOf at top level."""
    schema = ANY_OF_SCHEMA
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should produce valid TypeScript (may use type instead of interface for unions)
    assert result is not None
    assert len(result) > 0
    assert "Schema" in result


def test_typescript_top_level_oneof():
    """Test TypeScript formatter with oneOf at top level."""
    schema = ONE_OF_SCHEMA
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should produce valid TypeScript (may use type instead of interface for unions)
    assert result is not None
    assert len(result) > 0
    assert "Schema" in result


def test_typescript_top_level_allof():
    """Test TypeScript formatter with allOf at top level."""
    schema = ALL_OF_SCHEMA
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should produce valid TypeScript
    assert result is not None
    assert len(result) > 0
    assert "Schema" in result
    # allOf merges required fields - should have name and age
    assert "name" in result
    assert "age" in result


def test_typescript_const_keyword():
    """Test TypeScript formatter with const keyword."""
    schema = CONST_SCHEMA
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should produce valid TypeScript
    assert "interface Schema {" in result
    # const field should be present
    assert "api_version*:" in result


def test_typescript_dependencies():
    """Test TypeScript formatter with dependencies keyword."""
    schema = DEPENDENCY_SCHEMA
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should produce valid TypeScript
    assert "interface Schema {" in result
    # Properties should be present
    assert "name:" in result


# ============================================================================
# Edge Cases and Regression
# ============================================================================


def test_typescript_null_type():
    """Test TypeScript formatter with null type (optional fields)."""
    schema = RequiredOptionalModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Optional fields should not have asterisk
    assert "optional_field:" in result
    assert "optional_field*:" not in result


def test_typescript_type_array():
    """Test TypeScript formatter with type as array (e.g., ['string', 'null'])."""
    schema = {
        "type": "object",
        "properties": {"nullable_field": {"type": ["string", "null"]}},
        "required": [],
    }
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should produce valid TypeScript
    assert "interface Schema {" in result
    assert "nullable_field:" in result


def test_typescript_property_order_preservation():
    """Test that TypeScript formatter preserves property order."""
    schema = OrderedFieldsModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    fields = parse_typescript_interface_fields(result, interface_name="Schema")
    names = [n for n, _ in fields]
    assert names == ["first", "second", "third"], f"Unexpected field order: {names}"


def test_typescript_full_featured_model():
    """Test TypeScript formatter with kitchen sink model."""
    schema = FullFeaturedModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Should have many fields
    fields = parse_typescript_interface_fields(result, interface_name="Schema")
    assert len(fields) > 5, "FullFeaturedModel should have many fields"


def test_typescript_consistent_asterisk_usage():
    """Test that asterisk usage is consistent across formatter."""
    schema = RequiredOptionalModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Required field should have asterisk
    assert "required_field*:" in result
    # Optional field should not have asterisk
    assert "optional_field:" in result
    assert "optional_field*:" not in result


# ============================================================================
# Empty schema, additionalProperties, array parity with JSONish
# ============================================================================


def test_typescript_formatter_empty_schema_renders_as_any():
    """Test that empty schema {} renders as 'any' type."""
    schema = {
        "type": "object",
        "properties": {
            "value": {}  # Empty schema
        },
    }
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert "value: any" in result


def test_typescript_formatter_object_with_complex_additional_props_shows_placeholder():
    """Test that object with only complex additionalProperties shows placeholder key."""
    # Root has no properties and complex additionalProperties -> Branch 2 emits placeholder
    schema = {
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "properties": {"value": {}},
            "required": ["value"],
        },
    }
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Should show placeholder field with structure
    assert "<key>" in result
    assert "any properties allowed" in result


def test_typescript_formatter_additional_props_with_object_schema():
    """Test that additionalProperties with object schema shows structure details in comment."""
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
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # With placeholder we show "any properties allowed"; otherwise "additional: ..."
    assert "additional:" in result or "any properties allowed" in result
    assert "value" in result and "string" in result


def test_typescript_formatter_simple_additional_props_still_work():
    """Test that simple additionalProperties (type only) still render as comments."""
    # Root-level simple additionalProperties so the comment is emitted on the main interface
    schema = {
        "type": "object",
        "properties": {"config": {"type": "object"}},
        "additionalProperties": {"type": "string"},
    }
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert "additional: string" in result


def test_typescript_formatter_additional_props_false_still_works():
    """Test that additionalProperties: false still shows 'no additional properties'."""
    schema = {
        "type": "object",
        "additionalProperties": False,
    }
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert "no additional properties" in result


def test_typescript_formatter_array_of_objects_not_duplicated():
    """Verify array items render correctly as TypeScript Array<{{...}}> syntax."""
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
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert result.count("items*:") == 1
    assert "Array<" in result
    assert "product_name" in result and "quantity" in result and "price" in result
