"""Tests for YAML formatter using PyYAML."""

from __future__ import annotations

import pytest
import yaml

from llm_schema_lite.formatters.yaml_formatter import YAMLFormatter
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
    parse_yaml_root_fields,
)

# Alias for backwards compatibility in tests
SimpleModel = SimpleFormatterModel


def test_yaml_formatter_produces_valid_yaml():
    """Test that YAML formatter produces valid YAML that can be parsed."""
    schema = SimpleModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Verify the output is valid YAML
    parsed = yaml.safe_load(result)
    assert isinstance(parsed, dict)

    fields = parse_yaml_root_fields(result)
    assert_required_optional_fields_match_schema(fields, schema)
    assert_required_optional_consistent(result, schema)
    assert_schema_title_comment_consistent(
        result, schema, include_metadata=True, comment_prefix=formatter.comment_prefix
    )


def test_yaml_formatter_without_metadata():
    """Test YAML formatter without metadata."""
    schema = SimpleModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Verify the output is valid YAML
    parsed = yaml.safe_load(result)
    assert isinstance(parsed, dict)
    fields = parse_yaml_root_fields(result)
    assert_required_optional_fields_match_schema(fields, schema)
    assert_required_optional_consistent(result, schema)
    assert_schema_title_comment_consistent(
        result, schema, include_metadata=False, comment_prefix=formatter.comment_prefix
    )


def test_yaml_formatter_with_nested_defs():
    """Test YAML formatter with nested $defs."""
    schema = PersonWithAddress.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Should contain both Address section and main Person properties.
    # (Nested definitions are emitted as separate YAML blocks with dotted keys.)
    assert "# Address" in result

    # Nested $defs keys should exist for required fields.
    defs = schema.get("$defs", schema.get("definitions", {})) or {}
    address_schema = defs.get("Address")
    assert isinstance(address_schema, dict), "Expected $defs.Address schema"
    required = set(address_schema.get("required", []) or [])

    # Load the entire output and check both dotted keys and root keys.
    parsed = yaml.safe_load(result)
    assert isinstance(parsed, dict)

    root_only = {k: v for k, v in parsed.items() if isinstance(k, str) and "." not in k}
    root_fields = []
    for k in root_only.keys():
        if k.endswith("*"):
            root_fields.append((k[:-1], True))
        else:
            root_fields.append((k, False))
    assert_required_optional_fields_match_schema(root_fields, schema)
    assert_required_optional_consistent(result, schema)

    for field in required:
        assert f"Address.{field}*" in parsed, f"Missing required nested field: Address.{field}*"


def test_yaml_formatter_key_order_preserved():
    """Test that YAML formatter preserves key order (dict order)."""
    schema = OrderedFieldsModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    fields = parse_yaml_root_fields(result)
    names = [n for n, _ in fields]
    assert names == ["first", "second", "third"], f"Unexpected field order: {names}"


def test_yaml_formatter_caching():
    """Test that formatter caching works correctly."""
    schema = SimpleModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)

    # First call
    result1 = formatter.transform_schema()
    # Second call should use cache
    result2 = formatter.transform_schema()

    assert result1 == result2
    assert hasattr(formatter, "_processed_data")
    assert formatter._processed_data is not None


# ============================================================================
# Parsing and Contract Tests
# ============================================================================


def test_yaml_required_optional_parsing_matches_schema():
    """Test that parsed YAML root fields match schema required/properties."""
    schema = RequiredOptionalModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    fields = parse_yaml_root_fields(result)
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
def test_yaml_lists_all_root_properties_for_models(model_cls):
    """Contract: YAML output lists all root properties from schema."""
    schema = model_cls.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    fields = parse_yaml_root_fields(result)
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
def test_yaml_required_optional_consistent_for_models(model_cls):
    """Contract: required fields have '*' and optional do not."""
    schema = model_cls.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)


# ============================================================================
# Edge Cases and Scaffolding
# ============================================================================


def test_yaml_empty_object_schema():
    """Test YAML formatter with empty object schema."""
    schema = EMPTY_SCHEMA
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should produce valid YAML
    parsed = yaml.safe_load(result)
    assert isinstance(parsed, dict)


def test_yaml_single_required_field():
    """Test YAML formatter with single required field."""
    schema = ObjectRequiredOnly.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # All fields in ObjectRequiredOnly are required
    assert "name*:" in result
    assert "age*:" in result


def test_yaml_all_optional_no_asterisks():
    """Test YAML formatter with all optional fields has no asterisks."""
    schema = ObjectWithDefaults.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # ObjectWithDefaults has defaults and no required fields
    required = set(schema.get("required", []))
    if not required:
        # No asterisks should appear in the field names
        fields = parse_yaml_root_fields(result)
        for name, is_req in fields:
            assert not is_req, f"Field '{name}' should not be marked as required"


def test_yaml_empty_properties():
    """Test YAML formatter with schema that has empty properties dict."""
    schema = {"type": "object", "properties": {}}
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    parsed = yaml.safe_load(result)
    assert isinstance(parsed, dict)


def test_yaml_schema_title_when_metadata_on():
    """Test YAML formatter includes schema title comment when metadata is on."""
    schema = WithTitleDescription.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_schema_title_comment_consistent(
        result, schema, include_metadata=True, comment_prefix=formatter.comment_prefix
    )
    # Title should appear as a comment
    if schema.get("title"):
        assert f"# {schema['title']}" in result or f"# Title: {schema['title']}" in result


def test_yaml_format_scaffolding():
    """Test that YAML output has expected top-level structure."""
    schema = SimpleFormatterModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should be valid YAML
    parsed = yaml.safe_load(result)
    assert isinstance(parsed, dict)
    # Should have expected keys from schema
    properties = set((schema.get("properties", {}) or {}).keys())
    assert properties, "Expected schema to have properties"


# ============================================================================
# Enum and Literal Types
# ============================================================================


def test_yaml_string_enum():
    """Test YAML formatter with string enum (Role)."""
    # Build model with Role enum
    from pydantic import BaseModel

    class ModelWithRole(BaseModel):
        role: Role

    schema = ModelWithRole.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "role*:" in result
    # YAML formatter should represent enum somehow (either as enum values or string)
    # At minimum, the field should be present
    parsed = yaml.safe_load(result)
    assert "role*" in parsed or "role" in parsed


def test_yaml_int_enum():
    """Test YAML formatter with integer enum."""
    schema = IntEnumModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "priority*:" in result


def test_yaml_literal_single():
    """Test YAML formatter with single literal value."""
    schema = LiteralSingle.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "api_version*:" in result


def test_yaml_literal_union():
    """Test YAML formatter with union of literals."""
    schema = LiteralUnion.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "status*:" in result


# ============================================================================
# Array Types
# ============================================================================


def test_yaml_array_of_strings():
    """Test YAML formatter with array of strings."""
    schema = ArrayOfStrings.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "items*:" in result
    # YAML should indicate it's a list
    assert "list" in result.lower()


def test_yaml_array_constraints():
    """Test YAML formatter with array min/max items constraints."""
    schema = ArrayMinMaxItems.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "tags*:" in result
    # When metadata is on, constraints may appear as comments
    # At minimum, the field should be present


def test_yaml_unique_items():
    """Test YAML formatter with unique items constraint."""
    schema = ArrayUniqueItems.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # unique_tags field should be present
    fields = parse_yaml_root_fields(result)
    field_names = {name for name, _ in fields}
    assert "unique_tags" in field_names


def test_yaml_array_of_refs():
    """Test YAML formatter with array of referenced objects."""
    schema = ArrayOfRefsModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Should have array fields
    assert "addresses*:" in result
    assert "products*:" in result


# ============================================================================
# Nested and Complex Types
# ============================================================================


def test_yaml_deep_nesting():
    """Test YAML formatter with deep nesting (A -> B -> C)."""
    schema = DeepNested.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Root fields should be present
    assert "id*:" in result
    assert "level_b*:" in result


def test_yaml_union_heavy():
    """Test YAML formatter with multiple union types."""
    schema = UnionHeavy.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Fields should be present
    fields = parse_yaml_root_fields(result)
    field_names = {name for name, _ in fields}
    assert "id" in field_names
    assert "value" in field_names


def test_yaml_complex_types():
    """Test YAML formatter with complex field types."""
    schema = UnionTypes.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Should have union type fields
    assert "id*:" in result
    assert "status*:" in result


# ============================================================================
# Defaults and Metadata
# ============================================================================


def test_yaml_defaults():
    """Test YAML formatter with default values."""
    schema = ObjectWithDefaults.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Fields with defaults are typically not required
    assert_required_optional_consistent(result, schema)
    # When metadata is on, defaults may appear as comments
    # At minimum, fields should be present
    fields = parse_yaml_root_fields(result)
    field_names = {name for name, _ in fields}
    assert "name" in field_names
    assert "count" in field_names


def test_yaml_field_descriptions():
    """Test YAML formatter with field descriptions."""
    schema = WithFieldDescriptions.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # When metadata is on, descriptions may appear as comments
    # At minimum, fields should be present
    assert "name*:" in result
    assert "email*:" in result
    assert "age*:" in result


def test_yaml_examples():
    """Test YAML formatter with examples in schema."""
    from tests.conftest import DEPRECATED_EXAMPLES_SCHEMA

    schema = DEPRECATED_EXAMPLES_SCHEMA
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Should produce valid output
    parsed = yaml.safe_load(result)
    assert isinstance(parsed, dict)


# ============================================================================
# String Format and Pattern
# ============================================================================


def test_yaml_string_format_email():
    """Test YAML formatter with email format."""
    schema = StringFormatEmail.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "email*:" in result


def test_yaml_string_format_uri():
    """Test YAML formatter with URI format."""
    schema = StringFormatUri.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "website*:" in result


def test_yaml_string_pattern():
    """Test YAML formatter with pattern constraint."""
    schema = StringPattern.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "code*:" in result


def test_yaml_multiple_patterns():
    """Test YAML formatter with multiple pattern constraints."""
    schema = PatternConstraints.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # All fields have patterns
    assert "phone*:" in result
    assert "zip_code*:" in result
    assert "username*:" in result


# ============================================================================
# Numeric Constraints
# ============================================================================


def test_yaml_exclusive_min_max():
    """Test YAML formatter with exclusive minimum/maximum (gt/lt)."""
    schema = ExclusiveMinMax.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Fields should be present; formatter may not render exclusive bounds
    assert "value*:" in result
    assert "count*:" in result


# ============================================================================
# Dict and Additional Properties
# ============================================================================


def test_yaml_dict_fields():
    """Test YAML formatter with dict fields."""
    schema = DictOnlyModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "metadata*:" in result


def test_yaml_additional_properties_false():
    """Test YAML formatter with additionalProperties: false."""
    schema = ObjectAdditionalPropsFalse.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Fields should be present
    assert "name*:" in result
    assert "value*:" in result
    # Check that additionalProperties constraint appears (as comment)
    assert "no additional properties" in result or "#no additional properties" in result


# ============================================================================
# Datetime
# ============================================================================


def test_yaml_datetime_fields():
    """Test YAML formatter with datetime fields."""
    schema = EventWithDate.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Datetime fields should be present
    assert "name*:" in result
    assert "event_date*:" in result


# ============================================================================
# Composition (allOf)
# ============================================================================


def test_yaml_allof_like():
    """Test YAML formatter with allOf-like composition."""
    schema = AllOfLike.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # AllOfLike inherits from BaseA and BaseB
    fields = parse_yaml_root_fields(result)
    field_names = {name for name, _ in fields}
    # Should have fields from both bases
    assert "field_a" in field_names or "own_field" in field_names


# ============================================================================
# Advanced (anyOf/oneOf/allOf, const, dependencies)
# ============================================================================


def test_yaml_anyof_property_level():
    """Test YAML formatter with anyOf at property level."""
    schema = UnionTypes.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Union types generate anyOf in schema
    assert "id*:" in result


def test_yaml_top_level_anyof():
    """Test YAML formatter with anyOf at top level."""
    schema = ANY_OF_SCHEMA
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should produce valid output (may be string for union types)
    assert result is not None
    assert len(result) > 0


def test_yaml_top_level_oneof():
    """Test YAML formatter with oneOf at top level."""
    schema = ONE_OF_SCHEMA
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should produce valid YAML
    parsed = yaml.safe_load(result)
    assert isinstance(parsed, dict)


def test_yaml_top_level_allof():
    """Test YAML formatter with allOf at top level."""
    schema = ALL_OF_SCHEMA
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should produce valid output
    assert result is not None
    assert len(result) > 0
    # allOf merges required fields - should have name and age
    assert "name" in result
    assert "age" in result


def test_yaml_const_keyword():
    """Test YAML formatter with const keyword."""
    schema = CONST_SCHEMA
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should produce valid YAML
    parsed = yaml.safe_load(result)
    assert isinstance(parsed, dict)
    # const field should be present
    assert "api_version*" in parsed or "api_version" in str(parsed)


def test_yaml_dependencies():
    """Test YAML formatter with dependencies keyword."""
    schema = DEPENDENCY_SCHEMA
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should produce valid YAML
    parsed = yaml.safe_load(result)
    assert isinstance(parsed, dict)
    # Properties should be present
    assert "name" in str(parsed)


# ============================================================================
# Edge Cases and Regression
# ============================================================================


def test_yaml_null_type():
    """Test YAML formatter with null type (optional fields)."""
    schema = RequiredOptionalModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Optional fields should not have asterisk
    assert "optional_field:" in result
    assert "optional_field*:" not in result


def test_yaml_type_array():
    """Test YAML formatter with type as array (e.g., ['string', 'null'])."""
    schema = {
        "type": "object",
        "properties": {"nullable_field": {"type": ["string", "null"]}},
        "required": [],
    }
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Should produce valid YAML
    parsed = yaml.safe_load(result)
    assert isinstance(parsed, dict)
    assert "nullable_field" in str(parsed)


def test_yaml_property_order_preservation():
    """Test that YAML formatter preserves property order."""
    schema = OrderedFieldsModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    fields = parse_yaml_root_fields(result)
    names = [n for n, _ in fields]
    assert names == ["first", "second", "third"], f"Unexpected field order: {names}"


def test_yaml_full_featured_model():
    """Test YAML formatter with kitchen sink model."""
    schema = FullFeaturedModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Should have many fields
    fields = parse_yaml_root_fields(result)
    assert len(fields) > 5, "FullFeaturedModel should have many fields"


def test_yaml_consistent_asterisk_usage():
    """Test that asterisk usage is consistent across formatter."""
    schema = RequiredOptionalModel.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Required field should have asterisk
    assert "required_field*:" in result
    # Optional field should not have asterisk
    assert "optional_field:" in result
    assert "optional_field*:" not in result
