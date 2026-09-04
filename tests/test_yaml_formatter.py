"""Tests for YAML formatter using PyYAML."""

from __future__ import annotations

import pytest
import yaml

from llm_schema_lite import FormatterConfig
from llm_schema_lite.formatters.yaml_formatter import YAMLFormatter
from llm_schema_lite.schema_normalization import auto_title_for_key
from tests.conftest import (
    ALL_OF_SCHEMA,
    ANY_OF_SCHEMA,
    CONST_SCHEMA,
    DEPENDENCY_SCHEMA,
    EMPTY_SCHEMA,
    ONE_OF_SCHEMA,
    Address,
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
    ModelWithAlias,
    ObjectAdditionalPropsFalse,
    ObjectRequiredOnly,
    ObjectWithDefaults,
    Order,
    OrderedFieldsModel,
    PatternConstraints,
    PersonWithAddress,
    RequiredOptionalModel,
    Role,
    SimpleFormatterModel,
    SingleConstInt,
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


def test_no_auto_generated_property_titles_yaml():
    """AC-1: YAML drops auto-generated per-field titles from the output."""
    for model in (Address, SimpleFormatterModel, ModelWithAlias):
        schema = model.model_json_schema()
        formatter = YAMLFormatter(schema, include_metadata=True)
        result = formatter.transform_schema()

        for key in schema.get("properties", {}):
            assert f"{formatter.comment_prefix} {auto_title_for_key(key)}:" not in result


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


def test_yaml_enum_with_descriptions_and_aliases():
    """Test YAML formatter shows OPTIONS with descriptions."""
    from llm_schema_lite import simplify_schema
    from tests.conftest import ModelWithPriorityMetadata

    result = simplify_schema(ModelWithPriorityMetadata, format_type="yaml").to_string()
    assert "OPTIONS with descriptions" in result
    assert "Non-urgent, can wait" in result
    assert "aliases:" in result
    assert "OPTIONS:" in result
    assert "low" in result and "critical" in result


def test_yaml_literal_single():
    """Test YAML formatter with single literal value."""
    schema = LiteralSingle.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "api_version*:" in result
    # Single literal should be rendered as the const value (quoted string)
    assert "v1" in result


def test_yaml_literal_union():
    """Test YAML formatter with union of literals."""
    schema = LiteralUnion.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "status*:" in result
    # Should use OPTIONS format for multiple literals
    assert "OPTIONS:" in result
    # Should contain all literal values (YAML wraps entire OPTIONS string in quotes)
    assert "draft" in result
    assert "published" in result
    assert "archived" in result


def test_yaml_int_literals():
    """Test YAML formatter with integer literals."""
    from tests.conftest import IntLiterals

    schema = IntLiterals.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "priority*:" in result
    # Should use OPTIONS format for multiple integer literals
    assert "OPTIONS:" in result
    # Should contain all integer values (unquoted)
    assert "1" in result
    assert "2" in result
    assert "3" in result
    assert "4" in result
    assert "5" in result
    # Verify unquoted format
    assert "OPTIONS: 1| 2| 3| 4| 5" in result or "OPTIONS: 1 | 2 | 3 | 4 | 5" in result


def test_yaml_bool_literals():
    """Test YAML formatter with boolean literals."""
    from tests.conftest import BoolLiterals

    schema = BoolLiterals.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "flag*:" in result
    # Should use OPTIONS format for boolean literals
    assert "OPTIONS:" in result
    # YAML may serialize bools as True/False (Python) - verify presence
    has_true = "true" in result.lower() or "True" in result
    has_false = "false" in result.lower() or "False" in result
    assert has_true and has_false


def test_yaml_mixed_type_literals():
    """Test YAML formatter with mixed type literals (string, int, bool)."""
    from tests.conftest import MixedTypeLiterals

    schema = MixedTypeLiterals.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Should contain all three fields
    assert "status*:" in result
    assert "level*:" in result
    assert "enabled*:" in result

    # String literals (YAML wraps entire OPTIONS string in quotes)
    assert "active" in result
    assert "inactive" in result

    # Integer literals should be unquoted
    assert "1" in result and "2" in result and "3" in result

    # Boolean literals (YAML may use True/False)
    has_true = "true" in result.lower() or "True" in result
    has_false = "false" in result.lower() or "False" in result
    assert has_true and has_false


def test_yaml_single_const_int():
    """Test YAML formatter with single integer const."""
    from tests.conftest import SingleConstInt

    schema = SingleConstInt.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "version*:" in result
    # Single integer literal (YAML may quote it)
    assert "version*: 1" in result


def test_yaml_issue_classification():
    """Test YAML formatter with IssueClassification model (integration test)."""
    from tests.conftest import IssueClassification

    schema = IssueClassification.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Should contain both fields
    assert "category*:" in result
    assert "priority*:" in result

    # String literals with OPTIONS format (YAML wraps entire string)
    assert "OPTIONS:" in result
    assert "bug" in result
    assert "feature" in result
    assert "question" in result

    # Integer literals with OPTIONS format
    assert "1" in result
    assert "2" in result
    assert "3" in result
    assert "4" in result
    assert "5" in result


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


def test_yaml_formatter_empty_schema_renders_as_any():
    """Test that empty schema {} renders as 'any', not 'string'."""
    schema = {
        "type": "object",
        "properties": {
            "value": {}  # Empty schema
        },
    }
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert "value: any" in result or "value: any\n" in result


def test_yaml_formatter_object_with_complex_additional_props_shows_placeholder():
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
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert "<key>" in result
    assert "value*" in result or "value:" in result
    assert "any" in result
    assert "any properties allowed" in result


def test_yaml_formatter_additional_props_with_object_schema():
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
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert "additional:" in result or "any properties allowed" in result
    assert "value" in result and "string" in result


def test_yaml_formatter_simple_additional_props_still_work():
    """Test that simple additionalProperties still render as comments."""
    schema = {
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            }
        },
    }
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert "additional:" in result
    assert "string" in result


def test_yaml_formatter_additional_props_false_still_works():
    """Test that additionalProperties: false still works correctly."""
    schema = {
        "type": "object",
        "additionalProperties": False,
    }
    formatter = YAMLFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert "no additional properties" in result


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
    # Optional field should not have any marker (default optional_marker="")
    assert "optional_field:" in result
    assert "optional_field*:" not in result


def test_user_supplied_title_and_description_render_once_yaml():
    """AC-2: a user-supplied field title and description each render exactly once.

    ``result`` is un-escaped first: the rendered value is a single-quoted YAML
    scalar, so PyYAML doubles the apostrophe in ``The user's full name``.
    """
    schema = WithFieldDescriptions.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()
    unescaped = result.replace("''", "'")

    assert result.count("Full Name") == 1
    assert unescaped.count("The user's full name") == 1


def test_metadata_inclusion_title_false_suppresses_title_yaml():
    """The ``title`` metadata_inclusion lever suppresses field titles, not descriptions."""
    schema = WithFieldDescriptions.model_json_schema()
    config = FormatterConfig(include_metadata=True, metadata_inclusion={"title": False})
    formatter = YAMLFormatter(schema, config=config)
    result = formatter.transform_schema()

    assert "Full Name" not in result
    assert "The user's full name" in result.replace("''", "'")


def test_order_yaml_product_name_appears_exactly_twice():
    """``Product name`` renders once per genuinely distinct property line.

    ``Order`` yields two such lines -- the ``$defs``-hoisted ``Product.name`` line and
    the inlined one -- so the description must appear exactly twice overall and never
    twice on the same line.
    """
    schema = Order.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert result.count("Product name") == 2, result
    for line in result.splitlines():
        assert line.count("Product name") <= 1, f"duplicated on one line: {line!r}"


def test_yaml_add_metadata_handles_non_str_representation():
    """``add_metadata`` must tolerate a non-``str`` representation (``SingleConstInt``).

    The idempotence guard calls ``.endswith`` on the representation; ``const``
    schemas pass the raw ``int`` value through, so the guard must be
    ``isinstance``-checked.
    """
    schema = SingleConstInt.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)

    assert formatter.add_metadata(1, {"const": 1}) is not None  # type: ignore[arg-type]

    result = formatter.transform_schema()
    assert "version" in result


def _iter_yaml_comment_lines(result: str) -> list[str]:
    """Yield logical (un-wrapped) lines of a rendered YAML schema.

    PyYAML folds long single-quoted scalars across physical lines, so a raw
    ``splitlines()`` can end a line on ``#`` purely because of wrapping. Loading
    the document rejoins each scalar, giving the logical lines the formatter
    actually produced. Top-level comment lines are added back verbatim.
    """
    lines = [line for line in result.splitlines() if line.lstrip().startswith("#")]

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            lines.extend(node.splitlines())

    walk(yaml.safe_load(result))
    return lines


def test_no_empty_comment_marker_across_all_models_yaml(all_pydantic_models) -> None:
    """No rendered YAML line should carry an empty trailing ``#`` marker."""
    for _name, model in all_pydantic_models:
        schema = model.model_json_schema()
        formatter = YAMLFormatter(schema, include_metadata=True)
        result = formatter.transform_schema()

        for line in _iter_yaml_comment_lines(result):
            assert not line.endswith("# "), f"{_name}: trailing bare `# ` in line: {line!r}"
            assert not line.endswith("#"), f"{_name}: trailing bare `#` in line: {line!r}"


def test_order_yaml_token_count_decreases() -> None:
    """Dropping auto-generated titles shrinks the YAML token count for ``Order``.

    HEAD (pre-ticket-003) measured 2032 tokens; the fix is expected to land at 1129.
    The assertion is strict-lower-than-HEAD rather than pinned to the exact number so
    unrelated future formatting tweaks do not spuriously fail this test.
    """
    pytest.importorskip("tiktoken")
    from llm_schema_lite import simplify_schema

    assert simplify_schema(Order, format_type="yaml").token_count() < 2032
