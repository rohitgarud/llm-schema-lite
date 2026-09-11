"""Tests for YAML formatter using PyYAML."""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from pydantic import BaseModel, Field, TypeAdapter

from llm_schema_lite import FormatterConfig
from llm_schema_lite.formatters.yaml_formatter import YAMLFormatter
from llm_schema_lite.schema_normalization import auto_title_for_key
from tests.conftest import (
    ADDITIONAL_ITEMS_SCHEMA,
    ALL_OF_SCHEMA,
    ANY_OF_SCHEMA,
    CONST_SCHEMA,
    DEPENDENCY_SCHEMA,
    EMPTY_SCHEMA,
    ONE_OF_SCHEMA,
    PREFIX_ITEMS_SCHEMA,
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
    HashInPatternModel,
    IntEnumModel,
    LiteralSingle,
    LiteralUnion,
    ModelWithAlias,
    MultiLineDescriptionModel,
    ObjectAdditionalPropsFalse,
    ObjectRequiredOnly,
    ObjectWithDefaults,
    OptionalTree,
    Order,
    OrderedFieldsModel,
    PatternConstraints,
    PersonWithAddress,
    R1Model,
    RequiredOptionalModel,
    Role,
    Root,
    SimpleFormatterModel,
    SingleConstInt,
    StringFormatEmail,
    StringFormatUri,
    StringPattern,
    TreeNode,
    UnionHeavy,
    UnionTypes,
    User,
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
    """A ``$defs`` model renders INLINE, as a real nested block on the referring property.

    There is no longer a hoisted ``# Address`` section and no ``Address.street*`` key: the
    def's fields live under ``address*`` as a genuine YAML mapping, carrying **Address's own**
    required markers (D3) rather than the root model's.
    """
    schema = PersonWithAddress.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    defs = schema.get("$defs", schema.get("definitions", {})) or {}
    address_schema = defs.get("Address")
    assert isinstance(address_schema, dict), "Expected $defs.Address schema"

    parsed = yaml.safe_load(result)
    assert isinstance(parsed, dict)

    # No ``root_only`` filter any more: Axis 2 guarantees no key can contain a ``.``, so the
    # old ``"." not in k`` guard was a no-op. (It was a local comprehension, not a shared
    # helper -- nothing else used it.)
    root_fields = []
    for k in parsed:
        if k.endswith("*"):
            root_fields.append((k[:-1], True))
        else:
            root_fields.append((k, False))
    assert_required_optional_fields_match_schema(root_fields, schema)
    assert_required_optional_consistent(result, schema)

    # AC1: nothing is hoisted, and no key carries a ``Class.field`` prefix.
    assert "# Address" not in result
    assert "Address." not in result

    # AC1/D3: the nested block is a real dict whose keys are exactly Address's fields, marked
    # against Address's OWN required list rather than PersonWithAddress's.
    nested = parsed["address*"]
    assert isinstance(nested, dict), f"address* is {type(nested).__name__}, not a mapping"
    address_required = set(address_schema.get("required", []) or [])
    assert set(nested) == {
        f"{name}*" if name in address_required else name for name in address_schema["properties"]
    }


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
    assert "one of:" in result
    assert "(Non-urgent, can wait)" in result
    assert "aliases: urgent, blocker" in result
    assert "low" in result and "critical" in result


def test_yaml_literal_single():
    """Test YAML formatter with single literal value."""
    schema = LiteralSingle.model_json_schema()
    formatter = YAMLFormatter(schema)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "api_version*:" in result
    # Single literal should be rendered as the const value (quoted string)
    assert "v1" in result


def test_yaml_literal_union():
    """Test YAML formatter with union of literals."""
    schema = LiteralUnion.model_json_schema()
    formatter = YAMLFormatter(schema)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "status*:" in result
    # Should use the "one of:" trailing-comment format for multiple literals
    assert "string  # one of:" in result
    # Should contain all literal values (bare unquoted plain scalar, no wrapping)
    assert "draft" in result
    assert "published" in result
    assert "archived" in result


def test_yaml_int_literals():
    """Test YAML formatter with integer literals."""
    from tests.conftest import IntLiterals

    schema = IntLiterals.model_json_schema()
    formatter = YAMLFormatter(schema)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "priority*:" in result
    # Should use the "one of:" trailing-comment format for multiple integer literals
    assert "int  # one of:" in result
    # Should contain all integer values (unquoted)
    assert "1" in result
    assert "2" in result
    assert "3" in result
    assert "4" in result
    assert "5" in result
    # Verify exact unquoted format
    assert "int  # one of: 1, 2, 3, 4, 5" in result


def test_yaml_bool_literals():
    """Test YAML formatter with boolean literals."""
    from tests.conftest import BoolLiterals

    schema = BoolLiterals.model_json_schema()
    formatter = YAMLFormatter(schema)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "flag*:" in result
    # Should use the "one of:" trailing-comment format for boolean literals
    assert "bool  # one of: true, false" in result
    # YAML may serialize bools as True/False (Python) - verify presence
    has_true = "true" in result.lower() or "True" in result
    has_false = "false" in result.lower() or "False" in result
    assert has_true and has_false


def test_yaml_mixed_type_literals():
    """Test YAML formatter with mixed type literals (string, int, bool)."""
    from tests.conftest import MixedTypeLiterals

    schema = MixedTypeLiterals.model_json_schema()
    formatter = YAMLFormatter(schema)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Should contain all three fields
    assert "status*:" in result
    assert "level*:" in result
    assert "enabled*:" in result

    # String literals (bare, unquoted plain scalar with a trailing "# one of:" comment)
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
    formatter = YAMLFormatter(schema)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    assert "version*:" in result
    # Single integer literal renders with the base type token and a trailing comment
    assert "version*: int  # one of: 1" in result


def test_yaml_issue_classification():
    """Test YAML formatter with IssueClassification model (integration test)."""
    from tests.conftest import IssueClassification

    schema = IssueClassification.model_json_schema()
    formatter = YAMLFormatter(schema)
    result = formatter.transform_schema()

    assert_required_optional_consistent(result, schema)
    # Should contain both fields
    assert "category*:" in result
    assert "priority*:" in result

    # String literals with the "one of:" trailing-comment format (bare, unquoted)
    assert "  # one of:" in result
    assert "bug" in result
    assert "feature" in result
    assert "question" in result

    # Integer literals with the "one of:" trailing-comment format
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

    assert "<string>" in result
    assert "value*" in result or "value:" in result
    assert "any" in result


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


def test_yaml_root_additional_properties_survives_trailer_removal():
    """AC-2: deleting the document-tail trailer must not silence the ROOT channel.

    This schema has both a nested structural mapping (which used to bump the deleted
    placeholder counter) and a complex root ``additionalProperties``. At HEAD it printed
    ``any properties allowed`` TWICE -- once from the trailer, once from
    ``process_additional_properties``. Exactly one must survive: the root one.
    """
    schema = {
        "type": "object",
        "properties": {
            "m": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"],
                },
            }
        },
        "required": ["m"],
        "additionalProperties": {
            "type": "object",
            "properties": {"v": {"type": "string"}},
            "required": ["v"],
        },
    }
    result = YAMLFormatter(schema, include_metadata=False).transform_schema()

    assert result.count("any properties allowed") == 1, result
    # The surviving one is the ROOT channel's trailing comment, not a document tail.
    assert result.rstrip().endswith("# any properties allowed"), result


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

    assert "dict[string, string]" in result
    assert "additional:" not in result


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


def test_order_yaml_product_name_appears_exactly_once():
    """AC3: a field's description renders exactly ONCE in the whole document.

    ``Product.name`` used to appear twice -- once in the hoisted ``$defs`` section and once
    inline. With the section gone there is only the inlined occurrence, which makes this a
    strictly stronger oracle than the old ``== 2`` (that count was satisfied even in a
    duplication-riddled world, because it summed one hoisted and one inline occurrence).
    """
    schema = Order.model_json_schema()
    formatter = YAMLFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert result.count("Product name") == 1, result
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


# ============================================================================
# Container types (dict / tuple / set / Any) -- lsl-2026-09-04-015
# ============================================================================


class _AnyVariants(BaseModel):
    """Local model pinning that ``Any`` never renders as ``string`` (AC-2)."""

    plain: Any
    described: Any = Field(..., description="free form")
    titled: Any = Field(..., title="Custom Title")


class _HasDict(BaseModel):
    """Local model whose only field is a mapping (correction C3 / design risk 3)."""

    d: dict[str, int]


class _Wrapper(BaseModel):
    """Local model nesting :class:`_HasDict` so the mapping sits one level down."""

    inner: _HasDict


def _yaml_line_with(result: str, prefix: str) -> str:
    """Return the single rendered line whose key matches ``prefix``."""
    for line in result.splitlines():
        if line.strip().startswith(prefix):
            return line
    raise AssertionError(f"No line starting with {prefix!r} in:\n{result}")


def test_yaml_formatter_root_fixture_default():
    """Golden whole-string rendering of the container-types ``Root`` fixture."""
    result = YAMLFormatter(Root.model_json_schema()).transform_schema()

    expected = "\n".join(
        [
            # Deviation from the plan's golden (recorded in the Phase 5c progress notes):
            # the plan expected "# Title: Root". The `Root` fixture's auto-generated title is
            # stripped by ticket 003's `normalize_schema_titles`, and its class docstring
            # (landed with P4) becomes the Description comment instead. Both code paths are
            # untouched.
            #
            # lsl-2026-09-04-006 reshaped the rest of this golden. Four independent changes:
            #   1. The hoisted `# Inner` / `# Strict` / `# SubModel` $defs sections are GONE.
            #      `Inner` and `Strict` are now real nested blocks on the properties that
            #      reference them, and `SubModel` is inlined under `dict_of_models*`. That is
            #      the whole point of the ticket: no `Class.field` keys, no duplication.
            #   2. `#` now prefixes the description's SECOND paragraph. Previously that line
            #      was emitted bare, which made this render the one that failed
            #      `yaml.safe_load` (D8).
            #   3. `strict*` carries `# no additional properties`. The closed-world marker is
            #      structural, so it survives on the nested block (R1) instead of being lost
            #      with the deleted `Strict` section.
            #   4. `# any properties allowed` is GONE. lsl-2026-09-05-010 deleted the
            #      document-tail trailer and its placeholder counter outright; only the root
            #      `additionalProperties` channel may describe an open root now, and `Root`
            #      has no root `additionalProperties`.
            "# Description: Kitchen-sink fixture for lsl-2026-09-04-015 "
            "(dict/tuple/set/Any container rendering).",
            "#",
            "# Fields verbatim from the approved design (2026-09-04-design-discussion-v2.md 5.1).",
            "",
            "# Fields marked with * are required",
            "",
            "extra*: dict[string, int]",
            "dict_of_models*:",
            "  <string>:",
            "    a*: int",
            "    b*: string",
            "by_color*: dict[red OR green, int]",
            "pair*: tuple[int, string]",
            "var_tuple*: list[int]",
            "tags*: list[string] (unique)",
            "anything*: any",
            "described*: any  # free form",
            "opt_any: any OR null  # (default=null)",
            "any_list*: list[any]",
            "opt_extra: dict[string, int] OR null  # (default=null)",
            "inner*:",
            "  d*: dict[string, int]",
            "  t*: tuple[int, string]",
            "strict*:  # no additional properties",
            "  s*: string",
        ]
    )

    assert result == expected


def test_yaml_formatter_root_fixture_include_metadata_false():
    """With metadata off, container tokens survive unquoted and no comment leaks."""
    result = YAMLFormatter(Root.model_json_schema(), include_metadata=False).transform_schema()

    assert "described*: any\n" in result
    assert "opt_any: any OR null" in result
    assert "opt_extra: dict[string, int] OR null" in result
    assert "tags*: list[string]" in result
    assert "extra*: dict[string, int]" in result


def test_yaml_formatter_root_fixture_metadata_inclusion_variant():
    """``uniqueItems: False`` removes the ``(unique)`` token from the set field."""
    config = FormatterConfig(metadata_inclusion={"uniqueItems": False})
    result = YAMLFormatter(Root.model_json_schema(), config=config).transform_schema()

    assert "tags*: list[string]" in result
    assert "(unique)" not in result


def test_yaml_formatter_root_fixture_no_forbidden_substrings():
    """AC-1: the old YAML container vocabulary is gone."""
    result = YAMLFormatter(Root.model_json_schema()).transform_schema()

    assert "additional:" not in result
    assert "2-2" not in result
    assert "UNIQUE" not in result


def test_yaml_formatter_any_never_renders_as_string():
    """AC-2: an ``Any`` field never renders as ``string``, with or without metadata."""
    result = YAMLFormatter(_AnyVariants.model_json_schema()).transform_schema()

    for prefix in ("plain*", "described*", "titled*"):
        line = _yaml_line_with(result, prefix)
        assert "string" not in line, f"{prefix} rendered as string: {line!r}"


def test_yaml_formatter_no_dict_repr_leak_for_nested_model_dict_field():
    """Correction C3: a mapping inside a nested model never leaks a Python dict repr."""
    result = YAMLFormatter(_Wrapper.model_json_schema()).transform_schema()

    assert "{'" not in result


def test_yaml_formatter_draft7_tuple_items_no_crash():
    """Behaviour-change pin: HEAD rendered ``list[Any]  # additionalItems: False``."""
    result = YAMLFormatter(ADDITIONAL_ITEMS_SCHEMA).transform_schema()

    assert result == "tuple[string, int]"
    assert "additionalItems" not in result


def test_yaml_formatter_prefix_items_tuple_with_variadic_tail():
    """``prefixItems`` plus an ``items`` tail renders as a variadic tuple."""
    result = YAMLFormatter(PREFIX_ITEMS_SCHEMA).transform_schema()

    assert result == "tuple[string, int, bool, ...string]"


# ============================================================================
# Recursive models -- depth-limited rendering (lsl-2026-09-04-014)
# ============================================================================


class _ListNode(BaseModel):
    """Self-referencing model behind the D4 root-unwrap goldens."""

    label: str
    kids: list[_ListNode] = Field(default_factory=list)


class _MapNode(BaseModel):
    """Self-referencing model reached through a mapping value (D5).

    Deliberately NOT registered in ``all_pydantic_models``: PyYAML folds its
    rendering onto physical lines that end on ``#``, which would trip
    ``test_no_empty_comment_marker_across_all_models_yaml``.
    """

    name: str
    kids: dict[str, _MapNode] = Field(default_factory=dict)


_ListNode.model_rebuild()
_MapNode.model_rebuild()


def test_yaml_recursive_list_golden_default_depth():
    """Whole-string golden for a recursive list model at the default depth."""
    from llm_schema_lite import simplify_schema

    result = simplify_schema(_ListNode, format_type="yaml").to_string()

    expected = (
        "# Title: _ListNode, Description: Self-referencing model behind the D4"
        " root-unwrap goldens.\n"
        "\n"
        "# Fields marked with * are required\n"
        "\n"
        "label*: string\n"
        "kids:\n"
        "- label*: string\n"
        "  kids: list[object]  # recursive: _ListNode"
    )
    assert result == expected


def test_yaml_mapping_of_recursive_model_terminates():
    """D5: ``dict[str, RecursiveModel]`` terminates instead of RecursionError."""
    from llm_schema_lite import simplify_schema

    result = simplify_schema(
        _MapNode,
        config=FormatterConfig(max_recursion_depth=1),
        format_type="yaml",
    ).to_string()

    assert "object  # recursive: _MapNode" in result


def test_yaml_root_ref_model_renders_properties():
    """D4: a root-``$ref`` schema renders its properties, not an empty document."""
    result = YAMLFormatter(_ListNode.model_json_schema()).transform_schema()

    assert result != "{}"
    assert "label" in result


def test_yaml_recursive_output_round_trips_through_safe_load():
    """The recursion marker stays inert text: PyYAML quotes any scalar with ``#``."""
    from llm_schema_lite import simplify_schema

    result = simplify_schema(_ListNode, format_type="yaml").to_string()

    loaded = yaml.safe_load(result)
    assert isinstance(loaded, dict)
    assert loaded
    for key, value in loaded.items():
        assert value is not None, f"{key} round-tripped to None in:\n{result}"


def test_yaml_recursive_render_is_idempotent():
    """C3: the cached ``$defs`` walk seeds the depth counter like the main one."""
    formatter = YAMLFormatter(_ListNode.model_json_schema())

    first = formatter.transform_schema()
    second = formatter.transform_schema()

    assert first == second


# ============================================================================
# lsl-2026-09-04-006 — nested-model blocks and de-duplication
# ============================================================================


def test_yaml_r1_closed_world_markers_on_nested_blocks_metadata_off():
    """R1: a nested block carries its own def's closed-world marker, ungated by metadata.

    ``emits_closed_world_marker`` is structural, not metadata, so all three ``StrictSub``
    blocks are marked even with metadata off, while the ``extra="allow"`` sibling is not.
    """
    result = YAMLFormatter(R1Model.model_json_schema(), include_metadata=False).transform_schema()

    assert result == (
        "strict*:  # no additional properties\n"
        "  s*: string\n"
        "opt:  # OR null; no additional properties\n"
        "  s*: string\n"
        "many:  # no additional properties\n"
        "- s*: string\n"
        "open_one*:\n"
        "  s*: string"
    )


def test_yaml_r1_closed_world_markers_on_nested_blocks_metadata_on():
    """R1 with metadata on: the structural notes and the defaults share ONE slot per key."""
    result = YAMLFormatter(R1Model.model_json_schema()).transform_schema()

    assert result == (
        "# Description: R1 matrix: a closed-world nested block required, nullable, "
        "list-wrapped, and open.\n"
        "\n"
        "# Fields marked with * are required\n"
        "\n"
        "strict*:  # no additional properties\n"
        "  s*: string\n"
        "opt:  # OR null; no additional properties; (default=null)\n"
        "  s*: string\n"
        "many:  # no additional properties; (default=[])\n"
        "- s*: string\n"
        "open_one*:\n"
        "  s*: string"
    )


def test_yaml_array_of_refs_renders_list_of_mappings():
    """AC2: ``list[Model]`` is a YAML sequence of mappings, never the ``list[object]`` token.

    Asserted structurally through ``yaml.safe_load`` rather than by substring, so it cannot
    be satisfied by a quoted multi-line scalar that merely happens to contain the right text.
    """
    result = YAMLFormatter(ArrayOfRefsModel.model_json_schema()).transform_schema()
    parsed = yaml.safe_load(result)

    assert isinstance(parsed, dict)
    # Top level: list[Address] and list[Product] are real sequences of real mappings.
    assert isinstance(parsed["addresses*"], list)
    assert isinstance(parsed["addresses*"][0], dict)
    assert set(parsed["addresses*"][0]) == {
        "street*",
        "city*",
        "state*",
        "postal_code*",
        "country",
    }
    assert isinstance(parsed["products*"][0], dict)

    # Four levels deep, through a nullable ``list[User]`` (design v2 4.4).
    assert isinstance(parsed["users"], list)
    user = parsed["users"][0]
    assert isinstance(user, dict)
    assert isinstance(user["addresses*"], list)
    assert isinstance(user["addresses*"][0], dict)
    assert "street*" in user["addresses*"][0]
    assert isinstance(user["contact_info*"], dict)
    assert "email*" in user["contact_info*"]

    assert "list[object]" not in result


def test_yaml_tree_node_default_depth_block_shape():
    """Recursion at ``max_recursion_depth=2`` expands once inside the list, then truncates."""
    result = YAMLFormatter(TreeNode.model_json_schema(), include_metadata=False).transform_schema()

    assert result == (
        "name*: string\nchildren:\n- name*: string\n  children: list[object]  # recursive: TreeNode"
    )


def test_yaml_tree_node_default_depth_block_shape_metadata_on():
    """The ``(default=[])`` folds into the SAME slot as the recursion note (design v2 4.5)."""
    result = YAMLFormatter(TreeNode.model_json_schema()).transform_schema()

    assert "children:  # (default=[])" in result
    assert "- name*: string" in result
    assert "  children: list[object]  # recursive: TreeNode; (default=[])" in result


def test_yaml_optional_tree_nullable_recursive_block_shape():
    """``Model | None`` recursion: the block is real, and ``OR null`` rides the key's slot."""
    result = YAMLFormatter(OptionalTree.model_json_schema()).transform_schema()

    assert result.endswith(
        "value*: string\n"
        "left:  # OR null; (default=null)\n"
        "  value*: string\n"
        "  left: object OR null  # recursive: OptionalTree; (default=null)\n"
        "  right: object OR null  # recursive: OptionalTree; (default=null)\n"
        "right:  # OR null; (default=null)\n"
        "  value*: string\n"
        "  left: object OR null  # recursive: OptionalTree; (default=null)\n"
        "  right: object OR null  # recursive: OptionalTree; (default=null)"
    )
    parsed = yaml.safe_load(result)
    assert isinstance(parsed["left"], dict)
    assert isinstance(parsed["right"], dict)


def test_yaml_person_with_address_metadata_off():
    """The metadata-OFF companion to ``test_yaml_formatter_with_nested_defs``."""
    result = YAMLFormatter(
        PersonWithAddress.model_json_schema(), include_metadata=False
    ).transform_schema()

    # With metadata off the pattern constraints are gated away too, so every nested field is
    # a bare token -- the block structure and ``Address``'s OWN required markers are what this
    # pins.
    assert result == (
        "name*: string\n"
        "address*:\n"
        "  street*: string\n"
        "  city*: string\n"
        "  state*: string\n"
        "  postal_code*: string\n"
        "  country: string"
    )
    assert "Address." not in result


def test_yaml_formatter_root_fixture_metadata_off_golden():
    """Whole-string metadata-OFF companion to ``test_yaml_formatter_root_fixture_default``.

    With metadata off only the STRUCTURAL notes survive: ``strict*`` keeps its closed-world
    marker (R1/6.8) while every description, default and title is gated away.
    """
    result = YAMLFormatter(Root.model_json_schema(), include_metadata=False).transform_schema()

    assert result == "\n".join(
        [
            "extra*: dict[string, int]",
            "dict_of_models*:",
            "  <string>:",
            "    a*: int",
            "    b*: string",
            "by_color*: dict[red OR green, int]",
            "pair*: tuple[int, string]",
            "var_tuple*: list[int]",
            "tags*: list[string]",
            "anything*: any",
            "described*: any",
            "opt_any: any OR null",
            "any_list*: list[any]",
            "opt_extra: dict[string, int] OR null",
            "inner*:",
            "  d*: dict[string, int]",
            "  t*: tuple[int, string]",
            "strict*:  # no additional properties",
            "  s*: string",
        ]
    )


def test_yaml_formatter_does_not_mutate_caller_config():
    """D6: ``YAMLFormatter`` must not poison a ``FormatterConfig`` a sibling format reuses.

    HEAD rewrote ``config.union_separator`` in place, so a ``TypeScriptFormatter`` handed the
    same object afterwards emitted ``number OR string``.
    """
    from llm_schema_lite.formatters.typescript_formatter import TypeScriptFormatter

    config = FormatterConfig()
    assert config.union_separator == " | "

    YAMLFormatter(UnionTypes.model_json_schema(), config=config).transform_schema()

    assert config.union_separator == " | ", "YAMLFormatter mutated the caller's config"

    ts_result = TypeScriptFormatter(
        UnionTypes.model_json_schema(), config=config
    ).transform_schema()

    assert "number | string" in ts_result
    assert "number OR string" not in ts_result


def _conftest_base_models() -> list[type[BaseModel]]:
    """Every ``BaseModel`` defined at module level in ``tests/conftest.py``."""
    import inspect

    from tests import conftest

    return [
        obj
        for _, obj in sorted(vars(conftest).items())
        if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel
    ]


@pytest.mark.parametrize("include_metadata", [True, False], ids=["metadata_on", "metadata_off"])
def test_yaml_every_conftest_model_round_trips_through_safe_load(include_metadata: bool):
    """6.9 / A-D6: ``safe_load``-ability is a hard invariant, not a best effort.

    HEAD fails this for ``Root`` with metadata on (D8: the description's second paragraph is
    emitted as a bare, uncommented document line).
    """
    failures = []
    for model in _conftest_base_models():
        result = YAMLFormatter(
            model.model_json_schema(), include_metadata=include_metadata
        ).transform_schema()
        try:
            yaml.safe_load(result)
        except yaml.YAMLError as exc:  # pragma: no cover - only on regression
            failures.append(f"{model.__name__}: {exc}")

    assert not failures, "renders that failed yaml.safe_load:\n" + "\n".join(failures)


def test_yaml_render_contains_no_pyyaml_anchors():
    """6.4 / A-D12: the block builder bypasses ``_ref_cache``, so no ``&id001`` alias appears."""
    for model in _conftest_base_models():
        for include_metadata in (True, False):
            result = YAMLFormatter(
                model.model_json_schema(), include_metadata=include_metadata
            ).transform_schema()
            assert "&id0" not in result, f"{model.__name__} emitted a PyYAML anchor:\n{result}"
            assert "*id0" not in result, f"{model.__name__} emitted a PyYAML alias:\n{result}"


# ============================================================================
# lsl-2026-09-05-006 -- hash-in-pattern and multi-line descriptions
# ============================================================================


def _yaml(model: type[BaseModel], *, include_metadata: bool = True) -> str:
    """Render ``model`` as YAML for this section's assertions."""
    return YAMLFormatter(
        model.model_json_schema(), include_metadata=include_metadata
    ).transform_schema()


def test_yaml_hash_glued_pattern_renders_intact() -> None:
    """A ``#`` glued to a preceding character is never treated as a comment marker.

    Exact-line assertion, NOT routed through ``extract_comment_slots``: that helper is an
    assertion-side heuristic that cannot parse a ``#`` inside a quoted YAML scalar.
    """
    result = _yaml(HashInPatternModel)
    lines = result.split("\n")
    assert "tag*: 'string (PATTERN: ^#[0-9a-f]{6}$)'  # Hex colour" in lines
    yaml.safe_load(result)


def test_yaml_hash_glued_pattern_in_default_renders_intact() -> None:
    """A ``#`` inside a *default* value survives too (ticket Risk: defaults containing #)."""
    result = _yaml(HashInPatternModel)
    lines = result.split("\n")
    assert "shade: 'string (PATTERN: ^#[0-9a-f]{3,6}$)'  # (default='#ffffff')" in lines
    yaml.safe_load(result)


def test_yaml_whitespace_preceded_hash_in_pattern_renders_intact() -> None:
    """Research Q28's ``^a #b$``: a whitespace-preceded ``#`` inside a quoted scalar."""
    result = _yaml(HashInPatternModel)
    lines = result.split("\n")
    assert "spaced*: 'string (PATTERN: ^a #b$)'  # Whitespace before the hash" in lines
    yaml.safe_load(result)


def test_yaml_multiline_description_top_level_round_trips() -> None:
    """A top-level field's continuation line is a ``#`` comment at column 0."""
    result = _yaml(MultiLineDescriptionModel)
    lines = result.split("\n")
    assert "summary*: string  # line one" in lines
    assert "# line two" in lines
    assert "line two" not in [line.strip() for line in lines if not line.lstrip().startswith("#")]
    yaml.safe_load(result)


def test_yaml_multiline_description_nested_block_round_trips() -> None:
    """A nested block field's continuation line matches that field's own 2-space indent."""
    result = _yaml(MultiLineDescriptionModel)
    lines = result.split("\n")
    assert "  step*: string  # first step" in lines
    assert "  # second step" in lines
    yaml.safe_load(result)


def test_yaml_multiline_description_sequence_item_round_trips() -> None:
    """A ``- `` sequence item aligns its continuation past the dash, not to it."""
    result = _yaml(MultiLineDescriptionModel)
    lines = result.split("\n")
    assert "- step*: string  # first step" in lines
    assert "  # second step" in lines
    yaml.safe_load(result)


def test_yaml_multiline_description_emits_no_bare_document_line() -> None:
    """No physical line of the render is a bare, uncommented continuation fragment."""
    result = _yaml(MultiLineDescriptionModel)
    for line in result.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert ":" in stripped, f"bare document line in YAML render: {line!r}"


def test_yaml_multiline_description_whole_render_is_stable() -> None:
    """Whole-string golden for the three indent contexts in one model."""
    assert _yaml(MultiLineDescriptionModel).rstrip("\n") == (
        "# Description: Per-field descriptions containing real newlines "
        "(lsl-2026-09-05-006).\n"
        "\n"
        "# Fields marked with * are required\n"
        "\n"
        "summary*: string  # line one\n"
        "# line two\n"
        "nested*:\n"
        "  step*: string  # first step\n"
        "  # second step\n"
        "items*:\n"
        "- step*: string  # first step\n"
        "  # second step\n"
    ).rstrip("\n")


def test_yaml_multiline_description_metadata_off_is_plain() -> None:
    """With metadata off there is no comment at all and still no bare line."""
    result = _yaml(MultiLineDescriptionModel, include_metadata=False)
    assert "#" not in result
    yaml.safe_load(result)


# ============================================================================
# lsl-2026-09-05-006 -- pattern/format stated once per field (fix 3)
# ============================================================================


def _yaml_field_lines(result: str) -> list[str]:
    """Every physical line of ``result`` that renders a field (not a comment/blank)."""
    return [ln for ln in result.split("\n") if ln.strip() and not ln.lstrip().startswith("#")]


def test_yaml_pattern_is_not_restated_in_the_comment() -> None:
    """The token already renders ``(PATTERN: ...)``; the comment must not repeat it."""
    result = _yaml(PatternConstraints)
    assert "(PATTERN:" in result
    assert "pattern:" not in result


def test_yaml_format_is_not_restated_in_the_comment() -> None:
    """Same rule for ``format``, including the ``_format`` underscore variant."""
    for model in (StringFormatEmail, StringFormatUri):
        result = _yaml(model)
        assert "(FORMAT:" in result
        assert "format:" not in result


def test_yaml_pattern_appears_at_most_once_per_field_line() -> None:
    """3C acceptance: no field line states the same constraint keyword twice."""
    for model in (Address, PatternConstraints, StringPattern, User, HashInPatternModel):
        result = _yaml(model)
        for line in _yaml_field_lines(result):
            assert line.lower().count("pattern") <= 1, f"{model.__name__}: {line!r}"
            assert line.lower().count("format") <= 1, f"{model.__name__}: {line!r}"


def test_yaml_pattern_survives_on_a_non_string_typed_node() -> None:
    """Information-loss guard: a ``pattern`` NOT on a ``"type": "string"`` node still reports.

    ``_token_owned_metadata_keys`` gates on ``value.get("type") == "string"`` only, so a
    hand-authored schema without that key keeps its ``pattern:`` comment fragment.
    """
    schema: dict[str, Any] = {
        "type": "object",
        "title": "HandAuthored",
        "properties": {"code": {"pattern": r"^[A-Z]{3}$", "description": "Code"}},
        "required": ["code"],
    }
    result = YAMLFormatter(schema, include_metadata=True).transform_schema()
    assert "pattern: ^[A-Z]{3}$" in result


def test_yaml_metadata_off_is_unchanged_by_the_exclusion() -> None:
    """``include_metadata=False`` renders carry no constraint text at all, before or after."""
    for model in (PatternConstraints, StringFormatEmail):
        result = _yaml(model, include_metadata=False)
        assert "PATTERN:" not in result
        assert "pattern:" not in result
        assert "FORMAT:" not in result


def test_yaml_hash_pattern_model_whole_render_is_stable() -> None:
    """Whole-string golden: the ticket's headline fixture, all three bugs fixed."""
    assert _yaml(HashInPatternModel).rstrip("\n") == (
        "# Description: Regex patterns containing ``#`` (lsl-2026-09-05-006).\n"
        "#\n"
        "# Every field carries a SECOND metadata part on purpose: with pattern/format now owned\n"
        "# by the YAML type token, a pattern-only field mints no deferred marker at all and would\n"
        "# never enter ``_hoist_deferred_line``.\n"
        "\n"
        "# Fields marked with * are required\n"
        "\n"
        "tag*: 'string (PATTERN: ^#[0-9a-f]{6}$)'  # Hex colour\n"
        "shade: 'string (PATTERN: ^#[0-9a-f]{3,6}$)'  # (default='#ffffff')\n"
        "spaced*: 'string (PATTERN: ^a #b$)'  # Whitespace before the hash\n"
    ).rstrip("\n")


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
            "- name*: string  # Item name\n  qty: int  # Quantity, (default=1)",
        ),
        (
            Item | None,
            "# OR null\nname*: string  # Item name\nqty: int  # Quantity, (default=1)",
        ),
        (
            list[Item] | None,
            "# OR null\n- name*: string  # Item name\n  qty: int  # Quantity, (default=1)",
        ),
        (
            Node,
            "# Title: Node, Description: A tree node.\n"
            "\n"
            "# Fields marked with * are required\n"
            "\n"
            "value*: string  # Node value\n"
            "children:  # Child nodes\n"
            "- value*: string  # Node value\n"
            "  children: list[object]  # recursive: Node; Child nodes",
        ),
        (list[list[int]], "list[list[int]]"),
        (tuple[int, str], "tuple[int, string]"),
        (
            dict[str, Item],
            "<string>:\n  name*: string  # Item name\n  qty: int  # Quantity, (default=1)",
        ),
        (
            Item | str | None,
            "{name*: string, qty: int} OR string OR null  # Item name; Quantity, (default=1)",
        ),
    ],
)
def test_root_shape_golden(annotation: Any, expected: str) -> None:
    """Whole-string golden for a root-level shape, byte-for-byte against a measured run."""
    from llm_schema_lite import simplify_schema

    result = simplify_schema(TypeAdapter(annotation).json_schema(), format_type="yaml").to_string()

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
    from llm_schema_lite import simplify_schema

    root_render = simplify_schema(
        TypeAdapter(annotation).json_schema(), format_type="yaml"
    ).to_string()

    root_schema = TypeAdapter(annotation).json_schema()
    defs = root_schema.pop("$defs", None)
    nested = {"type": "object", "properties": {"f": root_schema}}
    if defs is not None:
        nested["$defs"] = defs
    nested_render = simplify_schema(nested, format_type="yaml").to_string()

    for line in root_render.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        normalized = stripped.rstrip(",")
        assert normalized in nested_render, f"{normalized!r} not found in nested render"


def test_list_of_optional_differs_from_optional_list() -> None:
    """AC: ``list[Item | None]`` and ``list[Item] | None`` must not collapse to one rendering.

    Reference values (design v2 §4.9):
      - ``list[Item | None]`` ->
        ``"list[{name*: string, qty: int} OR null]  # Item name; Quantity, (default=1)"``
      - ``list[Item] | None`` ->
        ``"# OR null\\n- name*: string  # Item name\\n  qty: int  # Quantity, (default=1)"``
    """
    from llm_schema_lite import simplify_schema

    render_a = simplify_schema(
        TypeAdapter(list[Item | None]).json_schema(), format_type="yaml"
    ).to_string()
    render_b = simplify_schema(
        TypeAdapter(list[Item] | None).json_schema(), format_type="yaml"
    ).to_string()

    assert render_a != render_b


def test_every_titled_root_is_decorated(
    all_pydantic_models: list[tuple[str, type[BaseModel]]],
) -> None:
    """The invariant that makes root cause #4 unrepeatable -- every root return path
    reaches ``root_decorations()``.
    """
    from llm_schema_lite import simplify_schema
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
        result = simplify_schema(model, format_type="yaml").to_string()
        assert result.startswith(f"# Title: {title}"), (
            f"{name}: expected render to start with '# Title: {title}', got {result[:80]!r}"
        )

    assert exercised > 0, "guard exercised zero models -- fixture or filter is broken"
