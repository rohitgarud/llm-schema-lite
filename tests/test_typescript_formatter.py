"""Tests for TypeScript formatter."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from pydantic import BaseModel, Field, TypeAdapter

from llm_schema_lite import FormatterConfig
from llm_schema_lite.formatters.typescript_formatter import TypeScriptFormatter
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
    IntEnumModel,
    LiteralSingle,
    LiteralUnion,
    ModelWithAlias,
    MultiLineDescriptionModel,
    ObjectAdditionalPropsFalse,
    ObjectRequiredOnly,
    ObjectWithDefaults,
    OrderedFieldsModel,
    PatternConstraints,
    PersonWithAddress,
    Product,
    RequiredOptionalModel,
    Role,
    Root,
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
    assert "age*: number (0 to 150)" in result
    assert re.search(
        r"score\*:\s*number\s*\(0(\.0)? to 100(\.0)?\)", result
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


def test_no_auto_generated_property_titles_typescript():
    """AC-1: TypeScript drops auto-generated per-field titles from the output.

    At HEAD, TypeScript renders no field titles at all (no `title` key in
    METADATA_MAP), so this is already green; it becomes load-bearing once Phase 4
    adds title rendering.
    """
    for model in (Address, SimpleFormatterModel, ModelWithAlias):
        schema = model.model_json_schema()
        formatter = TypeScriptFormatter(schema, include_metadata=True)
        result = formatter.transform_schema()

        for key in schema.get("properties", {}):
            assert f"{formatter.comment_prefix} {auto_title_for_key(key)}:" not in result


def test_user_supplied_title_and_description_render_once_typescript():
    """AC-2: a user-supplied field title and description each render exactly once.

    New capability: TypeScript emits no field titles at all until
    ``METADATA_MAP["title"]`` exists.
    """
    schema = WithFieldDescriptions.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert result.count("Full Name") == 1
    assert result.count("The user's full name") == 1


def test_metadata_inclusion_title_false_suppresses_title_typescript():
    """The ``title`` metadata_inclusion lever suppresses field titles, not descriptions."""
    schema = WithFieldDescriptions.model_json_schema()
    config = FormatterConfig(include_metadata=True, metadata_inclusion={"title": False})
    formatter = TypeScriptFormatter(schema, config=config)
    result = formatter.transform_schema()

    assert "Full Name" not in result
    assert "The user's full name" in result


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


def test_typescript_enum_with_descriptions_and_aliases():
    """Test TypeScript formatter shows OPTIONS with descriptions in comment."""
    from llm_schema_lite import simplify_schema
    from tests.conftest import ModelWithPriorityMetadata

    result = simplify_schema(ModelWithPriorityMetadata, format_type="typescript").to_string()
    assert "OPTIONS with descriptions" in result
    assert "Non-urgent" in result or "Urgent" in result
    assert "aliases:" in result
    assert "critical" in result


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
    assert "Array<object>" not in result


# ============================================================================
# Nested and Complex Types
# ============================================================================


def test_typescript_nested_optional_ref_renders_as_inline_object() -> None:
    """A nested optional `$ref` renders as an inline TS object literal, not a dict repr."""
    from llm_schema_lite import simplify_schema

    class Addr(BaseModel):
        street: str = Field(..., description="Street")
        city: str = Field(..., description="City")

    class P1(BaseModel):
        name: str = Field(..., description="Patient name")
        address: Addr | None = Field(None, description="Home address")

    expected = "\n".join(
        [
            "interface Addr {",
            "  street*: string  // Street;",
            "  city*: string  // City;",
            "}",
            "",
            "// Title: P1",
            "// Fields marked with * are required",
            "interface Schema {",
            "  name*: string  // Patient name;",
            (
                "  address: { street*: string /* Street */, "
                "city*: string /* City */ } | null  // Home address;"
            ),
            "}",
        ]
    )

    assert simplify_schema(P1, format_type="typescript").to_string() == expected


def test_typescript_nested_list_ref_renders_as_array_of_object() -> None:
    """A list of `$ref` objects renders as `Array<{ ... }>`, not `Array<object>`."""
    from llm_schema_lite import simplify_schema

    class Addr(BaseModel):
        street: str = Field(..., description="Street")
        city: str = Field(..., description="City")

    class P2(BaseModel):
        name: str = Field(..., description="Patient name")
        addresses: list[Addr] = Field(..., description="Addresses")

    expected = "\n".join(
        [
            "interface Addr {",
            "  street*: string  // Street;",
            "  city*: string  // City;",
            "}",
            "",
            "// Title: P2",
            "// Fields marked with * are required",
            "interface Schema {",
            "  name*: string  // Patient name;",
            (
                "  addresses*: Array<{ street*: string /* Street */, "
                "city*: string /* City */ }>  // Addresses;"
            ),
            "}",
        ]
    )

    assert simplify_schema(P2, format_type="typescript").to_string() == expected


def test_typescript_nested_optional_list_ref_renders_as_array_of_object() -> None:
    """An optional list of `$ref` objects renders as `Array<{ ... }> | null`."""
    from llm_schema_lite import simplify_schema

    class Addr(BaseModel):
        street: str = Field(..., description="Street")
        city: str = Field(..., description="City")

    class P3(BaseModel):
        name: str = Field(..., description="Patient name")
        addresses: list[Addr] | None = Field(None, description="Addresses")

    expected = "\n".join(
        [
            "interface Addr {",
            "  street*: string  // Street;",
            "  city*: string  // City;",
            "}",
            "",
            "// Title: P3",
            "// Fields marked with * are required",
            "interface Schema {",
            "  name*: string  // Patient name;",
            (
                "  addresses: Array<{ street*: string /* Street */, "
                "city*: string /* City */ }> | null  // Addresses;"
            ),
            "}",
        ]
    )

    assert simplify_schema(P3, format_type="typescript").to_string() == expected


def test_typescript_nested_required_ref_renders_as_inline_object() -> None:
    """A required nested `$ref` renders as an inline TS object literal."""
    from llm_schema_lite import simplify_schema

    class Addr(BaseModel):
        street: str = Field(..., description="Street")
        city: str = Field(..., description="City")

    class P4(BaseModel):
        name: str = Field(..., description="Patient name")
        address: Addr = Field(..., description="Home address")

    expected = "\n".join(
        [
            "interface Addr {",
            "  street*: string  // Street;",
            "  city*: string  // City;",
            "}",
            "",
            "// Title: P4",
            "// Fields marked with * are required",
            "interface Schema {",
            "  name*: string  // Patient name;",
            (
                "  address*: { street*: string /* Street */, "
                "city*: string /* City */ }  // Home address;"
            ),
            "}",
        ]
    )

    assert simplify_schema(P4, format_type="typescript").to_string() == expected


def test_typescript_nested_ref_two_sibling_fields_both_inline() -> None:
    """Two sibling fields sharing a `$ref` type both render inline (AC3).

    Before the fingerprint-discard fix, the second sibling (`work`) degraded to the
    bare `object` type instead of the full inline object literal the first sibling
    (`home`) gets.
    """
    from enum import Enum

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

    expected = "\n".join(
        [
            "interface Address {",
            "  street*: string  // Street;",
            "  city*: string  // City;",
            '  country*: "US" | "CA"  // Country code;',
            "}",
            "",
            "// Title: TwoSiblings",
            "// Fields marked with * are required",
            "interface Schema {",
            (
                "  home*: { street*: string /* Street */, city*: string /* City */, "
                'country*: "US" | "CA" /* Country code */ }  // Home;'
            ),
            (
                "  work*: { street*: string /* Street */, city*: string /* City */, "
                'country*: "US" | "CA" /* Country code */ }  // Work;'
            ),
            "}",
        ]
    )

    assert simplify_schema(TwoSiblings, format_type="typescript").to_string() == expected


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

    # Should show a real TypeScript index signature, not the old non-TS <key> placeholder.
    assert "[key: string]" in result


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


def test_no_empty_comment_marker_across_all_models_typescript(all_pydantic_models) -> None:
    """No rendered TypeScript line should carry an empty `// ;` / bare `//` marker.

    ``add_metadata`` used to emit a `// <parts>` comment whenever
    ``filtered_metadata`` was non-empty, even if every entry in ``format_metadata_parts``
    resolved to an empty string (e.g. min/max skipped for the field's own type) --
    producing lines like `age*: number (>=0)  // ;`. Sweeps every registered model.
    """
    for _name, model in all_pydantic_models:
        schema = model.model_json_schema()
        formatter = TypeScriptFormatter(schema, include_metadata=True)
        result = formatter.transform_schema()

        for line in result.splitlines():
            assert "// ;" not in line, f"{_name}: empty `// ;` marker in line: {line!r}"
            assert not line.endswith("// "), f"{_name}: trailing bare `// ` in line: {line!r}"
            assert not line.endswith("//"), f"{_name}: trailing bare `//` in line: {line!r}"


def test_product_typescript_token_count_decreases() -> None:
    """Dropping auto-generated titles shrinks the TypeScript token count for ``Product``.

    Anchored on ``Product``, not ``Order``: ``Order``'s TypeScript token count *rises*
    790 -> 792 because ``tests/conftest.py``'s ``User.name`` carries a user-supplied
    ``title="Full Name"`` that TypeScript now correctly renders (this is AC-2's new
    capability, not a regression -- TypeScript previously had no ``"title"`` entry in
    its ``METADATA_MAP`` at all). ``Product`` is a submodel of the same ``Order`` gist
    with no user-supplied field titles, so its TS token count genuinely falls, HEAD
    109 -> expected 105. The assertion is strict-lower-than-HEAD rather than pinned to
    the exact number so unrelated future formatting tweaks do not spuriously fail this
    test.
    """
    pytest.importorskip("tiktoken")
    from llm_schema_lite import simplify_schema

    assert simplify_schema(Product, format_type="typescript").token_count() < 109


def test_typescript_inline_comment_splits_on_add_metadata_separator() -> None:
    """`_inline_comment` rewrites a trailing `// ...` line comment as a block comment.

    A `//` comment inside a single-line inline object literal would swallow the
    remainder of the line, including the closing brace and every later field.
    """
    assert TypeScriptFormatter._inline_comment("string  // Street") == "string /* Street */"
    assert TypeScriptFormatter._inline_comment("string") == "string"
    assert (
        TypeScriptFormatter._inline_comment("number  // min: 0, max: 5")
        == "number /* min: 0, max: 5 */"
    )
    # a description containing a literal */ must not terminate the block comment early
    assert (
        TypeScriptFormatter._inline_comment("string  // ends with */ here")
        == "string /* ends with * / here */"
    )


def test_typescript_format_field_name_consults_nested_required_stack() -> None:
    """`format_field_name` honours the innermost `$defs` required set when one is live."""
    formatter = TypeScriptFormatter({"type": "object", "properties": {}, "required": ["root_only"]})
    formatter._nested_required_stack.append({"street"})
    assert formatter.format_field_name("street") == "street" + formatter.config.required_marker
    assert (
        formatter.format_field_name("root_only") == "root_only" + formatter.config.optional_marker
    )
    formatter._nested_required_stack.pop()
    assert (
        formatter.format_field_name("root_only") == "root_only" + formatter.config.required_marker
    )


def test_no_array_of_object_across_all_models_typescript(all_pydantic_models) -> None:
    """No registered model's TypeScript output degrades a `$ref` to `Array<object>`.

    Cross-cutting invariant over every model in the shared fixture registry: a `$ref`
    reached through a list slot must render as `Array<{ ... }>`, never as the bare
    `Array<object>` type or a hand-built `'k': 'v'` dict repr.
    """
    for name, model in all_pydantic_models:
        result = TypeScriptFormatter(
            model.model_json_schema(), include_metadata=True
        ).transform_schema()
        assert "Array<object>" not in result, f"{name}: Array<object> in TS output"
        assert "': '" not in result, f"{name}: quoted dict repr in TS output"


def test_cyclic_and_mutually_recursive_refs_terminate() -> None:
    """The fingerprint-discard fix must not turn a cycle into unbounded expansion."""
    from llm_schema_lite.formatters.jsonish_formatter import JSONishFormatter
    from llm_schema_lite.formatters.yaml_formatter import YAMLFormatter

    cyclic = {
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node"}},
        "required": ["root"],
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "child": {"$ref": "#/$defs/Node"},
                },
                "required": ["name"],
            }
        },
    }
    mutual = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/A"}},
        "$defs": {
            "A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}},
            "B": {"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}},
        },
    }
    for schema in (cyclic, mutual):
        for cls in (TypeScriptFormatter, YAMLFormatter, JSONishFormatter):
            out = cls(schema).transform_schema()
            assert "object" in out  # truncated at the cycle boundary, not expanded
            assert "recursive:" in out  # and says so, rather than dropping the type
            # Worst case measured at the default depth of 2 is 250 chars; growth
            # is linear (d3 -> 302, d4 -> 354, d5 -> 421), so this bound fails
            # loudly if the shipped default ever drifts upward.
            assert len(out) < 400  # terminates rather than growing without bound


# ============================================================================
# Container types (dict / tuple / set / Any) -- lsl-2026-09-04-015
# ============================================================================


def test_typescript_formatter_root_fixture_default():
    """Whole-string golden for the container-types Root fixture (default config).

    NOTE (Phase 5b golden deviation, recorded per the plan's golden-block mismatch
    protocol): the design v2 5.2 AFTER block was captured against a ``Root`` trio with
    **no docstrings**, so it shows a plain ``// Title: Root`` header. The Phase 4
    conftest fixture's ``Root`` carries a real docstring, which pydantic turns into
    ``description``; `schema_normalization._strip_model_title` (untouched,
    out-of-scope) then drops the now-redundant auto-generated ``title`` whenever a
    non-empty ``description`` is present, so `BaseFormatter.get_schema_info_comment`
    emits ``// Description: <docstring>`` instead of ``// Title: Root`` -- the
    docstring's own blank line becomes a literal blank line inside the header. This is
    the same fixture artifact P5a documented for JSONish; it is unrelated to container
    rendering. Expected (design): ``// Title: Root``. Actual: ``// Description:
    Kitchen-sink fixture for lsl-2026-09-04-015 (dict/tuple/set/Any container
    rendering).`` + a blank line + ``Fields verbatim from the approved design
    (2026-09-04-design-discussion-v2.md 5.1).``. Every container token below this
    header is the design's verbatim.
    """
    result = TypeScriptFormatter(Root.model_json_schema()).transform_schema()

    expected = "\n".join(
        [
            "interface Inner {",
            "  d*: Record<string, number>;",
            "  t*: [number, string];",
            "}",
            "",
            "interface Strict {",
            "  s*: string;",
            "}",
            " // no additional properties",
            "",
            "interface SubModel {",
            "  a*: number;",
            "  b*: string;",
            "}",
            "",
            "// Description: Kitchen-sink fixture for lsl-2026-09-04-015 "
            "(dict/tuple/set/Any container rendering).",
            "",
            "Fields verbatim from the approved design (2026-09-04-design-discussion-v2.md 5.1).",
            "// Fields marked with * are required",
            "interface Schema {",
            "  extra*: Record<string, number>;",
            "  dict_of_models*: Record<string, { a*: number, b*: string }>;",
            "  by_color*: Record<red | green, number>;",
            "  pair*: [number, string];",
            "  var_tuple*: Array<number>;",
            "  tags*: Array<string> (unique);",
            "  anything*: any;",
            "  described*: any  // free form;",
            "  opt_any: any | null;",
            "  any_list*: Array<any>;",
            "  opt_extra: Record<string, number> | null;",
            "  inner*: { d*: Record<string, number>, t*: [number, string] };",
            "  strict*: { s*: string };",
            "}",
        ]
    )

    assert result == expected


def test_typescript_formatter_root_fixture_include_metadata_false():
    """With metadata off, constraint/title comments drop but structure survives."""
    formatter = TypeScriptFormatter(Root.model_json_schema(), include_metadata=False)
    result = formatter.transform_schema()

    assert "tags*: Array<string>;" in result
    assert "described*: any;" in result
    assert "// Title:" not in result


def test_typescript_formatter_root_fixture_metadata_inclusion_variant():
    """Disabling uniqueItems/minItems/maxItems metadata drops the constraint, keeps structure."""
    config = FormatterConfig(
        metadata_inclusion={"uniqueItems": False, "minItems": False, "maxItems": False}
    )
    formatter = TypeScriptFormatter(Root.model_json_schema(), config=config)
    result = formatter.transform_schema()

    assert "tags*: Array<string>;" in result
    assert "[number, string]" in result


def test_typescript_formatter_root_fixture_no_forbidden_substrings():
    """AC-1: no leaked JSONish/base vocabulary, and the tags line is not truncated by `//`."""
    result = TypeScriptFormatter(Root.model_json_schema()).transform_schema()

    assert "additional:" not in result
    assert "length:" not in result

    tags_line = next(line for line in result.splitlines() if "tags" in line)
    assert tags_line.rstrip().endswith(";")


def test_typescript_formatter_any_never_renders_as_string():
    """AC-2: a bare Any (and its described/titled variants) never render as `string`."""

    class _AnyVariants(BaseModel):
        plain: Any
        described: Any = Field(..., description="free form")
        titled: Any = Field(..., title="Custom Title")

    result = TypeScriptFormatter(_AnyVariants.model_json_schema()).transform_schema()

    for field_name in ("plain", "described", "titled"):
        line = next(line for line in result.splitlines() if line.strip().startswith(field_name))
        assert "string" not in line


def test_typescript_formatter_dict_of_model_with_described_field_not_split_mid_generic():
    """Design §10 risk 6: a `//` comment inside a mapping's value must not truncate the generic."""

    class _Described(BaseModel):
        x: int = Field(..., description="a described field")

    class _Container(BaseModel):
        m: dict[str, _Described]

    result = TypeScriptFormatter(_Container.model_json_schema()).transform_schema()

    m_line = next(line for line in result.splitlines() if line.strip().startswith("m"))
    assert "/* a described field */" in m_line
    assert m_line.strip().endswith(">;")


def test_typescript_formatter_draft7_tuple_items_no_crash():
    """Behaviour-change pin: HEAD rendered ``Array<any>`` for a draft-7 tuple."""
    result = TypeScriptFormatter(ADDITIONAL_ITEMS_SCHEMA).transform_schema()

    assert result == "type Schema = [string, number];"


def test_typescript_formatter_prefix_items_tuple_with_variadic_tail():
    """``prefixItems`` plus an ``items`` tail renders as a TS variadic tuple."""
    result = TypeScriptFormatter(PREFIX_ITEMS_SCHEMA).transform_schema()

    assert result == "type Schema = [string, number, boolean, ...string[]];"


# ============================================================================
# Recursive models -- lsl-2026-09-04-014
# ============================================================================


class _RecursiveListNode(BaseModel):
    """Self-referencing model used by the recursion goldens below."""

    label: str
    kids: list[_RecursiveListNode] = Field(default_factory=list)


_RecursiveListNode.model_rebuild()


def _list_node_schema() -> dict[str, Any]:
    """Return a root-``$ref`` JSON schema whose def is named ``ListNode``."""
    schema = _RecursiveListNode.model_json_schema()
    return json.loads(json.dumps(schema).replace("_RecursiveListNode", "ListNode"))


def test_typescript_recursive_list_golden_default_depth() -> None:
    """Whole-string golden for a recursive list model at the default depth of 2."""
    result = TypeScriptFormatter(_list_node_schema()).transform_schema()

    assert result == (
        "// Title: ListNode, Description: Self-referencing model used by the recursion"
        " goldens below.\n"
        "// Fields marked with * are required\n"
        "interface Schema {\n"
        "  label*: string;\n"
        "  kids: Array<{ label*: string, "
        "kids: Array<object /* recursive: ListNode */> }>;\n"
        "}"
    )


def test_typescript_root_ref_model_renders_interface_body() -> None:
    """D4: a bare root ``$ref`` must not collapse to ``interface Schema {}``."""
    result = TypeScriptFormatter(_list_node_schema()).transform_schema()

    assert result != "interface Schema {}"
    assert "label" in result


def test_typescript_recursive_render_is_idempotent() -> None:
    """C3: the cached ``_processed_data`` branch must seed ``$defs`` like the main flow."""
    formatter = TypeScriptFormatter(_list_node_schema())

    first = formatter.transform_schema()
    second = formatter.transform_schema()

    assert first == second


def test_typescript_recursive_placeholder_preserves_array_of_object_invariant() -> None:
    """The placeholder sits between ``object`` and ``>``, so ``Array<object>`` never appears."""
    for depth in (1, 2, 3):
        config = FormatterConfig(max_recursion_depth=depth)
        result = TypeScriptFormatter(_list_node_schema(), config=config).transform_schema()

        assert "recursive: ListNode" in result, f"depth={depth}: no placeholder"
        assert "Array<object>" not in result, f"depth={depth}: degraded to Array<object>"


# ============================================================================
# lsl-2026-09-05-006 -- multi-line descriptions
# ============================================================================


def test_typescript_multiline_description_interface_member_continuation() -> None:
    """First line inline, rest as ``// `` comments at the fixed 2-space member indent."""
    result = TypeScriptFormatter(
        MultiLineDescriptionModel.model_json_schema(), include_metadata=True
    ).transform_schema()
    lines = result.split("\n")
    assert "  summary*: string  // line one" in lines
    assert "  // line two;" in lines


def test_typescript_multiline_description_emits_no_bare_document_line() -> None:
    """No physical line is a bare continuation fragment outside a comment."""
    result = TypeScriptFormatter(
        MultiLineDescriptionModel.model_json_schema(), include_metadata=True
    ).transform_schema()
    for line in result.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped in {"{", "}"}:
            continue
        assert ":" in stripped or stripped.startswith(
            "interface"
        ), f"bare document line in TypeScript render: {line!r}"


def test_typescript_multiline_description_nested_inline_literal_folds_to_block_comment() -> None:
    """A nested inline object literal folds to one ``/* a b */`` with no leaked ``//``."""
    result = TypeScriptFormatter(
        MultiLineDescriptionModel.model_json_schema(), include_metadata=True
    ).transform_schema()
    lines = result.split("\n")
    assert "  nested*: { step*: string /* first step second step */ };" in lines
    assert "  items*: Array<{ step*: string /* first step second step */ }>;" in lines
    for line in lines:
        if "/*" in line:
            block = line.split("/*", 1)[1].split("*/", 1)[0]
            assert "//" not in block, f"leaked // inside a block comment: {line!r}"


def test_typescript_multiline_description_whole_render_is_stable() -> None:
    """Whole-string golden for the interface-member and inline-literal shapes."""
    result = TypeScriptFormatter(
        MultiLineDescriptionModel.model_json_schema(), include_metadata=True
    ).transform_schema()
    assert "interface MultiLineDescriptionInner {" in result
    assert "  step*: string  // first step" in result
    assert "  // second step;" in result


def test_typescript_inline_comment_folds_continuation_lines() -> None:
    """``_inline_comment`` unit: the ``\n  // `` break becomes a single space."""
    assert (
        TypeScriptFormatter._inline_comment("  z*: string  // a\n  // b")
        == "  z*: string /* a b */"
    )
    assert (
        TypeScriptFormatter._inline_comment("  z*: string  // a\n  // b\n  // c")
        == "  z*: string /* a b c */"
    )


def test_typescript_inline_comment_escapes_star_slash_after_folding() -> None:
    """Ordering pin: the fold runs first, then the ``*/`` escape; both invariants hold."""
    assert (
        TypeScriptFormatter._inline_comment("  z*: string  // a */ b\n  // c")
        == "  z*: string /* a * / b c */"
    )


def test_typescript_inline_comment_passes_through_without_a_comment() -> None:
    """No ``  // `` separator means the value is returned unchanged."""
    assert TypeScriptFormatter._inline_comment("plain: string") == "plain: string"


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
            "type Schema = Array<{ name*: string /* Item name */,"
            " qty: number /* (defaults to 1), Quantity */ }>;",
        ),
        (
            Item | None,
            "type Schema = { name*: string /* Item name */,"
            " qty: number /* (defaults to 1), Quantity */ } | null;",
        ),
        (
            list[Item] | None,
            "type Schema = Array<{ name*: string /* Item name */,"
            " qty: number /* (defaults to 1), Quantity */ }> | null;",
        ),
        (
            Node,
            "// Title: Node, Description: A tree node.\n"
            "// Fields marked with * are required\n"
            "interface Schema {\n"
            "  value*: string  // Node value;\n"
            "  children: Array<{ value*: string /* Node value */,"
            " children: Array<object /* recursive: Node */> /* Child nodes */ }>"
            "  // Child nodes;\n"
            "}",
        ),
        (list[list[int]], "type Schema = Array<Array<number>>;"),
        (tuple[int, str], "type Schema = [number, string];"),
        (
            dict[str, Item],
            "interface Schema {\n"
            "  [key: string]: { name*: string /* Item name */,"
            " qty: number /* (defaults to 1), Quantity */ };\n"
            "}",
        ),
        (
            Item | str | None,
            "type Schema = { name*: string /* Item name */,"
            " qty: number /* (defaults to 1), Quantity */ } | string | null;",
        ),
    ],
)
def test_root_shape_golden(annotation: Any, expected: str) -> None:
    """Whole-string golden for a root-level shape, byte-for-byte against a measured run."""
    from llm_schema_lite import simplify_schema

    result = simplify_schema(
        TypeAdapter(annotation).json_schema(), format_type="typescript"
    ).to_string()

    assert result == expected


def test_list_of_optional_differs_from_optional_list() -> None:
    """AC: ``list[Item | None]`` and ``list[Item] | None`` must not collapse to one rendering."""
    from llm_schema_lite import simplify_schema

    render_a = simplify_schema(
        TypeAdapter(list[Item | None]).json_schema(), format_type="typescript"
    ).to_string()
    render_b = simplify_schema(
        TypeAdapter(list[Item] | None).json_schema(), format_type="typescript"
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
        result = simplify_schema(model, format_type="typescript").to_string()
        assert result.startswith(
            f"// Title: {title}"
        ), f"{name}: expected render to start with '// Title: {title}', got {result[:80]!r}"

    assert exercised > 0, "guard exercised zero models -- fixture or filter is broken"


@pytest.mark.xfail(
    strict=True,
    reason="FU-3: TypeScript's non-object-root branch stays undecorated by design "
    "(design v2 §6.8) -- Pydantic never emits title/required on a non-object root and "
    "the DSPy adapter never selects TypeScript.",
)
def test_root_array_title_is_rendered() -> None:
    from llm_schema_lite import simplify_schema

    schema = {
        "type": "array",
        "title": "Bag",
        "description": "A bag of items.",
        "items": {"type": "string"},
    }
    result = simplify_schema(schema, format_type="typescript").to_string()
    assert "// Title: Bag" in result
