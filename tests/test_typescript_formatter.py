"""Tests for TypeScript formatter."""

from llm_schema_lite.formatters.typescript_formatter import TypeScriptFormatter

# Import models from conftest
from tests.conftest import (
    ConstrainedFormatterModel,
    OrderedFieldsModel,
    PersonWithAddress,
    RequiredOptionalModel,
    SimpleFormatterModel,
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
    assert "name*:" in result
    assert "age*:" in result
    assert "email:" in result
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
    assert "name*:" in result
    assert "age*:" in result
    assert "email:" in result
    # Description metadata should not appear as comment
    assert "Name field" not in result


def test_typescript_formatter_with_nested_defs():
    """Test TypeScript formatter with nested $defs."""
    schema = PersonWithAddress.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert "interface Address {" in result
    assert "street*:" in result
    assert "city*:" in result
    assert "interface Schema {" in result or "interface PersonWithAddress {" in result
    assert "name*:" in result
    assert "address*:" in result


def test_typescript_formatter_key_order_preserved():
    """Test that TypeScript formatter preserves key order (dict order)."""
    schema = OrderedFieldsModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    first_pos = result.find("first*:")
    second_pos = result.find("second*:")
    third_pos = result.find("third*:")

    assert first_pos != -1
    assert second_pos != -1
    assert third_pos != -1
    assert first_pos < second_pos < third_pos


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
    assert "name*:" in result
    assert "age*:" in result
    assert "score*:" in result
    # Constraints appear in comments or inline (e.g. min/max values)
    assert "1" in result and "100" in result  # name length and age max
    assert "0" in result and "150" in result  # age range


def test_typescript_formatter_with_optional_union():
    """Test that TypeScript formatter handles optional fields (union with None)."""
    schema = RequiredOptionalModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert "required_field*:" in result
    assert "optional_field:" in result
    assert "optional_field*:" not in result
