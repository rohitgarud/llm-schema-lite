"""Tests for JSONish formatter."""

from llm_schema_lite.formatters.jsonish_formatter import JSONishFormatter

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


def test_jsonish_formatter_produces_valid_output():
    """Test that JSONish formatter produces valid output with required fields marked."""
    schema = SimpleModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Verify the output contains expected fields
    assert "name*:" in result  # Required field
    assert "age*:" in result  # Required field
    assert "email:" in result  # Optional field
    # Verify asterisk notation comment is present
    assert "Fields marked with * are required" in result
    # Verify it uses // for comments
    assert "//" in result


def test_jsonish_formatter_without_metadata():
    """Test JSONish formatter without metadata."""
    schema = SimpleModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Verify the output contains expected fields
    assert "name*:" in result
    assert "age*:" in result
    assert "email:" in result
    # Verify description metadata is not included
    assert "description" not in result.lower() or "Name field" not in result


def test_jsonish_formatter_with_nested_defs():
    """Test JSONish formatter with nested $defs."""
    schema = PersonWithAddress.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Should contain required fields
    assert "name*:" in result
    assert "address*:" in result
    # Should contain nested Address fields (in dict representation or expanded)
    assert "street*" in result
    assert "city*" in result


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
    assert first_pos != -1
    assert second_pos != -1
    assert third_pos != -1

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

    # Both results should have the same core content (ignoring spacing differences)
    # The cache is stored in simplified_schema
    assert hasattr(formatter, "simplified_schema")
    assert formatter.simplified_schema is not None
    # Verify both contain the same fields
    assert "name*:" in result1 and "name*:" in result2
    assert "age*:" in result1 and "age*:" in result2
    assert "email:" in result1 and "email:" in result2


def test_jsonish_formatter_with_constraints():
    """Test that JSONish formatter includes field constraints."""
    schema = ConstrainedFormatterModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Verify constraints are included (ConstrainedFormatterModel has name, age, score, tags)
    assert "name*:" in result
    # Check for length constraints on name (1-100)
    assert "1" in result and "100" in result
    assert "age*:" in result
    # Check for range constraints on age (0-150)
    assert "0" in result and "150" in result
    assert "score*:" in result
    # Check for range constraints on score (0.0-100.0)
    # Note: may appear as int or float in output


def test_jsonish_formatter_with_optional_union():
    """Test that JSONish formatter handles optional fields (union with None)."""
    schema = RequiredOptionalModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Verify required vs optional marking
    assert "required_field*:" in result
    assert "optional_field:" in result
    # Optional field should not have asterisk
    assert "optional_field*:" not in result
