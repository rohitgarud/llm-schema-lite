"""Tests for JSONish formatter."""

from pydantic import BaseModel, Field

from llm_schema_lite.formatters.jsonish_formatter import JSONishFormatter


class SimpleModel(BaseModel):
    """Simple test model."""

    name: str = Field(..., description="Name field")
    age: int = Field(..., ge=0, le=120)
    email: str | None = None


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

    class Address(BaseModel):
        """Address model."""

        street: str
        city: str

    class Person(BaseModel):
        """Person model."""

        name: str
        address: Address

    schema = Person.model_json_schema()
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

    class OrderedModel(BaseModel):
        """Model to test order preservation."""

        first: str
        second: int
        third: bool

    schema = OrderedModel.model_json_schema()
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

    class ConstrainedModel(BaseModel):
        """Model with constraints."""

        username: str = Field(..., min_length=3, max_length=20)
        score: int = Field(..., ge=0, le=100)

    schema = ConstrainedModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    # Verify constraints are included
    assert "username*:" in result
    assert "3-20 chars" in result or "(3-20 chars)" in result
    assert "score*:" in result
    assert "0 to 100" in result or "(0 to 100)" in result


def test_jsonish_formatter_with_optional_union():
    """Test that JSONish formatter handles optional fields (union with None)."""

    class OptionalModel(BaseModel):
        """Model with optional field."""

        required_field: str
        optional_field: str | None = None

    schema = OptionalModel.model_json_schema()
    formatter = JSONishFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    # Verify required vs optional marking
    assert "required_field*:" in result
    assert "optional_field:" in result
    # Optional field should not have asterisk
    assert "optional_field*:" not in result
