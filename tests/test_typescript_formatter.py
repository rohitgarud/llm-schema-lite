"""Tests for TypeScript formatter."""

from pydantic import BaseModel, Field

from llm_schema_lite.formatters.typescript_formatter import TypeScriptFormatter


class SimpleModel(BaseModel):
    """Simple test model."""

    name: str = Field(..., description="Name field")
    age: int = Field(..., ge=0, le=120)
    email: str | None = None


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

    class Address(BaseModel):
        """Address model."""

        street: str
        city: str

    class Person(BaseModel):
        """Person model."""

        name: str
        address: Address

    schema = Person.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert "interface Address {" in result
    assert "street*:" in result
    assert "city*:" in result
    assert "interface Schema {" in result
    assert "name*:" in result
    assert "address*:" in result


def test_typescript_formatter_key_order_preserved():
    """Test that TypeScript formatter preserves key order (dict order)."""

    class OrderedModel(BaseModel):
        """Model to test order preservation."""

        first: str
        second: int
        third: bool

    schema = OrderedModel.model_json_schema()
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

    class ConstrainedModel(BaseModel):
        """Model with constraints."""

        username: str = Field(..., min_length=3, max_length=20)
        score: int = Field(..., ge=0, le=100)

    schema = ConstrainedModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=True)
    result = formatter.transform_schema()

    assert "username*:" in result
    assert "score*:" in result
    # Constraints appear in comments or inline (e.g. "3-20 chars" or min/max)
    assert "3" in result and "20" in result
    assert "0" in result and "100" in result


def test_typescript_formatter_with_optional_union():
    """Test that TypeScript formatter handles optional fields (union with None)."""

    class OptionalModel(BaseModel):
        """Model with optional field."""

        required_field: str
        optional_field: str | None = None

    schema = OptionalModel.model_json_schema()
    formatter = TypeScriptFormatter(schema, include_metadata=False)
    result = formatter.transform_schema()

    assert "required_field*:" in result
    assert "optional_field:" in result
    assert "optional_field*:" not in result
