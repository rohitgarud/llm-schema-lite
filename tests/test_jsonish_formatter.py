"""Tests for JSONish formatter."""

from llm_schema_lite import simplify_schema
from llm_schema_lite.formatters import JSONishFormatter

from .fixtures import (
    Address,
    ComplexTypes,
    ConfigModel,
    Order,
    Profile,
    SimpleUser,
    User,
)


class TestJSONishFormatterBasic:
    """Test basic JSONish formatter functionality."""

    def test_simple_model(self):
        """Test JSONish output for simple model."""
        schema = simplify_schema(SimpleUser, format_type="jsonish", include_metadata=False)
        output = schema.to_string()

        assert "{" in output
        assert "name: string" in output
        assert "age: int" in output
        assert "email: string" in output

    def test_simple_model_with_metadata(self):
        """Test JSONish output with metadata."""
        schema = simplify_schema(User, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Check for metadata comments
        assert "//" in output
        assert "The user's full name" in output
        assert "maxLength: 100" in output
        assert "User role" in output

    def test_without_metadata(self):
        """Test JSONish output without metadata."""
        schema = simplify_schema(User, format_type="jsonish", include_metadata=False)
        output = schema.to_string()

        # Note: Enum values (//oneOf:) are part of the type info, not metadata
        # So they may still appear even with include_metadata=False
        # Real metadata like descriptions, min/max should not be present
        assert "The user's full name" not in output
        assert "maxLength: 100" not in output

    def test_enum_handling(self):
        """Test enum field handling."""
        schema_dict = {"type": "string", "enum": ["admin", "user", "guest"]}
        formatter = JSONishFormatter({"properties": {"role": schema_dict}})

        output = formatter.process_enum(schema_dict)
        assert "oneOf" in output
        assert "admin" in output
        assert "user" in output
        assert "guest" in output


class TestJSONishFormatterNested:
    """Test JSONish formatter with nested structures."""

    def test_nested_model(self):
        """Test nested model handling."""
        schema = simplify_schema(User, format_type="jsonish", include_metadata=False)
        output = schema.to_string()

        # Check for nested structures
        assert "addresses" in output
        assert "contact_info" in output
        assert "street: string" in output
        assert "city: string" in output

    def test_complex_order_model(self):
        """Test complex Order model from gist."""
        schema = simplify_schema(Order, format_type="jsonish", include_metadata=False)
        output = schema.to_string()

        assert "order_id: int" in output
        assert "user:" in output
        assert "products:" in output
        assert "total_price: float" in output


class TestJSONishFormatterArrays:
    """Test array/list handling."""

    def test_string_array(self):
        """Test string array formatting."""
        schema = simplify_schema(ComplexTypes, format_type="jsonish", include_metadata=False)
        output = schema.to_string()

        assert "string[]" in output or "string_list: string[]" in output

    def test_nested_array(self):
        """Test nested object arrays."""
        schema = simplify_schema(User, format_type="jsonish", include_metadata=False)
        output = schema.to_string()

        # addresses is a list of Address objects
        assert "addresses:" in output


class TestJSONishFormatterOptional:
    """Test optional field handling."""

    def test_optional_fields(self):
        """Test optional/union type fields."""
        schema = simplify_schema(Profile, format_type="jsonish", include_metadata=False)
        output = schema.to_string()

        # Should handle optional fields (str | None becomes "string or null")
        assert "name:" in output
        assert "email:" in output
        assert "age: int" in output

    def test_optional_with_none(self):
        """Test optional fields with None default."""
        schema = simplify_schema(ConfigModel, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Check for default values in metadata
        assert "(defaults to" in output


class TestJSONishFormatterValidation:
    """Test validation constraint handling."""

    def test_numeric_constraints(self):
        """Test numeric validation constraints."""
        schema = simplify_schema(ComplexTypes, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Check for min/max constraints
        assert "min:" in output or "ge:" in output
        assert "max:" in output or "le:" in output

    def test_string_constraints(self):
        """Test string validation constraints."""
        schema = simplify_schema(User, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Check for maxLength
        assert "maxLength: 100" in output

    def test_pattern_constraints(self):
        """Test pattern validation."""
        schema = simplify_schema(Address, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Check for pattern constraints
        assert "pattern:" in output


class TestJSONishFormatterDefaults:
    """Test default value handling."""

    def test_simple_defaults(self):
        """Test simple default values."""
        schema = simplify_schema(ConfigModel, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Check for default values
        assert "(defaults to" in output

    def test_boolean_default(self):
        """Test boolean default values."""
        schema = simplify_schema(ConfigModel, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        assert "enabled: bool" in output
        assert "(defaults to True)" in output


class TestJSONishFormatterEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_schema(self):
        """Test handling of schema with no properties."""
        schema_dict = {"type": "object", "properties": {}}
        formatter = JSONishFormatter(schema_dict)
        output = formatter.transform_schema()

        # Empty schema returns "object" or "" since no properties exist
        assert output in ["{}", "object", ""]

    def test_dict_to_string(self):
        """Test dict to string conversion."""
        test_dict = {"name": "string", "age": "int"}
        output = JSONishFormatter.dict_to_string(test_dict, indent=0)

        assert "{" in output
        assert "name: string" in output
        assert "age: int" in output

    def test_none_handling(self):
        """Test None value handling."""
        output = JSONishFormatter.dict_to_string(None)
        assert output == "null"

    def test_list_handling(self):
        """Test list value handling in dict_to_string."""
        test_list = ["string", "int", "float"]
        output = JSONishFormatter.dict_to_string(test_list)

        assert "[" in output
        assert "string" in output


class TestJSONishFormatterIntegration:
    """Integration tests with SchemaLite."""

    def test_to_dict(self):
        """Test SchemaLite to_dict() method."""
        schema = simplify_schema(SimpleUser, format_type="jsonish")
        result = schema.to_dict()

        assert isinstance(result, dict)
        assert "name" in result
        assert "age" in result
        assert "email" in result

    def test_to_json(self):
        """Test SchemaLite to_json() method."""
        schema = simplify_schema(SimpleUser, format_type="jsonish")
        result = schema.to_json()

        assert isinstance(result, str)
        assert "{" in result
        assert '"name"' in result

    def test_to_string(self):
        """Test SchemaLite to_string() method."""
        schema = simplify_schema(SimpleUser, format_type="jsonish")
        result = schema.to_string()

        assert isinstance(result, str)
        assert "{" in result
        assert "name:" in result
