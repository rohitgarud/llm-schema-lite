"""Tests for YAML formatter."""

from llm_schema_lite import simplify_schema
from llm_schema_lite.formatters import YAMLFormatter

from .fixtures import (
    Address,
    ComplexTypes,
    ConfigModel,
    Order,
    Profile,
    SimpleUser,
    User,
)


class TestYAMLFormatterBasic:
    """Test basic YAML formatter functionality."""

    def test_simple_model(self):
        """Test YAML output for simple model."""
        schema = simplify_schema(SimpleUser, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        assert "name: str" in output
        assert "age: int" in output
        assert "email: str" in output
        # Should not have braces/brackets for simple format
        assert "{" not in output
        assert "interface" not in output

    def test_simple_model_with_metadata(self):
        """Test YAML output with metadata."""
        schema = simplify_schema(User, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Check for metadata comments with #
        assert "#" in output
        assert "The user's full name" in output
        assert "maxLength: 100" in output

    def test_without_metadata(self):
        """Test YAML output without metadata."""
        schema = simplify_schema(User, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        # Should not contain comments
        assert "#" not in output


class TestYAMLFormatterTypes:
    """Test YAML type mapping."""

    def test_type_mapping(self):
        """Test correct type mapping to Python types."""
        schema = simplify_schema(ComplexTypes, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        # String should map to str
        assert "str" in output

        # Integer should map to int
        assert "int" in output

        # Float should map to float
        assert "float" in output

        # Boolean should map to bool
        assert "bool" in output

    def test_list_types(self):
        """Test list type mapping."""
        schema = simplify_schema(ComplexTypes, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        # Should have list[type] syntax
        assert "list[str]" in output or "list[int]" in output


class TestYAMLFormatterNested:
    """Test YAML formatter with nested structures."""

    def test_nested_references(self):
        """Test nested model references."""
        schema = simplify_schema(User, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        # Nested objects should be referenced
        assert "contact_info:" in output
        assert "addresses:" in output

    def test_complex_order_model(self):
        """Test complex Order model."""
        schema = simplify_schema(Order, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        assert "order_id: int" in output
        assert "user:" in output
        assert "products:" in output
        assert "total_price: float" in output


class TestYAMLFormatterEnums:
    """Test enum handling in YAML."""

    def test_enum_as_literal(self):
        """Test enum represented as Literal."""
        schema_dict = {"type": "string", "enum": ["admin", "user", "guest"]}
        formatter = YAMLFormatter({"properties": {"role": schema_dict}})

        output = formatter.process_enum(schema_dict)

        # Should create Literal union
        assert "Literal[" in output
        assert '"admin"' in output
        assert '"user"' in output
        assert '"guest"' in output


class TestYAMLFormatterOptional:
    """Test optional field handling."""

    def test_union_types(self):
        """Test union/optional types."""
        schema = simplify_schema(Profile, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        # Should handle union types
        assert "name:" in output
        assert "email:" in output

    def test_optional_with_default(self):
        """Test optional fields with default values."""
        schema = simplify_schema(ConfigModel, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Should have comments for defaults
        assert "(defaults to" in output


class TestYAMLFormatterValidation:
    """Test validation constraint comments."""

    def test_numeric_constraints(self):
        """Test numeric validation in comments."""
        schema = simplify_schema(ComplexTypes, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Constraints should be in comments
        assert "min:" in output or "ge:" in output

    def test_string_constraints(self):
        """Test string validation in comments."""
        schema = simplify_schema(User, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # MaxLength should be in comments
        assert "maxLength: 100" in output

    def test_pattern_constraints(self):
        """Test pattern validation in comments."""
        schema = simplify_schema(Address, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Pattern should be in comments
        assert "pattern:" in output


class TestYAMLFormatterFormatting:
    """Test YAML output formatting."""

    def test_simple_structure(self):
        """Test simple key: value structure."""
        schema = simplify_schema(SimpleUser, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        # Each line should be "key: type"
        lines = output.strip().split("\n")
        for line in lines:
            if line.strip():
                assert ":" in line
                parts = line.split(":")
                assert len(parts) >= 2

    def test_no_extra_braces(self):
        """Test that YAML format doesn't have JSON braces."""
        schema = simplify_schema(SimpleUser, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        # Should not have JSON-like braces
        assert "{" not in output
        assert "interface" not in output


class TestYAMLFormatterEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_schema(self):
        """Test handling of schema with no properties."""
        schema_dict = {"type": "object", "properties": {}}
        formatter = YAMLFormatter(schema_dict)
        output = formatter.transform_schema()

        assert output == "{}"

    def test_anyof_handling(self):
        """Test anyOf/union type handling."""
        anyof_schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"},
            ]
        }
        formatter = YAMLFormatter({"properties": {"field": anyof_schema}})
        output = formatter.process_anyof(anyof_schema)

        assert "str" in output
        assert "int" in output
        assert "|" in output

    def test_list_with_ref(self):
        """Test list with reference items."""
        schema = simplify_schema(User, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        # Should handle list of nested objects
        assert "addresses:" in output


class TestYAMLFormatterIntegration:
    """Integration tests with SchemaLite."""

    def test_full_workflow(self):
        """Test complete workflow from model to YAML."""
        schema = simplify_schema(SimpleUser, format_type="yaml")
        output = schema.to_string()

        # Should produce clean YAML-style output
        assert "name: str" in output
        assert "age: int" in output
        assert "{" not in output

    def test_complex_model_workflow(self):
        """Test complex model conversion."""
        schema = simplify_schema(Order, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Should handle nested structures
        lines = output.strip().split("\n")
        # Should have multiple fields
        assert len(lines) >= 4

    def test_with_lists(self):
        """Test handling of list fields."""
        schema = simplify_schema(ComplexTypes, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        # Should have list[type] notation
        assert "list[" in output
