"""Tests for TypeScript formatter."""

from llm_schema_lite import simplify_schema
from llm_schema_lite.formatters import TypeScriptFormatter

from .fixtures import (
    Address,
    ComplexTypes,
    ConfigModel,
    Order,
    Profile,
    SimpleUser,
    User,
)


class TestTypeScriptFormatterBasic:
    """Test basic TypeScript formatter functionality."""

    def test_simple_model(self):
        """Test TypeScript output for simple model."""
        schema = simplify_schema(SimpleUser, format_type="typescript", include_metadata=False)
        output = schema.to_string()

        assert "interface Schema {" in output
        assert "name: string;" in output
        assert "age: number;" in output
        assert "email: string;" in output
        assert "}" in output

    def test_simple_model_with_metadata(self):
        """Test TypeScript output with metadata."""
        schema = simplify_schema(User, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Check for metadata comments
        assert "//" in output
        assert "The user's full name" in output
        assert "maxLength: 100" in output

    def test_without_metadata(self):
        """Test TypeScript output without metadata."""
        schema = simplify_schema(User, format_type="typescript", include_metadata=False)
        output = schema.to_string()

        # Should have minimal or no comments
        comment_count = output.count("//")
        assert comment_count == 0


class TestTypeScriptFormatterTypes:
    """Test TypeScript type mapping."""

    def test_type_mapping(self):
        """Test correct type mapping from JSON schema to TypeScript."""
        schema = simplify_schema(ComplexTypes, format_type="typescript", include_metadata=False)
        output = schema.to_string()

        # String should map to string
        assert "string_field: string;" in output

        # Integer should map to number
        assert "int_field: number;" in output

        # Float should map to number
        assert "float_field: number;" in output

        # Boolean should map to boolean
        assert "bool_field: boolean;" in output

    def test_array_types(self):
        """Test array type mapping."""
        schema = simplify_schema(ComplexTypes, format_type="typescript", include_metadata=False)
        output = schema.to_string()

        # Should have array types
        assert "string[]" in output or "number[]" in output


class TestTypeScriptFormatterNested:
    """Test TypeScript formatter with nested structures."""

    def test_nested_references(self):
        """Test nested model references."""
        schema = simplify_schema(User, format_type="typescript", include_metadata=False)
        output = schema.to_string()

        # Nested objects should be referenced by name
        assert "contact_info:" in output
        assert "addresses:" in output

    def test_complex_order_model(self):
        """Test complex Order model."""
        schema = simplify_schema(Order, format_type="typescript", include_metadata=False)
        output = schema.to_string()

        assert "interface Schema {" in output
        assert "order_id: number;" in output
        assert "user:" in output
        assert "products:" in output


class TestTypeScriptFormatterEnums:
    """Test enum handling in TypeScript."""

    def test_enum_as_union(self):
        """Test enum represented as TypeScript union."""
        schema_dict = {"type": "string", "enum": ["admin", "user", "guest"]}
        formatter = TypeScriptFormatter({"properties": {"role": schema_dict}})

        output = formatter.process_enum(schema_dict)

        # Should create union of literals
        assert '"admin"' in output
        assert '"user"' in output
        assert '"guest"' in output
        assert "|" in output


class TestTypeScriptFormatterOptional:
    """Test optional field handling."""

    def test_union_types(self):
        """Test union/optional types."""
        schema = simplify_schema(Profile, format_type="typescript", include_metadata=False)
        output = schema.to_string()

        # Should handle union types
        assert "name:" in output
        assert "email:" in output

    def test_optional_with_default(self):
        """Test optional fields with default values."""
        schema = simplify_schema(ConfigModel, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Should have comments for defaults
        assert "(defaults to" in output


class TestTypeScriptFormatterValidation:
    """Test validation constraint comments."""

    def test_numeric_constraints(self):
        """Test numeric validation in comments."""
        schema = simplify_schema(ComplexTypes, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Constraints should be in comments
        assert "min:" in output or "ge:" in output

    def test_string_constraints(self):
        """Test string validation in comments."""
        schema = simplify_schema(User, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # MaxLength should be in comments
        assert "maxLength: 100" in output

    def test_pattern_constraints(self):
        """Test pattern validation in comments."""
        schema = simplify_schema(Address, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Pattern should be in comments
        assert "pattern:" in output


class TestTypeScriptFormatterFormatting:
    """Test TypeScript output formatting."""

    def test_interface_structure(self):
        """Test proper interface structure."""
        schema = simplify_schema(SimpleUser, format_type="typescript", include_metadata=False)
        output = schema.to_string()

        # Should have proper interface structure
        lines = output.strip().split("\n")
        assert lines[0] == "interface Schema {"
        assert lines[-1] == "}"

        # Each field should end with semicolon
        for line in lines[1:-1]:
            line = line.strip()
            if line and not line.startswith("//"):
                assert line.endswith(";")

    def test_indentation(self):
        """Test proper indentation."""
        schema = simplify_schema(SimpleUser, format_type="typescript", include_metadata=False)
        output = schema.to_string()

        lines = output.strip().split("\n")
        # Fields should be indented
        for line in lines[1:-1]:
            if line.strip() and not line.strip().startswith("//"):
                assert line.startswith("  ")  # 2 space indent


class TestTypeScriptFormatterEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_schema(self):
        """Test handling of schema with no properties."""
        schema_dict = {"type": "object", "properties": {}}
        formatter = TypeScriptFormatter(schema_dict)
        output = formatter.transform_schema()

        assert output == "interface Schema {}"

    def test_anyof_handling(self):
        """Test anyOf/union type handling."""
        anyof_schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
            ]
        }
        formatter = TypeScriptFormatter({"properties": {"field": anyof_schema}})
        output = formatter.process_anyof(anyof_schema)

        assert "string" in output
        assert "number" in output
        assert "|" in output


class TestTypeScriptFormatterIntegration:
    """Integration tests with SchemaLite."""

    def test_full_workflow(self):
        """Test complete workflow from model to TypeScript."""
        schema = simplify_schema(SimpleUser, format_type="typescript")
        output = schema.to_string()

        # Should produce valid-looking TypeScript interface
        assert "interface Schema {" in output
        assert "name: string;" in output
        assert "}" in output

    def test_complex_model_workflow(self):
        """Test complex model conversion."""
        schema = simplify_schema(Order, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Should handle nested structures
        assert "interface Schema {" in output
        assert "}" in output
        # Should have multiple fields
        assert output.count(";") >= 4
