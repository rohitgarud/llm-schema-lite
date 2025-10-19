"""Consolidated tests for TypeScript formatter using base classes."""

import pytest

from llm_schema_lite import simplify_schema

# Import models from conftest (fixtures are auto-discovered but models need explicit import)
from .conftest import ComplexTypes, Order, Profile, SimpleUser, User
from .test_base_formatter import TestTypeScriptFormatter


class TestTypeScriptFormatterConsolidated(TestTypeScriptFormatter):
    """Consolidated TypeScript formatter tests using base class."""

    # Inherits all common tests from TestTypeScriptFormatter base class
    # Only TypeScript-specific tests are defined here

    def test_typescript_interface_syntax(self):
        """Test TypeScript interface syntax."""
        schema = simplify_schema(User, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Should use TypeScript interface syntax
        assert "interface Schema {" in output
        assert "}" in output

        # Should have proper property syntax with semicolons
        lines = output.split("\n")
        property_lines = [
            line for line in lines if ":" in line and not line.strip().startswith("//")
        ]
        for line in property_lines:
            if line.strip() and not line.strip().endswith("{") and not line.strip().endswith("}"):
                assert line.strip().endswith(
                    ";"
                ), f"Property line missing semicolon: {line.strip()}"

    def test_typescript_type_mapping(self):
        """Test TypeScript-specific type mapping."""
        schema = simplify_schema(ComplexTypes, format_type="typescript", include_metadata=False)
        output = schema.to_string()

        # Should use TypeScript type mappings
        assert "number" in output  # integer -> number
        assert "boolean" in output  # boolean -> boolean
        assert "string" in output  # string -> string
        assert "Array" in output  # array -> Array

    def test_typescript_union_types(self):
        """Test TypeScript union type syntax."""
        schema = simplify_schema(ComplexTypes, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Should use TypeScript union syntax
        assert " | " in output or "Union[" in output

    def test_typescript_enum_literals(self):
        """Test TypeScript enum literal formatting."""
        schema = simplify_schema(User, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Should format enums as union literals
        assert '"admin" | "user" | "guest"' in output or '"user" | "admin" | "guest"' in output

    def test_typescript_array_types(self):
        """Test TypeScript array type formatting."""
        schema = simplify_schema(ComplexTypes, format_type="typescript", include_metadata=False)
        output = schema.to_string()

        # Should have array types (both Array<T> and T[] syntax are valid)
        assert (
            "Array<string>" in output
            or "string[]" in output
            or "Array<number>" in output
            or "number[]" in output
        )

    def test_typescript_nullable_types(self):
        """Test TypeScript nullable type formatting."""
        schema = simplify_schema(Profile, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Should handle nullable types
        assert " | null" in output or " | undefined" in output

    def test_typescript_nested_interfaces(self):
        """Test TypeScript nested interface generation."""
        schema = simplify_schema(Order, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Should generate nested interfaces
        assert "interface" in output
        assert "Schema" in output

        # Should handle nested objects
        assert "user" in output
        assert "products" in output

    def test_typescript_comment_syntax(self):
        """Test TypeScript comment syntax."""
        schema = simplify_schema(User, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Should use // for comments
        assert "//" in output

        # Comments should be properly formatted
        lines = output.split("\n")
        comment_lines = [line for line in lines if "//" in line]
        for line in comment_lines:
            assert line.strip().startswith("//") or "  //" in line, f"Malformed comment: {line}"

    def test_typescript_metadata_integration(self):
        """Test TypeScript metadata integration."""
        schema = simplify_schema(User, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Should integrate metadata properly
        assert "//" in output
        assert "description" in output or "The user's full name" in output
        assert "minLength:" in output or "maxLength:" in output
        assert "pattern:" in output

    def test_typescript_edge_cases(self):
        """Test TypeScript edge cases."""
        # Test empty schema
        empty_schema = {"type": "object"}
        schema = simplify_schema(empty_schema, format_type="typescript")
        output = schema.to_string()
        assert "interface Schema {}" in output

        # Test schema with no properties
        no_props_schema = {"type": "object", "properties": {}}
        schema = simplify_schema(no_props_schema, format_type="typescript")
        output = schema.to_string()
        assert "interface Schema {}" in output

    def test_typescript_integration_methods(self):
        """Test TypeScript integration with SchemaLite methods."""
        schema = simplify_schema(User, format_type="typescript", include_metadata=True)

        # Test to_dict
        schema_dict = schema.to_dict()
        assert isinstance(schema_dict, dict)
        assert len(schema_dict) > 0

        # Test to_json
        schema_json = schema.to_json()
        assert isinstance(schema_json, str)
        assert len(schema_json.strip()) > 0

        # Test to_string
        schema_string = schema.to_string()
        assert isinstance(schema_string, str)
        assert "interface" in schema_string

    def test_typescript_complex_workflow(self):
        """Test complete TypeScript workflow."""
        # Create a complex schema
        complex_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "age": {"type": "integer", "minimum": 0},
                    },
                },
                "items": {"type": "array", "items": {"type": "string"}},
            },
        }

        # Process with TypeScript formatter
        schema = simplify_schema(complex_schema, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Should handle complex structure
        assert "interface" in output
        assert "user" in output
        assert "items" in output
        assert ";" in output  # Semicolons after properties

    def test_typescript_performance(self):
        """Test TypeScript formatter performance."""
        import time

        # Create a moderately complex schema
        large_schema = {
            "type": "object",
            "properties": {
                f"field_{i}": {
                    "type": "string",
                    "description": f"Field {i} description",
                    "minLength": 1,
                    "maxLength": 100,
                }
                for i in range(50)
            },
        }

        start_time = time.time()
        schema = simplify_schema(large_schema, format_type="typescript", include_metadata=True)
        output = schema.to_string()
        end_time = time.time()

        processing_time = end_time - start_time

        # Should process within reasonable time
        assert processing_time < 1.0, f"Processing took too long: {processing_time:.3f}s"

        # Should produce substantial output
        assert len(output) > 1000, f"Output too short: {len(output)} characters"

        # Should include all fields
        for i in range(50):
            assert f"field_{i}" in output, f"Missing field_{i} in output"


# ============================================================================
# Parameterized Tests for TypeScript-Specific Features
# ============================================================================


class TestTypeScriptParameterized:
    """Parameterized tests for TypeScript-specific features."""

    @pytest.mark.parametrize("include_metadata", [True, False])
    def test_metadata_inclusion(self, user_model, include_metadata):
        """Test metadata inclusion/exclusion."""
        schema = simplify_schema(
            user_model, format_type="typescript", include_metadata=include_metadata
        )
        output = schema.to_string()

        if include_metadata:
            assert "//" in output
            assert "description" in output or "The user's full name" in output
        else:
            # Should not have metadata comments
            lines = output.split("\n")
            metadata_lines = [line for line in lines if "//" in line]
            assert (
                len(metadata_lines) == 0
            ), f"Found metadata when include_metadata=False: {metadata_lines}"

    @pytest.mark.parametrize("model_class", [SimpleUser, User, ComplexTypes, Order, Profile])
    def test_various_models(self, model_class):
        """Test TypeScript formatter with various model types."""
        schema = simplify_schema(model_class, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Should produce valid output
        assert len(output.strip()) > 0
        assert "interface" in output
        assert "{" in output
        assert "}" in output

        # Should use TypeScript syntax
        assert ":" in output  # Property type annotations
        assert ";" in output  # Semicolons after properties

    @pytest.mark.parametrize("format_type", ["typescript"])  # Only TypeScript for this test
    def test_typescript_specific_syntax(self, user_model, format_type):
        """Test TypeScript-specific syntax elements."""
        schema = simplify_schema(user_model, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should use TypeScript interface syntax
        assert "interface Schema {" in output

        # Should use TypeScript type mappings
        assert "number" in output  # integer -> number
        assert "boolean" in output  # boolean -> boolean
        assert "string" in output  # string -> string

        # Should have semicolons after properties
        lines = output.split("\n")
        property_lines = [
            line for line in lines if ":" in line and not line.strip().startswith("//")
        ]
        for line in property_lines:
            if line.strip() and not line.strip().endswith("{") and not line.strip().endswith("}"):
                assert line.strip().endswith(
                    ";"
                ), f"Property line missing semicolon: {line.strip()}"

    @pytest.mark.parametrize("nullable_type", ["string", "number", "boolean"])
    def test_nullable_types(self, nullable_type):
        """Test nullable type handling."""
        nullable_schema = {
            "type": "object",
            "properties": {"field": {"type": [nullable_type, "null"]}},
        }

        schema = simplify_schema(nullable_schema, format_type="typescript", include_metadata=False)
        output = schema.to_string()

        # Should handle nullable types
        assert " | null" in output

    @pytest.mark.parametrize("array_type", ["string", "number", "boolean"])
    def test_array_types(self, array_type):
        """Test array type handling."""
        array_schema = {
            "type": "object",
            "properties": {"field": {"type": "array", "items": {"type": array_type}}},
        }

        schema = simplify_schema(array_schema, format_type="typescript", include_metadata=False)
        output = schema.to_string()

        # Should handle array types (both Array<T> and T[] syntax are valid)
        assert f"Array<{array_type}>" in output or f"{array_type}[]" in output
