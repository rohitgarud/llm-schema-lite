"""Consolidated tests for JSONish formatter using base classes."""

import pytest

from llm_schema_lite import simplify_schema

# Import models from conftest (fixtures are auto-discovered but models need explicit import)
from .conftest import ComplexTypes, Order, Profile, SimpleUser, User
from .test_base_formatter import TestJSONishFormatter


class TestJSONishFormatterConsolidated(TestJSONishFormatter):
    """Consolidated JSONish formatter tests using base class."""

    # Inherits all common tests from TestJSONishFormatter base class
    # Only JSONish-specific tests are defined here

    def test_jsonish_enum_handling(self):
        """Test JSONish-specific enum handling."""
        schema = simplify_schema(User, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Should use //oneOf: syntax for enums
        assert "//oneOf:" in output
        assert "admin" in output
        assert "user" in output
        assert "guest" in output

    def test_jsonish_type_mapping(self):
        """Test JSONish-specific type mapping."""
        schema = simplify_schema(ComplexTypes, format_type="jsonish", include_metadata=False)
        output = schema.to_string()

        # Should use JSONish type mappings
        assert "int" in output  # integer -> int
        assert "float" in output  # number -> float
        assert "bool" in output  # boolean -> bool
        assert "string" in output  # string -> string

    def test_jsonish_array_handling(self):
        """Test JSONish array handling."""
        schema = simplify_schema(ComplexTypes, format_type="jsonish", include_metadata=False)
        output = schema.to_string()

        # Should handle arrays appropriately
        assert "array" in output or "[]" in output

    def test_jsonish_nested_objects(self):
        """Test JSONish nested object handling."""
        schema = simplify_schema(Order, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Should handle nested objects with proper indentation
        assert "{" in output
        assert "}" in output
        assert "user" in output
        assert "products" in output

    def test_jsonish_optional_fields(self):
        """Test JSONish optional field handling."""
        schema = simplify_schema(Profile, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Should handle optional fields
        assert "?" in output or "null" in output or "optional" in output.lower()

    def test_jsonish_validation_constraints(self):
        """Test JSONish validation constraint display."""
        schema = simplify_schema(User, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Should show constraints in comments
        assert "//" in output
        assert "minLength:" in output or "maxLength:" in output
        assert "pattern:" in output

    def test_jsonish_default_values(self):
        """Test JSONish default value display."""
        schema = simplify_schema(User, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Should show default values in comments
        assert "defaults to" in output or "default:" in output

    def test_jsonish_dict_to_string_formatting(self):
        """Test JSONish dict_to_string formatting."""
        schema = simplify_schema(Order, format_type="jsonish", include_metadata=False)
        output = schema.to_string()

        # Should have proper JSON-like formatting
        assert "{" in output
        assert "}" in output
        assert ":" in output

        # Should have proper indentation
        lines = output.split("\n")
        indented_lines = [line for line in lines if line.startswith(" ")]
        assert len(indented_lines) > 0, "No indented lines found"

    def test_jsonish_comment_syntax(self):
        """Test JSONish comment syntax."""
        schema = simplify_schema(User, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Should use // for comments
        assert "//" in output

        # Comments should be properly formatted
        lines = output.split("\n")
        comment_lines = [line for line in lines if "//" in line]
        for line in comment_lines:
            assert line.strip().startswith("//") or "  //" in line, f"Malformed comment: {line}"

    def test_jsonish_metadata_integration(self):
        """Test JSONish metadata integration."""
        schema = simplify_schema(User, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Should integrate metadata properly
        assert "//" in output
        assert "description" in output or "The user's full name" in output
        assert "minLength:" in output or "maxLength:" in output
        assert "pattern:" in output

    def test_jsonish_edge_cases(self):
        """Test JSONish edge cases."""
        # Test empty schema
        empty_schema = {"type": "object"}
        schema = simplify_schema(empty_schema, format_type="jsonish")
        output = schema.to_string()
        assert output == "{}"

        # Test schema with no properties
        no_props_schema = {"type": "object", "properties": {}}
        schema = simplify_schema(no_props_schema, format_type="jsonish")
        output = schema.to_string()
        assert output == "{}"

    def test_jsonish_integration_methods(self):
        """Test JSONish integration with SchemaLite methods."""
        schema = simplify_schema(User, format_type="jsonish", include_metadata=True)

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
        assert "{" in schema_string
        assert "}" in schema_string

    def test_jsonish_complex_workflow(self):
        """Test complete JSONish workflow."""
        # Create a complex schema with top-level metadata
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
                "count": {"type": "integer", "minimum": 0, "description": "Item count"},
            },
        }

        # Process with JSONish formatter
        schema = simplify_schema(complex_schema, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Should handle complex structure
        assert "user" in output
        assert "items" in output
        assert "count" in output
        assert "{" in output
        assert "}" in output
        assert "//" in output  # Should have metadata from count field

    def test_jsonish_performance(self):
        """Test JSONish formatter performance."""
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
        schema = simplify_schema(large_schema, format_type="jsonish", include_metadata=True)
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
# Parameterized Tests for JSONish-Specific Features
# ============================================================================


class TestJSONishParameterized:
    """Parameterized tests for JSONish-specific features."""

    @pytest.mark.parametrize("include_metadata", [True, False])
    def test_metadata_inclusion(self, user_model, include_metadata):
        """Test metadata inclusion/exclusion."""
        schema = simplify_schema(
            user_model, format_type="jsonish", include_metadata=include_metadata
        )
        output = schema.to_string()

        if include_metadata:
            assert "//" in output
            assert "description" in output or "The user's full name" in output
        else:
            # Should not have metadata comments (except enum values which are type info)
            lines = output.split("\n")
            metadata_lines = [line for line in lines if "//" in line and "oneOf:" not in line]
            assert (
                len(metadata_lines) == 0
            ), f"Found metadata when include_metadata=False: {metadata_lines}"

    @pytest.mark.parametrize("model_class", [SimpleUser, User, ComplexTypes, Order, Profile])
    def test_various_models(self, model_class):
        """Test JSONish formatter with various model types."""
        schema = simplify_schema(model_class, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Should produce valid output
        assert len(output.strip()) > 0
        assert "{" in output
        assert "}" in output

        # Should use JSONish syntax
        assert ":" in output  # Key-value pairs

    @pytest.mark.parametrize("format_type", ["jsonish"])  # Only JSONish for this test
    def test_jsonish_specific_syntax(self, user_model, format_type):
        """Test JSONish-specific syntax elements."""
        schema = simplify_schema(user_model, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should use JSONish comment syntax
        assert "//" in output

        # Should use JSONish type mappings
        assert "int" in output  # integer -> int
        assert "bool" in output  # boolean -> bool

        # Should have JSON-like structure
        assert "{" in output
        assert "}" in output
        assert ":" in output
