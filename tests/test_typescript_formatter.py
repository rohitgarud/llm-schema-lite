"""Consolidated tests for TypeScript formatter using base classes."""

import pytest

from llm_schema_lite import simplify_schema
from llm_schema_lite.formatters.jsonish_formatter import JSONishFormatter
from llm_schema_lite.formatters.typescript_formatter import TypeScriptFormatter
from llm_schema_lite.formatters.yaml_formatter import YAMLFormatter

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
        # Check for integrated constraints in type descriptions
        assert "chars" in output or "minLength:" in output or "maxLength:" in output
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
            # Should not have metadata comments (except required fields comment)
            lines = output.split("\n")
            metadata_lines = [
                line
                for line in lines
                if "//" in line and "Fields marked with * are required" not in line
            ]
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


class TestTypeScriptFormatterMissingCoverage:
    """Test missing coverage in TypeScriptFormatter class."""

    def test_format_enum_with_empty_values(self):
        """Test format_enum with empty enum values."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": []}

        result = formatter.process_enum(schema)
        assert result == "string"

    def test_format_enum_with_single_value(self):
        """Test format_enum with single enum value."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["single_value"]}

        result = formatter.process_enum(schema)
        assert "single_value" in result

    def test_format_enum_with_mixed_types(self):
        """Test format_enum with mixed type enum values."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["string_value", 123, True, None]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_unicode_values(self):
        """Test format_enum with unicode values."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["测试", "🚀", "unicode_value"]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_special_characters(self):
        """Test format_enum with special characters."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {
            "type": "string",
            "enum": [
                "value with spaces",
                "value-with-dashes",
                "value_with_underscores",
                "value.with.dots",
            ],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_numeric_values(self):
        """Test format_enum with numeric values."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "number", "enum": [1, 2, 3, 4, 5]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_boolean_values(self):
        """Test format_enum with boolean values."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "boolean", "enum": [True, False]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_null_values(self):
        """Test format_enum with null values."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["value1", None, "value2"]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_complex_values(self):
        """Test format_enum with complex values."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {
            "type": "string",
            "enum": [
                {"type": "object", "properties": {"name": {"type": "string"}}},
                {"type": "array", "items": {"type": "string"}},
            ],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_very_long_values(self):
        """Test format_enum with very long values."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        long_values = [f"very_long_value_{i}" * 10 for i in range(5)]
        schema = {"type": "string", "enum": long_values}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_duplicate_values(self):
        """Test format_enum with duplicate values."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["value1", "value2", "value1", "value3", "value2"]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_none_schema(self):
        """Test format_enum with None schema."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        # Test that None schema raises an error or returns empty string
        try:
            result = formatter.process_enum(None)
            assert result == ""
        except (AttributeError, TypeError):
            # Expected behavior when None is passed
            pass

    def test_format_enum_with_empty_schema(self):
        """Test format_enum with empty schema."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        result = formatter.process_enum({})
        assert result == "string"  # Fallback for empty enum

    def test_format_enum_without_enum_key(self):
        """Test format_enum without enum key."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "string"}

        result = formatter.process_enum(schema)
        assert result == "string"

    def test_format_enum_with_invalid_enum(self):
        """Test format_enum with invalid enum (not a list)."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": "not_a_list"}

        result = formatter.process_enum(schema)
        # When enum is not a list, it treats each character as a separate value
        assert "|" in result  # TypeScript union type

    def test_format_enum_with_nested_objects(self):
        """Test format_enum with nested objects."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {
            "type": "object",
            "enum": [
                {"name": "John", "age": 30},
                {"name": "Jane", "age": 25},
                {"name": "Bob", "age": 35},
            ],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_arrays(self):
        """Test format_enum with arrays."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "array", "enum": [[1, 2, 3], ["a", "b", "c"], [True, False]]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_mixed_primitive_types(self):
        """Test format_enum with mixed primitive types."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["string_value", 123, 45.67, True, False, None]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_very_large_enum(self):
        """Test format_enum with very large enum."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        large_enum = [f"value_{i}" for i in range(100)]
        schema = {"type": "string", "enum": large_enum}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_unicode_enum(self):
        """Test format_enum with unicode enum values."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["测试", "🚀", "unicode", "中文", "emoji 🎉"]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_special_json_characters(self):
        """Test format_enum with special JSON characters."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {
            "type": "string",
            "enum": [
                'value with "quotes"',
                "value with 'single quotes'",
                "value with\nnewlines",
                "value with\ttabs",
                "value with\\backslashes",
                "value with/slashes",
            ],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_numeric_strings(self):
        """Test format_enum with numeric strings."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["1", "2", "3", "10", "100"]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_boolean_strings(self):
        """Test format_enum with boolean strings."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["true", "false", "True", "False", "TRUE", "FALSE"]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_empty_strings(self):
        """Test format_enum with empty strings."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["", "value1", "", "value2", ""]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_whitespace_values(self):
        """Test format_enum with whitespace values."""
        formatter = TypeScriptFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": [" ", "  ", "\t", "\n", "value1", "value2"]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)


class TestYAMLFormatterMissingCoverage:
    """Test missing coverage in YAMLFormatter class."""

    def test_format_enum_with_empty_values(self):
        """Test format_enum with empty enum values."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": []}

        result = formatter.process_enum(schema)
        assert result == "str"

    def test_format_enum_with_single_value(self):
        """Test format_enum with single enum value."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["single_value"]}

        result = formatter.process_enum(schema)
        assert "single_value" in result

    def test_format_enum_with_mixed_types(self):
        """Test format_enum with mixed type enum values."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["string_value", 123, True, None]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_unicode_values(self):
        """Test format_enum with unicode values."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["测试", "🚀", "unicode_value"]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_special_characters(self):
        """Test format_enum with special characters."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {
            "type": "string",
            "enum": [
                "value with spaces",
                "value-with-dashes",
                "value_with_underscores",
                "value.with.dots",
            ],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_numeric_values(self):
        """Test format_enum with numeric values."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "number", "enum": [1, 2, 3, 4, 5]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_boolean_values(self):
        """Test format_enum with boolean values."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "boolean", "enum": [True, False]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_null_values(self):
        """Test format_enum with null values."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["value1", None, "value2"]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_complex_values(self):
        """Test format_enum with complex values."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {
            "type": "string",
            "enum": [
                {"type": "object", "properties": {"name": {"type": "string"}}},
                {"type": "array", "items": {"type": "string"}},
            ],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_very_long_values(self):
        """Test format_enum with very long values."""
        formatter = YAMLFormatter({}, include_metadata=True)

        long_values = [f"very_long_value_{i}" * 10 for i in range(5)]
        schema = {"type": "string", "enum": long_values}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_duplicate_values(self):
        """Test format_enum with duplicate values."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["value1", "value2", "value1", "value3", "value2"]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_none_schema(self):
        """Test format_enum with None schema."""
        formatter = YAMLFormatter({}, include_metadata=True)

        # Test that None schema raises an error or returns empty string
        try:
            result = formatter.process_enum(None)
            assert result == ""
        except (AttributeError, TypeError):
            # Expected behavior when None is passed
            pass

    def test_format_enum_with_empty_schema(self):
        """Test format_enum with empty schema."""
        formatter = YAMLFormatter({}, include_metadata=True)

        result = formatter.process_enum({})
        assert result == "str"  # Fallback for empty enum

    def test_format_enum_without_enum_key(self):
        """Test format_enum without enum key."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "string"}

        result = formatter.process_enum(schema)
        assert result == "str"

    def test_format_enum_with_invalid_enum(self):
        """Test format_enum with invalid enum (not a list)."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": "not_a_list"}

        result = formatter.process_enum(schema)
        # When enum is not a list, it treats each character as a separate value
        assert "Literal" in result  # YAML Literal type

    def test_format_enum_with_nested_objects(self):
        """Test format_enum with nested objects."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {
            "type": "object",
            "enum": [
                {"name": "John", "age": 30},
                {"name": "Jane", "age": 25},
                {"name": "Bob", "age": 35},
            ],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_arrays(self):
        """Test format_enum with arrays."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "array", "enum": [[1, 2, 3], ["a", "b", "c"], [True, False]]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_mixed_primitive_types(self):
        """Test format_enum with mixed primitive types."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["string_value", 123, 45.67, True, False, None]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_very_large_enum(self):
        """Test format_enum with very large enum."""
        formatter = YAMLFormatter({}, include_metadata=True)

        large_enum = [f"value_{i}" for i in range(100)]
        schema = {"type": "string", "enum": large_enum}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_unicode_enum(self):
        """Test format_enum with unicode enum values."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["测试", "🚀", "unicode", "中文", "emoji 🎉"]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_special_json_characters(self):
        """Test format_enum with special JSON characters."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {
            "type": "string",
            "enum": [
                'value with "quotes"',
                "value with 'single quotes'",
                "value with\nnewlines",
                "value with\ttabs",
                "value with\\backslashes",
                "value with/slashes",
            ],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_numeric_strings(self):
        """Test format_enum with numeric strings."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["1", "2", "3", "10", "100"]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_boolean_strings(self):
        """Test format_enum with boolean strings."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["true", "false", "True", "False", "TRUE", "FALSE"]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_empty_strings(self):
        """Test format_enum with empty strings."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["", "value1", "", "value2", ""]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_whitespace_values(self):
        """Test format_enum with whitespace values."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": [" ", "  ", "\t", "\n", "value1", "value2"]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_yaml_special_characters(self):
        """Test format_enum with YAML special characters."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {
            "type": "string",
            "enum": [
                "value:with:colons",
                "value|with|pipes",
                "value-with-dashes",
                "value_with_underscores",
                "value.with.dots",
            ],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_yaml_boolean_literals(self):
        """Test format_enum with YAML boolean literals."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {
            "type": "string",
            "enum": ["true", "false", "True", "False", "TRUE", "FALSE", "on", "off", "yes", "no"],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_yaml_null_literals(self):
        """Test format_enum with YAML null literals."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {"type": "string", "enum": ["null", "NULL", "Null", "~", "None", "NONE", "none"]}

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_yaml_numeric_literals(self):
        """Test format_enum with YAML numeric literals."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {
            "type": "string",
            "enum": ["123", "45.67", "1e10", "1.2e-5", "0x1A", "0o755", "0b1010"],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_yaml_quoted_strings(self):
        """Test format_enum with YAML quoted strings."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {
            "type": "string",
            "enum": [
                '"quoted string"',
                "'single quoted'",
                '"string with\nnewlines"',
                '"string with\ttabs"',
            ],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_yaml_multiline_strings(self):
        """Test format_enum with YAML multiline strings."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {
            "type": "string",
            "enum": ["line1\nline2\nline3", "line1\r\nline2\r\nline3", "line1\rline2\rline3"],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_yaml_anchor_references(self):
        """Test format_enum with YAML anchor references."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {
            "type": "string",
            "enum": ["&anchor value", "*reference", "&anchor1 value1", "*reference1"],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_yaml_tags(self):
        """Test format_enum with YAML tags."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {
            "type": "string",
            "enum": ["!!str string_value", "!!int 123", "!!float 45.67", "!!bool true"],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)

    def test_format_enum_with_yaml_complex_values(self):
        """Test format_enum with YAML complex values."""
        formatter = YAMLFormatter({}, include_metadata=True)

        schema = {
            "type": "string",
            "enum": [
                "key: value",
                "key: value\n  nested: value",
                "key: value\n  nested:\n    - item1\n    - item2",
            ],
        }

        result = formatter.process_enum(schema)
        assert isinstance(result, str)


class TestTypeScriptFormatterCoverage:
    """Tests for TypeScriptFormatter to increase coverage."""

    def test_interface_syntax(self):
        """Test TypeScript interface syntax."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        formatter = TypeScriptFormatter(schema)
        result = formatter.transform_schema()
        assert "interface" in result
        assert "name: string" in result
        assert "age: number" in result

    def test_type_mapping(self):
        """Test TypeScript type mapping."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "price": {"type": "number"},
                "active": {"type": "boolean"},
            },
        }
        formatter = TypeScriptFormatter(schema)
        result = formatter.transform_schema()
        assert "count: number" in result
        assert "price: number" in result
        assert "active: boolean" in result

    def test_union_types(self):
        """Test TypeScript union types."""
        schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
        formatter = TypeScriptFormatter(schema)
        result = formatter.transform_schema()
        assert "string" in result
        assert "number" in result
        assert "|" in result

    def test_array_types(self):
        """Test TypeScript array types."""
        schema = {"type": "array", "items": {"type": "string"}}
        formatter = TypeScriptFormatter(schema)
        result = formatter.transform_schema()
        assert "Array<string>" in result

    def test_comment_syntax(self):
        """Test TypeScript comment syntax."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "User name"}},
        }
        formatter = TypeScriptFormatter(schema, include_metadata=True)
        result = formatter.transform_schema()
        assert "//" in result
        assert "User name" in result

    def test_metadata_integration(self):
        """Test TypeScript metadata integration."""
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer", "minimum": 0, "maximum": 120}},
        }
        formatter = TypeScriptFormatter(schema, include_metadata=True)
        result = formatter.transform_schema()
        assert "age: number" in result
        assert "(0-120)" in result

    def test_edge_cases(self):
        """Test TypeScript formatter edge cases."""
        # Empty schema
        formatter = TypeScriptFormatter({})
        result = formatter.transform_schema()
        assert "interface Schema {}" in result

        # Schema with only type
        schema = {"type": "string"}
        formatter = TypeScriptFormatter(schema)
        result = formatter.transform_schema()
        assert "type Schema = string" in result

    def test_integration_methods(self):
        """Test TypeScript formatter integration methods."""
        schema = {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
            "required": ["id"],
        }
        formatter = TypeScriptFormatter(schema)

        # Test process_schema method
        processed = formatter.process_schema()
        assert isinstance(processed, dict)

    def test_performance(self):
        """Test TypeScript formatter performance."""
        schema = {
            "type": "object",
            "properties": {f"field_{i}": {"type": "string"} for i in range(100)},
        }
        formatter = TypeScriptFormatter(schema)
        result = formatter.transform_schema()
        assert "interface" in result
        assert len(result) > 1000


class TestYAMLFormatterCoverage:
    """Tests for YAMLFormatter to increase coverage."""

    def test_yaml_formatting_syntax(self):
        """Test YAML formatting syntax."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        formatter = YAMLFormatter(schema)
        result = formatter.transform_schema()
        assert "name: str" in result
        assert "age: int" in result

    def test_yaml_type_mapping(self):
        """Test YAML type mapping."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "price": {"type": "number"},
                "active": {"type": "boolean"},
            },
        }
        formatter = YAMLFormatter(schema)
        result = formatter.transform_schema()
        assert "count: int" in result
        assert "price: float" in result
        assert "active: bool" in result

    def test_yaml_union_types(self):
        """Test YAML union types."""
        schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
        formatter = YAMLFormatter(schema)
        result = formatter.transform_schema()
        assert "str" in result
        assert "int" in result
        assert "|" in result

    def test_yaml_list_types(self):
        """Test YAML list types."""
        schema = {"type": "array", "items": {"type": "string"}}
        formatter = YAMLFormatter(schema)
        result = formatter.transform_schema()
        assert "list[str]" in result

    def test_yaml_nested_sections(self):
        """Test YAML nested sections."""
        schema = {
            "type": "object",
            "properties": {"user": {"type": "object", "properties": {"name": {"type": "string"}}}},
        }
        formatter = YAMLFormatter(schema)
        result = formatter.transform_schema()
        assert "user:" in result
        assert "name: str" in result

    def test_yaml_comment_syntax(self):
        """Test YAML comment syntax."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "User name"}},
        }
        formatter = YAMLFormatter(schema, include_metadata=True)
        result = formatter.transform_schema()
        assert "#" in result
        assert "User name" in result

    def test_yaml_metadata_integration(self):
        """Test YAML metadata integration."""
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer", "minimum": 0, "maximum": 120}},
        }
        formatter = YAMLFormatter(schema, include_metadata=True)
        result = formatter.transform_schema()
        assert "age: int" in result
        assert "(0-120)" in result

    def test_yaml_edge_cases(self):
        """Test YAML formatter edge cases."""
        # Empty schema
        formatter = YAMLFormatter({})
        result = formatter.transform_schema()
        assert result == "{}"

        # Schema with only type
        schema = {"type": "string"}
        formatter = YAMLFormatter(schema)
        result = formatter.transform_schema()
        assert "str" in result

    def test_yaml_integration_methods(self):
        """Test YAML formatter integration methods."""
        schema = {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
            "required": ["id"],
        }
        formatter = YAMLFormatter(schema)

        # Test process_schema method
        processed = formatter.process_schema()
        assert isinstance(processed, dict)

    def test_yaml_performance(self):
        """Test YAML formatter performance."""
        schema = {
            "type": "object",
            "properties": {f"field_{i}": {"type": "string"} for i in range(100)},
        }
        formatter = YAMLFormatter(schema)
        result = formatter.transform_schema()
        assert "field_0: str" in result
        assert len(result) > 1000


class TestFormatterEdgeCases:
    """Test edge cases for all formatters."""

    def test_all_formatters_with_empty_schema(self):
        """Test all formatters with empty schema."""
        formatters = [JSONishFormatter({}), TypeScriptFormatter({}), YAMLFormatter({})]

        for formatter in formatters:
            result = formatter.transform_schema()
            # Each formatter handles empty schema differently
            assert isinstance(result, str)

    def test_all_formatters_with_invalid_schema(self):
        """Test all formatters with invalid schema."""
        invalid_schema = {"invalid": "schema"}
        formatters = [
            JSONishFormatter(invalid_schema),
            TypeScriptFormatter(invalid_schema),
            YAMLFormatter(invalid_schema),
        ]

        for formatter in formatters:
            result = formatter.transform_schema()
            # Should handle gracefully
            assert isinstance(result, str)

    def test_all_formatters_metadata_disabled(self):
        """Test all formatters with metadata disabled."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string", "minLength": 5, "maxLength": 20}},
        }

        formatters = [
            JSONishFormatter(schema, include_metadata=False),
            TypeScriptFormatter(schema, include_metadata=False),
            YAMLFormatter(schema, include_metadata=False),
        ]

        for formatter in formatters:
            result = formatter.transform_schema()
            # Should not include metadata comments
            assert "minLength" not in result
            assert "maxLength" not in result
