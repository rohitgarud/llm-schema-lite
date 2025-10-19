"""Consolidated tests for YAML formatter using base classes."""

import pytest

from llm_schema_lite import simplify_schema

# Import models from conftest (fixtures are auto-discovered but models need explicit import)
from .conftest import ComplexTypes, Order, Profile, SimpleUser, User
from .test_base_formatter import TestYAMLFormatter


class TestYAMLFormatterConsolidated(TestYAMLFormatter):
    """Consolidated YAML formatter tests using base class."""

    # Inherits all common tests from TestYAMLFormatter base class
    # Only YAML-specific tests are defined here

    def test_yaml_formatting_syntax(self):
        """Test YAML formatting syntax."""
        schema = simplify_schema(User, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Should use YAML key: value syntax
        assert ":" in output

        # Should use # for comments
        assert "#" in output

        # Should not have JSON-like braces in the main structure (but allow them in regex patterns)
        # Check that we don't have object definitions with curly braces
        lines = output.split("\n")
        for line in lines:
            if ":" in line and not line.strip().startswith("#"):
                # This is a main field definition line
                line.split(":")[0].strip()  # Extract field name
                value_part = line.split(":", 1)[1].strip() if ":" in line else ""
                # The value part should not contain curly braces (except in comments)
                if "{" in value_part and not value_part.strip().startswith("#"):
                    # Check if it's just a regex pattern in comments
                    if not ("pattern:" in value_part or "^" in value_part):
                        raise AssertionError(f"Found curly braces in main structure: {line}")

    def test_yaml_type_mapping(self):
        """Test YAML-specific type mapping."""
        schema = simplify_schema(ComplexTypes, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        # Should use YAML type mappings
        assert "str" in output  # string -> str
        assert "int" in output  # integer -> int
        assert "float" in output  # number -> float
        assert "bool" in output  # boolean -> bool
        assert "list" in output  # array -> list

    def test_yaml_union_types(self):
        """Test YAML union type syntax."""
        schema = simplify_schema(ComplexTypes, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Should use YAML union syntax
        assert " | " in output or "Union[" in output

    def test_yaml_literal_types(self):
        """Test YAML Literal type formatting."""
        schema = simplify_schema(User, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Should format enums as Literal types
        assert "Literal[" in output
        assert '"admin"' in output
        assert '"user"' in output
        assert '"guest"' in output

    def test_yaml_list_types(self):
        """Test YAML list type formatting."""
        schema = simplify_schema(ComplexTypes, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        # Should have list types
        assert "list[" in output or "list<" in output

    def test_yaml_nullable_types(self):
        """Test YAML nullable type formatting."""
        schema = simplify_schema(Profile, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Should handle nullable types
        assert " | None" in output

    def test_yaml_nested_sections(self):
        """Test YAML nested section generation."""
        schema = simplify_schema(Order, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Should generate nested sections
        assert "user" in output
        assert "products" in output

        # Should use dot notation for nested properties
        lines = output.split("\n")
        nested_lines = [line for line in lines if "." in line and ":" in line]
        assert len(nested_lines) > 0, "No nested properties found"

    def test_yaml_comment_syntax(self):
        """Test YAML comment syntax."""
        schema = simplify_schema(User, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Should use # for comments
        assert "#" in output

        # Comments should be properly formatted
        lines = output.split("\n")
        comment_lines = [line for line in lines if "#" in line]
        for line in comment_lines:
            assert line.strip().startswith("#") or "  #" in line, f"Malformed comment: {line}"

    def test_yaml_metadata_integration(self):
        """Test YAML metadata integration."""
        schema = simplify_schema(User, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Should integrate metadata properly
        assert "#" in output
        assert "description" in output or "The user's full name" in output
        assert "minLength:" in output or "maxLength:" in output
        assert "pattern:" in output

    def test_yaml_edge_cases(self):
        """Test YAML edge cases."""
        # Test empty schema
        empty_schema = {"type": "object"}
        schema = simplify_schema(empty_schema, format_type="yaml")
        output = schema.to_string()
        assert output == "{}"

        # Test schema with no properties
        no_props_schema = {"type": "object", "properties": {}}
        schema = simplify_schema(no_props_schema, format_type="yaml")
        output = schema.to_string()
        assert output == "{}"

    def test_yaml_integration_methods(self):
        """Test YAML integration with SchemaLite methods."""
        schema = simplify_schema(User, format_type="yaml", include_metadata=True)

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
        assert ":" in schema_string

    def test_yaml_complex_workflow(self):
        """Test complete YAML workflow."""
        # Create a complex schema
        complex_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1, "description": "User name"},
                        "age": {"type": "integer", "minimum": 0, "description": "User age"},
                    },
                },
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of items",
                },
                "count": {"type": "integer", "minimum": 0, "description": "Item count"},
            },
        }

        # Process with YAML formatter
        schema = simplify_schema(complex_schema, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Should handle complex structure
        assert "user" in output
        assert "items" in output
        assert ":" in output  # YAML key-value syntax
        assert "#" in output  # Comments

    def test_yaml_performance(self):
        """Test YAML formatter performance."""
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
        schema = simplify_schema(large_schema, format_type="yaml", include_metadata=True)
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
# Parameterized Tests for YAML-Specific Features
# ============================================================================


class TestYAMLParameterized:
    """Parameterized tests for YAML-specific features."""

    @pytest.mark.parametrize("include_metadata", [True, False])
    def test_metadata_inclusion(self, user_model, include_metadata):
        """Test metadata inclusion/exclusion."""
        schema = simplify_schema(user_model, format_type="yaml", include_metadata=include_metadata)
        output = schema.to_string()

        if include_metadata:
            assert "#" in output
            assert "description" in output or "The user's full name" in output
        else:
            # Should not have metadata comments
            lines = output.split("\n")
            metadata_lines = [line for line in lines if "#" in line]
            assert (
                len(metadata_lines) == 0
            ), f"Found metadata when include_metadata=False: {metadata_lines}"

    @pytest.mark.parametrize("model_class", [SimpleUser, User, ComplexTypes, Order, Profile])
    def test_various_models(self, model_class):
        """Test YAML formatter with various model types."""
        schema = simplify_schema(model_class, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Should produce valid output
        assert len(output.strip()) > 0
        assert ":" in output  # YAML key-value syntax

        # Should not have JSON-like braces in the main structure
        # Check that the main structure doesn't start with { or contain { at the beginning of lines
        lines = output.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                assert not stripped.startswith("{"), f"Line should not start with {{: {line}"
                assert not stripped.startswith("}"), f"Line should not start with }}: {line}"

    @pytest.mark.parametrize("format_type", ["yaml"])  # Only YAML for this test
    def test_yaml_specific_syntax(self, user_model, format_type):
        """Test YAML-specific syntax elements."""
        schema = simplify_schema(user_model, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should use YAML comment syntax
        assert "#" in output

        # Should use YAML type mappings
        assert "str" in output  # string -> str
        assert "int" in output  # integer -> int
        assert "bool" in output  # boolean -> bool

        # Should have YAML-like structure
        assert ":" in output  # Key-value pairs
        # Check that the main structure doesn't start with { or }
        lines = output.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                assert not stripped.startswith("{"), f"Line should not start with {{: {line}"
                assert not stripped.startswith("}"), f"Line should not start with }}: {line}"

    @pytest.mark.parametrize("nullable_type", ["string", "integer", "number", "boolean"])
    def test_nullable_types(self, nullable_type):
        """Test nullable type handling."""
        nullable_schema = {
            "type": "object",
            "properties": {"field": {"type": [nullable_type, "null"]}},
        }

        schema = simplify_schema(nullable_schema, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        # Should handle nullable types
        assert " | None" in output

    @pytest.mark.parametrize(
        "list_type,expected_type",
        [("string", "str"), ("integer", "int"), ("number", "float"), ("boolean", "bool")],
    )
    def test_list_types(self, list_type, expected_type):
        """Test list type handling."""
        list_schema = {
            "type": "object",
            "properties": {"field": {"type": "array", "items": {"type": list_type}}},
        }

        schema = simplify_schema(list_schema, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        # Should handle list types with Python type names
        assert f"list[{expected_type}]" in output

    @pytest.mark.parametrize(
        "enum_values",
        [["active", "inactive"], ["admin", "user", "guest"], ["red", "green", "blue", "yellow"]],
    )
    def test_enum_literals(self, enum_values):
        """Test enum literal formatting."""
        enum_schema = {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": enum_values}},
        }

        schema = simplify_schema(enum_schema, format_type="yaml", include_metadata=False)
        output = schema.to_string()

        # Should format enums as Literal types
        assert "Literal[" in output
        for value in enum_values:
            assert f'"{value}"' in output
