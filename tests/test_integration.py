"""Consolidated integration tests for all formatters."""

import json

import pytest

from llm_schema_lite import simplify_schema

# Import models from conftest (fixtures are auto-discovered but models need explicit import)
from .conftest import ComplexTypes, Order, Profile, User


class TestFormatterIntegration:
    """Integration tests for all formatters."""

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_complete_workflow(self, format_type):
        """Test complete workflow for each formatter."""
        # Test with a complex model
        schema = simplify_schema(Order, format_type=format_type, include_metadata=True)

        # Test all SchemaLite methods
        schema_dict = schema.to_dict()
        schema_json = schema.to_json()
        schema_string = schema.to_string()

        # All methods should return valid data
        assert isinstance(schema_dict, dict)
        assert isinstance(schema_json, str)
        assert isinstance(schema_string, str)

        # String output should be substantial
        assert len(schema_string.strip()) > 0

        # JSON should be valid
        parsed_json = json.loads(schema_json)
        assert isinstance(parsed_json, dict)

        # Dict should contain expected structure (formatter-specific)
        assert len(schema_dict) > 0

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_metadata_workflow(self, format_type):
        """Test metadata handling workflow."""
        # Test with metadata
        schema_with_metadata = simplify_schema(User, format_type=format_type, include_metadata=True)
        output_with_metadata = schema_with_metadata.to_string()

        # Test without metadata
        schema_without_metadata = simplify_schema(
            User, format_type=format_type, include_metadata=False
        )
        output_without_metadata = schema_without_metadata.to_string()

        # Output with metadata should be longer
        assert len(output_with_metadata) > len(output_without_metadata)

        # Both should include field names
        for field in ["name", "active", "role"]:
            assert field in output_with_metadata
            assert field in output_without_metadata

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_complex_model_workflow(self, format_type):
        """Test complex model workflow."""
        schema = simplify_schema(ComplexTypes, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should handle complex types
        assert len(output.strip()) > 0

        # Should include various field types
        field_indicators = ["string", "int", "float", "bool", "array", "object"]
        has_fields = any(indicator in output for indicator in field_indicators)
        assert has_fields, f"No field types found in {format_type} output"

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_nested_references_workflow(self, format_type):
        """Test nested references workflow."""
        schema = simplify_schema(Order, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should handle nested references
        assert "user" in output
        assert "products" in output

        # Should produce substantial output for nested structure
        assert len(output.strip()) > 100

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_validation_constraints_workflow(self, format_type):
        """Test validation constraints workflow."""
        schema = simplify_schema(User, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should handle validation constraints
        assert len(output.strip()) > 0

        # Should include constraint information when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            constraint_indicators = ["minLength:", "maxLength:", "pattern:", "min:", "max:"]
            has_constraints = any(indicator in output for indicator in constraint_indicators)
            assert has_constraints, f"No constraints found in {format_type} output"

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_union_types_workflow(self, format_type):
        """Test union types workflow."""
        schema = simplify_schema(ComplexTypes, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should handle union types
        assert len(output.strip()) > 0

        # Should include union type indicators
        if format_type == "typescript":
            assert " | " in output
        elif format_type == "yaml":
            assert " | " in output
        elif format_type == "jsonish":
            assert " or " in output or "oneOf:" in output

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_optional_fields_workflow(self, format_type):
        """Test optional fields workflow."""
        schema = simplify_schema(Profile, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should handle optional fields
        assert len(output.strip()) > 0

        # Should include optional field indicators
        if format_type == "typescript":
            assert " | null" in output or " | undefined" in output
        elif format_type == "yaml":
            assert " | None" in output
        elif format_type == "jsonish":
            assert "?" in output or "null" in output

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_empty_schema_workflow(self, format_type):
        """Test empty schema workflow."""
        empty_schema = {"type": "object"}
        schema = simplify_schema(empty_schema, format_type=format_type)
        output = schema.to_string()

        # Should handle empty schema gracefully
        assert output is not None
        assert isinstance(output, str)

        # Should produce appropriate empty representation
        if format_type == "jsonish":
            assert output == "{}"
        elif format_type == "typescript":
            assert "interface Schema {}" in output
        elif format_type == "yaml":
            assert output == "{}"


class TestCrossFormatterConsistency:
    """Test consistency across formatters."""

    def test_field_name_consistency(self):
        """Test that all formatters include the same field names."""
        model = User

        outputs = {}
        for format_type in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(model, format_type=format_type, include_metadata=False)
            outputs[format_type] = schema.to_string()

        # All formatters should include the same field names
        expected_fields = ["name", "active", "role"]
        for format_type, output in outputs.items():
            for field in expected_fields:
                assert field in output, f"Missing field '{field}' in {format_type} output"

    def test_metadata_consistency(self):
        """Test that metadata is consistently included/excluded."""
        model = User

        # Test with metadata
        outputs_with_metadata = {}
        for format_type in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(model, format_type=format_type, include_metadata=True)
            outputs_with_metadata[format_type] = schema.to_string()

        # Test without metadata
        outputs_without_metadata = {}
        for format_type in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(model, format_type=format_type, include_metadata=False)
            outputs_without_metadata[format_type] = schema.to_string()

        # All formatters should have longer output with metadata
        for format_type in ["jsonish", "typescript", "yaml"]:
            with_metadata = outputs_with_metadata[format_type]
            without_metadata = outputs_without_metadata[format_type]
            assert len(with_metadata) > len(without_metadata), f"{format_type} metadata not working"

    def test_type_mapping_consistency(self):
        """Test that type mappings are consistent within each formatter."""
        model = ComplexTypes

        # Test each formatter's type mapping
        jsonish_schema = simplify_schema(model, format_type="jsonish", include_metadata=False)
        jsonish_output = jsonish_schema.to_string()

        typescript_schema = simplify_schema(model, format_type="typescript", include_metadata=False)
        typescript_output = typescript_schema.to_string()

        yaml_schema = simplify_schema(model, format_type="yaml", include_metadata=False)
        yaml_output = yaml_schema.to_string()

        # JSONish should use its specific type mappings
        assert "int" in jsonish_output  # integer -> int
        assert "float" in jsonish_output  # number -> float
        assert "bool" in jsonish_output  # boolean -> bool

        # TypeScript should use its specific type mappings
        assert "number" in typescript_output  # integer -> number
        assert "boolean" in typescript_output  # boolean -> boolean
        assert "string" in typescript_output  # string -> string

        # YAML should use its specific type mappings
        assert "str" in yaml_output  # string -> str
        assert "int" in yaml_output  # integer -> int
        assert "float" in yaml_output  # number -> float
        assert "bool" in yaml_output  # boolean -> bool


class TestErrorHandlingIntegration:
    """Test error handling across all formatters."""

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_invalid_schema_handling(self, format_type):
        """Test handling of invalid schemas."""
        # Test with malformed schema
        invalid_schema = {"invalid": "schema"}

        # Should handle gracefully without crashing
        try:
            schema = simplify_schema(invalid_schema, format_type=format_type)
            output = schema.to_string()
            assert isinstance(output, str)
        except Exception as e:
            # If it raises an exception, it should be a reasonable one
            assert isinstance(e, ValueError | TypeError | KeyError)

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_missing_properties_handling(self, format_type):
        """Test handling of schemas with missing properties."""
        # Test with schema missing properties
        no_props_schema = {"type": "object"}

        schema = simplify_schema(no_props_schema, format_type=format_type)
        output = schema.to_string()

        # Should handle gracefully
        assert isinstance(output, str)
        assert len(output.strip()) > 0

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_circular_reference_handling(self, format_type):
        """Test handling of circular references."""
        # Test with potential circular reference
        circular_schema = {
            "type": "object",
            "properties": {
                "self_ref": {"anyOf": [{"type": "string"}, {"$ref": "#/properties/self_ref"}]}
            },
        }

        # Should handle gracefully without infinite recursion
        schema = simplify_schema(circular_schema, format_type=format_type)
        output = schema.to_string()

        assert isinstance(output, str)
        assert len(output.strip()) > 0
        assert "self_ref" in output


class TestPerformanceIntegration:
    """Performance integration tests."""

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_large_schema_performance(self, format_type):
        """Test performance with large schemas."""
        import time

        # Create a large schema
        large_schema = {
            "type": "object",
            "properties": {
                f"field_{i}": {
                    "type": "string",
                    "description": f"Field {i} description",
                    "minLength": 1,
                    "maxLength": 100,
                }
                for i in range(100)
            },
        }

        start_time = time.time()
        schema = simplify_schema(large_schema, format_type=format_type, include_metadata=True)
        output = schema.to_string()
        end_time = time.time()

        processing_time = end_time - start_time

        # Should process within reasonable time
        assert (
            processing_time < 2.0
        ), f"{format_type} processing took too long: {processing_time:.3f}s"

        # Should produce substantial output
        assert len(output) > 2000, f"{format_type} output too short: {len(output)} characters"

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_repeated_processing_performance(self, format_type):
        """Test performance with repeated processing."""
        import time

        test_schema = {
            "type": "object",
            "properties": {
                f"field_{i}": {"type": "string", "description": f"Field {i}"} for i in range(20)
            },
        }

        # Process multiple times
        times = []
        for _ in range(5):
            start_time = time.time()
            schema = simplify_schema(test_schema, format_type=format_type, include_metadata=True)
            schema.to_string()  # Process the schema
            end_time = time.time()
            times.append(end_time - start_time)

        # Average time should be reasonable
        avg_time = sum(times) / len(times)
        assert avg_time < 0.5, f"{format_type} average processing time too high: {avg_time:.3f}s"

        # Should be consistent
        max_time = max(times)
        min_time = min(times)
        assert max_time - min_time < 0.2, f"{format_type} processing time too inconsistent"


class TestSchemaLiteIntegration:
    """Test SchemaLite class integration."""

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_schema_lite_methods(self, format_type):
        """Test all SchemaLite methods."""
        schema = simplify_schema(User, format_type=format_type, include_metadata=True)

        # Test to_dict
        schema_dict = schema.to_dict()
        assert isinstance(schema_dict, dict)
        assert len(schema_dict) > 0

        # Test to_json
        schema_json = schema.to_json()
        assert isinstance(schema_json, str)
        parsed_json = json.loads(schema_json)
        assert isinstance(parsed_json, dict)

        # Test to_string
        schema_string = schema.to_string()
        assert isinstance(schema_string, str)
        assert len(schema_string.strip()) > 0

        # Test to_yaml (if available)
        if hasattr(schema, "to_yaml"):
            try:
                schema_yaml = schema.to_yaml()
                assert isinstance(schema_yaml, str)
                assert len(schema_yaml.strip()) > 0
            except ImportError:
                # PyYAML not available, skip this test
                pass

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_schema_lite_attributes(self, format_type):
        """Test SchemaLite attributes."""
        schema = simplify_schema(User, format_type=format_type, include_metadata=True)

        # Should have expected attributes
        assert hasattr(schema, "_data")
        assert hasattr(schema, "_formatter")
        assert hasattr(schema, "_original_schema")

        # Should have expected methods
        assert hasattr(schema, "to_dict")
        assert hasattr(schema, "to_json")
        assert hasattr(schema, "to_string")
        assert hasattr(schema, "to_yaml")

        # Data should be accessible
        assert isinstance(schema._data, dict)
        assert len(schema._data) > 0

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_schema_lite_consistency(self, format_type):
        """Test that SchemaLite objects maintain consistency."""
        schema = simplify_schema(User, format_type=format_type, include_metadata=True)

        # Test that methods return consistent results
        dict1 = schema.to_dict()
        dict2 = schema.to_dict()
        assert dict1 == dict2

        string1 = schema.to_string()
        string2 = schema.to_string()
        assert string1 == string2

        # Test that data is accessible
        assert isinstance(schema._data, dict)
        assert len(schema._data) > 0
