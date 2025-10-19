"""Comprehensive tests for advanced JSON Schema features."""

import pytest

from llm_schema_lite import simplify_schema


class TestAdvancedJSONSchemaFeatures:
    """Test advanced JSON Schema features across all formatters."""

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_dependencies_feature(self, dependency_schema, format_type):
        """Test dependencies feature handling."""
        schema = simplify_schema(dependency_schema, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should include the main properties
        assert "name" in output
        assert "credit_card" in output
        assert "billing_address" in output

        # Should include dependency information when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            assert "dependencies:" in output or "requires" in output

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_conditional_schemas(self, conditional_schema, format_type):
        """Test if/then/else conditional schemas."""
        schema = simplify_schema(conditional_schema, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should include the main properties
        assert "type" in output
        assert "permissions" in output

        # Should include conditional information when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            assert "if " in output and "then " in output

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_pattern_properties(self, pattern_properties_schema, format_type):
        """Test patternProperties feature."""
        schema = simplify_schema(
            pattern_properties_schema, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include pattern properties information when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            assert "patternProperties:" in output

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_contains_feature(self, contains_schema, format_type):
        """Test contains feature for arrays."""
        schema = simplify_schema(contains_schema, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should include contains information when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            assert "contains:" in output

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_unique_items(self, unique_items_schema, format_type):
        """Test uniqueItems feature."""
        schema = simplify_schema(
            unique_items_schema, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include unique items information when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            assert "unique items" in output or "uniqueItems:" in output

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_property_names(self, property_names_schema, format_type):
        """Test propertyNames feature."""
        schema = simplify_schema(
            property_names_schema, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include property names information when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            assert "propertyNames:" in output

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_unevaluated_properties(self, unevaluated_properties_schema, format_type):
        """Test unevaluatedProperties feature."""
        schema = simplify_schema(
            unevaluated_properties_schema, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include unevaluated properties information when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            assert "unevaluated" in output or "unevaluatedProperties:" in output

    def test_complex_anyof_combinations(self, format_type="jsonish"):
        """Test complex anyOf combinations."""
        complex_anyof_schema = {
            "type": "object",
            "properties": {
                "flexible_field": {
                    "anyOf": [
                        {"type": "string", "enum": ["option1", "option2"]},
                        {"type": "integer", "minimum": 0, "maximum": 100},
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                }
            },
        }

        schema = simplify_schema(
            complex_anyof_schema, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include the field
        assert "flexible_field" in output

        # Should handle the complex anyOf appropriately
        if format_type == "jsonish":
            assert " or " in output or "oneOf:" in output
        elif format_type == "typescript":
            assert " | " in output
        elif format_type == "yaml":
            assert " | " in output

    def test_complex_oneof_combinations(self, format_type="jsonish"):
        """Test complex oneOf combinations."""
        complex_oneof_schema = {
            "type": "object",
            "properties": {
                "exclusive_field": {
                    "oneOf": [
                        {"type": "string", "pattern": "^[A-Z]+$"},
                        {"type": "integer", "multipleOf": 5},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                }
            },
        }

        schema = simplify_schema(
            complex_oneof_schema, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include the field
        assert "exclusive_field" in output

        # Should handle the complex oneOf appropriately
        if format_type == "jsonish":
            assert "oneOf:" in output
        elif format_type == "typescript":
            assert " | " in output
        elif format_type == "yaml":
            assert " | " in output

    def test_complex_allof_combinations(self, format_type="jsonish"):
        """Test complex allOf combinations."""
        complex_allof_schema = {
            "type": "object",
            "properties": {
                "constrained_field": {
                    "allOf": [
                        {"type": "string"},
                        {"minLength": 5},
                        {"maxLength": 20},
                        {"pattern": "^[A-Za-z]+$"},
                    ]
                }
            },
        }

        schema = simplify_schema(
            complex_allof_schema, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include the field
        assert "constrained_field" in output

        # Should handle the complex allOf appropriately
        if format_type == "jsonish":
            assert "allOf:" in output or " & " in output
        elif format_type == "typescript":
            assert " & " in output
        elif format_type == "yaml":
            assert " & " in output

    def test_not_negation(self, format_type="jsonish"):
        """Test not (negation) feature."""
        not_schema = {
            "type": "object",
            "properties": {"non_numeric_field": {"not": {"type": "number"}}},
        }

        schema = simplify_schema(not_schema, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should include the field
        assert "non_numeric_field" in output

        # Should handle the not appropriately
        if format_type == "jsonish":
            assert "not:" in output
        elif format_type == "typescript":
            # TypeScript doesn't have direct not support, should fallback gracefully
            assert "string" in output or "object" in output
        elif format_type == "yaml":
            assert "not:" in output

    def test_nested_conditional_schemas(self, format_type="jsonish"):
        """Test nested conditional schemas."""
        nested_conditional_schema = {
            "type": "object",
            "properties": {
                "user_type": {"type": "string", "enum": ["admin", "user"]},
                "permissions": {"type": "array", "items": {"type": "string"}},
                "access_level": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "if": {"properties": {"user_type": {"const": "admin"}}},
            "then": {
                "properties": {"permissions": {"minItems": 1}, "access_level": {"minimum": 5}}
            },
            "else": {
                "properties": {"permissions": {"maxItems": 0}, "access_level": {"maximum": 3}}
            },
        }

        schema = simplify_schema(
            nested_conditional_schema, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include all properties
        assert "user_type" in output
        assert "permissions" in output
        assert "access_level" in output

        # Should include conditional information when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            assert "if " in output and "then " in output

    def test_multiple_pattern_properties(self, format_type="jsonish"):
        """Test multiple pattern properties."""
        multiple_patterns_schema = {
            "type": "object",
            "patternProperties": {
                "^[a-z]+$": {"type": "string", "minLength": 1},
                "^[0-9]+$": {"type": "number", "minimum": 0},
                "^[A-Z]+$": {"type": "boolean"},
            },
        }

        schema = simplify_schema(
            multiple_patterns_schema, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include pattern properties information when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            assert "patternProperties:" in output

    def test_array_with_contains_and_unique(self, format_type="jsonish"):
        """Test array with both contains and uniqueItems."""
        array_complex_schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "contains": {"type": "string", "pattern": "^admin_"},
                    "uniqueItems": True,
                    "minItems": 1,
                    "maxItems": 10,
                }
            },
        }

        schema = simplify_schema(
            array_complex_schema, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include the field
        assert "tags" in output

        # Should include array constraints when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            assert "contains:" in output or "unique items" in output or "uniqueItems:" in output

    def test_object_with_property_names_and_unevaluated(self, format_type="jsonish"):
        """Test object with both propertyNames and unevaluatedProperties."""
        object_complex_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "propertyNames": {"pattern": "^[a-zA-Z][a-zA-Z0-9_]*$"},
            "unevaluatedProperties": False,
        }

        schema = simplify_schema(
            object_complex_schema, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include the field
        assert "name" in output

        # Should include object constraints when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            assert (
                "propertyNames:" in output
                or "unevaluated" in output
                or "unevaluatedProperties:" in output
            )


class TestAdvancedFeatureCombinations:
    """Test combinations of advanced features."""

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_allof_with_anyof_and_oneof(self, format_type):
        """Test allOf combined with anyOf and oneOf."""
        complex_combination_schema = {
            "type": "object",
            "properties": {
                "complex_field": {
                    "allOf": [
                        {"anyOf": [{"type": "string"}, {"type": "number"}]},
                        {
                            "oneOf": [
                                {"type": "string", "minLength": 5},
                                {"type": "number", "minimum": 0},
                            ]
                        },
                    ]
                }
            },
        }

        schema = simplify_schema(
            complex_combination_schema, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include the field
        assert "complex_field" in output

        # Should handle the complex combination appropriately
        if format_type == "jsonish":
            assert "allOf:" in output or "anyOf:" in output or "oneOf:" in output
        elif format_type == "typescript":
            assert " | " in output or " & " in output
        elif format_type == "yaml":
            assert " | " in output or " & " in output

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_conditional_with_dependencies(self, format_type):
        """Test conditional schemas with dependencies."""
        conditional_dependency_schema = {
            "type": "object",
            "properties": {
                "user_type": {"type": "string", "enum": ["admin", "user"]},
                "permissions": {"type": "array"},
                "billing_info": {"type": "object"},
            },
            "if": {"properties": {"user_type": {"const": "admin"}}},
            "then": {"dependencies": {"permissions": ["billing_info"]}},
        }

        schema = simplify_schema(
            conditional_dependency_schema, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include all properties
        assert "user_type" in output
        assert "permissions" in output
        assert "billing_info" in output

        # Should include both conditional and dependency information when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            has_conditional = "if " in output and "then " in output
            has_dependencies = "dependencies:" in output or "requires" in output
            assert (
                has_conditional or has_dependencies
            ), f"Neither conditional nor dependency info found: {output}"

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_pattern_properties_with_conditional(self, format_type):
        """Test patternProperties with conditional schemas."""
        pattern_conditional_schema = {
            "type": "object",
            "properties": {"mode": {"type": "string", "enum": ["strict", "loose"]}},
            "patternProperties": {"^[a-z]+$": {"type": "string"}},
            "if": {"properties": {"mode": {"const": "strict"}}},
            "then": {"patternProperties": {"^[a-z]+$": {"minLength": 3}}},
        }

        schema = simplify_schema(
            pattern_conditional_schema, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include the field
        assert "mode" in output

        # Should include pattern properties information when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            assert "patternProperties:" in output or "if:" in output or "then:" in output


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling for advanced features."""

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_empty_anyof_array(self, format_type):
        """Test handling of empty anyOf array."""
        empty_anyof_schema = {"type": "object", "properties": {"empty_anyof": {"anyOf": []}}}

        schema = simplify_schema(empty_anyof_schema, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should handle empty anyOf gracefully
        assert "empty_anyof" in output
        assert len(output.strip()) > 0

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_empty_oneof_array(self, format_type):
        """Test handling of empty oneOf array."""
        empty_oneof_schema = {"type": "object", "properties": {"empty_oneof": {"oneOf": []}}}

        schema = simplify_schema(empty_oneof_schema, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should handle empty oneOf gracefully
        assert "empty_oneof" in output
        assert len(output.strip()) > 0

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_empty_allof_array(self, format_type):
        """Test handling of empty allOf array."""
        empty_allof_schema = {"type": "object", "properties": {"empty_allof": {"allOf": []}}}

        schema = simplify_schema(empty_allof_schema, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should handle empty allOf gracefully
        assert "empty_allof" in output
        assert len(output.strip()) > 0

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_malformed_conditional(self, format_type):
        """Test handling of malformed conditional schemas."""
        malformed_conditional_schema = {
            "type": "object",
            "properties": {"field": {"type": "string"}},
            "if": {"type": "string"},  # Missing then/else
            "then": {"type": "integer"},
        }

        schema = simplify_schema(
            malformed_conditional_schema, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should handle malformed conditional gracefully
        assert "field" in output
        assert len(output.strip()) > 0

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_circular_references_in_advanced_features(self, format_type):
        """Test handling of circular references in advanced features."""
        # This would be a complex schema with circular references
        # For now, we'll test that the system doesn't crash
        circular_schema = {
            "type": "object",
            "properties": {
                "self_ref": {"anyOf": [{"type": "string"}, {"$ref": "#/properties/self_ref"}]}
            },
        }

        schema = simplify_schema(circular_schema, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should handle circular references gracefully
        assert "self_ref" in output
        assert len(output.strip()) > 0
