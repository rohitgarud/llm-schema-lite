#!/usr/bin/env python3
"""
Comprehensive tests for the analyze_schema_features function.

This test suite covers all JSON Schema features and edge cases to ensure
accurate feature detection and recursive analysis.
"""

from ..base import analyze_schema_features


class TestAnalyzeSchemaFeatures:
    """Test suite for analyze_schema_features function."""

    def test_basic_features(self):
        """Test detection of basic JSON Schema features."""
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number", "minimum": 0, "maximum": 120},
                "email": {"type": "string", "format": "email"},
                "status": {"enum": ["active", "inactive"]},
                "id": {"const": "user123"},
            },
        }

        features = analyze_schema_features(schema)
        expected_features = {
            "type",
            "required",
            "properties",
            "minimum",
            "maximum",
            "format",
            "enum",
            "const",
        }

        assert set(features) == expected_features

    def test_array_features(self):
        """Test detection of array-related features."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10,
            "uniqueItems": True,
            "contains": {"type": "string", "pattern": "test"},
            "additionalItems": False,
        }

        features = analyze_schema_features(schema)
        expected_features = {
            "type",
            "items",
            "minItems",
            "maxItems",
            "uniqueItems",
            "contains",
            "pattern",
            "additionalItems",
        }

        assert set(features) == expected_features

    def test_array_items_as_list(self):
        """Test detection of tuple validation with items as array."""
        schema = {
            "type": "array",
            "items": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}],
            "additionalItems": {"type": "string"},
        }

        features = analyze_schema_features(schema)
        expected_features = {"type", "items", "additionalItems"}

        assert set(features) == expected_features

    def test_union_features(self):
        """Test detection of union/choice features."""
        schema = {
            "oneOf": [{"type": "string"}, {"type": "number"}],
            "anyOf": [{"type": "object"}, {"type": "array"}],
            "allOf": [{"type": "object"}, {"required": ["id"]}],
        }

        features = analyze_schema_features(schema)
        expected_features = {"oneOf", "anyOf", "allOf", "type", "required"}

        assert set(features) == expected_features

    def test_conditional_logic_valid(self):
        """Test detection of valid if/then/else conditional logic."""
        schema = {
            "type": "object",
            "if": {"properties": {"age": {"type": "number"}}},
            "then": {"properties": {"age": {"minimum": 0}}},
            "else": {"properties": {"age": {"type": "string"}}},
        }

        features = analyze_schema_features(schema)
        expected_features = {"type", "if", "then", "else", "properties", "minimum"}

        assert set(features) == expected_features

    def test_conditional_logic_if_then_only(self):
        """Test detection of if/then without else."""
        schema = {
            "type": "object",
            "if": {"properties": {"age": {"type": "number"}}},
            "then": {"properties": {"age": {"minimum": 0}}},
        }

        features = analyze_schema_features(schema)
        expected_features = {"type", "if", "then", "properties", "minimum"}

        assert set(features) == expected_features
        assert "else" not in features

    def test_conditional_logic_invalid_if_only(self):
        """Test that if without then is not detected as a feature."""
        schema = {"type": "object", "if": {"properties": {"age": {"type": "number"}}}}

        features = analyze_schema_features(schema)
        expected_features = {"type", "properties"}

        assert set(features) == expected_features
        assert "if" not in features
        assert "then" not in features
        assert "else" not in features

    def test_conditional_logic_invalid_then_only(self):
        """Test that then without if is not detected as a feature."""
        schema = {"type": "object", "then": {"properties": {"age": {"minimum": 0}}}}

        features = analyze_schema_features(schema)
        expected_features = {"type", "properties", "minimum"}

        assert set(features) == expected_features
        assert "if" not in features
        assert "then" not in features
        assert "else" not in features

    def test_not_keyword_valid(self):
        """Test detection of valid not keyword with schema."""
        schema = {"type": "object", "not": {"properties": {"secret": {"type": "string"}}}}

        features = analyze_schema_features(schema)
        expected_features = {"type", "not", "properties"}

        assert set(features) == expected_features

    def test_not_keyword_invalid_boolean(self):
        """Test that not with boolean value is not detected."""
        schema = {"type": "object", "not": True}

        features = analyze_schema_features(schema)
        expected_features = {"type"}

        assert set(features) == expected_features
        assert "not" not in features

    def test_not_keyword_invalid_string(self):
        """Test that not with string value is not detected."""
        schema = {"type": "object", "not": "invalid"}

        features = analyze_schema_features(schema)
        expected_features = {"type"}

        assert set(features) == expected_features
        assert "not" not in features

    def test_object_features(self):
        """Test detection of object-related features."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "patternProperties": {"^test_": {"type": "string"}},
            "additionalProperties": False,
            "minProperties": 1,
            "maxProperties": 5,
            "propertyNames": {"pattern": "^[a-zA-Z]"},
            "unevaluatedProperties": False,
            "dependencies": {"age": ["name"]},
        }

        features = analyze_schema_features(schema)
        expected_features = {
            "type",
            "properties",
            "patternProperties",
            "additionalProperties",
            "minProperties",
            "maxProperties",
            "propertyNames",
            "pattern",
            "unevaluatedProperties",
            "dependencies",
        }

        assert set(features) == expected_features

    def test_validation_constraints(self):
        """Test detection of validation constraint features."""
        schema = {"type": "string", "minLength": 1, "maxLength": 100, "pattern": "^[a-z]+$"}

        features = analyze_schema_features(schema)
        expected_features = {"type", "minLength", "maxLength", "pattern"}

        assert set(features) == expected_features

    def test_numeric_constraints(self):
        """Test detection of numeric validation constraints."""
        schema = {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 100,
            "multipleOf": 5,
        }

        features = analyze_schema_features(schema)
        expected_features = {
            "type",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
        }

        assert set(features) == expected_features

    def test_metadata_features(self):
        """Test detection of metadata features."""
        schema = {
            "type": "object",
            "title": "User Schema",
            "description": "A user object",
            "default": {"name": "John", "age": 30},
        }

        features = analyze_schema_features(schema)
        expected_features = {"type", "title", "description", "default"}

        assert set(features) == expected_features

    def test_ref_keyword(self):
        """Test detection of $ref keyword."""
        schema = {"type": "object", "properties": {"user": {"$ref": "#/definitions/User"}}}

        features = analyze_schema_features(schema)
        expected_features = {"type", "properties", "$ref"}

        assert set(features) == expected_features

    def test_recursive_nested_schemas(self):
        """Test recursive detection in deeply nested schemas."""
        schema = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "profile": {
                                "type": "object",
                                "properties": {
                                    "settings": {
                                        "type": "object",
                                        "properties": {"theme": {"enum": ["light", "dark"]}},
                                    }
                                },
                            }
                        },
                    },
                }
            },
        }

        features = analyze_schema_features(schema)
        expected_features = {"type", "properties", "items", "enum"}

        assert set(features) == expected_features

    def test_complex_conditional_with_nested_features(self):
        """Test complex conditional logic with nested features."""
        schema = {
            "type": "object",
            "if": {"properties": {"type": {"enum": ["admin"]}}},
            "then": {
                "properties": {
                    "permissions": {"type": "array", "items": {"type": "string"}, "minItems": 1}
                },
                "required": ["permissions"],
            },
            "else": {"properties": {"role": {"type": "string", "pattern": "^user_"}}},
        }

        features = analyze_schema_features(schema)
        expected_features = {
            "type",
            "if",
            "then",
            "else",
            "properties",
            "enum",
            "items",
            "minItems",
            "required",
            "pattern",
        }

        assert set(features) == expected_features

    def test_not_with_nested_features(self):
        """Test not keyword with nested schema features."""
        schema = {
            "type": "object",
            "not": {
                "type": "object",
                "properties": {"secret": {"type": "string", "pattern": ".*secret.*"}},
                "required": ["secret"],
            },
        }

        features = analyze_schema_features(schema)
        expected_features = {"type", "not", "properties", "pattern", "required"}

        assert set(features) == expected_features

    def test_dependencies_with_schema_values(self):
        """Test dependencies with schema values."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
            "dependencies": {"age": {"properties": {"birth_year": {"type": "number"}}}},
        }

        features = analyze_schema_features(schema)
        expected_features = {"type", "properties", "dependencies"}

        assert set(features) == expected_features

    def test_empty_schema(self):
        """Test empty schema."""
        schema = {}

        features = analyze_schema_features(schema)

        assert features == []

    def test_non_dict_input(self):
        """Test non-dict input (should not crash)."""
        schema = "not a dict"

        features = analyze_schema_features(schema)

        assert features == []

    def test_none_input(self):
        """Test None input (should not crash)."""
        schema = None

        features = analyze_schema_features(schema)

        assert features == []

    def test_all_features_comprehensive(self):
        """Test a comprehensive schema with all possible features."""
        schema = {
            "type": "object",
            "title": "Comprehensive Schema",
            "description": "A schema with all features",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "pattern": "^[0-9]+$"},
                "name": {"type": "string", "minLength": 1, "maxLength": 100},
                "age": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 120,
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 120,
                    "multipleOf": 1,
                },
                "email": {"type": "string", "format": "email"},
                "status": {"enum": ["active", "inactive", "pending"]},
                "role": {"const": "user"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 10,
                    "uniqueItems": True,
                    "contains": {"type": "string", "pattern": "important"},
                    "additionalItems": False,
                },
                "metadata": {
                    "type": "object",
                    "patternProperties": {"^meta_": {"type": "string"}},
                    "additionalProperties": False,
                    "minProperties": 0,
                    "maxProperties": 5,
                    "propertyNames": {"pattern": "^[a-z_]+$"},
                    "unevaluatedProperties": False,
                },
            },
            "minProperties": 1,
            "maxProperties": 10,
            "additionalProperties": False,
            "patternProperties": {"^extra_": {"type": "string"}},
            "propertyNames": {"pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$"},
            "unevaluatedProperties": False,
            "dependencies": {"age": ["birth_year"]},
            "oneOf": [{"type": "object"}, {"type": "array"}],
            "anyOf": [{"type": "string"}, {"type": "number"}],
            "allOf": [{"type": "object"}, {"required": ["id"]}],
            "not": {"type": "object", "properties": {"secret": {"type": "string"}}},
            "if": {"properties": {"age": {"type": "number"}}},
            "then": {"properties": {"age": {"minimum": 0}}},
            "else": {"properties": {"age": {"type": "string"}}},
            "default": {"id": "default", "name": "Default User"},
        }

        features = analyze_schema_features(schema)

        # Should detect all features present in the schema
        expected_features = {
            "type",
            "title",
            "description",
            "required",
            "properties",
            "pattern",
            "minLength",
            "maxLength",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "format",
            "enum",
            "const",
            "items",
            "minItems",
            "maxItems",
            "uniqueItems",
            "contains",
            "additionalItems",
            "patternProperties",
            "additionalProperties",
            "minProperties",
            "maxProperties",
            "propertyNames",
            "unevaluatedProperties",
            "dependencies",
            "oneOf",
            "anyOf",
            "allOf",
            "not",
            "if",
            "then",
            "else",
            "default",
        }

        assert set(features) == expected_features

    def test_feature_deduplication(self):
        """Test that features are not duplicated in the result."""
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"deep": {"type": "string", "pattern": "^test$"}},
                }
            },
        }

        features = analyze_schema_features(schema)

        # Each feature should appear only once
        assert len(features) == len(set(features))
        assert features.count("type") == 1
        assert features.count("properties") == 1
        assert features.count("pattern") == 1

    def test_malformed_schema_handling(self):
        """Test handling of malformed schema structures."""
        schema = {
            "type": "object",
            "properties": {
                "valid": {"type": "string"},
                "invalid_items": {
                    "items": "not_a_schema"  # items should be schema or array of schemas
                },
                "invalid_not": {
                    "not": "not_a_schema"  # not should be a schema
                },
            },
        }

        features = analyze_schema_features(schema)

        # Should still detect valid features and not crash
        expected_features = {"type", "properties"}
        assert set(features) == expected_features
        assert "not" not in features  # invalid not should not be detected
