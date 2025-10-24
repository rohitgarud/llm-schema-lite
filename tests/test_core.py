"""Tests for core functionality (SchemaLite and simplify_schema)."""

import json

import pytest

from llm_schema_lite import SchemaLite, simplify_schema
from llm_schema_lite.exceptions import ConversionError, UnsupportedModelError

# Import models from conftest (fixtures are auto-discovered but models need explicit import)
from .conftest import Order, SimpleUser, User


class TestSimplifySchema:
    """Test simplify_schema function."""

    def test_with_pydantic_model(self):
        """Test simplify_schema with Pydantic model."""
        schema = simplify_schema(SimpleUser)
        assert isinstance(schema, SchemaLite)

    def test_with_dict_schema(self):
        """Test simplify_schema with JSON schema dict."""
        schema_dict = SimpleUser.model_json_schema()
        schema = simplify_schema(schema_dict)
        assert isinstance(schema, SchemaLite)

    def test_with_string_schema(self):
        """Test simplify_schema with JSON schema string."""
        schema_string = (
            '{"type": "object", "properties": {"name": {"type": "string"}, '
            '"age": {"type": "integer"}}}'
        )
        schema = simplify_schema(schema_string)
        assert isinstance(schema, SchemaLite)

        # Test that it produces the same output as dict version
        schema_dict = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        schema_from_dict = simplify_schema(schema_dict)
        assert schema.to_string() == schema_from_dict.to_string()

    def test_with_invalid_string_schema(self):
        """Test simplify_schema with invalid JSON string."""
        with pytest.raises(ConversionError):
            simplify_schema('{"invalid": json}')  # Invalid JSON

    def test_string_schema_with_all_formats(self):
        """Test string schema with all format types."""
        schema_string = (
            '{"type": "object", "properties": {"name": {"type": "string"}, '
            '"age": {"type": "integer"}}}'
        )

        # JSONish format
        schema_jsonish = simplify_schema(schema_string, format_type="jsonish")
        assert "{" in schema_jsonish.to_string()

        # TypeScript format
        schema_ts = simplify_schema(schema_string, format_type="typescript")
        assert "interface Schema {" in schema_ts.to_string()

        # YAML format
        schema_yaml = simplify_schema(schema_string, format_type="yaml")
        output = schema_yaml.to_string()
        assert "{" not in output and "interface" not in output

    def test_string_schema_with_metadata(self):
        """Test string schema with metadata handling."""
        schema_string = (
            '{"type": "object", "properties": {"name": {"type": "string", '
            '"minLength": 1, "description": "User name"}}}'
        )

        # With metadata
        schema_with = simplify_schema(schema_string, include_metadata=True)
        output_with = schema_with.to_string()
        assert "//" in output_with

        # Without metadata
        schema_without = simplify_schema(schema_string, include_metadata=False)
        output_without = schema_without.to_string()
        assert len(output_with) > len(output_without)

    def test_string_schema_complex(self):
        """Test string schema with complex nested structure."""
        schema_string = """{
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    }
                },
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }"""

        schema = simplify_schema(schema_string)
        output = schema.to_string()
        assert "user:" in output
        assert "items:" in output

    def test_string_schema_with_array_types(self):
        """Test string schema with array types (union types like ['string', 'null'])."""
        schema_string = """{
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "age": {"type": ["integer", "null"]},
                "email": {"type": "string"}
            }
        }"""

        schema = simplify_schema(schema_string)
        output = schema.to_string()
        assert "name:" in output
        assert "age:" in output
        assert "email:" in output
        # Should handle nullable types properly
        assert "string?" in output or "int?" in output

    def test_string_schema_with_boolean_in_anyof(self):
        """Test string schema with boolean values in anyOf/oneOf arrays."""
        schema_string = """{
            "type": "object",
            "properties": {
                "field1": {
                    "anyOf": [
                        {"type": "string"},
                        true,
                        {"type": "integer"}
                    ]
                },
                "field2": {
                    "oneOf": [
                        {"type": "string"},
                        false,
                        {"type": "boolean"}
                    ]
                }
            }
        }"""

        schema = simplify_schema(schema_string)
        output = schema.to_string()
        assert "field1:" in output
        assert "field2:" in output
        # Should handle boolean values gracefully without crashing
        assert "string" in output

    def test_string_schema_with_boolean_property_values(self):
        """Test string schema with boolean property values (not just boolean types)."""
        schema_string = """{
            "type": "object",
            "properties": {
                "enabled": true,
                "disabled": false,
                "name": "test",
                "count": 42,
                "config": {
                    "type": "object",
                    "properties": {
                        "active": true
                    }
                }
            }
        }"""

        schema = simplify_schema(schema_string)
        output = schema.to_string()
        assert "enabled:" in output
        assert "disabled:" in output
        assert "name:" in output
        assert "count:" in output
        assert "config:" in output
        # Should handle boolean property values as "bool"
        assert "bool" in output

    def test_string_schema_with_single_element_type_array(self):
        """Test string schema with single-element type arrays."""
        schema_string = """{
            "type": "object",
            "properties": {
                "field1": {"type": ["string"]},
                "field2": {"type": ["integer"]},
                "field3": {"type": ["array"], "items": {"type": "string"}}
            }
        }"""

        schema = simplify_schema(schema_string)
        output = schema.to_string()
        assert "field1:" in output
        assert "field2:" in output
        assert "field3:" in output
        # Should handle single-element arrays correctly
        assert "string" in output
        assert "int" in output
        assert "string[]" in output

    def test_string_schema_with_enum_single_element_type_array(self):
        """Test string schema with enum and single-element type arrays."""
        schema_string = """{
            "type": "object",
            "properties": {
                "field1": {"type": ["boolean"], "enum": [true, false]},
                "field2": {"type": ["string"], "enum": ["option1", "option2"]},
                "field3": {"type": ["boolean"], "enum": [false]}
            }
        }"""

        schema = simplify_schema(schema_string)
        output = schema.to_string()
        assert "field1:" in output
        assert "field2:" in output
        assert "field3:" in output
        # Should handle enum with single-element type arrays correctly
        assert "bool" in output
        assert "string" in output
        assert "oneOf:" in output

    def test_string_schema_with_circular_references(self):
        """Test string schema with circular references to prevent infinite recursion."""
        schema_string = """{
            "type": "object",
            "properties": {
                "block": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"$ref": "#/definitions/task"},
                            {"$ref": "#/definitions/block"}
                        ]
                    }
                }
            },
            "definitions": {
                "task": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "block": {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"$ref": "#/definitions/task"},
                                    {"$ref": "#/definitions/block"}
                                ]
                            }
                        }
                    }
                },
                "block": {
                    "type": "object",
                    "properties": {
                        "block": {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"$ref": "#/definitions/task"},
                                    {"$ref": "#/definitions/block"}
                                ]
                            }
                        }
                    }
                }
            }
        }"""

        schema = simplify_schema(schema_string)
        output = schema.to_string()
        assert "block:" in output
        # Should handle circular references without crashing
        assert "[" in output

    def test_string_schema_with_boolean_items(self):
        """Test string schema with boolean items values in arrays."""
        schema_string = """{
            "type": "object",
            "properties": {
                "field1": {
                    "items": true,
                    "type": "array"
                },
                "field2": {
                    "items": false,
                    "type": "array"
                },
                "field3": {
                    "items": true,
                    "minItems": 1,
                    "maxItems": 10,
                    "type": "array"
                }
            }
        }"""

        schema = simplify_schema(schema_string)
        output = schema.to_string()
        assert "field1:" in output
        assert "field2:" in output
        assert "field3:" in output
        # Should handle boolean items without crashing
        assert "array" in output

    def test_unsupported_input(self):
        """Test simplify_schema with unsupported input."""
        with pytest.raises(UnsupportedModelError):
            simplify_schema(123)  # type: ignore

    def test_format_types(self):
        """Test all supported format types."""
        # JSONish (default)
        schema1 = simplify_schema(SimpleUser, format_type="jsonish")
        assert "{" in schema1.to_string()

        # TypeScript
        schema2 = simplify_schema(SimpleUser, format_type="typescript")
        assert "interface Schema {" in schema2.to_string()

        # YAML
        schema3 = simplify_schema(SimpleUser, format_type="yaml")
        output = schema3.to_string()
        assert "{" not in output and "interface" not in output

    def test_unsupported_format_type(self):
        """Test unsupported format type."""
        with pytest.raises(ValueError):
            simplify_schema(SimpleUser, format_type="invalid")  # type: ignore

    def test_include_metadata_flag(self):
        """Test include_metadata flag."""
        # With metadata
        schema_with = simplify_schema(User, include_metadata=True)
        output_with = schema_with.to_string()

        # Without metadata
        schema_without = simplify_schema(User, include_metadata=False)
        output_without = schema_without.to_string()

        # With metadata should have comments
        assert "//" in output_with
        # Without metadata should have fewer or no comments
        assert len(output_with) > len(output_without)


class TestSchemaLiteToDict:
    """Test SchemaLite.to_dict() method."""

    def test_to_dict_simple(self):
        """Test to_dict with simple model."""
        schema = simplify_schema(SimpleUser)
        result = schema.to_dict()

        assert isinstance(result, dict)
        assert "name" in result
        assert "age" in result
        assert "email" in result

    def test_to_dict_complex(self):
        """Test to_dict with complex model."""
        schema = simplify_schema(User)
        result = schema.to_dict()

        assert isinstance(result, dict)
        assert "name" in result
        assert "contact_info" in result


class TestSchemaLiteToJSON:
    """Test SchemaLite.to_json() method."""

    def test_to_json_simple(self):
        """Test to_json with simple model."""
        schema = simplify_schema(SimpleUser)
        result = schema.to_json()

        assert isinstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "name" in parsed

    def test_to_json_with_indent(self):
        """Test to_json with custom indentation."""
        schema = simplify_schema(SimpleUser)

        result_2 = schema.to_json(indent=2)
        result_4 = schema.to_json(indent=4)

        # More indentation should result in longer output
        assert len(result_4) > len(result_2)


class TestSchemaLiteToString:
    """Test SchemaLite.to_string() method."""

    def test_to_string_jsonish(self):
        """Test to_string with JSONish formatter."""
        schema = simplify_schema(SimpleUser, format_type="jsonish")
        result = schema.to_string()

        assert isinstance(result, str)
        assert "{" in result
        assert "name:" in result

    def test_to_string_typescript(self):
        """Test to_string with TypeScript formatter."""
        schema = simplify_schema(SimpleUser, format_type="typescript")
        result = schema.to_string()

        assert "interface Schema {" in result
        assert "name: string;" in result

    def test_to_string_yaml(self):
        """Test to_string with YAML formatter."""
        schema = simplify_schema(SimpleUser, format_type="yaml")
        result = schema.to_string()

        assert "name: str" in result
        assert "{" not in result


class TestSchemaLiteToYAML:
    """Test SchemaLite.to_yaml() method."""

    def test_to_yaml_requires_pyyaml(self):
        """Test that to_yaml() works if PyYAML is installed."""
        schema = simplify_schema(SimpleUser)

        try:
            result = schema.to_yaml()
            assert isinstance(result, str)
            # Should have YAML-like structure
            assert "name:" in result or "age:" in result
        except ImportError:
            # If PyYAML not installed, should raise ImportError with helpful message
            pytest.skip("PyYAML not installed")

    def test_to_yaml_flow_style(self):
        """Test to_yaml with default_flow_style."""
        schema = simplify_schema(SimpleUser)

        try:
            result_false = schema.to_yaml(default_flow_style=False)
            result_true = schema.to_yaml(default_flow_style=True)

            # Both should be strings
            assert isinstance(result_false, str)
            assert isinstance(result_true, str)
        except ImportError:
            pytest.skip("PyYAML not installed")


class TestSchemaLiteTokenCount:
    """Test token counting functionality."""

    def test_token_count_requires_tiktoken(self):
        """Test that token_count() requires tiktoken."""
        schema = simplify_schema(SimpleUser)

        try:
            count = schema.token_count()
            assert isinstance(count, int)
            assert count > 0
        except ImportError:
            # If tiktoken not installed, should raise ImportError with helpful message
            pytest.skip("tiktoken not installed")

    def test_token_count_with_encoding(self):
        """Test token_count with different encodings."""
        schema = simplify_schema(SimpleUser)

        try:
            count_default = schema.token_count()
            count_gpt4 = schema.token_count(encoding="cl100k_base")

            assert isinstance(count_default, int)
            assert isinstance(count_gpt4, int)
        except ImportError:
            pytest.skip("tiktoken not installed")


class TestSchemaLiteCompareTokens:
    """Test token comparison functionality."""

    def test_compare_tokens_basic(self):
        """Test basic token comparison."""
        schema = simplify_schema(SimpleUser)

        try:
            result = schema.compare_tokens()

            assert isinstance(result, dict)
            assert "original_tokens" in result
            assert "simplified_tokens" in result
            assert "tokens_saved" in result
            assert "reduction_percent" in result

            # Simplified should use fewer tokens
            assert result["simplified_tokens"] < result["original_tokens"]
            assert result["tokens_saved"] > 0
            assert result["reduction_percent"] > 0
        except ImportError:
            pytest.skip("tiktoken not installed")

    def test_compare_tokens_with_custom_schema(self):
        """Test token comparison with provided original schema."""
        schema = simplify_schema(SimpleUser)
        original = SimpleUser.model_json_schema()

        try:
            result = schema.compare_tokens(original_schema=original)

            assert isinstance(result, dict)
            assert result["original_tokens"] > 0
        except ImportError:
            pytest.skip("tiktoken not installed")

    def test_token_reduction(self):
        """Test that simplified schema has significant token reduction."""
        schema = simplify_schema(Order, include_metadata=False)

        try:
            result = schema.compare_tokens()

            # Should have at least 30% reduction
            assert result["reduction_percent"] >= 30
        except ImportError:
            pytest.skip("tiktoken not installed")


class TestSchemaLiteRepr:
    """Test SchemaLite string representations."""

    def test_str(self):
        """Test __str__ method."""
        schema = simplify_schema(SimpleUser)
        result = str(schema)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_repr(self):
        """Test __repr__ method."""
        schema = simplify_schema(SimpleUser)
        result = repr(schema)

        assert isinstance(result, str)
        assert "SchemaLite" in result
        assert "keys=" in result


class TestSchemaLiteEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_properties(self):
        """Test handling of model with no properties."""
        schema_dict = {"type": "object", "properties": {}}
        schema = simplify_schema(schema_dict)

        result = schema.to_string()
        # Empty schema returns "object" (the type) since no properties to format
        assert result in ["{}", "object"]

    def test_malformed_schema_dict(self):
        """Test handling of malformed schema dict."""
        # Schema with minimal structure
        schema_dict = {"properties": {"field": {"type": "string"}}}
        schema = simplify_schema(schema_dict)

        assert isinstance(schema, SchemaLite)


class TestFormatComparison:
    """Test comparison between different formats."""

    def test_all_formats_produce_output(self):
        """Test that all formats produce valid output."""
        jsonish = simplify_schema(Order, format_type="jsonish")
        typescript = simplify_schema(Order, format_type="typescript")
        yaml = simplify_schema(Order, format_type="yaml")

        # All should produce non-empty strings
        assert len(jsonish.to_string()) > 0
        assert len(typescript.to_string()) > 0
        assert len(yaml.to_string()) > 0

    def test_formats_are_different(self):
        """Test that different formats produce different outputs."""
        jsonish = simplify_schema(SimpleUser, format_type="jsonish").to_string()
        typescript = simplify_schema(SimpleUser, format_type="typescript").to_string()
        yaml = simplify_schema(SimpleUser, format_type="yaml").to_string()

        # All should be different
        assert jsonish != typescript
        assert typescript != yaml
        assert jsonish != yaml

    def test_metadata_affects_all_formats(self):
        """Test that metadata flag affects all formats."""
        for format_type in ["jsonish", "typescript", "yaml"]:
            with_meta = simplify_schema(
                User, format_type=format_type, include_metadata=True
            ).to_string()  # type: ignore
            without_meta = simplify_schema(
                User, format_type=format_type, include_metadata=False
            ).to_string()  # type: ignore

            # With metadata should be longer
            assert len(with_meta) > len(without_meta)


class TestSchemaLiteYAMLOutput:
    """Test YAML output methods."""

    def test_to_yaml_default_parameters(self):
        """Test to_yaml with default parameters."""
        schema = simplify_schema(SimpleUser)
        yaml_output = schema.to_yaml()
        assert isinstance(yaml_output, str)
        assert len(yaml_output) > 0

    def test_to_yaml_with_flow_style_true(self):
        """Test to_yaml with flow style enabled."""
        schema = simplify_schema(SimpleUser)
        yaml_output = schema.to_yaml(default_flow_style=True)
        assert isinstance(yaml_output, str)
        assert len(yaml_output) > 0

    def test_to_yaml_with_flow_style_false(self):
        """Test to_yaml with flow style disabled."""
        schema = simplify_schema(SimpleUser)
        yaml_output = schema.to_yaml(default_flow_style=False)
        assert isinstance(yaml_output, str)
        assert len(yaml_output) > 0


class TestSimplifySchemaInvalidInputs:
    """Test error handling for invalid inputs."""

    def test_invalid_format_type(self):
        """Test with invalid format_type parameter."""
        with pytest.raises(ValueError, match="Unsupported format_type"):
            simplify_schema(SimpleUser, format_type="invalid_format")

    def test_unsupported_model_type_integer(self):
        """Test with unsupported integer type."""
        with pytest.raises(UnsupportedModelError, match="Unsupported model type"):
            simplify_schema(42)

    def test_unsupported_model_type_list(self):
        """Test with unsupported list type."""
        with pytest.raises(UnsupportedModelError, match="Unsupported model type"):
            simplify_schema([1, 2, 3])

    def test_unsupported_model_type_none(self):
        """Test with None type."""
        with pytest.raises(UnsupportedModelError, match="Unsupported model type"):
            simplify_schema(None)

    def test_unsupported_model_type_float(self):
        """Test with unsupported float type."""
        with pytest.raises(UnsupportedModelError, match="Unsupported model type"):
            simplify_schema(3.14)


class TestSchemaLiteMultipleFormats:
    """Test SchemaLite with multiple format types."""

    def test_schema_from_dict_all_formats(self):
        """Test creating schema from dict with all format types."""
        schema_dict = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
        }

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(schema_dict, format_type=fmt)
            assert isinstance(schema, SchemaLite)
            output = schema.to_string()
            assert "name" in output
            assert "age" in output
            assert "email" in output

    def test_schema_from_string_all_formats(self):
        """Test creating schema from JSON string with all format types."""
        schema_str = '{"type": "object", "properties": {"title": {"type": "string"}, "count": {"type": "integer"}}}'  # noqa: E501

        for fmt in ["jsonish", "typescript", "yaml"]:
            schema = simplify_schema(schema_str, format_type=fmt)
            assert isinstance(schema, SchemaLite)
            output = schema.to_string()
            assert "title" in output
            assert "count" in output
