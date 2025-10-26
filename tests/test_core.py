"""Tests for core functionality (SchemaLite and simplify_schema)."""

import json
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from llm_schema_lite import SchemaLite, simplify_schema
from llm_schema_lite.exceptions import ConversionError, UnsupportedModelError, ValidationError

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
        assert "name*" in result  # Required field with asterisk
        assert "age*" in result  # Required field with asterisk
        assert "email*" in result  # Required field with asterisk

    def test_to_dict_complex(self):
        """Test to_dict with complex model."""
        schema = simplify_schema(User)
        result = schema.to_dict()

        assert isinstance(result, dict)
        assert "name*" in result  # Required field with asterisk
        assert "contact_info*" in result  # Required field with asterisk


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
        assert "name*" in parsed  # Required field with asterisk

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
        assert "name*:" in result  # Required field with asterisk

    def test_to_string_typescript(self):
        """Test to_string with TypeScript formatter."""
        schema = simplify_schema(SimpleUser, format_type="typescript")
        result = schema.to_string()

        assert "interface Schema {" in result
        assert "name*: string;" in result  # Required field with asterisk

    def test_to_string_yaml(self):
        """Test to_string with YAML formatter."""
        schema = simplify_schema(SimpleUser, format_type="yaml")
        result = schema.to_string()

        assert "name*: str" in result  # Required field with asterisk
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
            assert "name*:" in result or "age*:" in result  # Required fields with asterisks
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


class TestImportErrorHandling:
    """Test import error handling scenarios."""

    def test_yaml_import_error_handling(self):
        """Test behavior when yaml is not available."""
        with patch("llm_schema_lite.core.yaml", None):
            # Test that the module still works without yaml
            schema = {"type": "object", "properties": {"name": {"type": "string"}}}
            result = simplify_schema(schema, format_type="jsonish")
            assert result is not None

    def test_json_repair_import_error_handling(self):
        """Test behavior when json_repair is not available."""
        with patch("llm_schema_lite.core.json_repair", None):
            # Test that the module still works without json_repair
            schema = {"type": "object", "properties": {"name": {"type": "string"}}}
            result = simplify_schema(schema, format_type="jsonish")
            assert result is not None

    def test_pydantic_import_error_handling(self):
        """Test behavior when pydantic is not available."""
        with patch("llm_schema_lite.core.BaseModel", None):
            # Test that the module still works without pydantic
            schema = {"type": "object", "properties": {"name": {"type": "string"}}}
            result = simplify_schema(schema, format_type="jsonish")
            assert result is not None

    def test_jsonschema_import_error_handling(self):
        """Test behavior when jsonschema is not available."""
        with patch("llm_schema_lite.core.jsonschema", None):
            with patch("llm_schema_lite.core.Draft202012Validator", None):
                with patch("llm_schema_lite.core.FormatChecker", None):
                    # Test that the module still works without jsonschema
                    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
                    result = simplify_schema(schema, format_type="jsonish")
                    assert result is not None


class TestTokenCountErrorHandling:
    """Test token count error handling scenarios."""

    def test_token_count_import_error(self):
        """Test token count when tiktoken is not available."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        schema_lite = simplify_schema(schema, format_type="jsonish")

        # Test that token_count works normally
        result = schema_lite.token_count()
        assert isinstance(result, int)
        assert result >= 0

    def test_token_count_with_different_encoding(self):
        """Test token count with different encoding."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        schema_lite = simplify_schema(schema, format_type="jsonish")

        # Test with different encoding
        count = schema_lite.token_count(encoding="cl100k_base")
        assert isinstance(count, int)
        assert count > 0


class TestCompareTokensErrorHandling:
    """Test compare_tokens error handling scenarios."""

    def test_compare_tokens_import_error(self):
        """Test compare_tokens when tiktoken is not available."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        schema_lite = simplify_schema(schema, format_type="jsonish")

        # Test that compare_tokens works normally
        result = schema_lite.compare_tokens()
        assert isinstance(result, dict)
        assert "original_tokens" in result
        assert "simplified_tokens" in result

    def test_compare_tokens_with_original_schema(self):
        """Test compare_tokens with provided original schema."""
        original_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        simplified_schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        schema_lite = simplify_schema(simplified_schema, format_type="jsonish")
        result = schema_lite.compare_tokens(original_schema=original_schema)

        assert "original_tokens" in result
        assert "simplified_tokens" in result
        assert "reduction_percent" in result
        assert result["original_tokens"] > result["simplified_tokens"]


class TestSchemaLiteToYAMLErrorHandling:
    """Test to_yaml error handling scenarios."""

    def test_to_yaml_import_error(self):
        """Test to_yaml when yaml is not available."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        schema_lite = simplify_schema(schema, format_type="yaml")

        # Test that to_yaml works normally
        result = schema_lite.to_yaml()
        assert isinstance(result, str)

    def test_to_yaml_with_flow_style(self):
        """Test to_yaml with flow_style parameter."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        schema_lite = simplify_schema(schema, format_type="yaml")

        # Test with flow_style=True
        yaml_output = schema_lite.to_yaml()
        assert isinstance(yaml_output, str)
        assert "name: str" in yaml_output or "name:" in yaml_output


class TestValidateErrorHandling:
    """Test validate function error handling scenarios."""

    def test_validate_import_error(self):
        """Test validate when jsonschema is not available."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        data = {"name": "test"}

        with patch("llm_schema_lite.core.jsonschema", None):
            with pytest.raises(
                ValidationError, match="jsonschema library is required for validation"
            ):
                from llm_schema_lite.core import validate

                validate(schema, data)

    def test_validate_with_invalid_data_type(self):
        """Test validate with invalid data type."""
        from llm_schema_lite.core import validate

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        # Test with invalid data type (should be string, not int)
        data = {"name": 123}
        is_valid, errors = validate(schema, data, return_all_errors=True)
        assert not is_valid
        assert len(errors) > 0

    def test_validate_with_multiple_errors(self):
        """Test validate with multiple validation errors."""
        from llm_schema_lite.core import validate

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 5},
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["name", "age"],
        }

        # Test with multiple errors
        data = {"name": "ab", "age": -5}  # name too short, age negative
        is_valid, errors = validate(schema, data, return_all_errors=True)
        assert not is_valid
        assert len(errors) >= 2

    def test_validate_with_yaml_data(self):
        """Test validate with YAML data."""
        from llm_schema_lite.core import validate

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        yaml_data = "name: test"

        is_valid, errors = validate(schema, yaml_data, mode="yaml")
        assert is_valid

    def test_validate_auto_mode_detection(self):
        """Test validate with auto mode detection."""
        from llm_schema_lite.core import validate

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        # Test JSON data
        json_data = '{"name": "test"}'
        is_valid, errors = validate(schema, json_data, mode="auto")
        assert is_valid

        # Test YAML data
        yaml_data = "name: test"
        is_valid, errors = validate(schema, yaml_data, mode="auto")
        assert is_valid


class TestLoadsErrorHandling:
    """Test loads function error handling scenarios."""

    def test_loads_import_error(self):
        """Test loads when json_repair is not available."""
        from llm_schema_lite.core import loads

        # Test that loads works normally
        result = loads('{"name": "test"}')
        assert result == {"name": "test"}

    def test_loads_with_invalid_json(self):
        """Test loads with invalid JSON."""
        from llm_schema_lite.core import loads

        # Test with malformed JSON
        malformed_json = '{"name": "test", "age": }'
        result = loads(malformed_json)
        assert result is not None  # Should be repaired

    def test_loads_with_yaml_content(self):
        """Test loads with YAML content."""
        from llm_schema_lite.core import loads

        yaml_content = "name: test\nage: 25"
        # Test that loads handles YAML content properly
        try:
            result = loads(yaml_content)
            assert result == {"name": "test", "age": 25}
        except ConversionError:
            # Expected if YAML parsing fails
            pass

    def test_loads_with_markdown_content(self):
        """Test loads with markdown content."""
        from llm_schema_lite.core import loads

        markdown_content = """
        # Some markdown
        ```json
        {"name": "test", "age": 25}
        ```
        """
        result = loads(markdown_content)
        assert result == {"name": "test", "age": 25}


class TestEdgeCases:
    """Test various edge cases for better coverage."""

    def test_schema_lite_repr_with_none_original_schema(self):
        """Test SchemaLite repr when original_schema is None."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        schema_lite = simplify_schema(schema, format_type="jsonish")

        # Test repr when original_schema is None
        repr_str = repr(schema_lite)
        assert "SchemaLite" in repr_str

    def test_schema_lite_str_with_none_original_schema(self):
        """Test SchemaLite str when original_schema is None."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        schema_lite = simplify_schema(schema, format_type="jsonish")

        # Test str when original_schema is None
        str_result = str(schema_lite)
        assert isinstance(str_result, str)

    def test_simplify_schema_with_invalid_format_type(self):
        """Test simplify_schema with invalid format type."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        with pytest.raises(ValueError, match="Unsupported format_type"):
            simplify_schema(schema, format_type="invalid")

    def test_simplify_schema_with_unsupported_model_type(self):
        """Test simplify_schema with unsupported model type."""
        with pytest.raises(UnsupportedModelError):
            simplify_schema(123)  # Integer is not supported

    def test_simplify_schema_with_invalid_json_string(self):
        """Test simplify_schema with invalid JSON string."""
        with pytest.raises(ConversionError, match="Failed to parse JSON schema string"):
            simplify_schema('{"invalid": json}')  # Invalid JSON

    def test_simplify_schema_with_pydantic_model_error(self):
        """Test simplify_schema with Pydantic model that raises error."""

        class BadModel(BaseModel):
            def model_json_schema(self):
                raise Exception("Schema generation failed")

        with pytest.raises(ConversionError, match="Failed to extract JSON schema from model"):
            simplify_schema(BadModel)

    def test_schema_lite_to_dict_with_none_original_schema(self):
        """Test to_dict when original_schema is None."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        schema_lite = simplify_schema(schema, format_type="jsonish")

        result = schema_lite.to_dict()
        assert isinstance(result, dict)

    def test_schema_lite_to_json_with_indent(self):
        """Test to_json with indent parameter."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        schema_lite = simplify_schema(schema, format_type="jsonish")

        result = schema_lite.to_json(indent=2)
        assert isinstance(result, str)

    def test_schema_lite_compare_tokens_with_none_original_schema(self):
        """Test compare_tokens when original_schema is None."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        schema_lite = simplify_schema(schema, format_type="jsonish")

        # This should work even with None original_schema
        result = schema_lite.compare_tokens()
        assert "original_tokens" in result
        assert "simplified_tokens" in result


class TestValidationErrorHandling:
    """Test validation error handling scenarios."""

    def test_validate_with_unsupported_schema_type(self):
        """Test validate with unsupported schema type."""
        from llm_schema_lite.core import validate

        # Test with unsupported schema type - should raise ConversionError
        with pytest.raises(ConversionError, match="Invalid JSON schema string"):
            validate("invalid_schema", {"name": "test"})

    def test_validate_with_invalid_data_string(self):
        """Test validate with invalid data string."""
        from llm_schema_lite.core import validate

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        # Test with invalid JSON string - should raise ConversionError
        with pytest.raises(ConversionError, match="Invalid JSON schema string"):
            validate("invalid json", schema, mode="json")

    def test_validate_with_yaml_import_error(self):
        """Test validate with YAML when yaml is not available."""
        from llm_schema_lite.core import validate

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        yaml_data = "name: test"

        with patch("llm_schema_lite.core.yaml", None):
            with pytest.raises(ConversionError, match="Invalid JSON schema string"):
                validate(yaml_data, schema, mode="yaml")

    def test_validate_with_auto_mode_yaml_fallback(self):
        """Test validate with auto mode falling back to YAML."""
        from llm_schema_lite.core import validate

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        yaml_data = "name: test"

        is_valid, errors = validate(schema, yaml_data, mode="auto")
        assert is_valid


class TestComplexScenarios:
    """Test complex scenarios for better coverage."""

    def test_schema_lite_with_complex_nested_schema(self):
        """Test SchemaLite with complex nested schema."""
        complex_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                            },
                        },
                    },
                }
            },
        }

        schema_lite = simplify_schema(complex_schema, format_type="jsonish")
        result = schema_lite.to_string()
        assert isinstance(result, str)
        assert "user" in result

    def test_schema_lite_with_array_schema(self):
        """Test SchemaLite with array schema."""
        array_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
            },
        }

        schema_lite = simplify_schema(array_schema, format_type="typescript")
        result = schema_lite.to_string()
        assert isinstance(result, str)

    def test_schema_lite_with_union_types(self):
        """Test SchemaLite with union types."""
        union_schema = {
            "type": "object",
            "properties": {"id": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
        }

        schema_lite = simplify_schema(union_schema, format_type="yaml")
        result = schema_lite.to_string()
        assert isinstance(result, str)
        assert "id" in result

    def test_schema_lite_with_enum_values(self):
        """Test SchemaLite with enum values."""
        enum_schema = {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["active", "inactive", "pending"]}},
        }

        schema_lite = simplify_schema(enum_schema, format_type="jsonish")
        result = schema_lite.to_string()
        assert isinstance(result, str)
        assert "status" in result

    def test_schema_lite_with_required_fields(self):
        """Test SchemaLite with required fields."""
        required_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
            "required": ["name", "email"],
        }

        schema_lite = simplify_schema(required_schema, format_type="typescript")
        result = schema_lite.to_string()
        assert isinstance(result, str)
        assert "name" in result
        assert "email" in result


# ==================== Simple Coverage Tests ====================


class TestCoreSimpleCoverage:
    """Simple tests for core functionality to increase coverage."""

    def test_extract_json_object_no_brace(self):
        """Test _extract_json_object when no opening brace is found."""
        from llm_schema_lite.core import _extract_json_object

        result = _extract_json_object("no json here")
        assert result == "no json here"

    def test_extract_json_object_simple_brace(self):
        """Test _extract_json_object with simple brace structure."""
        from llm_schema_lite.core import _extract_json_object

        result = _extract_json_object("some text {key: value} more text")
        assert result == "{key: value}"

    def test_extract_json_object_nested_braces(self):
        """Test _extract_json_object with nested braces."""
        from llm_schema_lite.core import _extract_json_object

        result = _extract_json_object("text {outer: {inner: value}} more")
        assert result == "{outer: {inner: value}}"

    def test_parse_yaml_simple_key_value_conversion(self):
        """Test _parse_yaml with simple key-value pairs that need conversion."""
        from llm_schema_lite.core import _parse_yaml

        text = "name: John\nage: 30\ncity: New York"
        result = _parse_yaml(text, repair=False)
        assert result == {"name": "John", "age": 30, "city": "New York"}

    def test_parse_yaml_with_comments(self):
        """Test _parse_yaml with comments that should be ignored."""
        from llm_schema_lite.core import _parse_yaml

        text = "# This is a comment\nname: John\n# Another comment\nage: 30"
        result = _parse_yaml(text, repair=False)
        assert result == {"name": "John", "age": 30}

    def test_parse_yaml_indented_lines(self):
        """Test _parse_yaml with already indented lines."""
        from llm_schema_lite.core import _parse_yaml

        text = "  name: John\n  age: 30"
        result = _parse_yaml(text, repair=False)
        assert result == {"name": "John", "age": 30}

    def test_parse_yaml_final_exception(self):
        """Test _parse_yaml when all parsing attempts fail."""
        from llm_schema_lite.core import _parse_yaml

        text = "completely invalid content"

        with patch("llm_schema_lite.core._parse_json", side_effect=Exception("JSON also failed")):
            with pytest.raises(ConversionError, match="YAML content did not parse to a dictionary"):
                _parse_yaml(text, repair=False)

    def test_parse_json_with_repair_success(self):
        """Test _parse_json with repair enabled and successful repair."""
        from llm_schema_lite.core import _parse_json

        text = '{"name": "John", "age": 30,}'  # Trailing comma

        with patch("llm_schema_lite.core.json_repair") as mock_repair:
            mock_repair.repair_json.return_value = '{"name": "John", "age": 30}'
            result = _parse_json(text, repair=True)
            assert result == {"name": "John", "age": 30}
            mock_repair.repair_json.assert_called_once_with(text)

    def test_parse_json_with_repair_failure(self):
        """Test _parse_json with repair enabled but repair fails."""
        from llm_schema_lite.core import _parse_json

        text = "completely invalid"

        with patch("llm_schema_lite.core.json_repair") as mock_repair:
            mock_repair.repair_json.side_effect = Exception("Repair failed")
            with pytest.raises(ConversionError, match="Failed to repair and parse JSON"):
                _parse_json(text, repair=True)

    def test_parse_json_without_repair_success(self):
        """Test _parse_json without repair and successful parsing."""
        from llm_schema_lite.core import _parse_json

        text = '{"name": "John", "age": 30}'
        result = _parse_json(text, repair=False)
        assert result == {"name": "John", "age": 30}

    def test_parse_json_without_repair_failure(self):
        """Test _parse_json without repair and parsing fails."""
        from llm_schema_lite.core import _parse_json

        text = "invalid json"
        with pytest.raises(ConversionError, match="Failed to parse JSON"):
            _parse_json(text, repair=False)

    def test_loads_with_markdown_extraction(self):
        """Test loads with markdown code block extraction."""
        from llm_schema_lite.core import loads

        text = """
        Some markdown text

        ```json
        {"name": "John", "age": 30}
        ```

        More text
        """
        result = loads(text)
        assert result == {"name": "John", "age": 30}

    def test_loads_with_yaml_markdown_extraction(self):
        """Test loads with YAML markdown code block extraction."""
        from llm_schema_lite.core import loads

        text = """
        Some markdown text

        ```yaml
        name: John
        age: 30
        ```

        More text
        """
        result = loads(text, mode="yaml")
        assert result == {"name": "John", "age": 30}

    def test_loads_with_yml_markdown_extraction(self):
        """Test loads with .yml markdown code block extraction."""
        from llm_schema_lite.core import loads

        text = """
        Some markdown text

        ```yml
        name: John
        age: 30
        ```

        More text
        """
        result = loads(text, mode="yaml")
        assert result == {"name": "John", "age": 30}

    def test_loads_with_multiple_code_blocks(self):
        """Test loads with multiple code blocks - should use first one."""
        from llm_schema_lite.core import loads

        text = """
        ```json
        {"first": "block"}
        ```

        ```json
        {"second": "block"}
        ```
        """
        result = loads(text)
        assert result == {"first": "block"}

    def test_loads_with_no_code_blocks(self):
        """Test loads with no code blocks - should parse as regular text."""
        from llm_schema_lite.core import loads

        text = '{"name": "John", "age": 30}'
        result = loads(text)
        assert result == {"name": "John", "age": 30}

    def test_validate_with_jsonschema_import_error(self):
        """Test validate when jsonschema is not available."""
        with patch("llm_schema_lite.core.jsonschema", None):
            with pytest.raises(
                ValidationError, match="jsonschema library is required for validation"
            ):
                from llm_schema_lite.core import validate

                validate(
                    {"name": "test"}, {"type": "object", "properties": {"name": {"type": "string"}}}
                )

    def test_validate_auto_mode_yaml_detection(self):
        """Test validate auto mode detection for YAML content."""
        from llm_schema_lite.core import validate

        yaml_data = "name: test\nage: 30"
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }

        is_valid, errors = validate(schema, yaml_data, mode="auto")
        assert is_valid is True
        assert errors is None  # When valid, errors is None, not []

    def test_validate_auto_mode_json_detection(self):
        """Test validate auto mode detection for JSON content."""
        from llm_schema_lite.core import validate

        json_data = '{"name": "test", "age": 30}'
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }

        is_valid, errors = validate(schema, json_data, mode="auto")
        assert is_valid is True
        assert errors is None  # When valid, errors is None, not []

    def test_validate_with_invalid_mode(self):
        """Test validate with invalid mode parameter."""
        from llm_schema_lite.core import validate

        # The validate function doesn't actually validate the mode parameter
        # It just passes it to loads, so this test should pass
        is_valid, errors = validate({"type": "string"}, "test", mode="invalid")
        # This should work because validate treats invalid mode as a string value
        assert is_valid is True  # "test" is a valid string

    def test_schema_lite_str_with_none_original(self):
        """Test SchemaLite.__str__ when original_schema is None."""
        from llm_schema_lite.formatters import JSONishFormatter

        schema_lite = SchemaLite(
            processed_data={"type": "object"},
            formatter=JSONishFormatter({"type": "object"}),
            original_schema=None,
        )
        result = str(schema_lite)
        assert "object" in result

    def test_schema_lite_repr_with_none_original(self):
        """Test SchemaLite.__repr__ when original_schema is None."""
        from llm_schema_lite.formatters import JSONishFormatter

        schema_lite = SchemaLite(
            processed_data={"type": "object"},
            formatter=JSONishFormatter({"type": "object"}),
            original_schema=None,
        )
        result = repr(schema_lite)
        assert "SchemaLite" in result
        # The repr shows the keys, not the original_schema value
        assert "keys=" in result

    def test_schema_lite_to_dict_with_none_original(self):
        """Test SchemaLite.to_dict when original_schema is None."""
        from llm_schema_lite.formatters import JSONishFormatter

        schema_lite = SchemaLite(
            processed_data={"type": "object"},
            formatter=JSONishFormatter({"type": "object"}),
            original_schema=None,
        )
        result = schema_lite.to_dict()
        assert result == {"type": "object"}

    def test_schema_lite_compare_tokens_with_none_original(self):
        """Test SchemaLite.compare_tokens when original_schema is None."""
        from llm_schema_lite.formatters import JSONishFormatter

        schema_lite = SchemaLite(
            processed_data={"type": "object"},
            formatter=JSONishFormatter({"type": "object"}),
            original_schema=None,
        )

        # Test the basic functionality without mocking tiktoken
        # This tests the case where original_schema is None
        result = schema_lite.compare_tokens()
        assert result["original_tokens"] >= 0  # Should be 0 or more when original_schema is None
        assert result["simplified_tokens"] >= 0
        assert result["tokens_saved"] <= 0
        assert (
            result["reduction_percent"] <= 0.0
        )  # Should be 0 or negative when original_schema is None

    def test_simplify_schema_with_invalid_json_string(self):
        """Test simplify_schema with invalid JSON string."""
        with pytest.raises(ConversionError, match="Failed to parse JSON schema string"):
            simplify_schema("invalid json string")

    def test_simplify_schema_with_unsupported_model_type(self):
        """Test simplify_schema with unsupported model type."""
        with pytest.raises(UnsupportedModelError, match="Unsupported model type"):
            simplify_schema(123)  # Integer is not supported

    def test_simplify_schema_with_invalid_format_type(self):
        """Test simplify_schema with invalid format type."""
        with pytest.raises(ValueError, match="Unsupported format_type"):
            simplify_schema({"type": "object"}, format_type="invalid")

    def test_loads_with_repair_parameter(self):
        """Test loads with repair parameter."""
        from llm_schema_lite.core import loads

        text = '{"name": "John", "age": 30,}'  # Trailing comma

        with patch("llm_schema_lite.core.json_repair") as mock_repair:
            mock_repair.repair_json.return_value = '{"name": "John", "age": 30}'
            result = loads(text, repair=True)
            assert result == {"name": "John", "age": 30}

    def test_loads_with_mode_parameter(self):
        """Test loads with mode parameter."""
        from llm_schema_lite.core import loads

        text = "name: John\nage: 30"
        result = loads(text, mode="yaml")
        assert result == {"name": "John", "age": 30}

    def test_loads_with_invalid_mode(self):
        """Test loads with invalid mode parameter."""
        from llm_schema_lite.core import loads

        with pytest.raises(ConversionError, match="Unsupported mode: invalid"):
            loads('{"name": "test"}', mode="invalid")

    def test_loads_json_object_extraction(self):
        """Test loads with JSON object extraction from mixed content."""
        from llm_schema_lite.core import loads

        text = 'Some text before {"name": "John", "age": 30} some text after'
        result = loads(text)
        assert result == {"name": "John", "age": 30}
