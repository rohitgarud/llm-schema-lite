"""Tests for core functionality (SchemaLite and simplify_schema)."""

import json

import pytest

from llm_schema_lite import SchemaLite, simplify_schema
from llm_schema_lite.exceptions import UnsupportedModelError

from .fixtures import Order, SimpleUser, User


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

    def test_unsupported_input(self):
        """Test simplify_schema with unsupported input."""
        with pytest.raises(UnsupportedModelError):
            simplify_schema("invalid_input")  # type: ignore

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
