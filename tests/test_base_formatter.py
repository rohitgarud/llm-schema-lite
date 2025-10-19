"""Base test classes for formatter testing to reduce duplication."""

import pytest

from llm_schema_lite import simplify_schema
from llm_schema_lite.formatters.base import BaseFormatter
from llm_schema_lite.formatters.jsonish_formatter import JSONishFormatter
from llm_schema_lite.formatters.typescript_formatter import TypeScriptFormatter
from llm_schema_lite.formatters.yaml_formatter import YAMLFormatter


class BaseFormatterTest:
    """Base class for formatter tests with common functionality."""

    @pytest.fixture(scope="class")
    def formatter_class(self) -> type[BaseFormatter]:
        """Override this in subclasses to specify the formatter class."""
        raise NotImplementedError("Subclasses must implement formatter_class")

    @pytest.fixture(scope="class")
    def format_type(self) -> str:
        """Override this in subclasses to specify the format type."""
        raise NotImplementedError("Subclasses must implement format_type")

    def test_simple_model(self, simple_user_model, format_type):
        """Test basic model formatting across all formatters."""
        schema = simplify_schema(simple_user_model, format_type=format_type, include_metadata=False)
        output = schema.to_string()

        # All formatters should include the basic fields
        assert "name" in output
        assert "age" in output
        assert "email" in output

        # Output should not be empty
        assert len(output.strip()) > 0

    def test_simple_model_with_metadata(self, user_model, format_type):
        """Test model formatting with metadata inclusion."""
        schema = simplify_schema(user_model, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should include field names
        assert "name" in output
        assert "email" in output
        assert "is_active" in output
        assert "role" in output

        # Should include some metadata (description, constraints, etc.)
        metadata_indicators = ["description", "minLength", "maxLength", "pattern", "defaults to"]
        has_metadata = any(indicator in output for indicator in metadata_indicators)
        assert has_metadata, f"No metadata found in output: {output}"

    def test_without_metadata(self, user_model, format_type):
        """Test model formatting without metadata."""
        schema = simplify_schema(user_model, format_type=format_type, include_metadata=False)
        output = schema.to_string()

        # Should include field names
        assert "name" in output
        assert "email" in output
        assert "is_active" in output
        assert "role" in output

        # Should not include metadata indicators
        metadata_indicators = ["description", "minLength", "maxLength", "pattern", "defaults to"]
        for indicator in metadata_indicators:
            assert (
                indicator not in output
            ), f"Found metadata '{indicator}' in output when include_metadata=False"

    def test_complex_order_model(self, complex_order_model, format_type):
        """Test complex nested model formatting."""
        schema = simplify_schema(
            complex_order_model, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include main fields
        assert "order_id" in output
        assert "customer" in output
        assert "items" in output
        assert "total" in output
        assert "status" in output

        # Output should be substantial for complex model
        assert len(output.strip()) > 50

    def test_empty_schema(self, empty_schema, format_type):
        """Test handling of empty schema."""
        schema = simplify_schema(empty_schema, format_type=format_type)
        output = schema.to_string()

        # Should handle empty schema gracefully
        assert output is not None
        assert isinstance(output, str)

        # Specific assertions depend on formatter
        if format_type == "jsonish":
            assert output == "{}"
        elif format_type == "typescript":
            assert "interface Schema {}" in output
        elif format_type == "yaml":
            assert output == "{}"

    def test_numeric_constraints(self, numeric_constraints_model, format_type):
        """Test numeric validation constraint handling."""
        schema = simplify_schema(
            numeric_constraints_model, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include field names
        assert "int_field" in output
        assert "float_field" in output
        assert "optional_int" in output

        # Should include numeric constraints when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            # These formatters show constraints in comments
            constraint_indicators = ["min:", "max:", "ge:", "le:", "gt:", "lt:", "multipleOf:"]
            has_constraints = any(indicator in output for indicator in constraint_indicators)
            assert has_constraints, f"No numeric constraints found in output: {output}"

    def test_string_constraints(self, string_constraints_model, format_type):
        """Test string validation constraint handling."""
        schema = simplify_schema(
            string_constraints_model, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include field names
        assert "name" in output
        assert "email" in output
        assert "description" in output

        # Should include string constraints when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            constraint_indicators = ["minLength:", "maxLength:", "pattern:"]
            has_constraints = any(indicator in output for indicator in constraint_indicators)
            assert has_constraints, f"No string constraints found in output: {output}"

    def test_pattern_constraints(self, pattern_constraints_model, format_type):
        """Test pattern validation constraint handling."""
        schema = simplify_schema(
            pattern_constraints_model, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include field names
        assert "phone" in output
        assert "zip_code" in output
        assert "username" in output

        # Should include pattern constraints when metadata is enabled
        if format_type in ["jsonish", "yaml"]:
            assert "pattern:" in output, f"No pattern constraints found in output: {output}"

    def test_anyof_handling(self, union_types_model, format_type):
        """Test anyOf (union type) handling."""
        schema = simplify_schema(union_types_model, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should include field names
        assert "id" in output
        assert "status" in output
        assert "metadata" in output

        # Should handle union types appropriately
        if format_type == "typescript":
            assert " | " in output or "Union[" in output
        elif format_type == "yaml":
            assert " | " in output or "Union[" in output
        elif format_type == "jsonish":
            assert " or " in output or "oneOf:" in output

    def test_optional_with_default(self, optional_with_default_model, format_type):
        """Test optional fields with default values."""
        schema = simplify_schema(
            optional_with_default_model, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include field names
        assert "name" in output
        assert "age" in output
        assert "is_active" in output
        assert "tags" in output

        # Should handle optional/nullable types appropriately
        if format_type == "typescript":
            assert " | null" in output or " | undefined" in output
        elif format_type == "yaml":
            assert " | None" in output
        elif format_type == "jsonish":
            assert "?" in output or "null" in output

    def test_type_mapping(self, simple_user_model, format_type, formatter_expected_patterns):
        """Test correct type mapping from JSON schema to target format."""
        schema = simplify_schema(simple_user_model, format_type=format_type, include_metadata=False)
        output = schema.to_string()

        # expected_patterns = formatter_expected_patterns[format_type]  # Unused for now

        # Check that types are mapped correctly
        if format_type == "jsonish":
            assert "string" in output  # name, email
            assert "int" in output  # age
        elif format_type == "typescript":
            assert "string" in output  # name, email
            assert "number" in output  # age
        elif format_type == "yaml":
            assert "str" in output  # name, email
            assert "int" in output  # age

    def test_union_types(self, union_types_model, format_type):
        """Test union type handling."""
        schema = simplify_schema(union_types_model, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should include field names
        assert "id" in output
        assert "status" in output
        assert "metadata" in output

        # Should handle union types with appropriate syntax
        if format_type == "typescript":
            assert " | " in output
        elif format_type == "yaml":
            assert " | " in output
        elif format_type == "jsonish":
            assert " or " in output or "oneOf:" in output


class TestJSONishFormatter(BaseFormatterTest):
    """Test class for JSONish formatter."""

    @pytest.fixture(scope="class")
    def formatter_class(self) -> type[BaseFormatter]:
        return JSONishFormatter

    @pytest.fixture(scope="class")
    def format_type(self) -> str:
        return "jsonish"

    def test_jsonish_specific_formatting(self, user_model):
        """Test JSONish-specific formatting features."""
        schema = simplify_schema(user_model, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Should use JSONish comment syntax
        assert "//" in output

        # Should use JSONish type mappings
        assert "int" in output  # integer -> int
        assert "bool" in output  # boolean -> bool

        # Should have JSON-like structure
        assert "{" in output
        assert "}" in output


class TestTypeScriptFormatter(BaseFormatterTest):
    """Test class for TypeScript formatter."""

    @pytest.fixture(scope="class")
    def formatter_class(self) -> type[BaseFormatter]:
        return TypeScriptFormatter

    @pytest.fixture(scope="class")
    def format_type(self) -> str:
        return "typescript"

    def test_typescript_specific_formatting(self, user_model):
        """Test TypeScript-specific formatting features."""
        schema = simplify_schema(user_model, format_type="typescript", include_metadata=True)
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

    def test_typescript_enum_literals(self, user_model):
        """Test TypeScript enum literal formatting."""
        schema = simplify_schema(user_model, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Should format enums as union literals
        assert '"admin" | "user" | "guest"' in output or '"user" | "admin" | "guest"' in output


class TestYAMLFormatter(BaseFormatterTest):
    """Test class for YAML formatter."""

    @pytest.fixture(scope="class")
    def formatter_class(self) -> type[BaseFormatter]:
        return YAMLFormatter

    @pytest.fixture(scope="class")
    def format_type(self) -> str:
        return "yaml"

    def test_yaml_specific_formatting(self, user_model):
        """Test YAML-specific formatting features."""
        schema = simplify_schema(user_model, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Should use YAML comment syntax
        assert "#" in output

        # Should use YAML type mappings
        assert "str" in output  # string -> str
        assert "int" in output  # integer -> int
        assert "bool" in output  # boolean -> bool

        # Should have YAML-like structure (key: value)
        assert ":" in output

    def test_yaml_literal_types(self, user_model):
        """Test YAML Literal type formatting."""
        schema = simplify_schema(user_model, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Should format enums as Literal types
        assert "Literal[" in output
        assert '"admin"' in output
        assert '"user"' in output
        assert '"guest"' in output


# ============================================================================
# Parameterized Tests for Cross-Formatter Validation
# ============================================================================


class TestCrossFormatterValidation:
    """Cross-formatter validation tests using parameterization."""

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_simple_model_all_formatters(self, simple_user_model, format_type):
        """Test simple model across all formatters."""
        schema = simplify_schema(simple_user_model, format_type=format_type, include_metadata=False)
        output = schema.to_string()

        # All formatters should include the basic fields
        assert "name" in output
        assert "age" in output
        assert "email" in output

        # Output should not be empty
        assert len(output.strip()) > 0

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_metadata_handling_all_formatters(self, user_model, format_type):
        """Test metadata handling across all formatters."""
        # Test with metadata
        schema_with_metadata = simplify_schema(
            user_model, format_type=format_type, include_metadata=True
        )
        output_with_metadata = schema_with_metadata.to_string()

        # Test without metadata
        schema_without_metadata = simplify_schema(
            user_model, format_type=format_type, include_metadata=False
        )
        output_without_metadata = schema_without_metadata.to_string()

        # Output with metadata should be longer
        assert len(output_with_metadata) > len(output_without_metadata)

        # Both should include field names
        for field in ["name", "email", "is_active", "role"]:
            assert field in output_with_metadata
            assert field in output_without_metadata

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_validation_constraints_all_formatters(self, numeric_constraints_model, format_type):
        """Test validation constraints across all formatters."""
        schema = simplify_schema(
            numeric_constraints_model, format_type=format_type, include_metadata=True
        )
        output = schema.to_string()

        # Should include field names
        assert "int_field" in output
        assert "float_field" in output
        assert "optional_int" in output

        # Should handle constraints appropriately for each formatter
        if format_type in ["jsonish", "yaml"]:
            # These formatters show constraints in comments
            constraint_indicators = ["min:", "max:", "ge:", "le:", "gt:", "lt:", "multipleOf:"]
            has_constraints = any(indicator in output for indicator in constraint_indicators)
            assert has_constraints, f"No constraints found in {format_type} output: {output}"

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_union_types_all_formatters(self, union_types_model, format_type):
        """Test union types across all formatters."""
        schema = simplify_schema(union_types_model, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should include field names
        assert "id" in output
        assert "status" in output
        assert "metadata" in output

        # Should handle union types with appropriate syntax
        if format_type == "typescript":
            assert " | " in output
        elif format_type == "yaml":
            assert " | " in output
        elif format_type == "jsonish":
            assert " or " in output or "oneOf:" in output

    @pytest.mark.parametrize("format_type", ["jsonish", "typescript", "yaml"])
    def test_empty_schema_all_formatters(self, empty_schema, format_type):
        """Test empty schema handling across all formatters."""
        schema = simplify_schema(empty_schema, format_type=format_type)
        output = schema.to_string()

        # Should handle empty schema gracefully
        assert output is not None
        assert isinstance(output, str)

        # Specific assertions depend on formatter
        if format_type == "jsonish":
            assert output == "{}"
        elif format_type == "typescript":
            assert "interface Schema {}" in output
        elif format_type == "yaml":
            assert output == "{}"
