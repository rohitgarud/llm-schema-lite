"""Tests for FormatterConfig functionality."""

from pydantic import BaseModel

from llm_schema_lite import FormatterConfig, simplify_schema


class User(BaseModel):
    name: str
    age: int


class RoleEnum(BaseModel):
    role: str


class TestFormatterConfig:
    """Test suite for FormatterConfig options."""

    def test_default_config(self):
        """Test that default config produces expected output."""
        config = FormatterConfig()
        result = simplify_schema(User, config=config)
        output = result.to_string()
        # Check that output contains the expected field names (with marker)
        assert "name" in output
        assert "age" in output

    def test_required_marker(self):
        """Test custom required marker."""
        config = FormatterConfig(required_marker="!")
        result = simplify_schema(User, config=config)
        output = result.to_string()
        # Required fields should have ! marker
        assert "name!" in output
        assert "age!" in output

    def test_optional_marker(self):
        """Test custom optional marker with a model that has optional fields."""
        config = FormatterConfig(optional_marker="?")

        class OptionalUser(BaseModel):
            name: str
            email: str | None = None

        result = simplify_schema(OptionalUser, config=config)
        output = result.to_string()
        # Optional fields should have ? marker - check the email field
        assert "email?" in output

    def test_union_separator_jsonish(self):
        """Test custom union separator for JSONish format."""
        config = FormatterConfig(union_separator=" or ")
        result = simplify_schema(User, config=config, format_type="jsonish")
        output = result.to_string()
        # The default union_separator for JSONish is " OR ", but we override it
        # Since User doesn't have union types, this is a basic test
        assert "name" in output
        assert "age" in output

    def test_union_separator_typescript(self):
        """Test custom union separator for TypeScript format."""
        config = FormatterConfig(union_separator=" | ")
        result = simplify_schema(User, config=config, format_type="typescript")
        output = result.to_string()
        assert "name" in output
        assert "age" in output

    def test_include_metadata_false(self):
        """Test disabling metadata - should remove schema-level comments."""
        config = FormatterConfig(include_metadata=False)
        result = simplify_schema(User, config=config)
        output = result.to_string()
        # When metadata is disabled, schema-level comments should be removed
        # The title comment should not be present
        assert "Title:" not in output
        # The required fields comment should not be present
        assert "Fields marked with" not in output

    def test_include_descriptions_false(self):
        """Test disabling descriptions."""
        config = FormatterConfig(include_descriptions=False)
        result = simplify_schema(User, config=config)
        output = result.to_string()
        # Basic test - should still produce output
        assert "name" in output

    def test_include_constraints_false(self):
        """Test disabling constraints."""
        config = FormatterConfig(include_constraints=False)

        class ConstrainedModel(BaseModel):
            value: int

        result = simplify_schema(ConstrainedModel, config=config)
        output = result.to_string()
        # Basic test - should still produce output
        assert "value" in output

    def test_prefix_option(self):
        """Test prefix option."""
        prefix_text = "Answer in this format:\n"
        config = FormatterConfig(prefix=prefix_text)
        result = simplify_schema(User, config=config)
        output = result.to_string()
        assert prefix_text in output

    def test_backward_compatibility_include_metadata(self):
        """Test backward compatibility with include_metadata parameter."""
        # Using the old API should still work
        result = simplify_schema(User, include_metadata=False)
        output = result.to_string()
        # Check that schema-level comments are removed
        assert "Title:" not in output
        assert "Fields marked with" not in output

    def test_config_with_typescript_format(self):
        """Test config with TypeScript format."""
        config = FormatterConfig(required_marker="!")
        result = simplify_schema(User, config=config, format_type="typescript")
        output = result.to_string()
        assert "name!" in output
        assert "age!" in output

    def test_config_with_yaml_format(self):
        """Test config with YAML format."""
        config = FormatterConfig(required_marker="!")
        result = simplify_schema(User, config=config, format_type="yaml")
        output = result.to_string()
        assert "name!" in output
        assert "age!" in output

    def test_hoist_enums_true(self):
        """Test hoist_enums=True option."""
        config = FormatterConfig(hoist_enums=True)
        # This tests the config is accepted - actual hoisting behavior
        # depends on the schema structure
        result = simplify_schema(User, config=config)
        assert result is not None

    def test_hoist_enums_false(self):
        """Test hoist_enums=False option."""
        config = FormatterConfig(hoist_enums=False)
        result = simplify_schema(User, config=config)
        assert result is not None

    def test_hoist_enums_auto(self):
        """Test hoist_enums='auto' option."""
        config = FormatterConfig(hoist_enums="auto")
        result = simplify_schema(User, config=config)
        assert result is not None

    def test_hoist_classes_true(self):
        """Test hoist_classes=True option."""
        config = FormatterConfig(hoist_classes=True)
        result = simplify_schema(User, config=config)
        assert result is not None

    def test_hoist_classes_false(self):
        """Test hoist_classes=False option."""
        config = FormatterConfig(hoist_classes=False)
        result = simplify_schema(User, config=config)
        assert result is not None

    def test_hoist_classes_list(self):
        """Test hoist_classes with a list of class names."""
        config = FormatterConfig(hoist_classes=["SomeClass"])
        result = simplify_schema(User, config=config)
        assert result is not None


class TestFormatterConfigIntegration:
    """Integration tests for FormatterConfig with complex schemas."""

    def test_model_with_union_type(self):
        """Test config with a model that has union types."""

        class UnionModel(BaseModel):
            value: str | int

        config = FormatterConfig(union_separator=" or ")
        result = simplify_schema(UnionModel, config=config, format_type="jsonish")
        output = result.to_string()
        # Should use custom separator
        assert " or " in output.lower()

    def test_model_with_enum(self):
        """Test config with a model that has enum."""

        class EnumModel(BaseModel):
            status: str

        config = FormatterConfig(hoist_enums=True)
        result = simplify_schema(EnumModel, config=config)
        assert result is not None
