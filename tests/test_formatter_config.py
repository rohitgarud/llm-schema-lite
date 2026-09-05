"""Tests for FormatterConfig functionality."""

import pytest
from pydantic import BaseModel, Field

from llm_schema_lite import FormatterConfig, simplify_schema
from llm_schema_lite.formatters.config import (
    DEFAULT_METADATA_INCLUSION,
    DEFAULT_UNION_SEPARATOR,
    DESCRIPTION_KEYWORDS,
    with_format_default_separator,
)
from tests.formatter_helpers import render_all_formatters


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
        """Disabling descriptions removes description text but keeps constraints."""
        config = FormatterConfig(include_descriptions=False, include_constraints=True)

        class DescribedConstrainedModel(BaseModel):
            value: int = Field(default=5, description="A described value", ge=0, le=10)

        result = simplify_schema(DescribedConstrainedModel, config=config)
        output = result.to_string()
        # Description text must be gone...
        assert "A described value" not in output
        # ...but the constraint category (including the default) must remain.
        assert "0 to 10" in output
        assert "(default=" in output

    def test_include_constraints_false(self):
        """Disabling constraints removes defaults too (ticket AC#3), keeps descriptions."""
        config = FormatterConfig(include_constraints=False)

        class DescribedConstrainedModel(BaseModel):
            value: int = Field(default=5, description="A described value", ge=0, le=10)

        result = simplify_schema(DescribedConstrainedModel, config=config)
        output = result.to_string()
        # AC#3: no default text may survive when constraints are disabled.
        assert "(default=" not in output
        # Constraint text is gone...
        assert "0 to 10" not in output
        # ...but the description remains.
        assert "A described value" in output

    def test_include_constraints_false_removes_ref_defaults_end_to_end(self, patient_model):
        """AC#3 end-to-end: no `(default=` text survives `include_constraints=False`.

        Exercises both a `$ref` field carrying a default (`role`, an enum) and a plain
        optional with `default=None` (`nickname`) -- the two independently-spelled,
        previously-ungated `$ref` default sites that this ticket gates -- across all three
        formatters.
        """
        config = FormatterConfig(include_constraints=False)
        rendered = render_all_formatters(patient_model, config)

        assert "(default=" not in rendered["jsonish"]
        assert "(default=" not in rendered["yaml"]
        # TypeScript never spells defaults as "(default=" -- it always uses
        # "(defaults to ...)" -- so asserting only "(default=" absence would be vacuously
        # true here. Assert the TypeScript spelling too so this leg actually exercises the
        # gate (unifying the spellings is out of scope, see design FU-3).
        assert "(default=" not in rendered["typescript"]
        assert "(defaults to" not in rendered["typescript"]

    def test_include_descriptions_false_narrowing_end_to_end(self, patient_model):
        """A per-keyword `metadata_inclusion` True cannot restore a category-removed field.

        Phase 1 covers this narrowing rule at the `includes()` unit level only; this
        exercises the full render pipeline across all three formatters.
        """
        config = FormatterConfig(
            include_descriptions=False, metadata_inclusion={"description": True}
        )
        rendered = render_all_formatters(patient_model, config)

        for fmt, output in rendered.items():
            assert "Full name" not in output, f"{fmt} leaked a description"

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

    def test_max_recursion_depth_defaults_to_two(self):
        """Test that max_recursion_depth defaults to 2."""
        config = FormatterConfig()
        assert config.max_recursion_depth == 2

    def test_max_recursion_depth_negative_raises(self):
        """Test that a negative max_recursion_depth raises ValueError."""
        with pytest.raises(ValueError, match="max_recursion_depth must be >= 0"):
            FormatterConfig(max_recursion_depth=-1)

    def test_max_recursion_depth_zero_is_accepted(self):
        """Test that max_recursion_depth=0 constructs successfully."""
        config = FormatterConfig(max_recursion_depth=0)
        assert config.max_recursion_depth == 0


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


class TestFormatterConfigIncludes:
    """Truth table for the three-gate FormatterConfig.includes() category layer."""

    @pytest.mark.parametrize(
        "im,desc,cons,expect_description,expect_title,expect_default,expect_pattern",
        [
            (True, True, True, True, True, True, True),
            (True, True, False, True, True, False, False),
            (True, False, True, False, False, True, True),
            (True, False, False, False, False, False, False),
            (False, True, True, False, False, False, False),
            (False, True, False, False, False, False, False),
            (False, False, True, False, False, False, False),
            (False, False, False, False, False, False, False),
        ],
    )
    def test_includes_truth_table(
        self, im, desc, cons, expect_description, expect_title, expect_default, expect_pattern
    ):
        cfg = FormatterConfig(
            include_metadata=im, include_descriptions=desc, include_constraints=cons
        )
        assert cfg.includes("description") is expect_description
        assert cfg.includes("title") is expect_title
        assert cfg.includes("default") is expect_default
        assert cfg.includes("pattern") is expect_pattern
        # `examples` is False in DEFAULT_METADATA_INCLUSION -> third gate always removes it
        assert cfg.includes("examples") is False

    def test_narrowing_not_overriding(self):
        """A per-keyword True cannot restore what a category flag removed."""
        cfg = FormatterConfig(include_descriptions=False, metadata_inclusion={"description": True})
        assert cfg.includes("description") is False

    def test_default_metadata_inclusion_is_unchanged(self):
        """INV-2: the 13-entry default table is byte-identical."""
        assert DEFAULT_METADATA_INCLUSION == {
            "pattern": True,
            "format": True,
            "minimum": True,
            "maximum": True,
            "minLength": True,
            "maxLength": True,
            "minItems": True,
            "maxItems": True,
            "uniqueItems": True,
            "const": True,
            "default": True,
            "title": True,
            "examples": False,
        }

    def test_description_keywords_membership(self):
        assert DESCRIPTION_KEYWORDS == frozenset(
            {"title", "description", "id", "$comment", "x-enum-descriptions"}
        )


class TestWithFormatDefaultSeparator:
    """Unit tests for `with_format_default_separator` (design v2 behaviour table)."""

    def test_none_config_returns_fresh_config_with_separator(self):
        """`None` in yields a fresh `FormatterConfig` carrying the given separator."""
        result = with_format_default_separator(None, " OR ")

        assert isinstance(result, FormatterConfig)
        assert result.union_separator == " OR "

    def test_default_separator_is_replaced_without_mutating_input(self):
        """A config still at the package default yields a *different* object with the new
        separator; the input object's own `union_separator` is unchanged."""
        original = FormatterConfig()
        assert original.union_separator == DEFAULT_UNION_SEPARATOR

        result = with_format_default_separator(original, " OR ")

        assert result.union_separator == " OR "
        assert original.union_separator == DEFAULT_UNION_SEPARATOR
        # Row 2 of the behaviour table is the copy branch, so `is not` is CORRECT here.
        assert result is not original

    def test_explicit_separator_is_returned_unchanged_by_identity(self):
        """A config with a non-default separator is returned as the same object,
        untouched, so an explicit choice always wins."""
        original = FormatterConfig(union_separator=" or ")

        result = with_format_default_separator(original, " OR ")

        # Row 3 of the behaviour table is the passthrough branch: the helper returns the
        # caller's own object. `is` is CORRECT here; `is not` would be a bug in the test.
        assert result is original
        assert result.union_separator == " or "
