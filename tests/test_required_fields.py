"""Tests for required field highlighting functionality."""

import pytest
from pydantic import BaseModel, Field

from llm_schema_lite import simplify_schema


class TestRequiredFields:
    """Test required field highlighting across all formatters."""

    def test_required_fields_jsonish(self):
        """Test required field highlighting in JSONish formatter."""

        class TestModel(BaseModel):
            name: str = Field(description="User's name")
            age: int = Field(ge=0, le=120, description="User's age")
            email: str = Field(description="User's email")
            phone: str | None = Field(default=None, description="Optional phone")
            is_active: bool = Field(default=True, description="Active status")

        schema = simplify_schema(TestModel, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Should include required fields comment
        assert "// Fields marked with * are required" in output

        # Required fields should have asterisks
        assert "name*:" in output
        assert "age*:" in output
        assert "email*:" in output

        # Optional fields should not have asterisks
        assert "phone:" in output  # No asterisk
        assert "is_active:" in output  # No asterisk (has default)

    def test_required_fields_yaml(self):
        """Test required field highlighting in YAML formatter."""

        class TestModel(BaseModel):
            name: str = Field(description="User's name")
            age: int = Field(ge=0, le=120, description="User's age")
            email: str = Field(description="User's email")
            phone: str | None = Field(default=None, description="Optional phone")

        schema = simplify_schema(TestModel, format_type="yaml", include_metadata=True)
        output = schema.to_string()

        # Should include required fields comment
        assert "# Fields marked with * are required" in output

        # Required fields should have asterisks
        assert "name*:" in output
        assert "age*:" in output
        assert "email*:" in output

        # Optional fields should not have asterisks
        assert "phone:" in output  # No asterisk

    def test_required_fields_typescript(self):
        """Test required field highlighting in TypeScript formatter."""

        class TestModel(BaseModel):
            name: str = Field(description="User's name")
            age: int = Field(ge=0, le=120, description="User's age")
            email: str = Field(description="User's email")
            phone: str | None = Field(default=None, description="Optional phone")

        schema = simplify_schema(TestModel, format_type="typescript", include_metadata=True)
        output = schema.to_string()

        # Should include required fields comment
        assert "// Fields marked with * are required" in output

        # Required fields should have asterisks
        assert "name*:" in output
        assert "age*:" in output
        assert "email*:" in output

        # Optional fields should not have asterisks
        assert "phone:" in output  # No asterisk

    def test_no_required_fields_comment(self):
        """Test that comment is not added when there are no required fields."""

        class AllOptionalModel(BaseModel):
            name: str | None = Field(default=None, description="Optional name")
            age: int | None = Field(default=None, description="Optional age")

        schema = simplify_schema(AllOptionalModel, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Should not include required fields comment
        assert "// Fields marked with * are required" not in output

        # No fields should have asterisks
        assert "name:" in output  # No asterisk
        assert "age:" in output  # No asterisk

    def test_nested_required_fields(self):
        """Test required field highlighting in nested definitions."""

        class Address(BaseModel):
            street: str = Field(description="Street address")
            city: str = Field(description="City")
            zip_code: str | None = Field(default=None, description="Optional zip code")

        class User(BaseModel):
            name: str = Field(description="User's name")
            address: Address = Field(description="User's address")
            phone: str | None = Field(default=None, description="Optional phone")

        # Test JSONish formatter
        schema = simplify_schema(User, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Main schema required fields
        assert "name*:" in output
        assert "address*:" in output

        # Nested definition required fields (if processed)
        # Note: This depends on how nested definitions are handled

    def test_required_fields_without_metadata(self):
        """Test required field highlighting works without metadata."""

        class TestModel(BaseModel):
            name: str
            age: int
            email: str
            phone: str | None = None

        schema = simplify_schema(TestModel, format_type="jsonish", include_metadata=False)
        output = schema.to_string()

        # Should still include required fields comment even when metadata is disabled
        assert "// Fields marked with * are required" in output

        # Required fields should have asterisks
        assert "name*:" in output
        assert "age*:" in output
        assert "email*:" in output

        # Optional fields should not have asterisks
        assert "phone:" in output  # No asterisk

    @pytest.mark.parametrize("format_type", ["jsonish", "yaml", "typescript"])
    def test_required_fields_all_formatters(self, format_type):
        """Test required field highlighting across all formatters."""

        class TestModel(BaseModel):
            required_field: str = Field(description="This field is required")
            optional_field: str | None = Field(default=None, description="This field is optional")

        schema = simplify_schema(TestModel, format_type=format_type, include_metadata=True)
        output = schema.to_string()

        # Should include required fields comment
        assert "Fields marked with * are required" in output

        # Required field should have asterisk
        assert "required_field*:" in output

        # Optional field should not have asterisk
        assert "optional_field:" in output  # No asterisk

    def test_mixed_required_optional_fields(self):
        """Test schema with mix of required and optional fields."""

        class MixedModel(BaseModel):
            # Required fields
            id: int = Field(description="Unique identifier")
            name: str = Field(description="User name")
            email: str = Field(description="Email address")

            # Optional fields
            phone: str | None = Field(default=None, description="Phone number")
            age: int | None = Field(default=None, description="Age")
            is_active: bool = Field(default=True, description="Active status")

        schema = simplify_schema(MixedModel, format_type="jsonish", include_metadata=True)
        output = schema.to_string()

        # Should include required fields comment
        assert "// Fields marked with * are required" in output

        # Required fields should have asterisks
        assert "id*:" in output
        assert "name*:" in output
        assert "email*:" in output

        # Optional fields should not have asterisks
        assert "phone:" in output  # No asterisk
        assert "age:" in output  # No asterisk
        assert "is_active:" in output  # No asterisk (has default)
