"""Tests for the validate() function."""

import pytest
from pydantic import BaseModel, Field

from llm_schema_lite import validate
from llm_schema_lite.exceptions import ConversionError, UnsupportedModelError


class TestValidateWithPydantic:
    """Test validate() with Pydantic models."""

    def test_valid_data(self):
        """Test validation with valid data."""

        class User(BaseModel):
            name: str
            age: int

        is_valid, errors = validate(User, {"name": "John", "age": 30})
        assert is_valid is True
        assert errors is None

    def test_missing_required_field(self):
        """Test validation with missing required field."""

        class User(BaseModel):
            name: str
            age: int

        is_valid, errors = validate(User, {"name": "John"})
        assert is_valid is False
        assert errors is not None

    def test_wrong_type(self):
        """Test validation with wrong type."""

        class User(BaseModel):
            name: str
            age: int

        is_valid, errors = validate(User, {"name": "John", "age": "thirty"})
        assert is_valid is False
        assert errors is not None

    def test_extra_fields_allowed(self):
        """Test validation with extra fields (Pydantic allows by default)."""

        class User(BaseModel):
            name: str
            age: int

        # Pydantic will ignore extra fields by default
        is_valid, errors = validate(User, {"name": "John", "age": 30, "extra": "field"})
        assert is_valid is True
        assert errors is None

    def test_nested_model(self):
        """Test validation with nested models."""

        class Address(BaseModel):
            street: str
            city: str

        class User(BaseModel):
            name: str
            address: Address

        valid_data = {"name": "John", "address": {"street": "123 Main St", "city": "NYC"}}
        is_valid, errors = validate(User, valid_data)
        assert is_valid is True
        assert errors is None

        invalid_data = {"name": "John", "address": {"street": "123 Main St"}}
        is_valid, errors = validate(User, invalid_data)
        assert is_valid is False
        assert errors is not None

    def test_with_field_constraints(self):
        """Test validation with field constraints."""

        class Product(BaseModel):
            name: str = Field(..., min_length=1)
            price: float = Field(..., ge=0)

        is_valid, errors = validate(Product, {"name": "Widget", "price": 10.0})
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(Product, {"name": "", "price": 10.0})
        assert is_valid is False
        assert errors is not None
        is_valid, errors = validate(Product, {"name": "Widget", "price": -5.0})
        assert is_valid is False
        assert errors is not None

    def test_with_json_string_data(self):
        """Test validation with JSON string data."""

        class User(BaseModel):
            name: str
            age: int

        is_valid, errors = validate(User, '{"name": "John", "age": 30}')
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(User, '{"name": "John"}')
        assert is_valid is False
        assert errors is not None


class TestValidateWithJSONSchema:
    """Test validate() with JSON schema dicts."""

    def test_simple_object_valid(self):
        """Test validation with simple object schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }

        is_valid, errors = validate(schema, {"name": "John", "age": 30})
        assert is_valid is True
        assert errors is None

    def test_simple_object_missing_required(self):
        """Test validation with missing required field."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }

        is_valid, errors = validate(schema, {"name": "John"})
        assert is_valid is False
        assert errors is not None

    def test_simple_object_wrong_type(self):
        """Test validation with wrong type."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }

        is_valid, errors = validate(schema, {"name": "John", "age": "thirty"})
        assert is_valid is False
        assert errors is not None

    def test_nested_object(self):
        """Test validation with nested objects."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    "required": ["name"],
                }
            },
        }

        is_valid, errors = validate(schema, {"user": {"name": "John", "age": 30}})
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, {"user": {"age": 30}})
        assert is_valid is False
        assert errors is not None

    def test_array_validation(self):
        """Test validation with arrays."""
        schema = {"type": "array", "items": {"type": "string"}}

        is_valid, errors = validate(schema, ["a", "b", "c"])
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, ["a", 1, "c"])
        assert is_valid is False
        assert errors is not None

    def test_array_with_constraints(self):
        """Test validation with array constraints."""
        schema = {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5}

        is_valid, errors = validate(schema, ["a", "b"])
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, ["a"])
        assert is_valid is False
        assert errors is not None  # Too few items
        is_valid, errors = validate(schema, ["a", "b", "c", "d", "e", "f"])
        assert is_valid is False
        assert errors is not None  # Too many items

    def test_string_constraints(self):
        """Test validation with string constraints."""
        schema = {"type": "string", "minLength": 3, "maxLength": 10}

        is_valid, errors = validate(schema, "hello")
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, "hi")
        assert is_valid is False
        assert errors is not None  # Too short
        is_valid, errors = validate(schema, "hello world!")
        assert is_valid is False
        assert errors is not None  # Too long

    def test_string_pattern(self):
        """Test validation with string pattern."""
        schema = {"type": "string", "pattern": r"^[A-Z][a-z]+$"}

        is_valid, errors = validate(schema, "Hello")
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, "hello")
        assert is_valid is False
        assert errors is not None
        is_valid, errors = validate(schema, "HELLO")
        assert is_valid is False
        assert errors is not None

    def test_number_constraints(self):
        """Test validation with number constraints."""
        schema = {"type": "number", "minimum": 0, "maximum": 100}

        is_valid, errors = validate(schema, 50)
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, 0)
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, 100)
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, -1)
        assert is_valid is False
        assert errors is not None
        is_valid, errors = validate(schema, 101)
        assert is_valid is False
        assert errors is not None

    def test_integer_constraints(self):
        """Test validation with integer constraints."""
        schema = {"type": "integer", "minimum": 1, "maximum": 10}

        is_valid, errors = validate(schema, 5)
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, 0)
        assert is_valid is False
        assert errors is not None
        is_valid, errors = validate(schema, 11)
        assert is_valid is False
        assert errors is not None

    def test_exclusive_minimum_maximum(self):
        """Test validation with exclusive minimum/maximum."""
        schema = {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 100}

        is_valid, errors = validate(schema, 50)
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, 0.1)
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, 0)
        assert is_valid is False
        assert errors is not None
        is_valid, errors = validate(schema, 100)
        assert is_valid is False
        assert errors is not None

    def test_multiple_types(self):
        """Test validation with multiple types (union)."""
        schema = {"type": ["string", "null"]}

        is_valid, errors = validate(schema, "hello")
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, None)
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, 123)
        assert is_valid is False
        assert errors is not None

    def test_boolean_type(self):
        """Test validation with boolean type."""
        schema = {"type": "boolean"}

        is_valid, errors = validate(schema, True)
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, False)
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, "true")
        assert is_valid is False
        assert errors is not None

    def test_null_type(self):
        """Test validation with null type."""
        schema = {"type": "null"}

        is_valid, errors = validate(schema, None)
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, "null")
        assert is_valid is False
        assert errors is not None


class TestValidateWithJSONSchemaString:
    """Test validate() with JSON schema strings."""

    def test_valid_schema_string(self):
        """Test validation with JSON schema string."""
        schema_str = (
            '{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}'
        )

        is_valid, errors = validate(schema_str, {"name": "John"})
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema_str, {})
        assert is_valid is False
        assert errors is not None

    def test_invalid_schema_string(self):
        """Test validation with invalid JSON schema string."""
        schema_str = '{"type": "object", invalid json'

        with pytest.raises(ConversionError, match="Invalid JSON schema string"):
            validate(schema_str, {"name": "John"})


class TestValidateErrorHandling:
    """Test error handling in validate()."""

    def test_unsupported_schema_type(self):
        """Test validation with unsupported schema type."""
        with pytest.raises(UnsupportedModelError, match="Unsupported schema type"):
            validate(123, {"name": "John"})  # type: ignore

    def test_invalid_data_string(self):
        """Test validation with invalid JSON data string."""

        class User(BaseModel):
            name: str

        # Invalid JSON that can't be repaired should return False
        is_valid, errors = validate(User, "{invalid json}")
        assert is_valid is False
        assert errors is not None


class TestValidateWithYAML:
    """Test validate() with YAML data."""

    def test_yaml_string_data(self):
        """Test validation with YAML string data."""

        class User(BaseModel):
            name: str
            age: int

        yaml_data = """
name: John
age: 30
"""
        is_valid, errors = validate(User, yaml_data, mode="yaml")
        assert is_valid is True
        assert errors is None

    def test_yaml_missing_field(self):
        """Test validation with YAML missing required field."""

        class User(BaseModel):
            name: str
            age: int

        yaml_data = "name: John"
        is_valid, errors = validate(User, yaml_data, mode="yaml")
        assert is_valid is False
        assert errors is not None

    def test_yaml_wrong_type(self):
        """Test validation with YAML wrong type."""

        class User(BaseModel):
            name: str
            age: int

        yaml_data = """
name: John
age: thirty
"""
        is_valid, errors = validate(User, yaml_data, mode="yaml")
        assert is_valid is False
        assert errors is not None

    def test_yaml_with_json_schema(self):
        """Test YAML validation with JSON schema dict."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }

        yaml_data = """
name: Jane
age: 25
"""
        is_valid, errors = validate(schema, yaml_data, mode="yaml")
        assert is_valid is True
        assert errors is None

    def test_yaml_nested_structure(self):
        """Test YAML validation with nested structures."""

        class Address(BaseModel):
            street: str
            city: str

        class User(BaseModel):
            name: str
            address: Address

        yaml_data = """
name: John
address:
  street: 123 Main St
  city: NYC
"""
        is_valid, errors = validate(User, yaml_data, mode="yaml")
        assert is_valid is True
        assert errors is None

    def test_yaml_with_lists(self):
        """Test YAML validation with lists."""

        class User(BaseModel):
            name: str
            tags: list[str]

        yaml_data = """
name: John
tags:
  - python
  - testing
  - yaml
"""
        is_valid, errors = validate(User, yaml_data, mode="yaml")
        assert is_valid is True
        assert errors is None

    def test_auto_mode_json(self):
        """Test auto mode with JSON data."""

        class User(BaseModel):
            name: str
            age: int

        # JSON string should be detected automatically
        json_data = '{"name": "John", "age": 30}'
        is_valid, errors = validate(User, json_data, mode="auto")
        assert is_valid is True
        assert errors is None

    def test_auto_mode_yaml(self):
        """Test auto mode with YAML data."""

        class User(BaseModel):
            name: str
            age: int

        # YAML string should be detected automatically
        yaml_data = "name: John\nage: 30"
        is_valid, errors = validate(User, yaml_data, mode="auto")
        assert is_valid is True
        assert errors is None

    def test_auto_mode_fallback_to_yaml(self):
        """Test auto mode falling back to YAML when JSON fails."""

        class User(BaseModel):
            name: str
            age: int

        # This looks like JSON but is actually YAML
        yaml_data = """
name: John
age: 30
"""
        is_valid, errors = validate(User, yaml_data, mode="auto")
        assert is_valid is True
        assert errors is None


class TestValidateMultipleErrors:
    """Test validate() with multiple errors and return_all_errors parameter."""

    def test_multiple_errors_all_returned(self):
        """Test that all errors are returned when return_all_errors=True."""

        class User(BaseModel):
            name: str
            age: int
            email: str

        # Missing all required fields
        is_valid, errors = validate(User, {}, return_all_errors=True)
        assert is_valid is False
        assert errors is not None
        assert isinstance(errors, list)
        assert len(errors) == 3  # Three missing required fields
        # Check that all field names are mentioned
        error_text = " ".join(errors)
        assert "name" in error_text
        assert "age" in error_text
        assert "email" in error_text

    def test_multiple_errors_first_only(self):
        """Test that only first error is returned when return_all_errors=False."""

        class User(BaseModel):
            name: str
            age: int
            email: str

        # Missing all required fields
        is_valid, errors = validate(User, {}, return_all_errors=False)
        assert is_valid is False
        assert errors is not None
        assert isinstance(errors, list)
        assert len(errors) == 1  # Only first error

    def test_multiple_type_errors(self):
        """Test multiple type validation errors."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"},
            },
            "required": ["name", "age", "active"],
        }

        # All fields have wrong types
        is_valid, errors = validate(
            schema, {"name": 123, "age": "thirty", "active": "yes"}, return_all_errors=True
        )
        assert is_valid is False
        assert errors is not None
        assert isinstance(errors, list)
        assert len(errors) >= 3  # At least three type errors

    def test_mixed_errors(self):
        """Test mix of missing fields and constraint violations."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 3},
                "age": {"type": "integer", "minimum": 0},
                "email": {"type": "string"},
            },
            "required": ["name", "age", "email"],
        }

        # Short name, negative age, missing email
        is_valid, errors = validate(schema, {"name": "Jo", "age": -5}, return_all_errors=True)
        assert is_valid is False
        assert errors is not None
        assert isinstance(errors, list)
        assert len(errors) >= 2  # At least missing email and one constraint violation


class TestValidateErrorMessages:
    """Test validate() error messages for LLM feedback."""

    def test_missing_required_field_error(self):
        """Test error message for missing required field."""

        class User(BaseModel):
            name: str
            age: int

        is_valid, errors = validate(User, {"name": "John"})
        assert is_valid is False
        assert errors is not None
        assert isinstance(errors, list)
        assert any("'age' is a required property" in err for err in errors)

    def test_wrong_type_error(self):
        """Test error message for wrong type."""
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}

        is_valid, errors = validate(schema, {"age": "thirty"})
        assert is_valid is False
        assert errors is not None
        assert isinstance(errors, list)
        assert any("type" in err.lower() or "integer" in err.lower() for err in errors)

    def test_constraint_violation_error(self):
        """Test error message for constraint violation."""
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer", "minimum": 0, "maximum": 120}},
        }

        is_valid, errors = validate(schema, {"age": 150})
        assert is_valid is False
        assert errors is not None
        assert isinstance(errors, list)
        assert any("maximum" in err.lower() or "120" in err for err in errors)

    def test_pattern_error(self):
        """Test error message for pattern mismatch."""
        schema = {"type": "string", "pattern": "^[A-Z][a-z]+$"}

        is_valid, errors = validate(schema, "hello")
        assert is_valid is False
        assert errors is not None
        assert isinstance(errors, list)
        assert any("pattern" in err.lower() for err in errors)

    def test_nested_field_error(self):
        """Test error message shows path for nested fields."""

        class Address(BaseModel):
            street: str
            city: str

        class User(BaseModel):
            name: str
            address: Address

        is_valid, errors = validate(User, {"name": "John", "address": {"street": "123 Main"}})
        assert is_valid is False
        assert errors is not None
        assert isinstance(errors, list)
        assert any("address" in err.lower() or "city" in err.lower() for err in errors)


class TestValidateComplexScenarios:
    """Test validate() with complex scenarios."""

    def test_deeply_nested_structure(self):
        """Test validation with deeply nested structure."""
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {"level3": {"type": "string"}},
                            "required": ["level3"],
                        }
                    },
                    "required": ["level2"],
                }
            },
            "required": ["level1"],
        }

        valid_data = {"level1": {"level2": {"level3": "value"}}}
        is_valid, errors = validate(schema, valid_data)
        assert is_valid is True
        assert errors is None

        invalid_data = {"level1": {"level2": {}}}
        is_valid, errors = validate(schema, invalid_data)
        assert is_valid is False
        assert errors is not None

    def test_array_of_objects(self):
        """Test validation with array of objects."""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name"],
            },
        }

        valid_data = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
        is_valid, errors = validate(schema, valid_data)
        assert is_valid is True
        assert errors is None

        invalid_data = [{"name": "John"}, {"age": 25}]  # Missing name in second item
        is_valid, errors = validate(schema, invalid_data)
        assert is_valid is False
        assert errors is not None

    def test_mixed_valid_and_invalid_properties(self):
        """Test validation with mix of valid and invalid properties."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 3},
                "age": {"type": "integer", "minimum": 0},
                "email": {"type": "string"},
            },
            "required": ["name", "age"],
        }

        # All valid
        is_valid, errors = validate(
            schema, {"name": "John", "age": 30, "email": "john@example.com"}
        )
        assert is_valid is True
        assert errors is None

        # Invalid name (too short)
        is_valid, errors = validate(schema, {"name": "Jo", "age": 30})
        assert is_valid is False
        assert errors is not None

        # Invalid age (negative)
        is_valid, errors = validate(schema, {"name": "John", "age": -5})
        assert is_valid is False
        assert errors is not None

    def test_optional_fields(self):
        """Test validation with optional fields."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
            "required": ["name"],
        }

        # Only required field
        is_valid, errors = validate(schema, {"name": "John"})
        assert is_valid is True
        assert errors is None

        # With optional fields
        is_valid, errors = validate(schema, {"name": "John", "age": 30})
        assert is_valid is True
        assert errors is None
        is_valid, errors = validate(schema, {"name": "John", "email": "john@example.com"})
        assert is_valid is True
        assert errors is None

        # Missing required field
        is_valid, errors = validate(schema, {"age": 30})
        assert is_valid is False
        assert errors is not None
