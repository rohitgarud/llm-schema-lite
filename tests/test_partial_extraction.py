"""Tests for partial extraction functionality."""

import json

import pytest

from llm_schema_lite import (
    ConversionError,
    ParseConfig,
    _build_result,
    _get_json_schema,
    _validate_field,
    loads,
)


class TestLoadsWithSchema:
    """Tests for loads function with schema (partial mode via ParseConfig)."""

    def test_partial_extraction_with_invalid_optional_field(self):
        """Required field with invalid value raises error in partial mode."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int
            email: str | None = None

        # age is required, so invalid value should raise error
        with pytest.raises(ConversionError) as exc_info:
            loads(
                '{"name": "Alice", "age": "invalid", "email": "alice@example.com"}',
                schema=Person,
                parse_config=ParseConfig(partial=True),
            )
        assert "age" in str(exc_info.value)

    def test_partial_extraction_required_field_fails(self):
        """Required field with invalid value raises error when coercion is disabled."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        # With coercion disabled, passing int for string field should fail
        with pytest.raises(ConversionError) as exc_info:
            loads(
                '{"name": 123, "age": 30}',
                schema=Person,
                parse_config=ParseConfig(partial=True, allow_coercion=False),
            )
        assert "name" in str(exc_info.value)

    def test_partial_extraction_all_valid(self):
        """All valid fields returns full result with empty failed_fields."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        result, metadata = loads(
            '{"name": "Bob", "age": 25}',
            schema=Person,
            parse_config=ParseConfig(partial=True),
        )
        assert result.name == "Bob"
        assert result.age == 25
        assert "failed_fields" in metadata
        assert metadata["failed_fields"] == {}

    def test_partial_extraction_with_dict_input(self):
        """Works with dict input instead of string."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            email: str | None = None

        # Use only optional field for testing - convert dict to JSON string
        result, metadata = loads(
            json.dumps({"name": "Alice", "email": "alice@example.com"}),
            schema=Person,
            parse_config=ParseConfig(partial=True),
        )
        assert result.name == "Alice"
        assert result.email == "alice@example.com"
        assert "failed_fields" in metadata

    def test_partial_false_full_validation(self):
        """partial=False (default) raises error on invalid field."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        with pytest.raises(ConversionError):
            loads(
                '{"name": "John", "age": "not_a_number"}',
                schema=Person,
                parse_config=ParseConfig(partial=False),
            )

    def test_partial_false_valid_data(self):
        """partial=False (default) works with valid data."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        result, metadata = loads(
            '{"name": "John", "age": 30}',
            schema=Person,
            parse_config=ParseConfig(partial=False),
        )
        assert result.name == "John"
        assert result.age == 30
        assert metadata == {}

    def test_partial_extraction_multiple_invalid_fields(self):
        """Multiple invalid optional fields are handled correctly."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            email: str | None = None
            city: str | None = None

        # Disable coercion so integers don't get coerced to strings
        result, metadata = loads(
            '{"name": "Alice", "email": 123, "city": 456}',
            schema=Person,
            parse_config=ParseConfig(partial=True, allow_coercion=False),
        )
        assert result.name == "Alice"
        assert result.email is None
        assert result.city is None
        assert "email" in metadata["failed_fields"]
        assert "city" in metadata["failed_fields"]

    def test_partial_with_json_schema_dict(self):
        """Works with JSON schema dict instead of Pydantic model."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["name"],
        }

        # Use only required field
        result, metadata = loads(
            '{"name": "Bob"}',
            schema=schema,
            parse_config=ParseConfig(partial=True),
        )
        assert result["name"] == "Bob"
        assert "failed_fields" in metadata

    def test_partial_with_json_schema_string(self):
        """Works with JSON schema string."""
        schema_str = (
            '{"type": "object", "properties": '
            '{"name": {"type": "string"}, "email": {"type": "string"}}, '
            '"required": ["name"]}'
        )

        result, metadata = loads(
            '{"name": "Charlie"}',
            schema=schema_str,
            parse_config=ParseConfig(partial=True),
        )
        assert result["name"] == "Charlie"
        assert "failed_fields" in metadata

    def test_partial_missing_required_field_raises_error(self):
        """Missing required field raises error in partial mode."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        with pytest.raises(ConversionError) as exc_info:
            loads(
                '{"name": "John"}',
                schema=Person,
                parse_config=ParseConfig(partial=True),
            )
        assert "Required field 'age' is missing" in str(exc_info.value)

    def test_partial_with_coercion_config(self):
        """Works with custom ParseConfig for coercion."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            email: str | None = None

        # With coercion disabled, passing wrong type to optional field
        result, metadata = loads(
            '{"name": "John", "email": 123}',
            schema=Person,
            parse_config=ParseConfig(partial=True, allow_coercion=False),
        )
        assert result.name == "John"
        assert result.email is None  # Failed coercion
        assert "email" in metadata["failed_fields"]

    def test_partial_with_coercion_enabled(self):
        """Coercion works in partial mode."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        # With coercion enabled (default), string "30" should be coerced to int
        result, metadata = loads(
            '{"name": "John", "age": "30"}',
            schema=Person,
            parse_config=ParseConfig(partial=True),
        )
        assert result.name == "John"
        assert result.age == 30  # Successfully coerced
        assert "failed_fields" in metadata
        assert metadata["failed_fields"] == {}


class TestLoadsWithPartial:
    """Tests for loads function with partial mode."""

    def test_loads_with_partial(self):
        """loads function supports partial mode."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            email: str | None = None

        # Only use optional field for test
        result, metadata = loads(
            '{"name": "Alice", "email": "alice@example.com"}',
            parse_config=ParseConfig(partial=True),
            schema=Person,
        )
        assert result.name == "Alice"
        assert result.email == "alice@example.com"
        assert "failed_fields" in metadata

    def test_loads_without_partial(self):
        """loads function without partial returns dict (backward compatible)."""
        result = loads('{"name": "John", "age": 30}')
        assert result == {"name": "John", "age": 30}


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_json_schema_with_pydantic(self):
        """_get_json_schema works with Pydantic model."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        schema = _get_json_schema(Person)
        assert schema["type"] == "object"
        assert "properties" in schema

    def test_get_json_schema_with_dict(self):
        """_get_json_schema works with dict."""
        schema_dict = {"type": "object", "properties": {"name": {"type": "string"}}}
        schema = _get_json_schema(schema_dict)
        assert schema == schema_dict

    def test_get_json_schema_with_string(self):
        """_get_json_schema works with JSON string."""
        schema_str = '{"type": "object", "properties": {"name": {"type": "string"}}}'
        schema = _get_json_schema(schema_str)
        assert schema["type"] == "object"

    def test_validate_field_valid(self):
        """_validate_field returns True for valid value."""
        field_schema = {"type": "string"}
        is_valid, errors = _validate_field("hello", field_schema)
        assert is_valid is True
        assert errors == []

    def test_validate_field_invalid_type(self):
        """_validate_field returns False for invalid type."""
        field_schema = {"type": "integer"}
        is_valid, errors = _validate_field("not_a_number", field_schema)
        assert is_valid is False
        assert len(errors) > 0

    def test_build_result_with_pydantic(self):
        """_build_result returns Pydantic model when schema is BaseModel."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        data = {"name": "John", "age": 30}
        result = _build_result(data, Person)
        assert isinstance(result, Person)
        assert result.name == "John"

    def test_build_result_with_dict_schema(self):
        """_build_result returns dict when schema is dict."""
        schema_dict = {"type": "object"}
        data = {"name": "John", "age": 30}
        result = _build_result(data, schema_dict)
        assert result == data
        assert isinstance(result, dict)


class TestEdgeCases:
    """Edge case tests for partial extraction."""

    def test_empty_output(self):
        """Empty output with partial mode."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        with pytest.raises(ConversionError):
            loads(
                "{}",
                schema=Person,
                parse_config=ParseConfig(partial=True),
            )

    def test_whitespace_only_string(self):
        """Whitespace only string raises error."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        with pytest.raises(ConversionError):
            loads(
                "   ",
                schema=Person,
                parse_config=ParseConfig(partial=True),
            )

    def test_all_optional_fields(self):
        """All optional fields work correctly."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str | None = None
            age: int | None = None

        result, metadata = loads(
            '{"name": "Alice", "age": "invalid"}',
            schema=Person,
            parse_config=ParseConfig(partial=True),
        )
        assert result.name == "Alice"
        assert result.age is None

    def test_no_properties_schema(self):
        """Schema with no properties handled correctly."""
        schema = {"type": "object", "properties": {}}

        result, metadata = loads(
            '{"any": "data"}',
            schema=schema,
            parse_config=ParseConfig(partial=True),
        )
        assert result == {}
        assert "failed_fields" in metadata

    def test_non_dict_data_handled(self):
        """Non-dict data handled gracefully."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int | None = None

        # Pass a string that is not a valid JSON object - should raise error
        # for missing required field (since list is not valid object)
        with pytest.raises(ConversionError):
            loads(
                "[1, 2, 3]",  # String representation of a list, not an object
                schema=Person,
                parse_config=ParseConfig(partial=True),
            )


class TestRequiredMarkerKeys:
    """SchemaParser normalizes trailing required-marker reply keys (lsl-2026-09-04-008)."""

    def test_trailing_marker_key_maps_to_schema_field(self):
        """A reply key with the trailing marker maps onto the matching schema field."""
        from pydantic import BaseModel

        class M(BaseModel):
            name: str
            age: int | None = None

        result, _ = loads('{"name*": "x"}', schema=M)
        assert result.name == "x"

    def test_verbatim_schema_key_never_altered(self):
        """A reply key that already matches the schema verbatim is never altered."""
        from pydantic import BaseModel

        class M(BaseModel):
            name: str
            age: int | None = None

        result, _ = loads('{"name": "x"}', schema=M)
        assert result.name == "x"

    def test_marker_collision_prefers_verbatim_key(self):
        """When both the verbatim and the marked key are present, verbatim wins."""
        from pydantic import BaseModel

        class M(BaseModel):
            name: str
            age: int | None = None

        result, _ = loads('{"name": "a", "name*": "b"}', schema=M)
        assert result.name == "a"

    def test_marker_stripped_in_both_partial_and_full_modes(self):
        """Marker stripping applies exactly once, on both the partial and full routes."""
        from pydantic import BaseModel

        class M(BaseModel):
            name: str
            age: int | None = None

        result_full, _ = loads('{"name*": "x"}', schema=M, parse_config=ParseConfig(partial=False))
        result_partial, _ = loads(
            '{"name*": "x"}', schema=M, parse_config=ParseConfig(partial=True)
        )
        assert result_full.name == "x"
        assert result_partial.name == "x"

    def test_unknown_marked_key_is_not_invented(self):
        """A marked key with no matching schema property is left for downstream to drop."""
        from pydantic import BaseModel

        class M(BaseModel):
            name: str
            age: int | None = None

        result, _ = loads('{"name": "x", "bogus*": 1}', schema=M)
        assert result.name == "x"
        assert not hasattr(result, "bogus")

    def test_custom_marker_via_parse_config(self):
        """strip_required_marker overrides the default "*" marker."""
        from pydantic import BaseModel

        class M(BaseModel):
            name: str
            age: int | None = None

        result, _ = loads(
            '{"name!": "x"}', schema=M, parse_config=ParseConfig(strip_required_marker="!")
        )
        assert result.name == "x"

    def test_empty_marker_disables_stripping(self):
        """strip_required_marker="" is the escape hatch: "name*" stays an unknown key."""
        from pydantic import BaseModel

        class M(BaseModel):
            name: str
            age: int | None = None

        with pytest.raises(ConversionError, match="Required field 'name' is missing"):
            loads(
                '{"name*": "x"}',
                schema=M,
                parse_config=ParseConfig(strip_required_marker="", partial=True),
            )
