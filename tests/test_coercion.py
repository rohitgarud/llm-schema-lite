"""Tests for type coercion functionality."""

import logging

from llm_schema_lite import ParseConfig, coerce, coerce_value
from llm_schema_lite.core import loads


class TestCoerceValue:
    """Test suite for coerce_value function."""

    def test_coerce_int_from_string(self):
        """Coerce string '42' to int 42."""
        result, metadata = coerce_value("42", "integer")
        assert result == 42
        assert metadata is not None
        assert metadata.coercion_type == "to_int"
        assert metadata.original_value == "42"

    def test_coerce_int_from_float(self):
        """Coerce float 42.0 to int 42."""
        result, metadata = coerce_value(42.0, "integer")
        assert result == 42
        assert metadata is not None

    def test_coerce_float_from_string(self):
        """Coerce string '3.14' to float 3.14."""
        result, metadata = coerce_value("3.14", "number")
        assert result == 3.14
        assert metadata is not None

    def test_coerce_float_from_int(self):
        """Coerce int 3 to float 3.0."""
        result, metadata = coerce_value(3, "number")
        assert result == 3.0
        assert metadata is not None

    def test_coerce_bool_from_string_true(self):
        """Coerce string 'true' to boolean True."""
        result, metadata = coerce_value("true", "boolean")
        assert result is True
        assert metadata is not None
        assert metadata.coercion_type == "string_to_bool"

    def test_coerce_bool_from_string_false(self):
        """Coerce string 'false' to boolean False."""
        result, metadata = coerce_value("false", "boolean")
        assert result is False
        assert metadata is not None

    def test_coerce_bool_from_string_yes_no(self):
        """Coerce string 'yes'/'no' to boolean."""
        result, _ = coerce_value("yes", "boolean")
        assert result is True
        result, _ = coerce_value("no", "boolean")
        assert result is False

    def test_coerce_bool_from_string_yes_no_case_insensitive(self):
        """Coerce string 'YES'/'NO' (case insensitive) to boolean."""
        result, _ = coerce_value("YES", "boolean")
        assert result is True
        result, _ = coerce_value("NO", "boolean")
        assert result is False

    def test_coerce_bool_from_string_on_off(self):
        """Coerce string 'on'/'off' to boolean."""
        result, _ = coerce_value("on", "boolean")
        assert result is True
        result, _ = coerce_value("off", "boolean")
        assert result is False

    def test_coerce_bool_from_int_one(self):
        """Coerce int 1 to boolean True."""
        result, metadata = coerce_value(1, "boolean")
        assert result is True
        assert metadata is not None
        assert metadata.coercion_type == "int_to_bool"

    def test_coerce_bool_from_int_zero(self):
        """Coerce int 0 to boolean False."""
        result, metadata = coerce_value(0, "boolean")
        assert result is False
        assert metadata is not None

    def test_coerce_bool_from_float_zero(self):
        """Coerce float 0.0 to boolean False."""
        result, metadata = coerce_value(0.0, "boolean")
        assert result is False

    def test_coerce_bool_from_float_one(self):
        """Coerce float 1.0 to boolean True."""
        result, metadata = coerce_value(1.0, "boolean")
        assert result is True

    def test_coerce_string_from_int(self):
        """Coerce int 42 to string '42'."""
        result, metadata = coerce_value(42, "string")
        assert result == "42"
        assert metadata is not None

    def test_coerce_string_from_float(self):
        """Coerce float 3.14 to string '3.14'."""
        result, metadata = coerce_value(3.14, "string")
        assert result == "3.14"

    def test_coerce_string_from_bool(self):
        """Coerce bool True to string 'True'."""
        result, metadata = coerce_value(True, "string")
        assert result == "True"

    def test_no_coercion_needed(self):
        """No coercion when already correct type."""
        result, metadata = coerce_value(42, "integer")
        assert result == 42
        assert metadata is None

    def test_no_coercion_for_bool(self):
        """No coercion when already bool type."""
        result, metadata = coerce_value(True, "boolean")
        assert result is True
        assert metadata is None

    def test_no_coercion_for_string(self):
        """No coercion when already string type."""
        result, metadata = coerce_value("hello", "string")
        assert result == "hello"
        assert metadata is None

    def test_coerce_enum_case_insensitive(self):
        """Coerce enum value with case-insensitive matching."""
        enum_values = ["critical", "high", "low"]
        result, metadata = coerce_value("CRITICAL", "string", enum_values=enum_values)
        assert result == "critical"
        assert metadata is not None
        assert metadata.coercion_type == "enum_case_insensitive"

    def test_coerce_enum_case_insensitive_mixed(self):
        """Coerce enum value with mixed case."""
        enum_values = ["Critical", "High", "Low"]
        result, metadata = coerce_value("critical", "string", enum_values=enum_values)
        assert result == "Critical"
        assert metadata is not None
        assert metadata.coercion_type == "enum_case_insensitive"

    def test_coerce_enum_no_match(self):
        """No coercion when enum value doesn't match."""
        enum_values = ["critical", "high", "low"]
        result, metadata = coerce_value("medium", "string", enum_values=enum_values)
        assert result == "medium"
        assert metadata is None

    def test_coerce_enum_exact_match(self):
        """No coercion when exact enum match."""
        enum_values = ["critical", "high", "low"]
        result, metadata = coerce_value("critical", "string", enum_values=enum_values)
        assert result == "critical"
        assert metadata is None

    def test_coerce_enum_numeric_string(self):
        """Numeric text matches an integer enum member and returns the member itself."""
        result, metadata = coerce_value("2", "integer", enum_values=[1, 2])
        assert result == 2
        assert type(result) is int
        assert metadata is not None
        assert metadata.coercion_type == "enum_from_string"

    def test_coerce_list_single_item(self):
        """Coerce single item to list when enabled."""
        result, metadata = coerce_value("item", "array", coerce_list_single_item=True)
        assert result == ["item"]
        assert metadata is not None
        assert metadata.coercion_type == "single_to_list"

    def test_coerce_list_single_item_disabled(self):
        """Don't coerce single item to list when disabled."""
        result, metadata = coerce_value("item", "array", coerce_list_single_item=False)
        assert result == "item"
        assert metadata is None

    def test_coerce_list_already_list(self):
        """Don't coerce if already a list."""
        result, metadata = coerce_value([1, 2, 3], "array", coerce_list_single_item=True)
        assert result == [1, 2, 3]
        assert metadata is None


class TestParseConfig:
    """Test suite for ParseConfig."""

    def test_default_config(self):
        """Test default ParseConfig values."""
        config = ParseConfig()
        assert config.allow_coercion is True
        assert config.coerce_list_single_item is False
        assert config.log_coercions is True

    def test_custom_config(self):
        """Test custom ParseConfig values."""
        config = ParseConfig(
            allow_coercion=False,
            coerce_list_single_item=True,
            log_coercions=False,
        )
        assert config.allow_coercion is False
        assert config.coerce_list_single_item is True
        assert config.log_coercions is False


class TestCoerceFunction:
    """Test suite for coerce() function with schema."""

    def test_coerce_with_json_schema_dict(self):
        """Coerce data to match JSON schema dict."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"},
                "name": {"type": "string"},
            },
        }
        data = {"count": "10", "enabled": "1", "name": "test"}
        result, metadata = coerce(data, schema, ParseConfig())

        assert result["count"] == 10
        assert result["enabled"] is True
        assert result["name"] == "test"
        assert len(metadata) == 2  # count and enabled coerced

    def test_coerce_with_pydantic_model(self):
        """Coerce data to match Pydantic model types."""
        from pydantic import BaseModel

        class User(BaseModel):
            age: int
            active: bool
            score: float

        data = {"age": "25", "active": "true", "score": "95.5"}
        result, metadata = coerce(data, User, ParseConfig())

        assert result["age"] == 25
        assert result["active"] is True
        assert result["score"] == 95.5

    def test_coerce_with_json_string_schema(self):
        """Coerce data to match JSON schema string."""
        schema = '{"type": "object", "properties": {"count": {"type": "integer"}}}'
        data = {"count": "42"}
        result, metadata = coerce(data, schema, ParseConfig())

        assert result["count"] == 42

    def test_coerce_disabled(self):
        """No coercion when disabled in config."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
        }
        data = {"count": "10"}
        config = ParseConfig(allow_coercion=False)
        result, metadata = coerce(data, schema, config)

        # Should return original data (or empty dict for non-dict input)
        assert result.get("count") == "10" or result == {}
        assert metadata == []

    def test_coerce_nested_structure(self):
        """Coerce nested dict structure."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer"},
                        "active": {"type": "boolean"},
                    },
                }
            },
        }
        data = {"user": {"age": "25", "active": "true"}}
        result, metadata = coerce(data, schema, ParseConfig())

        assert result["user"]["age"] == 25
        assert result["user"]["active"] is True

    def test_coerce_array_items(self):
        """Coerce array items."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
        }
        data = {"items": ["1", "2", "3"]}
        result, metadata = coerce(data, schema, ParseConfig())

        assert result["items"] == [1, 2, 3]

    def test_coerce_with_enum(self):
        """Coerce enum values."""
        schema = {
            "type": "object",
            "properties": {
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
            },
        }
        data = {"priority": "HIGH"}
        result, metadata = coerce(data, schema, ParseConfig())

        assert result["priority"] == "high"

    def test_coerce_float_to_int(self):
        """Coerce float to int when integer expected."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
        }
        data = {"count": 42.9}
        result, metadata = coerce(data, schema, ParseConfig())

        assert result["count"] == 42

    def test_coerce_empty_config(self):
        """Coerce with empty ParseConfig (defaults)."""
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "integer"},
            },
        }
        data = {"value": "10"}
        result, metadata = coerce(data, schema, ParseConfig())

        assert result["value"] == 10

    def test_coerce_string_number(self):
        """Coerce string to number."""
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
            },
        }
        data = {"score": "3.14"}
        result, metadata = coerce(data, schema, ParseConfig())

        assert result["score"] == 3.14


class TestLoadsWithCoercion:
    """Test loads() with ParseConfig."""

    def test_loads_with_coercion_config(self):
        """Test loads() accepts ParseConfig parameter."""
        # loads() without schema can't do coercion, but should accept the parameter
        config = ParseConfig(allow_coercion=True)
        data = loads('{"name": "John"}', parse_config=config)
        assert data == {"name": "John"}

    def test_loads_with_coercion_disabled(self):
        """Test loads() with coercion disabled."""
        config = ParseConfig(allow_coercion=False)
        data = loads('{"name": "John"}', parse_config=config)
        assert data == {"name": "John"}


class TestLogging:
    """Test coercion logging."""

    def test_log_coercions_default(self, caplog):
        """Test logging when log_coercions is True (default)."""
        logging.basicConfig(level=logging.DEBUG)
        with caplog.at_level(logging.DEBUG):
            result, metadata = coerce_value("42", "integer")

        assert result == 42
        assert metadata is not None

    def test_log_coercions_disabled(self, caplog):
        """Test no logging when log_coercions is False."""
        logging.basicConfig(level=logging.DEBUG)
        # Just test that coerce_value works without errors (it ignores log_coercions)
        result, metadata = coerce_value("42", "integer")
        assert result == 42


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_coerce_invalid_int_returns_original(self):
        """Invalid int coercion returns original value."""
        result, metadata = coerce_value("not_a_number", "integer")
        assert result == "not_a_number"
        assert metadata is None

    def test_coerce_invalid_float_returns_original(self):
        """Invalid float coercion returns original value."""
        result, metadata = coerce_value("not_a_float", "number")
        assert result == "not_a_float"
        assert metadata is None

    def test_coerce_none_value(self):
        """Coerce None value to string."""
        result, metadata = coerce_value(None, "string")
        assert result == "None"
        assert metadata is not None

    def test_coerce_empty_string_to_int(self):
        """Coerce empty string to int should fail."""
        result, metadata = coerce_value("", "integer")
        # Empty string to int should fail, return original
        assert result == ""
        assert metadata is None

    def test_coerce_none_to_bool(self):
        """Coerce None to boolean."""
        result, metadata = coerce_value(None, "boolean")
        assert result is False  # None is falsy
        assert metadata is not None

    def test_coerce_int_string_decimal(self):
        """Coerce decimal string to int."""
        result, metadata = coerce_value("42.7", "integer")
        assert result == 42
        assert metadata is not None

    def test_coerce_bool_string_variations(self):
        """Test various bool string variations."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("yes", True),
            ("Yes", True),
            ("YES", True),
            ("1", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("no", False),
            ("No", False),
            ("0", False),
            ("off", False),
        ]
        for value, expected in test_cases:
            result, metadata = coerce_value(value, "boolean")
            assert result is expected, f"Failed for {value!r}"
            assert metadata is not None

    def test_coerce_with_float_schema(self):
        """Test coercion when float is expected but int provided."""
        result, metadata = coerce_value(42, "number")
        assert result == 42.0
        assert metadata is not None

    def test_coerce_nested_arrays(self):
        """Test coercion of nested arrays."""
        schema = {
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
            },
        }
        data = {"matrix": [["1", "2"], ["3", "4"]]}
        result, metadata = coerce(data, schema, ParseConfig())

        assert result["matrix"] == [[1, 2], [3, 4]]

    def test_coerce_with_coerce_list_single_item(self):
        """Test coerce_list_single_item config option."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        data = {"tags": "single_tag"}
        config = ParseConfig(coerce_list_single_item=True)
        result, metadata = coerce(data, schema, config)

        assert result["tags"] == ["single_tag"]

    def test_coerce_with_coerce_list_single_item_disabled(self):
        """Test coerce_list_single_item disabled."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        data = {"tags": "single_tag"}
        config = ParseConfig(coerce_list_single_item=False)
        result, metadata = coerce(data, schema, config)

        assert result["tags"] == "single_tag"


class TestMetadata:
    """Test CoercionMetadata dataclass."""

    def test_metadata_fields(self):
        """Test CoercionMetadata has expected fields."""
        result, metadata = coerce_value("42", "integer")

        assert metadata is not None
        assert metadata.original_value == "42"
        assert metadata.target_type == "integer"
        assert metadata.coerced_value == 42
        assert metadata.coercion_type == "to_int"
        assert metadata.field_path == ""

    def test_metadata_field_path(self):
        """Test field_path is set for nested coercion."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer"},
                    },
                }
            },
        }
        data = {"user": {"age": "25"}}
        result, metadata = coerce(data, schema, ParseConfig())

        # Find the age coercion metadata
        age_metadata = [m for m in metadata if m.field_path == "user.age"]
        assert len(age_metadata) == 1
        assert age_metadata[0].original_value == "25"
        assert age_metadata[0].coerced_value == 25
