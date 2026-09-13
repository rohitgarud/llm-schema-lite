"""Tests for type coercion functionality."""

import logging

import pytest

from llm_schema_lite import ConversionError, ParseConfig, coerce, coerce_value
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


class TestNestedModelRefs:
    """A nested BaseModel reaches coercion as {"$ref": "#/$defs/Name"}.

    Until the ref was resolved that node carried no "type", so every field inside it
    fell through to the untyped-object branch and was coerced against a hardcoded
    {"type": "string"} -- including values that already had the declared type.
    """

    @staticmethod
    def _user_model():
        from pydantic import BaseModel

        class Address(BaseModel):
            street: str
            zipcode: int
            verified: bool

        class User(BaseModel):
            name: str
            address: Address

        return User

    def test_correct_nested_types_are_preserved(self):
        """Values already of the declared type are left alone, not stringified."""
        data = {
            "name": "Ada",
            "address": {"street": "1 Main St", "zipcode": 12345, "verified": True},
        }
        result, _ = coerce(data, self._user_model(), ParseConfig())

        assert result["address"]["zipcode"] == 12345
        assert result["address"]["verified"] is True

    def test_nested_values_coerce_to_their_declared_types(self):
        """A nested int-as-string becomes an int once the $ref resolves."""
        data = {
            "name": "Ada",
            "address": {"street": "1 Main St", "zipcode": "12345", "verified": "true"},
        }
        result, _ = coerce(data, self._user_model(), ParseConfig())

        assert result["address"]["zipcode"] == 12345
        assert result["address"]["verified"] is True

    def test_nested_coercion_reports_a_dotted_field_path(self):
        """Metadata for a coercion inside a nested model carries the full path."""
        data = {
            "name": "Ada",
            "address": {"street": "1 Main St", "zipcode": "12345", "verified": True},
        }
        _, metadata = coerce(data, self._user_model(), ParseConfig())

        assert "address.zipcode" in {m.field_path for m in metadata}


class TestNoneIsNotCoerced:
    """null is a value, not a type mismatch to repair.

    str(None) yields the string "None", which validates against {"type": "string"};
    coercing it therefore hid a missing value from the validator completely.
    """

    def test_none_is_left_alone_for_a_string_field(self):
        """A null stays null instead of becoming the string "None"."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result, metadata = coerce({"name": None}, schema, ParseConfig())

        assert result["name"] is None
        assert metadata == []

    def test_none_is_left_alone_inside_a_nested_model(self):
        """The guard applies at every level, not just the top one."""
        from pydantic import BaseModel

        class Address(BaseModel):
            street: str

        class User(BaseModel):
            address: Address

        result, _ = coerce({"address": {"street": None}}, User, ParseConfig())

        assert result["address"]["street"] is None

    def test_coerce_value_keeps_its_own_contract(self):
        """The guard lives in coerce_recursive; the primitive is deliberately unchanged."""
        result, metadata = coerce_value(None, "string")

        assert result == "None"
        assert metadata is not None


class TestOptionalFieldsAreNotStringified:
    """`X | None` is an anyOf node carrying no "type" of its own.

    Without the unwrap it fell through to the scalar branch and was coerced against
    the default {"type": "string"}, so a valid `int | None` of 7 became "7" -- which
    then failed validation and, in partial mode, nulled the field and recorded the
    reply as at fault.
    """

    def test_a_valid_optional_int_survives_coercion(self):
        """The regression: a correctly-typed optional value is left exactly alone."""
        from pydantic import BaseModel

        class Rec(BaseModel):
            name: str
            count: int | None = None

        result, metadata = coerce({"name": "Ada", "count": 7}, Rec, ParseConfig())

        assert result["count"] == 7
        assert metadata == []

    def test_an_optional_int_is_still_coerced_from_a_string(self):
        """Unwrapping must not cost the coercion the union was hiding."""
        from pydantic import BaseModel

        class Rec(BaseModel):
            name: str
            count: int | None = None

        result, _ = coerce({"name": "Ada", "count": "7"}, Rec, ParseConfig())

        assert result["count"] == 7

    def test_an_optional_nested_model_resolves_through_the_union(self):
        """anyOf wrapping a $ref: both hops have to happen, in that order."""
        from pydantic import BaseModel

        class Address(BaseModel):
            zipcode: int

        class User(BaseModel):
            address: Address | None = None

        result, _ = coerce({"address": {"zipcode": 12345}}, User, ParseConfig())

        assert result["address"]["zipcode"] == 12345

    def test_a_genuine_multi_type_union_is_left_alone(self):
        """Two non-null members give no single target; guessing one would corrupt."""
        from pydantic import BaseModel

        class Rec(BaseModel):
            value: str | int

        result, metadata = coerce({"value": 7}, Rec, ParseConfig())

        assert result["value"] == 7
        assert metadata == []


class TestTypelessNodesAreNotStringified:
    """A schema node with no declared type is left alone, not stringified.

    The untyped-object branch hard-coded {"type": "string"} for every dict value
    instead of consulting additionalProperties, and the scalar branch defaulted to
    "string" for any node with no "type" at all. Both silently rewrote a value that
    was already correct, which then failed validation downstream.
    """

    def test_dict_values_coerce_to_additional_properties_type(self):
        """additionalProperties declares the value type; it is consulted, not ignored."""
        schema = {"type": "object", "additionalProperties": {"type": "integer"}}
        result, metadata = coerce({"a": "1"}, schema, ParseConfig())

        assert result == {"a": 1}
        assert [m.coercion_type for m in metadata] == ["to_int"]
        assert metadata[0].field_path == "a"

    def test_correctly_typed_dict_values_are_untouched(self):
        """The core symptom: a value that was already correct must not be rewritten."""
        schema = {"type": "object", "additionalProperties": {"type": "integer"}}
        result, metadata = coerce({"a": 1, "b": 2}, schema, ParseConfig())

        assert result == {"a": 1, "b": 2}
        assert metadata == []

    def test_declared_string_values_still_coerce(self):
        """The fix must not over-decline: a genuinely declared string type still coerces."""
        schema = {"type": "object", "additionalProperties": {"type": "string"}}
        result, metadata = coerce({"a": 1}, schema, ParseConfig())

        assert result == {"a": "1"}
        assert [m.coercion_type for m in metadata] == ["to_string"]

    def test_a_typeless_scalar_node_is_left_alone(self):
        """A node with no "type" key at all is not assumed to be a string."""
        schema = {"type": "object", "properties": {"value": {}}}
        result, metadata = coerce({"value": 5}, schema, ParseConfig())

        assert result["value"] == 5
        assert metadata == []

    def test_a_typeless_item_schema_leaves_list_items_alone(self):
        """The bare-list symptom, reached through the array branch."""
        schema = {"type": "object", "properties": {"items": {"type": "array", "items": {}}}}
        result, metadata = coerce({"items": [1, 2]}, schema, ParseConfig())

        assert result["items"] == [1, 2]
        assert metadata == []

    def test_a_typed_item_schema_still_coerces(self):
        """The scalar default must not reach into a declared item type."""
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "integer"}}},
        }
        result, metadata = coerce({"items": ["1", "2"]}, schema, ParseConfig())

        assert result["items"] == [1, 2]
        assert [m.coercion_type for m in metadata] == ["to_int", "to_int"]

    def test_a_mixed_type_literal_still_coerces_through_its_enum(self):
        """A mixed Literal emits a typeless enum node; its coercion must survive."""
        from typing import Literal

        from pydantic import BaseModel

        class Lit(BaseModel):
            k: Literal[1, "a"]

        result, metadata = coerce({"k": "1"}, Lit, ParseConfig())

        assert result["k"] == 1
        assert [m.coercion_type for m in metadata] == ["enum_from_string"]

    def test_a_mixed_type_literal_matches_case_insensitively(self):
        """The second enum coercion mode on the same typeless node."""
        from typing import Literal

        from pydantic import BaseModel

        class Lit(BaseModel):
            k: Literal[1, "a"]

        result, metadata = coerce({"k": "A"}, Lit, ParseConfig())

        assert result["k"] == "a"
        assert [m.coercion_type for m in metadata] == ["enum_case_insensitive"]

    def test_an_optional_mixed_type_literal_still_coerces(self):
        """The second entry path: the anyOf unwrap lands on the same typeless enum node."""
        from typing import Literal

        from pydantic import BaseModel

        class LitOpt(BaseModel):
            k: Literal[1, "a"] | None = None

        result, _ = coerce({"k": "1"}, LitOpt, ParseConfig())

        assert result["k"] == 1

    def test_additional_properties_true_leaves_values_alone(self):
        """pydantic 2.12's dict[str, Any] shape: additionalProperties is the boolean True."""
        schema = {"type": "object", "additionalProperties": True}
        result, metadata = coerce({"a": 5}, schema, ParseConfig())

        assert result == {"a": 5}
        assert metadata == []

    def test_object_node_with_no_value_schema_leaves_values_alone(self):
        """pydantic 2.10's shape for dict[str, Any] / bare dict: neither key present."""
        schema = {"type": "object"}
        result, metadata = coerce({"a": 5}, schema, ParseConfig())

        assert result == {"a": 5}
        assert metadata == []

    def test_additional_properties_false_leaves_values_alone(self):
        """Closed-world additionalProperties constrains keys, not value types."""
        schema = {"type": "object", "additionalProperties": False}
        result, metadata = coerce({"a": 5}, schema, ParseConfig())

        assert result == {"a": 5}
        assert metadata == []


class TestNullIsNotCoercedToBoolean:
    """The recursive walker and the primitive deliberately disagree on None.

    coerce_recursive's None guard returns None untouched before the scalar branch is
    ever reached, so a boolean field receiving null stays null. The primitive
    coerce_value(None, "boolean") still returns False -- a separate, already-pinned
    contract (test_coerce_none_to_bool) that this class does not touch, only restates
    alongside the walker's contract so both layers' intentional disagreement is
    documented in one place.
    """

    def test_null_for_a_boolean_field_is_left_alone(self):
        """The None guard sits above the scalar branch; it must stay there."""
        schema = {"type": "object", "properties": {"flag": {"type": "boolean"}}}
        result, metadata = coerce({"flag": None}, schema, ParseConfig())

        assert result["flag"] is None
        assert metadata == []

    def test_coerce_value_still_converts_null_to_false(self):
        """The primitive keeps its opposite contract; the existing test is not modified."""
        result, metadata = coerce_value(None, "boolean")

        assert result is False
        assert metadata is not None
        assert metadata.coercion_type == "none_to_bool"

    def test_null_for_a_boolean_field_is_rejected_by_both_routes(self):
        """End to end: both parse routes reject a null for a required boolean."""
        from pydantic import BaseModel

        class Boolish(BaseModel):
            flag: bool

        with pytest.raises(ConversionError) as strict_exc:
            loads('{"flag": null}', schema=Boolish)

        with pytest.raises(ConversionError) as partial_exc:
            loads('{"flag": null}', schema=Boolish, parse_config=ParseConfig(partial=True))

        assert "is not of type 'boolean'" in str(strict_exc.value)
        assert "is not of type 'boolean'" in str(partial_exc.value)
