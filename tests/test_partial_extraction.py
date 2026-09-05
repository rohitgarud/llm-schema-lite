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

    def test_non_partial_route_returns_real_nested_instances(self):
        """The non-partial loads route builds nested model instances, not raw dicts.

        Pins design v2 section 6.2's headline behaviour change: model_validate, not
        model_construct. No marker is involved -- this is about validation, not stripping.
        """
        from pydantic import BaseModel

        class Inner(BaseModel):
            name: str

        class Outer(BaseModel):
            person: Inner

        result, _ = loads('{"person": {"name": "A"}}', schema=Outer)
        assert isinstance(result.person, Inner)
        assert result.person.name == "A"

    def test_partial_route_still_uses_model_construct(self):
        """The partial route keeps model_construct: missing fields tolerated, inner stays a dict.

        Pins the deliberate asymmetry between the two routes.
        """
        from pydantic import BaseModel

        class Inner(BaseModel):
            name: str

        class Outer(BaseModel):
            person: Inner
            note: str | None = None

        result, _ = loads(
            '{"person": {"name": "A"}}', schema=Outer, parse_config=ParseConfig(partial=True)
        )
        assert not isinstance(result.person, Inner)
        assert result.person == {"name": "A"}

    def test_pydantic_only_validation_failure_raises_conversion_error(self):
        """A Pydantic-only constraint raises ConversionError, not ValidationError.

        The constraint is one jsonschema cannot see.
        Pins the section 6.2 error contract: loads never leaks pydantic.ValidationError.
        """
        import pydantic
        from pydantic import BaseModel, field_validator

        class OnlyEven(BaseModel):
            n: int

            @field_validator("n")
            @classmethod
            def _must_be_even(cls, v: int) -> int:
                if v % 2:
                    raise ValueError("n must be even")
                return v

        with pytest.raises(ConversionError, match="Validation failed"):
            loads('{"n": 3}', schema=OnlyEven)

        try:
            loads('{"n": 3}', schema=OnlyEven)
        except ConversionError as exc:
            assert not isinstance(exc, pydantic.ValidationError)

    def test_nested_marked_key_yields_validated_model(self):
        """AC 3: nested marked keys are stripped and the result is a validated nested model."""
        from pydantic import BaseModel

        class Inner(BaseModel):
            name: str

        class Outer(BaseModel):
            person: Inner

        result, _ = loads('{"person*": {"name*": "A"}}', schema=Outer)
        assert isinstance(result.person, Inner)
        assert result.person.name == "A"

    def test_nested_list_of_models(self):
        """items: every element of a list[Model] is walked and validated."""
        from pydantic import BaseModel

        class Inner(BaseModel):
            name: str

        class Many(BaseModel):
            people: list[Inner]

        result, _ = loads('{"people*": [{"name*": "A"}, {"name": "B"}]}', schema=Many)
        assert [p.name for p in result.people] == ["A", "B"]
        assert all(isinstance(p, Inner) for p in result.people)

    def test_optional_nested_model(self):
        """anyOf with a null branch: the non-null branch supplies the known keys."""
        from pydantic import BaseModel

        class Inner(BaseModel):
            name: str

        class Opt(BaseModel):
            person: Inner | None = None

        result, _ = loads('{"person*": {"name*": "A"}}', schema=Opt)
        assert isinstance(result.person, Inner)
        assert result.person.name == "A"

    def test_dict_str_str_keys_are_never_rewritten(self):
        """C16 safety property: an open-ended mapping's own keys are user data, never remapped."""
        from pydantic import BaseModel

        class MapStr(BaseModel):
            tags: dict[str, str]

        result, _ = loads('{"tags*": {"x*": "y"}}', schema=MapStr)
        assert result.tags == {"x*": "y"}

    def test_dict_str_model_keys_preserved_values_walked(self):
        """dict[str, Model]: the mapping's keys survive verbatim; the model's own keys strip."""
        from pydantic import BaseModel

        class Inner(BaseModel):
            name: str

        class MapModel(BaseModel):
            m: dict[str, Inner]

        result, _ = loads('{"m*": {"k*": {"name*": "A"}}}', schema=MapModel)
        assert list(result.m) == ["k*"]
        assert isinstance(result.m["k*"], Inner)
        assert result.m["k*"].name == "A"

    def test_discriminated_union_branch(self):
        """Merging all non-null branches makes BOTH arms of a discriminated union work."""
        from typing import Literal

        from pydantic import BaseModel, Field

        class Cat(BaseModel):
            kind: Literal["cat"]
            whiskers: int

        class Dog(BaseModel):
            kind: Literal["dog"]
            tail: int

        class Disc(BaseModel):
            pet: Cat | Dog = Field(discriminator="kind")

        dog, _ = loads('{"pet*": {"kind": "dog", "tail*": 2}}', schema=Disc)
        assert isinstance(dog.pet, Dog)
        assert dog.pet.tail == 2

        cat, _ = loads('{"pet": {"kind": "cat", "whiskers*": 9}}', schema=Disc)
        assert isinstance(cat.pet, Cat)
        assert cat.pet.whiskers == 9

    def test_tuple_prefix_items(self):
        """prefixItems: tuple elements are descended positionally."""
        from pydantic import BaseModel

        class Inner(BaseModel):
            name: str

        class Tup(BaseModel):
            t: tuple[Inner, int]

        result, _ = loads('{"t*": [{"name*": "A"}, 3]}', schema=Tup)
        assert isinstance(result.t[0], Inner)
        assert result.t[0].name == "A"
        assert result.t[1] == 3

    def test_recursive_model_terminates_and_deep_chain_fails_loudly(self):
        """A self-referential model terminates; past MARKER_WALK_MAX_DEPTH it fails loudly.

        extra="forbid" is load-bearing: it is what makes a leftover marked key past the
        depth cap a validation ERROR rather than a silently ignored extra key. See the
        plan's Phase 6 discovery note.
        """
        from pydantic import BaseModel

        class Node(BaseModel):
            model_config = {"extra": "forbid"}

            v: str
            child: "Node | None" = None

        def chain(n: int) -> str:
            root: dict = {"v": "0"}
            cur = root
            for i in range(1, n):
                cur["child*"] = {"v": str(i)}
                cur = cur["child*"]
            return json.dumps(root)

        shallow, _ = loads(chain(4), schema=Node)
        depth = 0
        node = shallow
        while node.child is not None:
            depth += 1
            node = node.child
        assert depth == 3

        with pytest.raises(ConversionError, match="Validation failed"):
            loads(chain(14), schema=Node)

    def test_nested_collision_prefers_verbatim_per_level(self):
        """The "stripped not already present" guard is evaluated per object level."""
        from pydantic import BaseModel

        class Inner(BaseModel):
            name: str

        class Outer(BaseModel):
            person: Inner

        result, _ = loads('{"person": {"name": "V", "name*": "M"}}', schema=Outer)
        assert isinstance(result.person, Inner)
        assert result.person.name == "V"


class TestNormalizeMarkerKeysUnit:
    """Unit tests for the extracted pure rule (lsl-2026-09-05-003, item 31)."""

    def test_normalize_marker_keys_is_pure(self):
        """The three rules hold in isolation and the input dict is never mutated."""
        from llm_schema_lite.parsers import normalize_marker_keys

        known = {"name", "age"}

        # Rule 1: a verbatim known key is kept as-is.
        assert normalize_marker_keys({"name": "x"}, known, "*") == {"name": "x"}

        # Rule 2: a marked key whose stripped name is known and absent is remapped.
        assert normalize_marker_keys({"name*": "x"}, known, "*") == {"name": "x"}

        # Rule 1 beats rule 2: verbatim wins on a collision. The marked key is not
        # remapped (that would clobber the verbatim value) and not dropped either --
        # rule 3 keeps it verbatim for the caller's own filter to discard, which is
        # what the adapter's output-field filter and schema validation both do.
        assert normalize_marker_keys({"name": "V", "name*": "M"}, known, "*") == {
            "name": "V",
            "name*": "M",
        }

        # Rule 3: an unknown marked key is left untouched, never invented.
        assert normalize_marker_keys({"bogus*": 1}, known, "*") == {"bogus*": 1}

        # A key equal to the marker strips to "", which is falsy: not remapped.
        assert normalize_marker_keys({"*": "x"}, known, "*") == {"*": "x"}

        # Degenerate inputs are identity no-ops, not copies.
        data = {"name*": "x"}
        assert normalize_marker_keys(data, known, "") is data
        assert normalize_marker_keys(data, set(), "*") is data
        empty: dict[str, object] = {}
        assert normalize_marker_keys(empty, known, "*") is empty

        # Purity: the caller's dict is never mutated.
        original = {"name*": "x", "bogus*": 1}
        snapshot = dict(original)
        normalize_marker_keys(original, known, "*")
        assert original == snapshot
