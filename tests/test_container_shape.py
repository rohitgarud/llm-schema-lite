"""Unit tests for the shared container classifier and array-constraint token source.

These cover the pure, non-emitting primitives added to
``llm_schema_lite.formatters.base`` (ticket lsl-2026-09-04-015, phases 1-3):

* ``classify_container`` / ``ContainerShape`` -- the ten ordered classification rules.
* ``array_constraint_tokens`` / ``format_array_constraints`` -- the single source of
  array constraint vocabulary.
* ``key_token`` / ``render_mapping`` / ``render_tuple`` -- the base renderer defaults.
"""

from typing import Any

import pytest
from pydantic import BaseModel

from llm_schema_lite.formatters.base import BaseFormatter, classify_container
from llm_schema_lite.formatters.config import FormatterConfig
from llm_schema_lite.formatters.jsonish_formatter import JSONishFormatter
from llm_schema_lite.formatters.typescript_formatter import TypeScriptFormatter
from llm_schema_lite.formatters.yaml_formatter import YAMLFormatter
from tests.conftest import Color

EMPTY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _host(config: FormatterConfig | None = None) -> JSONishFormatter:
    """A concrete formatter instance used purely as a host for base-class methods."""
    return JSONishFormatter(EMPTY_OBJECT_SCHEMA, config=config)


class EnumKeyWrappers(BaseModel):
    """Enum-keyed mappings behind an ``Optional[...]`` and a ``list[...]`` wrapper.

    conftest's ``Root.by_color`` covers the bare ``dict[Color, int]``; no existing fixture
    wraps one, and the wrapper paths (``anyOf`` branch, array ``items``) are the ones that
    could bypass ``key_token``. Defined here because this is the only module that needs it.
    """

    opt_map: dict[Color, int] | None = None
    map_list: list[dict[Color, int]]


# ---------------------------------------------------------------------------
# classify_container -- rule 1: nothing / non-dict -> "any"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schema", [None, "x", [], {}, 42, True])
def test_rule1_empty_or_non_dict_is_any(schema: Any) -> None:
    assert classify_container(schema).kind == "any"


# ---------------------------------------------------------------------------
# classify_container -- rule 2: composition / reference keywords -> "scalar"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "#/$defs/X"},
        {"enum": [1]},
        {"const": 1},
        {"anyOf": []},
        {"oneOf": []},
        {"allOf": []},
        {"not": {}},
    ],
)
def test_rule2_composition_keywords_are_scalar(schema: dict[str, Any]) -> None:
    assert classify_container(schema).kind == "scalar"


# ---------------------------------------------------------------------------
# classify_container -- rules 3/3b/4: tuples
# ---------------------------------------------------------------------------


def test_rule3_prefix_items_is_tuple() -> None:
    shape = classify_container(
        {
            "type": "array",
            "prefixItems": [{"type": "integer"}, {"type": "string"}],
            "minItems": 2,
            "maxItems": 2,
        }
    )
    assert shape.kind == "tuple"
    assert shape.prefix_schemas == ({"type": "integer"}, {"type": "string"})
    assert shape.rest_schema is None


def test_rule3b_prefix_items_beats_items() -> None:
    shape = classify_container(
        {
            "type": "array",
            "prefixItems": [{"type": "string"}],
            "items": {"type": "integer"},
        }
    )
    assert shape.kind == "tuple"
    assert shape.prefix_schemas == ({"type": "string"},)
    assert shape.rest_schema == {"type": "integer"}


def test_rule4_draft7_list_items_is_tuple() -> None:
    shape = classify_container(
        {
            "type": "array",
            "items": [{"type": "string"}, {"type": "integer"}],
            "additionalItems": False,
        }
    )
    assert shape.kind == "tuple"
    assert shape.prefix_schemas == ({"type": "string"}, {"type": "integer"})
    assert shape.rest_schema is None


def test_rule4_draft7_additional_items_dict_is_rest() -> None:
    shape = classify_container(
        {
            "type": "array",
            "items": [{"type": "string"}],
            "additionalItems": {"type": "integer"},
        }
    )
    assert shape.kind == "tuple"
    assert shape.rest_schema == {"type": "integer"}


# ---------------------------------------------------------------------------
# classify_container -- rules 5/5b: lists
# ---------------------------------------------------------------------------


def test_rule5_array_with_dict_items_is_list() -> None:
    shape = classify_container({"type": "array", "items": {"type": "string"}})
    assert shape.kind == "list"
    assert shape.item_schema == {"type": "string"}


@pytest.mark.parametrize("schema", [{"type": "array", "items": {}}, {"type": "array"}])
def test_rule5b_array_without_usable_items_is_list(schema: dict[str, Any]) -> None:
    shape = classify_container(schema)
    assert shape.kind == "list"
    assert shape.item_schema is None


# ---------------------------------------------------------------------------
# classify_container -- rules 6/6b/6c/6d/7: mappings and the C1 guard
# ---------------------------------------------------------------------------


def test_rule6_open_object_is_mapping() -> None:
    shape = classify_container({"type": "object", "additionalProperties": {"type": "integer"}})
    assert shape.kind == "mapping"
    assert shape.value_schema == {"type": "integer"}
    assert shape.key_schema is None


def test_rule6b_property_names_becomes_key_schema() -> None:
    shape = classify_container(
        {
            "type": "object",
            "additionalProperties": {"type": "integer"},
            "propertyNames": {"$ref": "#/$defs/Color"},
        }
    )
    assert shape.kind == "mapping"
    assert shape.key_schema == {"$ref": "#/$defs/Color"}


def test_rule6c_c1_guard_properties_wins_over_mapping() -> None:
    """Decision C1: a schema with declared ``properties`` is an object, never a mapping."""
    shape = classify_container(
        {
            "type": "object",
            "properties": {"config": {"type": "object"}},
            "additionalProperties": {"type": "string"},
        }
    )
    assert shape.kind == "object"


def test_rule6d_pattern_properties_is_object() -> None:
    shape = classify_container(
        {
            "type": "object",
            "patternProperties": {"^a": {}},
            "additionalProperties": {"type": "string"},
        }
    )
    assert shape.kind == "object"


def test_rule7_additional_properties_true_is_open_mapping() -> None:
    shape = classify_container({"type": "object", "additionalProperties": True})
    assert shape.kind == "mapping"
    assert shape.value_schema is None


# ---------------------------------------------------------------------------
# classify_container -- rules 8/9/10
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "additionalProperties": False},
        {"type": "object", "properties": {"a": {"type": "string"}}},
    ],
)
def test_rule8_closed_object_is_object(schema: dict[str, Any]) -> None:
    assert classify_container(schema).kind == "object"


@pytest.mark.parametrize("schema", [{"type": "string"}, {"type": "integer", "minimum": 0}])
def test_rule9_typed_leaf_is_scalar(schema: dict[str, Any]) -> None:
    assert classify_container(schema).kind == "scalar"


@pytest.mark.parametrize(
    "schema",
    [{"description": "free form"}, {"title": "Custom Title"}, {"pattern": "^a"}],
)
def test_rule10_typeless_non_empty_is_any(schema: dict[str, Any]) -> None:
    """The ``Any``-with-description fix: a typeless schema is ``any``, not ``string``."""
    assert classify_container(schema).kind == "any"


def test_classify_container_never_raises() -> None:
    bad_inputs: list[Any] = [
        None,
        0,
        "",
        b"",
        [1, 2],
        {"items": None},
        {"additionalProperties": None},
        {"type": "array", "prefixItems": "nope"},
    ]
    valid_kinds = {"mapping", "tuple", "list", "any", "object", "scalar"}
    for bad in bad_inputs:
        assert classify_container(bad).kind in valid_kinds


# ---------------------------------------------------------------------------
# array_constraint_tokens / format_array_constraints
# ---------------------------------------------------------------------------


ARRAY_CONSTRAINT_CASES: list[tuple[dict[str, Any], list[str], str]] = [
    ({"type": "array", "uniqueItems": True}, ["unique"], " (unique)"),
    ({"type": "array", "_uniqueItems": True}, ["unique"], " (unique)"),
    ({"type": "array", "minItems": 2, "maxItems": 2}, ["2-2 items"], " (2-2 items)"),
    ({"type": "array", "minItems": 2}, [">= 2 items"], " (>= 2 items)"),
    ({"type": "array", "maxItems": 5}, ["<= 5 items"], " (<= 5 items)"),
    (
        {"type": "array", "uniqueItems": True, "minItems": 2, "maxItems": 2},
        ["unique", "2-2 items"],
        " (unique, 2-2 items)",
    ),
    ({"type": "array"}, [], ""),
]


@pytest.mark.parametrize(("schema", "tokens", "rendered"), ARRAY_CONSTRAINT_CASES)
def test_array_constraint_tokens(schema: dict[str, Any], tokens: list[str], rendered: str) -> None:
    formatter = _host()
    assert formatter.array_constraint_tokens(schema) == tokens
    assert formatter.format_array_constraints(schema) == rendered


@pytest.mark.parametrize("schema", [case[0] for case in ARRAY_CONSTRAINT_CASES])
def test_array_constraint_tokens_suppressed_without_metadata(schema: dict[str, Any]) -> None:
    formatter = _host(FormatterConfig(include_metadata=False))
    assert formatter.array_constraint_tokens(schema) == []
    assert formatter.format_array_constraints(schema) == ""


def test_array_constraint_tokens_respects_metadata_inclusion() -> None:
    formatter = _host(FormatterConfig(metadata_inclusion={"uniqueItems": False}))
    schema = {"type": "array", "uniqueItems": True}
    assert formatter.array_constraint_tokens(schema) == []
    assert formatter.format_array_constraints(schema) == ""


# ---------------------------------------------------------------------------
# key_token / render_mapping / render_tuple -- base defaults
# ---------------------------------------------------------------------------


def test_render_tuple_base_default() -> None:
    formatter = _host()
    shape = classify_container(
        {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}]}
    )
    assert formatter.render_tuple(shape) == "[int, string]"


def test_render_tuple_with_rest_schema() -> None:
    formatter = _host()
    shape = classify_container(
        {
            "type": "array",
            "prefixItems": [{"type": "integer"}, {"type": "string"}],
            "items": {"type": "string"},
        }
    )
    assert formatter.render_tuple(shape) == "[int, string, ...string]"


def test_render_mapping_base_default() -> None:
    formatter = _host()
    shape = classify_container({"type": "object", "additionalProperties": {"type": "integer"}})
    assert formatter.render_mapping(shape) == "{ <string>: int }"


def test_render_mapping_open_value_is_any() -> None:
    formatter = _host()
    shape = classify_container({"type": "object", "additionalProperties": True})
    assert formatter.render_mapping(shape) == "{ <string>: any }"


def test_key_token_defaults_to_string() -> None:
    formatter = _host()
    shape = classify_container({"type": "object", "additionalProperties": {"type": "integer"}})
    assert formatter.key_token(shape) == "string"


@pytest.mark.parametrize(
    ("key_schema", "expected"),
    [
        ({"type": "string"}, "string"),
        ({"pattern": "^a"}, "string"),
        ({"enum": ["x", "y"]}, "x OR y"),
    ],
)
def test_key_token_direct_schema(key_schema: dict[str, Any], expected: str) -> None:
    formatter = _host()
    shape = classify_container(
        {
            "type": "object",
            "additionalProperties": {"type": "integer"},
            "propertyNames": key_schema,
        }
    )
    assert formatter.key_token(shape) == expected


def test_key_token_resolves_ref_through_defs() -> None:
    formatter = JSONishFormatter(
        {
            "type": "object",
            "properties": {},
            "$defs": {"Color": {"enum": ["red"], "type": "string"}},
        }
    )
    shape = classify_container(
        {
            "type": "object",
            "additionalProperties": {"type": "integer"},
            "propertyNames": {"$ref": "#/$defs/Color"},
        }
    )
    assert formatter.key_token(shape) == "red"


@pytest.mark.parametrize(
    ("defs", "ref_name", "expected"),
    [
        # dict[IntEnum, V]: non-string values route through ``format_literal_value``.
        ({"Num": {"enum": [1, 2], "type": "integer"}}, "Num", "1 OR 2"),
        # A def carrying ``enum`` but no ``type`` still yields the union, not "string".
        ({"Bare": {"enum": ["x"]}}, "Bare", "x"),
        # An unresolvable $ref returns "string" WITHOUT ever reading the wrapper's own type.
        ({}, "Nope", "string"),
    ],
)
def test_key_token_ref_enum_variants(defs: dict[str, Any], ref_name: str, expected: str) -> None:
    formatter = JSONishFormatter({"type": "object", "properties": {}, "$defs": defs})
    shape = classify_container(
        {
            "type": "object",
            "additionalProperties": {"type": "integer"},
            "propertyNames": {"$ref": f"#/$defs/{ref_name}"},
        }
    )
    assert formatter.key_token(shape) == expected


@pytest.mark.parametrize(
    "config",
    [
        None,
        FormatterConfig(include_metadata=False),
        FormatterConfig(include_constraints=False),
        FormatterConfig(metadata_inclusion={"propertyNames": False}),
    ],
    ids=["default", "no-metadata", "no-constraints", "propertyNames-off"],
)
def test_key_token_enum_ignores_metadata_gates(config: FormatterConfig | None) -> None:
    """AC-1b: the key's value set is STRUCTURAL -- no metadata gate can suppress it."""
    formatter = JSONishFormatter(
        {
            "type": "object",
            "properties": {},
            "$defs": {"Color": {"enum": ["red", "green"], "type": "string"}},
        },
        config=config,
    )
    shape = classify_container(
        {
            "type": "object",
            "additionalProperties": {"type": "integer"},
            "propertyNames": {"$ref": "#/$defs/Color"},
        }
    )
    assert formatter.key_token(shape) == "red OR green"


@pytest.mark.parametrize(
    ("formatter_cls", "union", "null_marker"),
    [
        (JSONishFormatter, "red OR green", "OR null"),
        (YAMLFormatter, "red OR green", "OR null"),
        (TypeScriptFormatter, "red | green", "| null"),
    ],
)
def test_enum_key_survives_container_wrappers(
    formatter_cls: type[BaseFormatter], union: str, null_marker: str
) -> None:
    """The enum key set survives an ``Optional[...]`` wrapper and a ``list[...]`` wrapper."""
    result = formatter_cls(EnumKeyWrappers.model_json_schema()).transform_schema()

    # Once for ``opt_map``, once for ``map_list``.
    assert result.count(union) == 2, result
    assert "string" not in result.replace("map_list", ""), result
    assert null_marker in result, result


# ===========================================================================
# PHASE 2 -- base.py rewiring behaviour
# ===========================================================================


def test_typeless_property_renders_any_not_string() -> None:
    """AC-2: a typeless, non-empty property is ``any``, never ``string``."""
    result = YAMLFormatter(
        {
            "type": "object",
            "properties": {"described": {"description": "free form"}},
            "required": ["described"],
        }
    ).transform_schema()
    described_lines = [line for line in result.splitlines() if line.strip().startswith("described")]
    assert described_lines, result
    assert "string" not in described_lines[0], described_lines[0]
    assert "any" in described_lines[0], described_lines[0]


def test_base_array_constraints_use_single_token_source() -> None:
    token = _host().process_type_value(
        {"type": "array", "uniqueItems": True, "items": {"type": "string"}}
    )
    assert token.endswith(" (unique)"), token
    assert "//unique items" not in token
    assert "length: " not in token


def test_base_array_range_is_ascii() -> None:
    token = _host().process_type_value(
        {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "string"}}
    )
    assert token.endswith(" (2-4 items)"), token
    assert "≥" not in token
    assert "≤" not in token


def test_base_process_type_value_tuple_early_return() -> None:
    token = _host().process_type_value(
        {"type": "array", "items": [{"type": "string"}, {"type": "integer"}]}
    )
    assert token == "[string, int]", token


def test_mapping_property_renders_structurally() -> None:
    prop = _host().process_property({"type": "object", "additionalProperties": {"type": "integer"}})
    assert prop == "{ <string>: int }", prop


def test_property_names_metadata_suppressed_for_mapping() -> None:
    parts = _host().format_metadata_parts(
        {
            "type": "object",
            "additionalProperties": {"type": "integer"},
            "propertyNames": {"$ref": "#/$defs/Color"},
        }
    )
    assert not any("propertyNames" in part for part in parts), parts


def test_property_names_never_leaks_on_object_node() -> None:
    """The old ``kind == "mapping"`` guard let an OBJECT-kind node leak a raw dict repr."""
    parts = _host().format_metadata_parts(
        {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "propertyNames": {"pattern": "^[a-z]+$"},
        }
    )
    assert not any("propertyNames" in part for part in parts), parts


# ===========================================================================
# PHASE 3 -- process_additional_properties pure-mapping split
# ===========================================================================


PURE_MAPPING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": {"type": "integer"},
}
CLOSED_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": False}
OBJECT_WITH_PROPS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"config": {"type": "object"}},
    "additionalProperties": {"type": "string"},
}


@pytest.mark.parametrize(
    ("formatter_cls", "expected_false", "expected_with_props"),
    [
        (JSONishFormatter, " //no additional properties", " //additional: string"),
        (TypeScriptFormatter, " // no additional properties", " // additional: string"),
        (YAMLFormatter, " # no additional properties", " # additional: string"),
    ],
)
def test_additional_properties_split(
    formatter_cls: type[BaseFormatter],
    expected_false: str,
    expected_with_props: str,
) -> None:
    formatter = formatter_cls(EMPTY_OBJECT_SCHEMA)
    # A pure mapping is rendered structurally by the caller, never as a trailing comment.
    assert formatter.process_additional_properties(PURE_MAPPING_SCHEMA) == ""
    # The `false` path is byte-identical to HEAD.
    assert formatter.process_additional_properties(CLOSED_OBJECT_SCHEMA) == expected_false
    # The object-with-properties path (decision C1) is byte-identical to HEAD.
    assert formatter.process_additional_properties(OBJECT_WITH_PROPS_SCHEMA) == expected_with_props


@pytest.mark.parametrize("formatter_cls", [JSONishFormatter, TypeScriptFormatter, YAMLFormatter])
def test_additional_properties_show_structure_false_still_guarded(
    formatter_cls: type[BaseFormatter],
) -> None:
    """A pure mapping returns "" regardless of ``show_structure`` -- the guard runs first."""
    formatter = formatter_cls(EMPTY_OBJECT_SCHEMA)
    assert formatter.process_additional_properties(PURE_MAPPING_SCHEMA, show_structure=False) == ""
