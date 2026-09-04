"""Recursive-model rendering tests (ticket lsl-2026-09-04-014).

This module starts with the ``BaseFormatter`` depth machinery that every
formatter shares: the ``max_recursion_depth`` truncation predicate, the
``_expanding`` bookkeeping context manager, the root-``$ref`` adoption helper
and the per-render state reset. The cross-format golden suite is layered on
top of these tests by the formatter phases of the same ticket.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from llm_schema_lite import FormatterConfig, loads, simplify_schema, validate
from llm_schema_lite.formatters.base import BaseFormatter
from llm_schema_lite.formatters.jsonish_formatter import JSONishFormatter
from llm_schema_lite.formatters.typescript_formatter import TypeScriptFormatter
from llm_schema_lite.formatters.yaml_formatter import YAMLFormatter

# ---------------------------------------------------------------------------
# Schemas used by the base-formatter unit tests
# ---------------------------------------------------------------------------

ROOT_REF_SCHEMA: dict[str, Any] = {
    "$ref": "#/$defs/T",
    "$defs": {
        "T": {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        }
    },
}

INLINE_ROOT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"b": {"type": "string"}},
    "required": ["b"],
}

RECURSIVE_DEFS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"node": {"$ref": "#/$defs/Node"}},
    "$defs": {
        "Node": {
            "type": "object",
            "title": "Node",
            "properties": {
                "name": {"type": "string"},
                "child": {"$ref": "#/$defs/Node"},
            },
            "required": ["name"],
        },
        "Addr": {
            "type": "object",
            "title": "Addr",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

# The guard attributes retired by lsl-2026-09-04-014. Research measured that
# only ``_processed_refs`` ever fired, and it is subsumed by the depth
# predicate, so none of these may come back.
RETIRED_GUARD_ATTRIBUTES: tuple[str, ...] = (
    "_recursion_depth",
    "_max_recursion_depth",
    "_processed_refs",
    "_expansion_count",
    "_max_expansions",
    "_ref_depth_tracker",
    "_max_ref_depth",
    "_expansion_fingerprints",
)


def _formatter(
    formatter_cls: type[BaseFormatter],
    schema: dict[str, Any],
    depth: int = 2,
) -> BaseFormatter:
    """Build a formatter with an explicit ``max_recursion_depth``.

    Args:
        formatter_cls: Formatter class to instantiate.
        schema: JSON schema handed to the formatter.
        depth: Value for ``FormatterConfig.max_recursion_depth``.

    Returns:
        The constructed formatter instance.
    """
    return formatter_cls(schema, FormatterConfig(max_recursion_depth=depth))


# ---------------------------------------------------------------------------
# recursion_placeholder()
# ---------------------------------------------------------------------------


def test_recursion_placeholder_form_a() -> None:
    """Form A uses the formatter's own comment prefix."""
    assert YAMLFormatter({}).recursion_placeholder("Node") == "object  # recursive: Node"
    assert JSONishFormatter({}).recursion_placeholder("Node") == "object  // recursive: Node"


# ---------------------------------------------------------------------------
# _expanding()
# ---------------------------------------------------------------------------


def test_expanding_context_manager_pushes_and_pops() -> None:
    """``_expanding`` pushes the key for the block and pops it afterwards."""
    formatter = YAMLFormatter({})
    assert formatter._ref_expansion_path == []
    with formatter._expanding("Node"):
        assert formatter._ref_expansion_path == ["Node"]
    assert formatter._ref_expansion_path == []

    with formatter._expanding(None):
        assert formatter._ref_expansion_path == []
    assert formatter._ref_expansion_path == []


def test_expanding_pops_on_exception() -> None:
    """The expansion path is restored when the block raises."""
    formatter = YAMLFormatter({})
    with pytest.raises(RuntimeError), formatter._expanding("Node"):
        assert formatter._ref_expansion_path == ["Node"]
        raise RuntimeError("boom")
    assert formatter._ref_expansion_path == []


# ---------------------------------------------------------------------------
# _reset_ref_state()
# ---------------------------------------------------------------------------


def test_reset_ref_state_clears_everything() -> None:
    """Every per-render ``$ref`` counter is zeroed."""
    formatter = YAMLFormatter({})
    formatter._ref_cache["Node"] = "object"
    formatter._ref_expansion_path.append("Node")
    formatter._global_expansion_count = 7
    formatter._truncation_epoch = 3

    formatter._reset_ref_state()

    assert formatter._ref_cache == {}
    assert formatter._ref_expansion_path == []
    assert formatter._global_expansion_count == 0
    assert formatter._truncation_epoch == 0


# ---------------------------------------------------------------------------
# _adopt_root_ref()
# ---------------------------------------------------------------------------


def test_adopt_root_ref_promotes_definition() -> None:
    """A bare root ``$ref`` becomes the effective root object."""
    formatter = YAMLFormatter(ROOT_REF_SCHEMA)
    assert formatter.properties == {}

    assert formatter._adopt_root_ref() == "T"
    assert formatter.properties == {"a": {"type": "string"}}
    assert formatter.required_fields == {"a"}


def test_adopt_root_ref_is_a_noop_for_inline_properties() -> None:
    """A schema that already has ``properties`` is left untouched."""
    formatter = YAMLFormatter(INLINE_ROOT_SCHEMA)

    assert formatter._adopt_root_ref() is None
    assert formatter.properties == {"b": {"type": "string"}}
    assert formatter.required_fields == {"b"}


# ---------------------------------------------------------------------------
# The depth predicate in BaseFormatter.process_ref
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("formatter_cls", [YAMLFormatter, TypeScriptFormatter])
def test_process_ref_truncates_on_same_type_reentry(
    formatter_cls: type[BaseFormatter],
) -> None:
    """Re-entering the same ``$ref`` beyond the budget yields a placeholder."""
    formatter = _formatter(formatter_cls, RECURSIVE_DEFS_SCHEMA, depth=1)
    out = formatter.process_ref({"$ref": "#/$defs/Node"})

    assert "recursive: Node" in out
    assert "name" in out


@pytest.mark.parametrize("formatter_cls", [YAMLFormatter, TypeScriptFormatter])
def test_first_expansion_is_unconditional_at_depth_zero(
    formatter_cls: type[BaseFormatter],
) -> None:
    """Depth 0 still renders the first body — the guard fires only on re-entry."""
    formatter = _formatter(formatter_cls, RECURSIVE_DEFS_SCHEMA, depth=0)
    out = formatter.process_ref({"$ref": "#/$defs/Node"})

    assert "name" in out
    assert "recursive: Node" in out


@pytest.mark.parametrize("formatter_cls", [YAMLFormatter, TypeScriptFormatter])
def test_non_recursive_ref_is_identical_at_every_depth(
    formatter_cls: type[BaseFormatter],
) -> None:
    """The knob cannot degrade a non-recursive ``$ref`` at any value."""
    renderings = {
        depth: _formatter(formatter_cls, RECURSIVE_DEFS_SCHEMA, depth=depth).process_ref(
            {"$ref": "#/$defs/Addr"}
        )
        for depth in (0, 1, 2, 3, 99)
    }

    assert len(set(renderings.values())) == 1
    assert "recursive:" not in renderings[0]


@pytest.mark.parametrize("formatter_cls", [YAMLFormatter, TypeScriptFormatter])
def test_truncated_rendering_is_never_cached(
    formatter_cls: type[BaseFormatter],
) -> None:
    """Taint-and-skip: a truncated body stays out of ``_ref_cache``."""
    formatter = _formatter(formatter_cls, RECURSIVE_DEFS_SCHEMA, depth=1)
    formatter.process_ref({"$ref": "#/$defs/Node"})

    assert "Node" not in formatter._ref_cache


@pytest.mark.parametrize("formatter_cls", [YAMLFormatter, TypeScriptFormatter])
def test_clean_sibling_is_still_cached_after_a_truncation(
    formatter_cls: type[BaseFormatter],
) -> None:
    """The epoch counter must not over-taint refs rendered after a truncation."""
    formatter = _formatter(formatter_cls, RECURSIVE_DEFS_SCHEMA, depth=1)
    formatter.process_ref({"$ref": "#/$defs/Node"})
    formatter.process_ref({"$ref": "#/$defs/Addr"})

    assert "Addr" in formatter._ref_cache


# ---------------------------------------------------------------------------
# Retired guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("formatter_cls", [JSONishFormatter, TypeScriptFormatter, YAMLFormatter])
def test_retired_guard_attributes_are_gone(formatter_cls: type[BaseFormatter]) -> None:
    """The dead ``$ref`` guards are removed, not merely bypassed."""
    formatter = formatter_cls({})
    for attribute in RETIRED_GUARD_ATTRIBUTES:
        assert not hasattr(formatter, attribute), attribute


@pytest.mark.parametrize("formatter_cls", [JSONishFormatter, TypeScriptFormatter, YAMLFormatter])
def test_retained_expansion_budget_attributes_survive(
    formatter_cls: type[BaseFormatter],
) -> None:
    """The global budget stays — TypeScript tiers ``anyOf`` caps off it."""
    formatter = formatter_cls({})

    assert formatter._global_expansion_budget == 150
    assert formatter._global_expansion_count == 0
    assert formatter._ref_expansion_path == []
    assert formatter._truncation_epoch == 0
    assert formatter._root_ref_key is None


# ---------------------------------------------------------------------------
# Cross-format golden suite (lsl-2026-09-04-014, Phase 8)
#
# These models are defined LOCALLY, with the names the golden strings embed
# (the placeholder renders ``recursive: <TypeName>``). They deliberately carry
# ``#`` comments instead of docstrings: pydantic promotes a model docstring to
# ``description``, which every formatter renders as a trailing comment and so
# would break the whole-string goldens.
# ---------------------------------------------------------------------------

FORMATS: tuple[str, ...] = ("jsonish", "typescript", "yaml")


# Self-referencing list with an ``= []`` default (root model, root-``$ref``).
class Node(BaseModel):
    name: str
    children: list[Node] = []


# The same shape reached through one wrapping property.
class RootNode(BaseModel):
    root: Node


# Mutually recursive pair A -> B -> A.
class A(BaseModel):
    name: str
    b: B | None = None


class B(BaseModel):
    tag: str
    a: A | None = None


# Self-reference through Optional, twice.
class Tree(BaseModel):
    value: str
    left: Tree | None = None
    right: Tree | None = None


# The primary AC2/AC3 golden: ``default_factory`` means pydantic emits no
# ``default`` key, so the output is free of ``(default=[])`` noise.
class ListNode(BaseModel):
    label: str
    kids: list[ListNode] = Field(default_factory=list)


# D5: recursion reached through a ``dict[str, T]`` mapping.
class MapNode(BaseModel):
    name: str
    kids: dict[str, MapNode] = {}


# C4: a nested list must keep its ``[] []`` postfix OUTSIDE the comment.
class LLNode(BaseModel):
    grid: list[list[LLNode]] = []


# C4: a mapping of optionals must keep `` OR null`` OUTSIDE the comment.
class N(BaseModel):
    m: dict[str, N | None] = {}


for _model in (Node, RootNode, A, B, Tree, ListNode, MapNode, LLNode, N):
    _model.model_rebuild()


JSONISH_D1 = "{\n  label*: string,\n  kids: object [] // recursive: ListNode\n}"
YAML_D1 = (
    "# ListNode\nListNode.label*: string\nListNode.kids: 'list[object  # recursive:"
    " ListNode]'\n\nlabel*: string\nkids: 'list[object  # recursive: ListNode]'"
)
TS_D1 = (
    "interface ListNode {\n  label*: string;\n  kids: Array<object /* recursive: ListNode"
    " */>;\n}\n\n// Fields marked with * are required\ninterface Schema {\n  label*:"
    " string;\n  kids: Array<object /* recursive: ListNode */>;\n}"
)

JSONISH_D2 = (
    "{\n  label*: string,\n  kids: [{\n    label*: string,\n    kids: object [] //"
    " recursive: ListNode\n  }] // recursive: ListNode\n}"
)
YAML_D2 = (
    "# ListNode\nListNode.label*: string\nListNode.kids: 'list[label*: string\n\n  kids:"
    " list[object  # recursive: ListNode]]'\n\nlabel*: string\nkids: 'list[label*:"
    " string\n\n  kids: list[object  # recursive: ListNode]]'"
)
TS_D2 = (
    "interface ListNode {\n  label*: string;\n  kids: Array<{ label*: string, kids:"
    " Array<object /* recursive: ListNode */> }>;\n}\n\n// Fields marked with * are"
    " required\ninterface Schema {\n  label*: string;\n  kids: Array<{ label*: string,"
    " kids: Array<object /* recursive: ListNode */> }>;\n}"
)

JSONISH_D3 = (
    "{\n  label*: string,\n  kids: [{\n    label*: string,\n    kids: [{\n      label*:"
    " string,\n      kids: object [] // recursive: ListNode\n    }] // recursive:"
    " ListNode\n  }] // recursive: ListNode\n}"
)
YAML_D3 = (
    "# ListNode\nListNode.label*: string\nListNode.kids: 'list[label*: string\n\n  kids:"
    " list[label*: string\n\n  kids: list[object  # recursive: ListNode]]]'\n\nlabel*:"
    " string\nkids: 'list[label*: string\n\n  kids: list[label*: string\n\n  kids:"
    " list[object  # recursive: ListNode]]]'"
)
TS_D3 = (
    "interface ListNode {\n  label*: string;\n  kids: Array<{ label*: string, kids:"
    " Array<{ label*: string, kids: Array<object /* recursive: ListNode */> }> }>;\n}\n\n"
    "// Fields marked with * are required\ninterface Schema {\n  label*: string;\n  kids:"
    " Array<{ label*: string, kids: Array<{ label*: string, kids: Array<object /*"
    " recursive: ListNode */> }> }>;\n}"
)

LIST_NODE_GOLDENS: dict[int, dict[str, str]] = {
    1: {"jsonish": JSONISH_D1, "yaml": YAML_D1, "typescript": TS_D1},
    2: {"jsonish": JSONISH_D2, "yaml": YAML_D2, "typescript": TS_D2},
    3: {"jsonish": JSONISH_D3, "yaml": YAML_D3, "typescript": TS_D3},
}

# The §2.4 sibling schema: a recursive ref, two clean refs, then the recursive
# ref again. ``rec2`` must not be served ``rec``'s truncated rendering.
SIBLING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rec": {"$ref": "#/$defs/Node"},
        "a1": {"$ref": "#/$defs/Addr"},
        "a2": {"$ref": "#/$defs/Addr"},
        "rec2": {"$ref": "#/$defs/Node"},
    },
    "required": ["rec", "a1", "a2", "rec2"],
    "$defs": {
        "Node": {
            "type": "object",
            "title": "Node",
            "properties": {"name": {"type": "string"}, "child": {"$ref": "#/$defs/Node"}},
            "required": ["name"],
        },
        "Addr": {
            "type": "object",
            "title": "Addr",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

# R7: JSONish memoises in ``processed_ref_cache``; YAML and TypeScript use
# ``_ref_cache``. Asserting against the wrong one silently passes.
CACHE_ATTRIBUTE: dict[type[BaseFormatter], str] = {
    JSONishFormatter: "processed_ref_cache",
    TypeScriptFormatter: "_ref_cache",
    YAMLFormatter: "_ref_cache",
}


def _registered(all_pydantic_models: list[tuple[str, type[BaseModel]]], name: str):
    """Look a conftest-registered model up by its registry name.

    The recursive fixtures are registered in ``all_pydantic_models`` rather than
    exposed as individual fixtures, so this doubles as a registration check.

    Args:
        all_pydantic_models: The ``(name, model)`` registry fixture.
        name: Registry name of the wanted model.

    Returns:
        The registered Pydantic model class.
    """
    registry = dict(all_pydantic_models)
    assert name in registry, f"{name} is not registered in all_pydantic_models"
    return registry[name]


def _render(model: type[BaseModel], fmt: str, depth: int | None = None) -> str:
    """Render a model through the public API.

    Args:
        model: Pydantic model to simplify.
        fmt: One of ``"jsonish"``, ``"typescript"``, ``"yaml"``.
        depth: ``max_recursion_depth``; ``None`` keeps the shipped default.

    Returns:
        The rendered schema string.
    """
    config = None if depth is None else FormatterConfig(max_recursion_depth=depth)
    return simplify_schema(model, config=config, format_type=fmt).to_string()  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AC1 — recursive models render instead of raising
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_recursive_models_do_not_raise(fmt: str, depth: int, all_pydantic_models) -> None:
    """AC1: every recursive shape renders at every depth, in every format."""
    fixtures = [
        _registered(all_pydantic_models, name) for name in ("TreeNode", "MutualA", "OptionalTree")
    ]
    for model in [*fixtures, ListNode]:
        out = _render(model, fmt, depth)
        assert out
        assert "recursive:" in out


def test_enrichment_terminates_on_recursive_annotations(all_pydantic_models) -> None:
    """The two unwrapped enrichment call sites terminate on a recursive model."""
    tree_node = _registered(all_pydantic_models, "TreeNode")

    parsed, _partial = loads('{"name": "a", "children": []}', schema=tree_node)
    assert parsed.name == "a"

    is_valid, errors = validate(tree_node, {"name": "a", "children": []})
    assert is_valid is True
    assert errors is None


# ---------------------------------------------------------------------------
# AC2 / AC3 — whole-string goldens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_default_depth_expands_twice_then_placeholder(fmt: str, all_pydantic_models) -> None:
    """AC2: the shipped default expands a recursive type twice, then truncates."""
    assert _render(ListNode, fmt) == LIST_NODE_GOLDENS[2][fmt]

    for name in ("TreeNode", "MutualA", "OptionalTree"):
        out = _render(_registered(all_pydantic_models, name), fmt)
        assert f"recursive: {name}" in out


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_depth_one_two_three(fmt: str, depth: int) -> None:
    """AC3: ``max_recursion_depth`` selects exactly how many bodies are emitted."""
    assert _render(ListNode, fmt, depth) == LIST_NODE_GOLDENS[depth][fmt]


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_root_model_and_wrapped_model_agree(fmt: str, depth: int) -> None:
    """§2.2: the root-``$ref`` unwrap makes ``Node`` and ``RootNode`` agree."""
    bare = _render(Node, fmt, depth)
    wrapped = _render(RootNode, fmt, depth)

    assert bare.count("recursive:") == wrapped.count("recursive:")
    assert bare.count("children") == wrapped.count("children")
    assert bare.count("name") == wrapped.count("name")


# ---------------------------------------------------------------------------
# C4 — the placeholder annotates, it never swallows structure
# ---------------------------------------------------------------------------


def test_placeholder_never_swallows_delimiters() -> None:
    """JSONish keeps ``[] []`` and `` OR null`` outside the block comment."""
    nested_list = _render(LLNode, "jsonish", 1)
    assert "grid: object /* recursive: LLNode */ [] [] // (default=[])" in nested_list

    optional_map = _render(N, "jsonish", 1)
    assert "<string>: object /* recursive: N */ OR null" in optional_map


# ---------------------------------------------------------------------------
# D5 — the mapping path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ("jsonish", "kids: {\n    <string>: object /* recursive: MapNode */\n  }"),
        ("typescript", "kids: Record<string, object /* recursive: MapNode */>"),
        ("yaml", "<string>: 'object  # recursive: MapNode'"),
    ],
    ids=["jsonish", "typescript", "yaml"],
)
def test_mapping_of_recursive_model_terminates(fmt: str, expected: str) -> None:
    """D5: ``dict[str, Recursive]`` truncates rather than recursing forever."""
    assert expected in _render(MapNode, fmt, 1)


# ---------------------------------------------------------------------------
# The knob's boundary values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_max_recursion_depth_zero_equals_one(fmt: str) -> None:
    """B2: the first expansion is unconditional, so 0 and 1 are byte-identical."""
    for model in (Node, MapNode, ListNode):
        assert _render(model, fmt, 0) == _render(model, fmt, 1)


def test_max_recursion_depth_negative_raises() -> None:
    """A negative budget is rejected at config-construction time."""
    with pytest.raises(ValueError, match="max_recursion_depth must be >= 0"):
        FormatterConfig(max_recursion_depth=-1)


# ---------------------------------------------------------------------------
# §2.4 — truncated renderings must never reach the cache
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("formatter_cls", [JSONishFormatter, TypeScriptFormatter, YAMLFormatter])
def test_truncated_rendering_is_not_cached(formatter_cls: type[BaseFormatter]) -> None:
    """A later sibling gets a full body, and the clean sibling is still cached."""
    formatter = _formatter(formatter_cls, SIBLING_SCHEMA, depth=2)
    out = formatter.transform_schema()

    tail = out.split("rec2")[-1]
    assert "recursive: Node" in tail
    assert tail.count("name") == 2, tail

    cache = getattr(formatter, CACHE_ATTRIBUTE[formatter_cls])
    assert "Addr" in cache
    assert "Node" not in cache


# ---------------------------------------------------------------------------
# The safety property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_non_recursive_schema_unchanged_at_every_depth(fmt: str, deep_nested_model) -> None:
    """The knob is provably incapable of degrading a non-recursive schema."""
    renderings = {depth: _render(deep_nested_model, fmt, depth) for depth in (1, 2, 3, 99)}

    assert len(set(renderings.values())) == 1
    assert "recursive:" not in renderings[1]
