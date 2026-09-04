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

from llm_schema_lite import FormatterConfig
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
