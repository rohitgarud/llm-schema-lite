"""Characterisation of the `$ref` back-reference path.

Written BEFORE changing `_truncation_epoch` scoping, because this path had no test
coverage whatsoever: no test in the repo contained the string ``defined above:``, and
the commit that introduced ``_emitted_refs``/``BACKREFERENCE_MIN_CHARS`` touched no test
file. Widening an untested path is how the taint bug survived this long.

Two populations of test live here, and the distinction matters when reading a diff:

* Most tests pin behaviour that MUST SURVIVE the taint-scoping fix -- naming a large
  repeat, inlining a small one, and still naming a clean sibling that merely follows an
  unrelated truncation.
* `test_def_spanning_a_truncation_is_named` and `test_ref_expansions_are_counted` were
  written to pin the DEFECTIVE behaviour, then flipped when the taint was scoped. That
  history is the point: the fix showed up as a readable, intentional diff instead of
  mystery churn in a golden, and the flip is recorded in each docstring.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

from llm_schema_lite import FormatterConfig, simplify_schema
from llm_schema_lite.formatters.base import BaseFormatter
from llm_schema_lite.formatters.jsonish_formatter import JSONishFormatter
from llm_schema_lite.formatters.typescript_formatter import TypeScriptFormatter
from llm_schema_lite.formatters.yaml_formatter import YAMLFormatter

# `simplify_schema` takes a Literal, so parametrized format names must be typed as one
# rather than plain `str`.
FormatName = Literal["jsonish", "typescript", "yaml"]

# Rendered well past BACKREFERENCE_MIN_CHARS (200), so a repeat is worth naming.
BIG_DEF: dict[str, Any] = {
    "type": "object",
    "title": "Address",
    "properties": {
        "street": {"type": "string", "description": "Street name and house number"},
        "city": {"type": "string", "description": "City or town of residence"},
        "region": {"type": "string", "description": "State, province or region"},
        "postcode": {"type": "string", "description": "Postal or ZIP code"},
        "country": {"type": "string", "description": "ISO 3166-1 alpha-2 country code"},
    },
    "required": ["street", "city"],
}

# Comfortably under the threshold: duplicating this reads better than naming it.
SMALL_DEF: dict[str, Any] = {
    "type": "object",
    "title": "Tag",
    "properties": {"k": {"type": "string"}},
}

KEYED_TWICE: dict[str, Any] = {
    "type": "object",
    "properties": {"home": {"$ref": "#/$defs/Address"}, "work": {"$ref": "#/$defs/Address"}},
    "required": ["home", "work"],
    "$defs": {"Address": BIG_DEF},
}

ARRAY_REPEAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "first": {"$ref": "#/$defs/Address"},
        "others": {"type": "array", "items": {"$ref": "#/$defs/Address"}},
    },
    "required": ["first", "others"],
    "$defs": {"Address": BIG_DEF},
}

SMALL_TWICE: dict[str, Any] = {
    "type": "object",
    "properties": {"a": {"$ref": "#/$defs/Tag"}, "b": {"$ref": "#/$defs/Tag"}},
    "required": ["a", "b"],
    "$defs": {"Tag": SMALL_DEF},
}

# A truncation happens FIRST (Node), and only then is Address expanded. Address's own
# expansion does not span the truncation, so it stays cacheable -- the anti-over-taint
# case, the other half of tests/test_recursive_models.py:245.
TRUNCATION_THEN_CLEAN: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rec": {"$ref": "#/$defs/Node"},
        "h": {"$ref": "#/$defs/Address"},
        "w": {"$ref": "#/$defs/Address"},
    },
    "required": ["rec", "h", "w"],
    "$defs": {
        "Node": {
            "type": "object",
            "title": "Node",
            "properties": {"name": {"type": "string"}, "child": {"$ref": "#/$defs/Node"}},
            "required": ["name"],
        },
        "Address": BIG_DEF,
    },
}

# The bug, in miniature. `Branch` is large enough to be worth naming AND contains a
# recursive ref, so its OWN expansion spans the truncation it causes. The global epoch
# therefore differs from its entry epoch, nothing is cached, nothing is named, and the
# body is emitted once per use site. This is the corpus monsters' pathology at small
# scale: o48404 takes it to 51,808 expansions of which 103 are cached.
SPANS_A_TRUNCATION: dict[str, Any] = {
    "type": "object",
    "properties": {"left": {"$ref": "#/$defs/Branch"}, "right": {"$ref": "#/$defs/Branch"}},
    "required": ["left", "right"],
    "$defs": {
        "Branch": {
            "type": "object",
            "title": "Branch",
            "properties": {
                "label": {"type": "string", "description": "Human readable branch label"},
                "weight": {"type": "number", "description": "Relative weight of this branch"},
                "owner": {"type": "string", "description": "Team or person owning the branch"},
                "child": {"$ref": "#/$defs/Branch"},
            },
            "required": ["label", "weight"],
        }
    },
}

CACHE_ATTRIBUTE: dict[type[BaseFormatter], str] = {
    JSONishFormatter: "processed_ref_cache",
    TypeScriptFormatter: "_ref_cache",
    YAMLFormatter: "_ref_cache",
}


def _render(formatter_cls: type[BaseFormatter], schema: dict[str, Any], depth: int = 2):
    """Return ``(rendered, formatter)`` so state can be asserted alongside output."""
    formatter = formatter_cls(schema, FormatterConfig(max_recursion_depth=depth))
    return formatter.transform_schema(), formatter


# ---------------------------------------------------------------------------
# Must survive the taint-scoping fix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ("jsonish", "work*: object // defined above: Address"),
        ("typescript", "work*: object /* defined above: Address */;"),
    ],
)
def test_large_repeated_def_is_named_not_inlined(fmt: FormatName, expected: str) -> None:
    """A repeat of a >200-char definition is replaced by a name, in field-key position.

    TypeScript needs the block comment form: a `//` would swallow the `;` that
    `interface Schema { ... }` appends after the field.
    """
    out = simplify_schema(KEYED_TWICE, format_type=fmt).to_string()
    assert expected in out
    # The body appears exactly once -- that is the whole point of naming the repeat.
    assert out.count("Street name and house number") == 1


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ("jsonish", "others*: object [] // defined above: Address"),
        ("typescript", "others*: Array<object /* defined above: Address */>;"),
    ],
)
def test_repeat_inside_an_array_is_named(fmt: FormatName, expected: str) -> None:
    """The array element carries the name, and the element type survives it.

    TypeScript keeps `Array<...>` rather than collapsing to bare `object`, which is what
    tests/test_typescript_formatter.py:1359 forbids across the corpus.
    """
    out = simplify_schema(ARRAY_REPEAT, format_type=fmt).to_string()
    assert expected in out
    assert out.count("Street name and house number") == 1


@pytest.mark.parametrize("fmt", ["jsonish", "yaml", "typescript"])
def test_small_repeated_def_is_inlined_twice(fmt: FormatName) -> None:
    """Under BACKREFERENCE_MIN_CHARS a definition is duplicated, never named.

    This is the sibling-inline acceptance criterion the threshold exists to protect:
    naming a tiny definition costs more than repeating it and reads worse.
    """
    out = simplify_schema(SMALL_TWICE, format_type=fmt).to_string()
    assert "defined above" not in out
    assert out.count("k: string") == 2


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ("jsonish", "w*: object // defined above: Address"),
        ("typescript", "w*: object /* defined above: Address */;"),
    ],
)
def test_clean_sibling_after_a_truncation_is_still_named(fmt: FormatName, expected: str) -> None:
    """A truncation must not poison a definition expanded entirely after it.

    `Node` truncates first and bumps the global epoch, but `Address` is expanded
    afterwards, so its own entry epoch is clean and it caches normally. The mirror of
    tests/test_recursive_models.py:245, asserted on rendered output rather than cache
    state -- and the invariant that stops a scoping fix from over-correcting.
    """
    out = simplify_schema(TRUNCATION_THEN_CLEAN, format_type=fmt).to_string()
    assert "recursive: Node" in out
    assert expected in out


def test_yaml_block_path_never_back_references() -> None:
    """YAML duplicates object bodies by design; it has no repeat-suppression.

    `_properties_block` deliberately bypasses `_ref_cache` (yaml_formatter.py:482-483)
    because caching a real dict lets PyYAML emit `&id001`/`*id001` aliases, which
    tests/test_yaml_formatter.py:1635 forbids. So YAML pays duplication to keep its
    output alias-free. Pinned here so the trade is visible and not "fixed" by accident.
    """
    out = simplify_schema(KEYED_TWICE, format_type="yaml").to_string()
    assert "defined above" not in out
    assert out.count("Street name and house number") == 2
    assert "&id001" not in out and "*id001" not in out


# ---------------------------------------------------------------------------
# Flipped when the taint was scoped -- these pinned the bug, now they pin the fix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("formatter_cls", "expected"),
    [
        (JSONishFormatter, "right*: object // defined above: Branch"),
        (TypeScriptFormatter, "right*: object /* defined above: Branch */;"),
    ],
)
def test_def_spanning_a_truncation_is_named(
    formatter_cls: type[BaseFormatter], expected: str
) -> None:
    """A self-recursive definition is cached, and its repeat named.

    `Branch` causes its own truncation, so the truncation is self-inflicted and the body
    renders identically wherever it appears -- `_body_is_replayable` allows the cache
    write. Before the taint was scoped, the global epoch blocked it and the body was
    emitted once per use site: this schema rendered 4 bodies and cached nothing, which is
    o48404's 51,808-expansions-with-103-cached pathology in miniature.
    """
    out, formatter = _render(formatter_cls, SPANS_A_TRUNCATION)

    assert "recursive: Branch" in out  # the truncation itself is correct and unchanged
    assert expected in out
    assert formatter._emitted_refs == {"Branch"}
    assert sorted(getattr(formatter, CACHE_ATTRIBUTE[formatter_cls])) == ["Branch"]
    assert out.count("Human readable branch label") == 2


def test_yaml_still_duplicates_a_self_recursive_def() -> None:
    """YAML is untouched by the scoping fix, because its block path never caches at all.

    Not an oversight: `_properties_block` bypasses `_ref_cache` so PyYAML cannot emit
    `&id001` aliases. YAML keeps paying duplication where JSONish and TypeScript now pay
    a name, which is why the test above cannot be parametrized over all three formats.
    """
    out, formatter = _render(YAMLFormatter, SPANS_A_TRUNCATION)

    assert "defined above" not in out
    assert formatter._emitted_refs == set()
    assert out.count("Human readable branch label") == 4


@pytest.mark.parametrize(
    ("formatter_cls", "expected_count"),
    [(JSONishFormatter, 2), (TypeScriptFormatter, 2), (YAMLFormatter, 4)],
)
def test_ref_expansions_are_counted(
    formatter_cls: type[BaseFormatter], expected_count: int
) -> None:
    """Every formatter now charges `$ref` expansions against the global budget.

    JSONish used not to increment `_global_expansion_count` at all (contrast
    base.py:727), so the 150-expansion budget governed YAML and TypeScript but not the
    default mode -- the mode that rendered 51,808 expansions on o48404. It read 0 here
    where the others read 4.

    Both numbers moved: JSONish 0 -> 2 because it now counts, and TypeScript 4 -> 2
    because caching a self-recursive body means it is expanded once rather than per use
    site. YAML stays at 4, since its block path neither caches nor reuses.
    """
    _, formatter = _render(formatter_cls, SPANS_A_TRUNCATION)
    assert formatter._global_expansion_budget == 150
    assert formatter._global_expansion_count == expected_count


# A schema whose distinct definitions outnumber the 150-expansion budget, so the guard is
# reachable without depending on a corpus file. Each def is referenced once from the root.
BUDGET_BUSTER: dict[str, Any] = {
    "type": "object",
    "properties": {f"f{i}": {"$ref": f"#/$defs/D{i}"} for i in range(200)},
    "$defs": {
        f"D{i}": {
            "type": "object",
            "title": f"D{i}",
            "properties": {"v": {"type": "string", "description": f"value number {i}"}},
        }
        for i in range(200)
    },
}


@pytest.mark.parametrize(
    ("fmt", "marker"),
    [
        ("jsonish", "// budget exhausted:"),
        ("yaml", "# budget exhausted:"),
        ("typescript", "/* budget exhausted:"),
    ],
)
def test_budget_exhaustion_is_marked_not_silent(fmt: FormatName, marker: str) -> None:
    """Past the expansion budget a `$ref` is NAMED as dropped, not left a bare `object`.

    The guard used to `return "object"`, which the model cannot tell apart from an
    unresolvable ref or a genuinely untyped one -- it could not know content was missing.
    This matters more now that the budget is enforced in JSONish too: it fires on 32 of
    the 9,542 corpus schemas, which were previously truncated silently in the default
    mode.

    Each format carries the note in its own comment syntax, mirroring `recursive:` and
    `defined above:`. TypeScript needs the block form so a `//` cannot swallow the `;`,
    and YAML defers the note so a later `OR null` folds into the same slot.
    """
    out = simplify_schema(BUDGET_BUSTER, format_type=fmt).to_string()
    assert marker in out
    # The budget stopped expansion; it must not have stopped the render.
    assert out.count("budget exhausted:") < 200
    assert "value number 0" in out
