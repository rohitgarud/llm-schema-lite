"""`{"type": "object", "allOf": [...]}` with no `properties` must not self-cycle.

The `$ref` sibling of this bug was already fixed by hoisting `$ref` above `type` in
`_process_schema_recursive_inner`'s dispatch chain; `allOf` was left below it. A node
carrying both `type: object` and `allOf` therefore took the `type` arm, and
`process_types`' object branch calls straight back into the dispatcher with the SAME
node. Nothing could break the loop: the empty-object base case excludes `allOf` nodes
explicitly, and the inline `id()` cycle guard is disabled while a `$ref` is expanding.

Four JSONSchemaBench Github_easy schemas (o68471, o5966, o74547, o64856) died with
`RecursionError` on this shape, which is what makes it worth a permanent check.
"""

from __future__ import annotations

import pytest

from llm_schema_lite import simplify_schema

FORMATS = ("jsonish", "yaml", "typescript")

# `C` is the shape that recursed: type/allOf, no properties of its own.
ALLOF_OBJECT_SCHEMA = {
    "definitions": {
        "T": {"type": "object", "properties": {"v": {"type": "string"}}},
    },
    "properties": {
        "C": {"type": "object", "allOf": [{"$ref": "#/definitions/T"}]},
    },
}


@pytest.mark.parametrize("fmt", FORMATS)
def test_allof_object_without_properties_terminates(fmt: str) -> None:
    """The render must finish at all -- this raised RecursionError before the fix."""
    out = simplify_schema(ALLOF_OBJECT_SCHEMA, format_type=fmt).to_string()
    assert out


def test_allof_object_inlines_the_referenced_members() -> None:
    """Terminating is not enough: the allOf member's fields must actually survive."""
    out = simplify_schema(ALLOF_OBJECT_SCHEMA, format_type="jsonish").to_string()
    assert "C" in out
    assert "v" in out
    assert "string" in out
