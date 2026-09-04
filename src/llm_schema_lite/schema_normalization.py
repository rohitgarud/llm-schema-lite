"""Pure, pydantic-free normalization of Pydantic-emitted auto-generated JSON-schema titles.

This module contains no pydantic import. It operates only on plain ``dict``/``list``
JSON-schema documents (Pydantic-produced, hand-written, or JSON-decoded) and is safe to
import from ``formatters/base.py``, which itself has no pydantic dependency.
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = ["auto_title_for_key", "normalize_schema_titles"]


def auto_title_for_key(key: str) -> str:
    """Return the title Pydantic would auto-generate for a property key.

    Mirrors ``pydantic.json_schema.GenerateJsonSchema.get_title_from_name``:
    title-case the key, then replace underscores with spaces, then strip.
    """
    return key.title().replace("_", " ").strip()


def normalize_schema_titles(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied schema with auto-generated titles removed.

    Applies R1 (property-level auto titles, at every depth) and R2 (root model-level
    auto title) to a deep copy of ``schema``, leaving the caller's dict untouched.
    """
    result = copy.deepcopy(schema)
    _strip_property_titles(result)
    _strip_model_title(result)
    return result


def _strip_property_titles(node: Any) -> None:
    """Recursively strip auto-generated titles from every ``properties`` mapping.

    Recurses generically over ``dict`` values and ``list`` elements so that it reaches
    ``$defs/*/properties``, ``items``, ``anyOf``/``oneOf``/``allOf`` members and
    ``additionalProperties`` without enumerating JSON-schema keywords by name.
    """
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            for key, prop_node in properties.items():
                if not isinstance(prop_node, dict):
                    continue
                title = prop_node.get("title")
                if isinstance(title, str) and title == auto_title_for_key(key):
                    del prop_node["title"]
        for value in node.values():
            _strip_property_titles(value)
    elif isinstance(node, list):
        for item in node:
            _strip_property_titles(item)


def _strip_model_title(schema: dict[str, Any]) -> None:
    """Strip the root schema's auto-generated model-level title (R2), in place."""
    title = schema.get("title")
    description = schema.get("description")
    if (
        isinstance(title, str)
        and title.isidentifier()
        and isinstance(description, str)
        and description.strip() != ""
    ):
        del schema["title"]
