"""Enum alias normalization for validation: replace alias values with canonical enum values."""

from typing import Any


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve #/$defs/Name or #/definitions/Name to the definition."""
    if not ref.startswith("#/"):
        return None
    parts = ref.split("/")[1:]
    if len(parts) >= 2 and parts[0] in ("$defs", "definitions"):
        key = parts[1]
        return defs.get(key) if isinstance(defs.get(key), dict) else None
    return None


def normalize_enum_aliases(
    data: Any,
    schema: dict[str, Any],
    defs: dict[str, Any] | None = None,
) -> Any:
    """
    Return a copy of data with enum alias values replaced by canonical values.

    Walks data and schema in parallel. At any location where the schema
    has an enum with x-enum-aliases, if the data value is an alias, it is
    replaced with the canonical value. $ref is resolved against defs.

    Args:
        data: Parsed data (dict, list, or primitive).
        schema: JSON schema node for this data (root or subschema).
        defs: Optional $defs/definitions for $ref resolution. If None, taken from schema.

    Returns:
        New structure with aliases normalized; primitives and non-enum parts unchanged.
    """
    if defs is None:
        defs = schema.get("$defs", schema.get("definitions", {})) or {}
    if not isinstance(defs, dict):
        defs = {}

    # Resolve $ref
    if isinstance(schema, dict) and "$ref" in schema and schema["$ref"]:
        resolved = _resolve_ref(schema["$ref"], defs)
        if resolved is not None:
            schema = resolved

    if not isinstance(schema, dict):
        return data

    # Enum with aliases: replace if data is an alias
    if "enum" in schema and "x-enum-aliases" in schema:
        aliases = schema["x-enum-aliases"]
        if isinstance(aliases, dict) and data is not None:
            # x-enum-aliases is {canonical: [alias, ...]}
            for canonical, alias_list in aliases.items():
                if isinstance(alias_list, list) and data in alias_list:
                    return str(canonical)
        return data

    if isinstance(data, dict) and "properties" in schema:
        props = schema.get("properties") or {}
        if not isinstance(props, dict):
            return data
        out = {}
        for k, v in data.items():
            sub_schema = props.get(k)
            if isinstance(sub_schema, dict) and "$ref" in sub_schema:
                resolved = _resolve_ref(sub_schema["$ref"], defs)
                sub_schema = resolved if resolved is not None else sub_schema
            if not isinstance(sub_schema, dict):
                sub_schema = {}
            out[k] = normalize_enum_aliases(v, sub_schema, defs)
        return out

    if isinstance(data, list) and "items" in schema:
        items_schema = schema["items"]
        if isinstance(items_schema, dict):
            return [normalize_enum_aliases(item, items_schema, defs) for item in data]
        return data

    return data
