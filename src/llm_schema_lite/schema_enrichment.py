"""Schema enrichment: inject enum metadata (_descriptions, _aliases) into JSON schema."""

from enum import Enum
from typing import Any, get_args, get_origin

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None  # type: ignore[assignment, misc]


def _get_enum_class(annotation: Any) -> type[Enum] | None:
    """Return the Enum class if annotation is an Enum or Optional[Enum], else None."""
    if annotation is None:
        return None
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        for arg in args:
            if isinstance(arg, type) and issubclass(arg, Enum):
                return arg
        return None
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    return None


def _collect_enum_classes_from_model(
    model: type[Any], _seen: set[type[Any]] | None = None
) -> set[type[Enum]]:
    """Recursively collect all Enum classes used in a Pydantic model.

    Args:
        model: Pydantic BaseModel class (or arbitrary type) to walk.
        _seen: Internal recursion guard tracking models already visited on
            this walk. Callers must not pass this argument; it defaults to
            ``None`` and is lazily initialized to an empty set. A single,
            global (not per-path) visited set is correct here because enum
            collection is a pure set union with no path-dependent semantics,
            so a self-referencing or mutually recursive model graph
            terminates instead of raising ``RecursionError``.

    Returns:
        Set of Enum classes reachable from the model's fields.
    """
    enums: set[type[Enum]] = set()
    if BaseModel is None or not isinstance(model, type) or not issubclass(model, BaseModel):
        return enums
    if _seen is None:
        _seen = set()
    if model in _seen:
        return enums
    _seen.add(model)
    for _name, field_info in model.model_fields.items():
        ann = field_info.annotation
        enum_cls = _get_enum_class(ann)
        if enum_cls is not None:
            enums.add(enum_cls)
        # Recurse into nested BaseModels
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            enums |= _collect_enum_classes_from_model(ann, _seen)
        origin = get_origin(ann)
        if origin is not None:
            for arg in get_args(ann):
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    enums |= _collect_enum_classes_from_model(arg, _seen)
    return enums


def extract_enum_metadata(enum_class: type[Enum]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Extract _descriptions and _aliases from an enum class.

    Args:
        enum_class: Enum class that may define _descriptions (member name -> str)
            and _aliases (member name -> list of alias strings).

    Returns:
        Tuple of (descriptions_by_value, aliases_by_canonical_value).
        descriptions_by_value: JSON enum value -> description string.
        aliases_by_canonical_value: canonical enum value -> list of accepted aliases.
    """
    descriptions_by_value: dict[str, str] = {}
    aliases_by_value: dict[str, list[str]] = {}

    raw_descriptions = getattr(enum_class, "_descriptions", None)
    if isinstance(raw_descriptions, Enum):
        raw_descriptions = raw_descriptions.value
    if isinstance(raw_descriptions, dict):
        for member_name, desc in raw_descriptions.items():
            if not isinstance(desc, str):
                continue
            try:
                member = getattr(enum_class, member_name, None)
            except AttributeError:
                continue
            if isinstance(member, Enum):
                descriptions_by_value[str(member.value)] = desc

    raw_aliases = getattr(enum_class, "_aliases", None)
    if isinstance(raw_aliases, Enum):
        raw_aliases = raw_aliases.value
    if isinstance(raw_aliases, dict):
        for member_name, alias_list in raw_aliases.items():
            if not isinstance(alias_list, list):
                continue
            member = getattr(enum_class, member_name, None)
            if isinstance(member, Enum):
                canonical = str(member.value)
                aliases_by_value[canonical] = [str(a) for a in alias_list if a is not None]

    return descriptions_by_value, aliases_by_value


def inject_enum_metadata(
    schema_node: dict[str, Any],
    descriptions: dict[str, str],
    aliases: dict[str, list[str]],
) -> None:
    """
    Inject x-enum-descriptions and x-enum-aliases into a schema node (mutates in place).

    Args:
        schema_node: A JSON schema object that contains "enum" (e.g. a $defs entry).
        descriptions: Map from enum value to description string.
        aliases: Map from canonical enum value to list of alias strings.
    """
    if descriptions:
        schema_node["x-enum-descriptions"] = dict(descriptions)
    if aliases:
        schema_node["x-enum-aliases"] = dict(aliases)


def enrich_schema_with_enum_metadata(model: type[Any], schema: dict[str, Any]) -> dict[str, Any]:
    """
    Enrich a JSON schema with enum metadata from a Pydantic model.

    Traverses the model to find Enum fields, extracts _descriptions and _aliases
    from those enum classes, and injects them into the schema using custom
    extension fields (x-enum-descriptions, x-enum-aliases). Works on $defs
    entries that correspond to enum types.

    Args:
        model: Pydantic BaseModel class used to generate the schema.
        schema: JSON schema dict (e.g. from model.model_json_schema()).

    Returns:
        The same schema dict (mutated in place); returned for convenience.
    """
    if BaseModel is None or not isinstance(model, type) or not issubclass(model, BaseModel):
        return schema

    defs = schema.get("$defs", schema.get("definitions", {}))
    if not defs:
        return schema

    enum_classes = _collect_enum_classes_from_model(model)
    for enum_cls in enum_classes:
        def_name = enum_cls.__name__
        if def_name not in defs:
            continue
        node = defs[def_name]
        if not isinstance(node, dict) or "enum" not in node:
            continue
        descriptions, aliases = extract_enum_metadata(enum_cls)
        if descriptions or aliases:
            inject_enum_metadata(node, descriptions, aliases)

    return schema
