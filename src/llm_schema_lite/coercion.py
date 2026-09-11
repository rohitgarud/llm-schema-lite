"""Type coercion functionality for LLM output handling."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Mapping of bool string values
BOOL_TRUE_VALUES = {"true", "yes", "1", "on", "y"}
BOOL_FALSE_VALUES = {"false", "no", "0", "off", "n"}


@dataclass
class CoercionMetadata:
    """Metadata about a coercion operation for debugging."""

    original_value: Any
    target_type: str
    coerced_value: Any
    coercion_type: str  # e.g., "to_int", "string_to_bool"
    field_path: str = ""  # Path to field for nested coercions


@dataclass
class ParseConfig:
    """Configuration for parsing/coercion behavior."""

    allow_coercion: bool = True  # Try coercion before rejecting type mismatch
    coerce_list_single_item: bool = False  # Wrap single item in list
    log_coercions: bool = True  # Log coercion events for debugging
    # Keep what validates instead of failing: loads(schema=...) drops a failing optional
    # field; StructuredOutputAdapter first nulls just the invalid values inside a field.
    partial: bool = False
    strip_required_marker: str = "*"  # Trailing marker stripped from reply keys


def _is_correct_type(value: Any, target_type: str, enum_values: list[Any] | None = None) -> bool:
    """Check if value is already of the correct type."""
    # If enum values are provided, check if value matches one of them exactly
    if enum_values is not None and value in enum_values:
        return True

    if target_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    elif target_type == "number":
        # Only float is correct for number type, int should be coerced
        return isinstance(value, float) and not isinstance(value, bool)
    elif target_type == "boolean":
        return isinstance(value, bool)
    elif target_type == "string":
        # Only return True for string if no enum values provided
        # or if value is already an exact match (handled above)
        return isinstance(value, str) and enum_values is None
    elif target_type == "array":
        return isinstance(value, list)
    elif target_type == "object":
        return isinstance(value, dict)
    return False


def _coerce_to_int(value: Any) -> tuple[int | Any, CoercionMetadata | None]:
    """Coerce a value to integer."""
    try:
        if isinstance(value, str):
            # Handle "42.0" string by converting through float first
            coerced = int(float(value))
        elif isinstance(value, float):
            coerced = int(value)
        elif isinstance(value, int):
            coerced = value
        else:
            coerced = int(value)
        return coerced, CoercionMetadata(value, "integer", coerced, "to_int")
    except (ValueError, TypeError):
        return value, None


def _coerce_to_float(value: Any) -> tuple[float | Any, CoercionMetadata | None]:
    """Coerce a value to float."""
    try:
        coerced = float(value)
        return coerced, CoercionMetadata(value, "number", coerced, "to_float")
    except (ValueError, TypeError):
        return value, None


def _coerce_to_bool(value: Any) -> tuple[bool | Any, CoercionMetadata | None]:
    """Coerce a value to boolean."""
    if isinstance(value, bool):
        return value, None  # Already bool

    if value is None:
        # None is falsy
        return False, CoercionMetadata(value, "boolean", False, "none_to_bool")

    if isinstance(value, str):
        lower = value.lower().strip()
        if lower in BOOL_TRUE_VALUES:
            return True, CoercionMetadata(value, "boolean", True, "string_to_bool")
        elif lower in BOOL_FALSE_VALUES:
            return False, CoercionMetadata(value, "boolean", False, "string_to_bool")

    if isinstance(value, int | float):
        # 0, 0.0 -> False; anything else -> True
        coerced = bool(value)
        return coerced, CoercionMetadata(value, "boolean", coerced, "int_to_bool")

    return value, None


def _coerce_to_string(value: Any) -> tuple[str, CoercionMetadata]:
    """Coerce a value to string."""
    coerced = str(value)
    return coerced, CoercionMetadata(value, "string", coerced, "to_string")


def _coerce_single_to_list(
    value: Any,
) -> tuple[list[Any] | Any, CoercionMetadata | None]:
    """Coerce single item to list."""
    if isinstance(value, list):
        return value, None

    coerced = [value]
    return coerced, CoercionMetadata(value, "array", coerced, "single_to_list")


_TYPE_COERCERS: dict[str, Callable[[Any], tuple[Any, CoercionMetadata | None]]] = {
    "integer": _coerce_to_int,
    "number": _coerce_to_float,
    "boolean": _coerce_to_bool,
    "string": _coerce_to_string,
}


def _coerce_to_enum(value: Any, enum_values: list[Any]) -> tuple[Any, CoercionMetadata | None]:
    """Coerce to enum value (case-insensitive matching for strings)."""
    # If already a valid enum value, return as-is
    if value in enum_values:
        return value, None

    # Try case-insensitive string matching
    if isinstance(value, str):
        value_lower = value.lower()
        for enum_val in enum_values:
            if isinstance(enum_val, str) and enum_val.lower() == value_lower:
                return enum_val, CoercionMetadata(
                    original_value=value,
                    target_type="enum",
                    coerced_value=enum_val,
                    coercion_type="enum_case_insensitive",
                )

    return value, None


def coerce_value(
    value: Any,
    target_type: str,
    enum_values: list[Any] | None = None,
    coerce_list_single_item: bool = False,
) -> tuple[Any, CoercionMetadata | None]:
    """
    Coerce a value to the expected type.

    Args:
        value: The value to coerce
        target_type: JSON Schema type ("integer", "number", "boolean", "string", "array")
        enum_values: Optional list of valid enum values for enum matching
        coerce_list_single_item: Whether to wrap single items in list

    Returns:
        Tuple of (coerced_value, metadata)
        metadata is None if no coercion was performed
    """
    # If value is already of the correct type, no coercion needed
    if _is_correct_type(value, target_type, enum_values):
        return value, None

    # Handle enum coercion first (before type coercion)
    if enum_values is not None:
        result, metadata = _coerce_to_enum(value, enum_values)
        if metadata is not None:
            logger.debug(f"Coerced value {value!r} to enum {result!r} (case-insensitive)")
            return result, metadata
        # If no enum match, return original value without coercion
        # (don't fall through to type coercion when enum is specified)
        return value, None

    # Handle specific coercions based on target type (a JSON-schema type list matches none)
    coercer: Callable[[Any], tuple[Any, CoercionMetadata | None]] | None
    if target_type == "array" and coerce_list_single_item:
        coercer = _coerce_single_to_list
    else:
        coercer = _TYPE_COERCERS.get(target_type) if isinstance(target_type, str) else None
    if coercer is None:
        # No coercion possible
        return value, None

    result, metadata = coercer(value)
    if metadata is not None:
        logger.debug(f"Coerced value {value!r} to {target_type} {result!r}")
    return result, metadata


def coerce_recursive(
    data: Any,
    schema: dict[str, Any],
    config: ParseConfig,
    field_path: str = "",
) -> tuple[Any, list[CoercionMetadata]]:
    """
    Recursively coerce data to match schema types.

    Args:
        data: The data to coerce
        schema: JSON schema dict
        config: ParseConfig with coercion settings
        field_path: Current field path for tracking nested fields

    Returns:
        Tuple of (coerced_data, list of CoercionMetadata)
    """
    metadata_list: list[CoercionMetadata] = []

    # Handle object (dict) with properties
    if isinstance(data, dict) and schema.get("type") == "object" and "properties" in schema:
        result_dict: dict[str, Any] = {}
        for key, value in data.items():
            prop_schema = schema["properties"].get(key, {})
            new_path = f"{field_path}.{key}" if field_path else key
            coerced_value, metadata = coerce_recursive(value, prop_schema, config, new_path)
            result_dict[key] = coerced_value
            if metadata:
                metadata_list.extend(metadata)
        return result_dict, metadata_list

    # Handle array with items
    if isinstance(data, list) and schema.get("type") == "array" and "items" in schema:
        result_list: list[Any] = []
        for i, item in enumerate(data):
            new_path = f"{field_path}[{i}]"
            coerced_item, metadata = coerce_recursive(item, schema["items"], config, new_path)
            result_list.append(coerced_item)
            if metadata:
                metadata_list.extend(metadata)
        return result_list, metadata_list

    # Handle object without explicit properties - try to coerce values
    if isinstance(data, dict):
        result_obj: dict[str, Any] = {}
        for key, value in data.items():
            new_path = f"{field_path}.{key}" if field_path else key
            # Try to determine type from value pattern or default to no coercion
            coerced_value, metadata = coerce_recursive(value, {"type": "string"}, config, new_path)
            result_obj[key] = coerced_value
            if metadata:
                metadata_list.extend(metadata)
        return result_obj, metadata_list

    # Scalar value - apply coercion
    target_type = schema.get("type", "string")
    enum_values = schema.get("enum")

    coerced, single_metadata = coerce_value(
        data,
        target_type,
        enum_values,
        config.coerce_list_single_item,
    )
    if single_metadata is not None:
        single_metadata.field_path = field_path
        metadata_list.append(single_metadata)

    return coerced, metadata_list


def coerce_to_schema(
    data: Any,
    schema: type["BaseModel"] | dict[str, Any] | str,
    config: ParseConfig | None = None,
) -> tuple[dict[str, Any], list[CoercionMetadata]]:
    """
    Coerce data to match schema types. Also exported as ``llm_schema_lite.coerce``.

    This function converts input data to match the expected types defined in a schema.
    It handles various type coercions like string to int, string to bool, etc.

    Args:
        data: Data to coerce (can be dict, list, string, number, boolean, null,
              or JSON string)
        schema: Pydantic BaseModel class, JSON schema dict, or JSON schema string
        config: ParseConfig with coercion settings (optional, uses default if None)

    Returns:
        Tuple of (coerced_data, list of CoercionMetadata)

    Raises:
        ConversionError: If the schema is invalid

    Example:
        >>> from pydantic import BaseModel
        >>> from llm_schema_lite import coerce
        >>>
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        ...
        >>> # Coerce data with type mismatches
        >>> coerced, metadata = coerce({"name": "John", "age": "30"}, User)
        >>> print(coerced)
        {'name': 'John', 'age': 30}
        >>> # Coerce from JSON string
        >>> coerced, metadata = coerce('{"name": "Jane", "age": "25"}', User)
        >>> print(coerced)
        {'name': 'Jane', 'age': 25}
        >>> # With custom config
        >>> config = ParseConfig(coerce_list_single_item=True)
        >>> coerced, metadata = coerce({"name": "John"}, User, config)
    """
    if config is None:
        config = ParseConfig()

    if not config.allow_coercion:
        return data if isinstance(data, dict) else {}, []

    # Parse schema if it's a string
    schema_dict: dict[str, Any]
    if isinstance(schema, str):
        import json

        try:
            schema_dict = json.loads(schema)
        except json.JSONDecodeError as e:
            from .exceptions import ConversionError

            raise ConversionError(f"Invalid JSON schema string: {e}") from e
    elif isinstance(schema, dict):
        schema_dict = schema
    else:
        # Assume it's a Pydantic model
        try:
            schema_dict = schema.model_json_schema()
        except Exception as e:
            from .exceptions import ConversionError

            raise ConversionError(f"Failed to get schema from model: {e}") from e

    # Parse data if it's a string
    data_dict: dict[str, Any]
    if isinstance(data, str):
        import json

        try:
            data_dict = json.loads(data)
        except json.JSONDecodeError:
            # If it looks like a simple value, wrap it
            data_dict = {"value": data}
    elif isinstance(data, dict):
        data_dict = data
    else:
        data_dict = {"value": data}

    # Apply recursive coercion
    return coerce_recursive(data_dict, schema_dict, config)
