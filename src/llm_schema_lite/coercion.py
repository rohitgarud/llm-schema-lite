"""Type coercion functionality for LLM output handling."""

import logging
from dataclasses import dataclass
from typing import Any

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None  # type: ignore[assignment, misc]

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
    original = value
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
        return coerced, CoercionMetadata(original, "integer", coerced, "to_int")
    except (ValueError, TypeError):
        return value, None


def _coerce_to_float(value: Any) -> tuple[float | Any, CoercionMetadata | None]:
    """Coerce a value to float."""
    original = value
    try:
        coerced = float(value)
        return coerced, CoercionMetadata(original, "number", coerced, "to_float")
    except (ValueError, TypeError):
        return value, None


def _coerce_to_bool(value: Any) -> tuple[bool | Any, CoercionMetadata | None]:
    """Coerce a value to boolean."""
    original = value
    if isinstance(value, bool):
        return value, None  # Already bool

    if value is None:
        # None is falsy
        return False, CoercionMetadata(original, "boolean", False, "none_to_bool")

    if isinstance(value, str):
        lower = value.lower().strip()
        if lower in BOOL_TRUE_VALUES:
            return True, CoercionMetadata(original, "boolean", True, "string_to_bool")
        elif lower in BOOL_FALSE_VALUES:
            return False, CoercionMetadata(original, "boolean", False, "string_to_bool")

    if isinstance(value, int | float):
        # 0, 0.0 -> False; anything else -> True
        coerced = bool(value)
        return coerced, CoercionMetadata(original, "boolean", coerced, "int_to_bool")

    return value, None


def _coerce_to_string(value: Any) -> tuple[str, CoercionMetadata]:
    """Coerce a value to string."""
    original = value
    coerced = str(value)
    return coerced, CoercionMetadata(original, "string", coerced, "to_string")


def _coerce_single_to_list(
    value: Any,
) -> tuple[list[Any] | Any, CoercionMetadata | None]:
    """Coerce single item to list."""
    if isinstance(value, list):
        return value, None

    original = value
    coerced = [value]
    return coerced, CoercionMetadata(original, "array", coerced, "single_to_list")


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

    # Handle specific coercions based on target type
    if target_type == "integer":
        result, metadata = _coerce_to_int(value)
        if metadata is not None:
            logger.debug(f"Coerced value {value!r} to int {result!r}")
        return result, metadata
    elif target_type == "number":
        result, metadata = _coerce_to_float(value)
        if metadata is not None:
            logger.debug(f"Coerced value {value!r} to float {result!r}")
        return result, metadata
    elif target_type == "boolean":
        result, metadata = _coerce_to_bool(value)
        if metadata is not None:
            logger.debug(f"Coerced value {value!r} to bool {result!r}")
        return result, metadata
    elif target_type == "string":
        result, metadata = _coerce_to_string(value)
        logger.debug(f"Coerced value {value!r} to string {result!r}")
        return result, metadata
    elif target_type == "array" and coerce_list_single_item:
        result, metadata = _coerce_single_to_list(value)
        if metadata is not None:
            logger.debug(f"Coerced single value {value!r} to list {result!r}")
        return result, metadata

    # No coercion possible
    return value, None


def _get_type_from_schema(schema: dict[str, Any]) -> str:
    """Extract type from JSON schema."""
    return schema.get("type", "string")  # type: ignore[no-any-return]


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
    target_type = _get_type_from_schema(schema)
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
    Coerce data to match schema types.

    Args:
        data: The data to coerce (dict or JSON string)
        schema: Pydantic BaseModel, JSON schema dict, or JSON schema string
        config: ParseConfig with coercion settings (uses default if None)

    Returns:
        Tuple of (coerced_data, list of CoercionMetadata)

    Raises:
        ConversionError: If schema is invalid
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
        if BaseModel is None:
            from .exceptions import ConversionError

            raise ConversionError("Pydantic is not installed")
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
