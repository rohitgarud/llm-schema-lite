"""Schema parser for parsing with schema validation and partial extraction."""

import json
import re
from typing import Any, cast

from ..coercion import ParseConfig, coerce_to_schema
from ..exceptions import ConversionError
from ..schema_enrichment import enrich_schema_with_enum_metadata
from .base import BaseParser
from .json_parser import JSONParser

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None  # type: ignore[assignment, misc]


# =============================================================================
# Module-level helper functions (for internal use and re-export)
# =============================================================================


def _get_json_schema(schema: "type[BaseModel] | dict[str, Any] | str") -> dict[str, Any]:
    """Convert schema input to JSON schema dict."""
    if BaseModel is not None and isinstance(schema, type) and issubclass(schema, BaseModel):
        json_schema = schema.model_json_schema()
        enrich_schema_with_enum_metadata(schema, json_schema)
        return json_schema
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, str):
        try:
            return cast(dict[str, Any], json.loads(schema))
        except json.JSONDecodeError as e:
            raise ConversionError(f"Invalid JSON schema string: {e}") from e
    raise ConversionError(
        f"Unsupported schema type: {type(schema)}. Expected Pydantic BaseModel, dict, or str."
    )


def _validate_field(value: Any, field_schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a single field against its schema."""
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError:
        # If jsonschema not available, assume valid
        return True, []

    try:
        format_checker = FormatChecker()
        validator = Draft202012Validator(field_schema, format_checker=format_checker)
        errors = list(validator.iter_errors(value))
        if not errors:
            return True, []
        error_messages = [err.message for err in errors]
        return False, error_messages
    except Exception:
        return True, []


def _build_result(
    data: dict[str, Any],
    schema: "type[BaseModel] | dict[str, Any] | str",
) -> "BaseModel | dict[str, Any]":
    """Build result as Pydantic model if schema is a BaseModel, otherwise return dict."""
    if BaseModel is not None and isinstance(schema, type) and issubclass(schema, BaseModel):
        # Use model_construct to bypass validation - allows partial fields with None
        return schema.model_construct(**data)
    return data


# =============================================================================
# Internal function for schema-aware parsing (called by core.loads)
# =============================================================================


def parse_with_schema(
    text: str,
    schema: "type[BaseModel] | dict[str, Any] | str",
    parse_config: ParseConfig,
) -> tuple["BaseModel | dict[str, Any]", dict[str, Any]]:
    """
    Parse LLM output with schema validation and optional partial extraction.

    This is an internal function called by core.loads() when a schema is provided.
    It implements the full parsing logic that was previously in parse_llm_output().

    Args:
        text: The text content to parse
        schema: Pydantic BaseModel, JSON schema dict, or JSON schema string
        parse_config: Configuration for parsing/coercion behavior (includes partial flag)

    Returns:
        Tuple of (parsed_result, metadata)
        - parsed_result: Model instance or dict with extracted fields
        - metadata: Dict with failed_fields info (empty dict if all fields valid)

    Raises:
        ConversionError: If partial=False and parsing fails, or if required fields fail
    """
    # Local import to avoid circular import
    from ..validators import JSONValidator

    # Create SchemaParser with the schema
    schema_parser = SchemaParser(schema=schema, parse_config=parse_config)

    if parse_config.partial:
        # Partial mode: use field-by-field extraction
        result_dict, failed_fields = schema_parser._parse_partial(text, repair=True)
        final_result = schema_parser.build_result(result_dict)
        return final_result, {"failed_fields": failed_fields}
    else:
        # Full validation mode: parse and validate
        result_dict = schema_parser.parse(text, repair=True)

        # Validate against full schema
        is_valid, errors = JSONValidator(schema).validate(result_dict, return_all_errors=True)
        if not is_valid:
            error_list = errors if errors else []
            raise ConversionError(f"Validation failed: {'; '.join(error_list)}")

        final_result = schema_parser.build_result(result_dict)
        return final_result, {}


class SchemaParser(BaseParser):
    """
    Schema-aware parser with support for partial extraction.

    Parses text content and validates/coerces against a provided schema.
    Supports partial extraction mode to extract valid fields even when
    some fields fail validation.
    """

    def __init__(
        self,
        schema: "type[BaseModel] | dict[str, Any] | str",
        parse_config: ParseConfig | None = None,
    ):
        """
        Initialize SchemaParser with schema.

        Args:
            schema: Pydantic BaseModel, JSON schema dict, or JSON schema string
            parse_config: Configuration for parsing/coercion behavior
        """
        self._schema = schema
        self._parse_config = parse_config or ParseConfig()
        self._json_parser = JSONParser()
        # Use module-level function
        self._json_schema = _get_json_schema(schema)

    def parse(self, text: str, repair: bool = True) -> dict[str, Any]:
        """
        Parse text content with schema validation.

        When partial mode is enabled in config, extracts valid fields even
        when some fields fail validation.

        Args:
            text: The text content to parse
            repair: Whether to attempt repair for malformed content

        Returns:
            Tuple of (parsed_result, metadata)
            - parsed_result: Dict with extracted fields
            - metadata: Dict with failed_fields info (empty if all fields valid)

        Raises:
            ConversionError: If parsing fails and partial mode is disabled,
                            or if required fields fail in partial mode
        """
        from ..validators import JSONValidator

        if self._parse_config.partial:
            # For partial mode, return just the dict (callers should use _parse_partial directly)
            return self._parse_partial(text, repair)[0]
        else:
            # Full validation mode - parse then validate
            parsed = self._json_parser.parse(text, repair)

            # Validate against full schema
            is_valid, errors = JSONValidator(self._schema).validate(parsed, return_all_errors=True)
            if not is_valid:
                error_list = errors if errors else []
                raise ConversionError(f"Validation failed: {'; '.join(error_list)}")

            return parsed

    def _parse_partial(self, text: str, repair: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Parse with partial extraction - extract valid fields even if some fail.

        Args:
            text: The text content to parse
            repair: Whether to attempt repair for malformed content

        Returns:
            Tuple of (result_dict, failed_fields_dict)

        Raises:
            ConversionError: If required fields are missing or fail validation
        """
        # First, parse the text to dict
        try:
            parsed = self._json_parser.parse(text, repair)
            data: dict[str, Any] = parsed if isinstance(parsed, dict) else {}
        except ConversionError:
            # Try to extract JSON from the string even if it's embedded
            data = self._extract_json_from_text(text)

        if not isinstance(data, dict):
            data = {}

        # Get required fields
        required_fields = set(self._json_schema.get("required", []))
        properties = self._json_schema.get("properties", {})

        # Track results
        result_dict: dict[str, Any] = {}
        failed_fields: dict[str, Any] = {}

        # Process each field in schema
        for field_name, field_schema in properties.items():
            # Check if field is present in data
            has_field = field_name in data

            # If field not present in data
            if not has_field:
                if field_name in required_fields:
                    raise ConversionError(f"Required field '{field_name}' is missing")
                # Optional field not provided - will default to None
                continue

            value = data[field_name]

            # Try to coerce and validate the field
            try:
                # Coerce the value - need to wrap in proper object schema
                if self._parse_config.allow_coercion:
                    field_object_schema = {
                        "type": "object",
                        "properties": {field_name: field_schema},
                    }
                    coerced_value, _ = coerce_to_schema(
                        {field_name: value}, field_object_schema, self._parse_config
                    )
                    field_value = coerced_value.get(field_name, value)
                else:
                    field_value = value

                # Validate the field
                is_valid, errors = self._validate_field(field_value, field_schema)
                if is_valid:
                    result_dict[field_name] = field_value
                else:
                    # Field failed validation
                    if field_name in required_fields:
                        raise ConversionError(
                            f"Required field '{field_name}' failed validation: "
                            f"{errors[0] if errors else 'unknown error'}"
                        )
                    # Optional field - set to None and track failure
                    result_dict[field_name] = None
                    failed_fields[field_name] = value
            except ConversionError:
                # Re-raise ConversionError (for required fields)
                raise
            except Exception as err:
                # Any other exception means validation/coercion failed
                if field_name in required_fields:
                    raise ConversionError(
                        f"Required field '{field_name}' failed validation"
                    ) from err
                result_dict[field_name] = None
                failed_fields[field_name] = value

        return result_dict, failed_fields

    def _validate_field(self, value: Any, field_schema: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate a single field against its schema."""
        # Delegate to module-level function
        return _validate_field(value, field_schema)

    def _extract_json_from_text(self, text: str) -> dict[str, Any]:
        """Extract JSON object from text that may contain extra content."""
        # Try to find JSON object in the text
        json_match = re.search(r"\{[^{}]*\}", text)
        if json_match:
            try:
                parsed: dict[str, Any] = JSONParser().parse(json_match.group(), repair=True)
                return parsed
            except ConversionError:
                pass

        # If no JSON found, return empty dict
        return {}

    def build_result(self, data: dict[str, Any]) -> "BaseModel | dict[str, Any]":
        """
        Build result as Pydantic model if schema is a BaseModel, otherwise return dict.

        Args:
            data: The parsed data dictionary

        Returns:
            Pydantic model instance or dict
        """
        # Delegate to module-level function
        return _build_result(data, self._schema)
