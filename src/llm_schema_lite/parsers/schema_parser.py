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

    def _parse_to_dict(self, text: str, repair: bool, *, rescue_embedded: bool) -> Any:
        """Sole text->dict entry point for both SchemaParser routes.

        Parses via self._json_parser, optionally rescues embedded JSON on failure, then
        normalizes required-marker keys via _normalize_marker_keys exactly once.

        `rescue_embedded` PRESERVES an existing asymmetry between SchemaParser.parse's
        non-partial branch and _parse_partial rather than introducing a new one: the
        non-partial route lets a ConversionError from self._json_parser.parse
        propagate; the partial route catches it and falls back to
        self._extract_json_from_text. Folding both into one unconditional behaviour
        would silently change the non-partial route's failure mode for garbage input
        from a parse error into a validation error, which is out of scope for
        lsl-2026-09-04-008. _extract_json_from_text itself is unchanged.

        Args:
            text: The text content to parse.
            repair: Whether to attempt repair for malformed content, forwarded to
                self._json_parser.parse.
            rescue_embedded: True for the _parse_partial route (catches ConversionError
                and retries via self._extract_json_from_text); False for the
                SchemaParser.parse non-partial route (lets ConversionError propagate).

        Returns:
            Whatever self._json_parser.parse (or the embedded-JSON rescue) produced, run
            through _normalize_marker_keys if it is a dict. Non-dict results are
            returned unchanged -- both callers have their own non-dict handling.

        Raises:
            ConversionError: propagates from self._json_parser.parse when
                rescue_embedded is False, or from the embedded-JSON rescue when
                rescue_embedded is True and the rescue also fails.
        """
        if rescue_embedded:
            try:
                parsed = self._json_parser.parse(text, repair)
            except ConversionError:
                parsed = self._extract_json_from_text(text)
        else:
            parsed = self._json_parser.parse(text, repair)

        if isinstance(parsed, dict):
            return self._normalize_marker_keys(parsed)
        return parsed

    def _normalize_marker_keys(self, data: dict[str, Any]) -> dict[str, Any]:
        """Map trailing-marker reply keys onto their schema property names (top level).

        Scope: self._json_schema["properties"] at the TOP LEVEL only. Nested marked
        keys (e.g. {"person": {"name*": "x"}}) are deliberately not walked -- reaching
        them would need $ref/anyOf/items traversal, which is a recorded follow-up and
        not part of lsl-2026-09-04-008.

        Uses self._parse_config.strip_required_marker as the marker. An empty marker,
        or an empty properties dict, is a no-op escape hatch.

        Rule order -- verbatim match wins FIRST, so a schema that legitimately declares
        a property literally named "name*" is never disturbed, and a collision between
        a verbatim key and its own marked form always keeps the verbatim key's value
        (last-writer-wins would depend on dict iteration order and is rejected):

          1. key already in properties -> keep as-is (checked FIRST).
          2. key ends with marker AND stripping it yields a non-empty name that IS a
             schema property AND that stripped name is NOT already a key in `data` ->
             remap to the stripped name.
          3. otherwise -> keep the key unchanged; it is left for downstream to drop as
             the unknown key it is. Stripping must never invent a schema key that was
             not already a property.

        Args:
            data: The parsed (dict) reply, pre-filtering, in whatever key order the LM
                or self._json_parser produced.

        Returns:
            A new dict with the same values and remapped keys. Never mutates `data`.
        """
        marker = self._parse_config.strip_required_marker
        properties = self._json_schema.get("properties", {})
        if not marker or not properties:
            return data

        result: dict[str, Any] = {}
        for key, value in data.items():
            if key in properties:
                result[key] = value
                continue
            if key.endswith(marker):
                stripped = key[: -len(marker)]
                if stripped and stripped in properties and stripped not in data:
                    result[stripped] = value
                    continue
            result[key] = value
        return result

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
            parsed: dict[str, Any] = self._parse_to_dict(text, repair, rescue_embedded=False)

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
        # First, parse the text to dict (marker-key normalization happens inside)
        data: dict[str, Any] = self._parse_to_dict(text, repair, rescue_embedded=True)

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
