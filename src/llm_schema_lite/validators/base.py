"""Base validator interface."""

from abc import ABC, abstractmethod
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import BaseModel

from ..exceptions import UnsupportedModelError, ValidationError
from ..parsers.schema_parser import _get_json_schema
from .enum_aliases import normalize_enum_aliases


class BaseValidator(ABC):
    """
    Abstract base class for schema validators.

    Subclasses implement parse_data() for format-specific input parsing; validate()
    and its jsonschema (Draft 2020-12) error formatting are shared.
    """

    def __init__(
        self,
        schema: type[Any] | dict[str, Any] | str,
    ) -> None:
        """
        Initialize the validator with a schema.

        Args:
            schema: Pydantic BaseModel class, JSON schema dict, or JSON schema string.
        """
        self._schema_input = schema
        self._json_schema = self._parse_schema()

    def _parse_schema(self) -> dict[str, Any]:
        """Convert schema input to JSON schema dict."""
        schema = self._schema_input
        if isinstance(schema, dict | str) or (
            isinstance(schema, type) and issubclass(schema, BaseModel)
        ):
            return _get_json_schema(schema)
        raise UnsupportedModelError(
            f"Unsupported schema type: {type(schema)}. Expected Pydantic BaseModel, dict, or str."
        )

    @abstractmethod
    def parse_data(
        self,
        data: dict[str, Any] | str | list[Any] | int | float | bool | None,
    ) -> dict[str, Any] | str | list[Any] | int | float | bool | None:
        """
        Parse string data into a structure suitable for jsonschema validation.

        Subclasses must implement this to handle format-specific parsing (JSON, YAML, etc.).
        Non-string data should be returned as-is.

        Args:
            data: Raw data (string or already-parsed structure).

        Returns:
            Parsed data ready for jsonschema validation.
        """
        pass

    def _format_validation_error(self, error: Any) -> str:
        """Format a jsonschema ValidationError into a human-readable message for LLMs."""
        path_parts = list(error.absolute_path)
        if path_parts:
            path_str = "." + ".".join(str(p) for p in path_parts)
        else:
            path_str = " (root)"
        message = error.message
        if hasattr(error, "instance"):
            instance = error.instance
            if isinstance(instance, dict | list):
                instance_str = f" (got {type(instance).__name__})"
            elif instance is None:
                instance_str = " (got null)"
            elif isinstance(instance, str):
                instance_str = (
                    f" (got '{instance[:47]}...')" if len(instance) > 50 else f" (got '{instance}')"
                )
            else:
                instance_str = f" (got {instance})"
        else:
            instance_str = ""
        schema_info = ""
        if hasattr(error, "validator") and hasattr(error, "validator_value"):
            validator = error.validator
            validator_value = error.validator_value
            if validator == "required":
                schema_info = f" - Required properties: {validator_value}"
            elif validator == "type":
                schema_info = f" - Expected type: {validator_value}"
            elif validator in (
                "minimum",
                "maximum",
                "minLength",
                "maxLength",
                "minItems",
                "maxItems",
            ):
                schema_info = f" - Constraint: {validator} = {validator_value}"
            elif validator == "pattern":
                schema_info = f" - Expected pattern: {validator_value}"
            elif validator == "enum":
                schema_info = f" - Allowed values: {validator_value}"
        return f"Validation error at '{path_str}': {message}{instance_str}{schema_info}"

    def validate(
        self,
        data: dict[str, Any] | str | list[Any] | int | float | bool | None,
        return_all_errors: bool = True,
    ) -> tuple[bool, list[str] | None]:
        """
        Validate data against the schema using jsonschema (Draft 2020-12).

        Args:
            data: Data to validate (string or already-parsed structure).
            return_all_errors: If True, return all validation errors; if False, only the first.

        Returns:
            Tuple of (is_valid, errors). If valid, errors is None.
            If invalid, errors is a list of strings.

        Raises:
            ValidationError: If validation fails or the schema is invalid.
        """
        parsed = self.parse_data(data)
        json_schema = self._json_schema
        normalized = normalize_enum_aliases(parsed, json_schema)
        try:
            Draft202012Validator.check_schema(json_schema)
            format_checker = FormatChecker()
            validator = Draft202012Validator(json_schema, format_checker=format_checker)
            errors = list(validator.iter_errors(normalized))
            if not errors:
                return (True, None)
            error_messages = [self._format_validation_error(err) for err in errors]
            if return_all_errors:
                return (False, error_messages)
            return (False, [error_messages[0]])
        except jsonschema.exceptions.SchemaError as e:
            raise ValidationError("Invalid JSON schema") from e
        except Exception as e:
            raise ValidationError(f"Validation failed: {e}") from e
