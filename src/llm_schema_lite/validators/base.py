"""Base validator interface."""

import json
from abc import ABC, abstractmethod
from typing import Any, cast

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None  # type: ignore[assignment, misc]

from ..exceptions import UnsupportedModelError


class BaseValidator(ABC):
    """
    Abstract base class for schema validators.

    Subclasses implement parse_data() for input parsing, validate() for
    format-specific validation, and _format_validation_error() for
    format-specific error formatting (e.g. jsonschema for JSON/YAML).
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

        if BaseModel is not None and isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_json_schema()
        if isinstance(schema, dict):
            return schema
        if isinstance(schema, str):
            try:
                return cast(dict[str, Any], json.loads(schema))
            except json.JSONDecodeError as e:
                from ..exceptions import ConversionError

                raise ConversionError(f"Invalid JSON schema string: {e}") from e

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

    @abstractmethod
    def _format_validation_error(self, error: Any) -> str:
        """
        Format a validation error into a human-readable message for LLMs.

        Subclasses implement this for their validation backend (e.g. jsonschema
        for JSON/YAML, or custom formatting for TypeScript, SQL, etc.).
        """
        pass

    @abstractmethod
    def validate(
        self,
        data: dict[str, Any] | str | list[Any] | int | float | bool | None,
        return_all_errors: bool = True,
    ) -> tuple[bool, list[str] | None]:
        """
        Validate data against the schema.

        Args:
            data: Data to validate (string or already-parsed structure).
            return_all_errors: If True, return all validation errors; if False, only the first.

        Returns:
            Tuple of (is_valid, errors). If valid, errors is None.
            If invalid, errors is a list of strings.

        Raises:
            ValidationError: If validation fails or the schema is invalid.
        """
        pass
