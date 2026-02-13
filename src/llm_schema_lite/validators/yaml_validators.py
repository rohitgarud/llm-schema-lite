"""YAML schema validator."""

from typing import Any

try:
    import jsonschema
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError:
    jsonschema = None
    Draft202012Validator = None
    FormatChecker = None

from ..exceptions import ConversionError, ValidationError
from ..parsers import YAMLParser
from .base import BaseValidator


class YAMLValidator(BaseValidator):
    """
    Validator for data that is or can be parsed as YAML.

    Uses YAMLParser for string inputs. Supports repair for malformed YAML.
    """

    def __init__(
        self,
        schema: type[Any] | dict[str, Any] | str,
        repair: bool = True,
    ) -> None:
        """
        Initialize the YAML validator.

        Args:
            schema: Pydantic BaseModel class, JSON schema dict, or JSON schema string.
            repair: Whether to attempt repair for malformed YAML. Default is True.
        """
        super().__init__(schema)
        self._repair = repair
        self._parser = YAMLParser()

    def parse_data(
        self,
        data: dict[str, Any] | str | list[Any] | int | float | bool | None,
    ) -> dict[str, Any] | str | list[Any] | int | float | bool | None:
        """
        Parse string data as YAML; on failure treat as plain value.

        Args:
            data: Raw data. If string, parsed as YAML.

        Returns:
            Parsed data for validation, or original data if not a string / parse fails.
        """
        if not isinstance(data, str):
            return data
        try:
            return self._parser.parse(data.strip(), repair=self._repair)
        except ConversionError:
            return data

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
        """Validate data against the schema using jsonschema (Draft 2020-12)."""
        if jsonschema is None:
            raise ValidationError(
                "jsonschema library is required for validation. "
                "Install it with: pip install jsonschema"
            )
        parsed = self.parse_data(data)
        json_schema = self._json_schema
        try:
            Draft202012Validator.check_schema(json_schema)
            format_checker = FormatChecker()
            validator = Draft202012Validator(json_schema, format_checker=format_checker)
            errors = list(validator.iter_errors(parsed))
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
