"""Core functionality for LLM Schema Lite."""

import json
from typing import Any, Literal

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None  # type: ignore[assignment, misc]

try:
    import jsonschema
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError:
    jsonschema = None
    Draft202012Validator = None
    FormatChecker = None

from .exceptions import ConversionError, UnsupportedModelError, ValidationError
from .formatters import JSONishFormatter, TypeScriptFormatter, YAMLFormatter
from .formatters.base import BaseFormatter
from .parsers import BaseParser, JSONParser, YAMLParser


class SchemaLite:
    """
    Simplified schema representation with multiple output formats.

    This class provides a unified interface for converting schemas to different
    string representations.
    """

    def __init__(
        self,
        formatter: BaseFormatter,
        original_schema: dict[str, Any],
    ):
        """
        Initialize SchemaLite with formatter and original schema.

        Args:
            formatter: The formatter instance for this schema.
            original_schema: The original JSON schema.
        """
        self._formatter = formatter
        self._original_schema = original_schema
        self._string_representation: str | None = None
        self._original_token_count: int | None = None
        self._simplified_token_count: int | None = None

    def to_string(self) -> str:
        """
        Get the simplified schema as a formatted string.

        Uses the processed data from the formatter.

        Returns:
            String representation of the schema.
        """
        if self._string_representation is None:
            self._string_representation = self._formatter.transform_schema()

        return self._string_representation

    def token_count(self, encoding: str = "cl100k_base") -> int:
        """
        Estimate token count for the simplified schema.

        Args:
            encoding: Tokenizer encoding to use (default: cl100k_base for GPT-4).

        Returns:
            Estimated token count.

        Raises:
            ImportError: If tiktoken is not installed.
        """
        # Delegate to formatter if it has token_count method
        if hasattr(self._formatter, "token_count") and callable(self._formatter.token_count):
            return self._formatter.token_count(encoding)  # type: ignore[no-any-return]

        # Fallback to default implementation
        try:
            import tiktoken

            enc = tiktoken.get_encoding(encoding)
            return len(enc.encode(self.to_string()))
        except ImportError as e:
            raise ImportError(
                "tiktoken is required for token counting. Install it with: pip install tiktoken"
            ) from e

    def compare_tokens(
        self,
        original_schema: dict[str, Any] | None = None,
        simplified_schema: str | None = None,
        encoding: str = "cl100k_base",
    ) -> dict[str, Any]:
        """
        Compare token counts between original and simplified schemas.

        Args:
            original_schema: Original schema dict (uses stored if not provided).
            simplified_schema: Simplified schema string (uses generated if not provided).
            encoding: Tokenizer encoding to use.

        Returns:
            Dictionary with original, simplified, and reduction metrics.
        """
        # Delegate to formatter if it has compare_tokens method
        if hasattr(self._formatter, "compare_tokens") and callable(self._formatter.compare_tokens):
            return self._formatter.compare_tokens(  # type: ignore[no-any-return]
                original_schema or self._original_schema, simplified_schema, encoding
            )

        # Fallback to default implementation
        try:
            import tiktoken

            enc = tiktoken.get_encoding(encoding)
            schema_to_compare = original_schema or self._original_schema

            original_str = json.dumps(schema_to_compare)
            simplified_str = simplified_schema or self.to_string()

            if self._original_token_count is None:
                self._original_token_count = len(enc.encode(original_str))
            if self._simplified_token_count is None:
                self._simplified_token_count = len(enc.encode(simplified_str))
            reduction_percent = (
                (self._original_token_count - self._simplified_token_count)
                / self._original_token_count
                * 100
            )

            return {
                "original_tokens": self._original_token_count,
                "simplified_tokens": self._simplified_token_count,
                "tokens_saved": self._original_token_count - self._simplified_token_count,
                "reduction_percent": round(reduction_percent, 2),
            }
        except ImportError as e:
            raise ImportError(
                "tiktoken is required for token comparison. Install it with: pip install tiktoken"
            ) from e

    def __str__(self) -> str:
        """String representation of the schema."""
        return self.to_string()

    def __repr__(self) -> str:
        """Developer representation of the SchemaLite object."""
        formatter_name = type(self._formatter).__name__
        return f"SchemaLite(formatter={formatter_name})"


def simplify_schema(
    model: type["BaseModel"] | dict[str, Any] | str,
    include_metadata: bool = True,
    format_type: Literal["jsonish", "typescript", "yaml"] = "jsonish",
) -> SchemaLite:
    """
    Convert Pydantic model, JSON schema dict, or JSON schema string to simplified schema.

    Args:
        model: Pydantic BaseModel class, JSON schema dict, or JSON schema string.
        include_metadata: Include validation rules as inline comments.
        format_type: Output format preference:
            - 'jsonish': JSONish/BAML-like format with inline comments (default)
            - 'typescript': TypeScript interface format
            - 'yaml': YAML-style format with Python type hints

    Returns:
        SchemaLite object with conversion methods.

    Raises:
        UnsupportedModelError: If the model type is not supported.
        ConversionError: If schema conversion fails.

    Examples:
        >>> from pydantic import BaseModel
        >>> from llm_schema_lite import simplify_schema
        >>>
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        ...
        >>> # JSONish format (default)
        >>> schema = simplify_schema(User)
        >>> print(schema.to_string())
        {
         name: string,
         age: int
        }

        >>> # From JSON schema dict
        >>> schema_dict = {
        ...     "type": "object",
        ...     "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}
        ... }
        >>> schema = simplify_schema(schema_dict)
        >>> print(schema.to_string())
        {
         name: string,
         age: int
        }

        >>> # From JSON schema string
        >>> schema_string = (
        ...     '{"type": "object", "properties": {"name": {"type": "string"}, '
        ...     '"age": {"type": "integer"}}}'
        ... )
        >>> schema = simplify_schema(schema_string)
        >>> print(schema.to_string())
        {
         name: string,
         age: int
        }

        >>> # TypeScript format
        >>> ts_schema = simplify_schema(User, format_type="typescript")
        >>> print(ts_schema.to_string())
        interface Schema {
          name: string;
          age: number;
        }

        >>> # YAML format
        >>> yaml_schema = simplify_schema(User, format_type="yaml")
        >>> print(yaml_schema.to_string())
        name: str
        age: int
    """
    # Handle BaseModel
    if BaseModel is not None and isinstance(model, type) and issubclass(model, BaseModel):
        try:
            original_schema = model.model_json_schema()
        except Exception as e:
            raise ConversionError(f"Failed to extract JSON schema from model: {e}") from e
    # Handle dict (already a JSON schema)
    elif isinstance(model, dict):
        original_schema = model
    # Handle string (JSON schema string)
    elif isinstance(model, str):
        try:
            original_schema = json.loads(model)
        except json.JSONDecodeError as e:
            raise ConversionError(f"Failed to parse JSON schema string: {e}") from e
    else:
        raise UnsupportedModelError(
            f"Unsupported model type: {type(model)}. Expected Pydantic BaseModel, dict, or str."
        )

    # Select formatter based on format_type
    formatter: BaseFormatter
    if format_type == "jsonish":
        formatter = JSONishFormatter(original_schema, include_metadata=include_metadata)
    elif format_type == "typescript":
        formatter = TypeScriptFormatter(original_schema, include_metadata=include_metadata)
    elif format_type == "yaml":
        formatter = YAMLFormatter(original_schema, include_metadata=include_metadata)
    else:
        raise ValueError(
            f"Unsupported format_type: {format_type}. "
            f"Supported formats: 'jsonish', 'typescript', 'yaml'"
        )

    # Let the formatter handle all processing logic
    try:
        return SchemaLite(
            formatter=formatter,
            original_schema=original_schema,
        )
    except Exception as e:
        raise ConversionError(f"Failed to convert schema: {e}") from e


def loads(
    text: str,
    mode: Literal["json", "yaml"] = "json",
    repair: bool = True,
) -> dict[str, Any]:
    """
    Parse structured text (JSON or YAML) with robust error handling and content extraction.

    This function provides a unified interface for parsing JSON and YAML content with
    automatic repair, smart content extraction, and fallback mechanisms. It automatically
    handles various LLM response formats including markdown code blocks, embedded JSON/YAML,
    and text with explanatory content.

    Args:
        text: The text content to parse
        mode: The parsing mode - "json" or "yaml"
        repair: Whether to attempt repair for malformed content

    Returns:
        Parsed dictionary content

    Raises:
        ConversionError: If parsing fails and repair is disabled or unsuccessful

    Examples:
        >>> # Parse JSON with automatic extraction
        >>> data = loads('{"name": "John", "age": 30}')

        >>> # Parse JSON from markdown with extra text
        >>> data = loads('Here is the result: ```json\\n{"name": "Jane", "age": 25}\\n```')

        >>> # Parse JSON embedded in explanatory text
        >>> data = loads('The user data is: {"name": "Bob", "age": 35} and that\'s all.')

        >>> # Parse YAML with automatic extraction
        >>> data = loads('```yaml\\nname: Alice\\nage: 28\\n```', mode="yaml")

        >>> # Parse with repair disabled
        >>> data = loads('{"name": "John"}', repair=False)
    """
    if not text or not text.strip():
        raise ConversionError("Empty or whitespace-only text provided")

    # Select parser based on mode
    parser: BaseParser
    if mode == "json":
        parser = JSONParser()
    elif mode == "yaml":
        parser = YAMLParser()
    else:
        raise ConversionError(f"Unsupported mode: {mode}. Supported modes: 'json', 'yaml'")

    # Delegate parsing to the selected parser
    return parser.parse(text.strip(), repair)


def validate(
    schema: BaseModel | dict[str, Any] | str,
    data: dict[str, Any] | str | list[Any] | int | float | bool | None,
    mode: Literal["json", "yaml", "auto"] = "auto",
    return_all_errors: bool = True,
) -> tuple[bool, list[str] | None]:
    """
    Validate data against a schema using jsonschema library.

    This function uses the jsonschema library (Draft 2020-12) for comprehensive
    JSON Schema validation, including format checking. Supports both JSON and YAML
    data formats. Returns detailed error messages that LLMs can use to improve output.

    Args:
        schema: Pydantic BaseModel class, JSON schema dict, or JSON schema string
        data: Data to validate (can be dict, list, string, number, boolean, null,
              JSON string, or YAML string)
        mode: Parsing mode for string data - "json", "yaml", or "auto" (default).
              In "auto" mode, tries JSON first, then YAML if JSON fails.
        return_all_errors: If True, return all validation errors. If False, return
              only the first error. Default is True.

    Returns:
        tuple[bool, list[str] | None]: A tuple of (is_valid, errors).
            - If valid: (True, None)
            - If invalid and return_all_errors=True: (False, [error1, error2, ...])
            - If invalid and return_all_errors=False: (False, [first_error])

    Raises:
        UnsupportedModelError: If schema type is not supported
        ConversionError: If data cannot be parsed
        ValidationError: If jsonschema library is not installed or schema is invalid

    Example:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> # Valid data
        >>> is_valid, errors = validate(User, {"name": "John", "age": 30})
        >>> print(is_valid)  # True
        >>> print(errors)  # None
        >>> # Invalid data with all errors
        >>> is_valid, errors = validate(User, {})
        >>> print(is_valid)  # False
        >>> print(errors)  # ["'name' is a required property", "'age' is a required property"]
        >>> # Invalid data with first error only
        >>> is_valid, errors = validate(User, {}, return_all_errors=False)
        >>> print(errors)  # ["'name' is a required property"]
        >>> # JSON/YAML modes
        >>> validate(User, '{"name": "John", "age": 30}', mode="json")
        (True, None)
        >>> validate(User, "name: John\\nage: 30", mode="yaml")
        (True, None)
    """
    # Check if jsonschema is available
    if jsonschema is None:
        raise ValidationError(
            "jsonschema library is required for validation. Install it with: pip install jsonschema"
        )

    # Parse data if it's a string
    if isinstance(data, str):
        data_str = data
        # Determine parsing strategy
        if mode == "json":
            # Try JSON parsing only
            if data_str.strip().startswith(("{", "[")):
                try:
                    data = loads(data_str, mode="json")
                except ConversionError:
                    # If it fails to parse, treat it as a plain string value
                    pass
        elif mode == "yaml":
            # Try YAML parsing
            try:
                data = loads(data_str, mode="yaml")
            except ConversionError:
                # If it fails to parse, treat it as a plain string value
                pass
        else:  # mode == "auto"
            # Try JSON first (faster and more common)
            if data_str.strip().startswith(("{", "[")):
                try:
                    data = loads(data_str, mode="json")
                except ConversionError:
                    # Try YAML as fallback
                    try:
                        data = loads(data_str, mode="yaml")
                    except ConversionError:
                        # If both fail, treat it as a plain string value
                        pass
            else:
                # Doesn't look like JSON, try YAML
                try:
                    data = loads(data_str, mode="yaml")
                except ConversionError:
                    # If it fails to parse, treat it as a plain string value
                    pass

    # Get JSON schema dict
    json_schema: dict[str, Any]

    # Handle Pydantic models
    if BaseModel is not None and isinstance(schema, type) and issubclass(schema, BaseModel):
        json_schema = schema.model_json_schema()  # type: ignore[union-attr]
    # Handle JSON schema dict
    elif isinstance(schema, dict):
        json_schema = schema
    # Handle JSON schema string
    elif isinstance(schema, str):
        try:
            json_schema = json.loads(schema)
        except json.JSONDecodeError as e:
            raise ConversionError(f"Invalid JSON schema string: {e}") from e
    else:
        raise UnsupportedModelError(
            f"Unsupported schema type: {type(schema)}. Expected Pydantic BaseModel, dict, or str."
        )

    # Validate using jsonschema
    try:
        # Check if schema itself is valid
        Draft202012Validator.check_schema(json_schema)

        # Create validator with format checker for additional validation
        format_checker = FormatChecker()
        validator = Draft202012Validator(json_schema, format_checker=format_checker)

        # Collect all validation errors
        errors = list(validator.iter_errors(data))

        if not errors:
            # Data is valid
            return (True, None)

        # Format error messages
        error_messages = [_format_validation_error(err) for err in errors]

        if return_all_errors:
            # Return all errors
            return (False, error_messages)
        else:
            # Return only the first error
            return (False, [error_messages[0]])

    except jsonschema.exceptions.SchemaError as e:
        # Schema itself is invalid
        raise ValidationError("Invalid JSON schema") from e
    except Exception as e:
        # Other errors (not validation errors)
        raise ValidationError(f"Validation failed: {e}") from e


def _format_validation_error(error: Any) -> str:
    """
    Format a jsonschema ValidationError into a human-readable message for LLMs.

    Args:
        error: jsonschema ValidationError instance

    Returns:
        str: Formatted error message with path, issue, and context
    """
    # Build the path to the error
    path_parts = list(error.absolute_path)
    if path_parts:
        path_str = "." + ".".join(str(p) for p in path_parts)
    else:
        path_str = " (root)"

    # Get the error message
    message = error.message

    # Add context about the failing value if available
    if hasattr(error, "instance"):
        instance = error.instance
        if isinstance(instance, dict | list):
            # For complex types, show the type
            instance_str = f" (got {type(instance).__name__})"
        elif instance is None:
            instance_str = " (got null)"
        elif isinstance(instance, str):
            # Truncate long strings
            if len(instance) > 50:
                instance_str = f" (got '{instance[:47]}...')"
            else:
                instance_str = f" (got '{instance}')"
        else:
            instance_str = f" (got {instance})"
    else:
        instance_str = ""

    # Add schema context if available
    schema_info = ""
    if hasattr(error, "validator") and hasattr(error, "validator_value"):
        validator = error.validator
        validator_value = error.validator_value

        if validator == "required":
            # For required fields, list what's missing
            schema_info = f" - Required properties: {validator_value}"
        elif validator == "type":
            # For type errors, show expected type
            schema_info = f" - Expected type: {validator_value}"
        elif validator in ("minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems"):
            # For constraint errors, show the constraint
            schema_info = f" - Constraint: {validator} = {validator_value}"
        elif validator == "pattern":
            # For pattern errors, show the pattern
            schema_info = f" - Expected pattern: {validator_value}"
        elif validator == "enum":
            # For enum errors, show allowed values
            schema_info = f" - Allowed values: {validator_value}"

    # Combine all parts
    full_message = f"Validation error at '{path_str}': {message}{instance_str}{schema_info}"

    return full_message
