"""Core functionality for LLM Schema Lite."""

import json
from typing import Any, Literal, cast

from pydantic import BaseModel

from .coercion import CoercionMetadata, ParseConfig, coerce_to_schema
from .exceptions import ConversionError, UnsupportedModelError
from .formatters import FormatterConfig, JSONishFormatter, TypeScriptFormatter, YAMLFormatter
from .formatters.base import BaseFormatter
from .parsers import BaseParser, JSONParser, YAMLParser
from .parsers.schema_parser import parse_with_schema
from .schema_enrichment import enrich_schema_with_enum_metadata
from .validators import JSONValidator, YAMLValidator


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
    config: "FormatterConfig | None" = None,
    include_metadata: bool | None = None,
    format_type: Literal["jsonish", "typescript", "yaml"] = "jsonish",
) -> SchemaLite:
    """
    Convert Pydantic model, JSON schema dict, or JSON schema string to simplified schema.

    Args:
        model: Pydantic BaseModel class, JSON schema dict, or JSON schema string.
        config: FormatterConfig for customizing formatter behavior.
        include_metadata: Deprecated. Use config.include_metadata instead.
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
    if isinstance(model, type) and issubclass(model, BaseModel):
        try:
            original_schema = model.model_json_schema()
            enrich_schema_with_enum_metadata(model, original_schema)
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
    formatter_kwargs: dict[str, Any] = {"schema": original_schema, "config": config}
    # Handle backward compatibility for include_metadata parameter
    if include_metadata is not None:
        formatter_kwargs["include_metadata"] = include_metadata

    if format_type == "jsonish":
        formatter = JSONishFormatter(**formatter_kwargs)
    elif format_type == "typescript":
        formatter = TypeScriptFormatter(**formatter_kwargs)
    elif format_type == "yaml":
        formatter = YAMLFormatter(**formatter_kwargs)
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
    parse_config: ParseConfig | None = None,
    schema: type[BaseModel] | dict[str, Any] | str | None = None,
) -> dict[str, Any] | tuple[dict[str, Any] | BaseModel, dict[str, Any]]:
    """
    Parse structured text (JSON or YAML) with robust error handling and content extraction.

    This function provides a unified interface for parsing JSON and YAML content with
    automatic repair, smart content extraction, and fallback mechanisms. It automatically
    handles various LLM response formats including markdown code blocks, embedded JSON/YAML,
    and text with explanatory content.

    When parse_config.partial=True and schema is provided, enables partial extraction mode
    to extract valid fields even when some fields fail validation.

    Args:
        text: The text content to parse
        mode: The parsing mode - "json" or "yaml"
        repair: Whether to attempt repair for malformed content
        parse_config: Configuration for coercion behavior (optional, includes partial flag)
        schema: Pydantic BaseModel, JSON schema dict, or JSON schema string

    Returns:
        Tuple of (parsed_result, metadata)
        - parsed_result: Parsed dictionary or model instance
        - metadata: Dict with failed_fields info (empty if not partial or all fields valid)

    Raises:
        ConversionError: If parsing fails and repair is disabled or unsuccessful,
                        or if partial=True and required fields fail validation

    Examples:
        >>> # Parse JSON with automatic extraction
        >>> data, metadata = loads('{"name": "John", "age": 30}')

        >>> # Parse JSON from markdown with extra text
        >>> data, metadata = loads(
        ...     'Here is the result: ```json\\n{"name": "Jane", "age": 25}\\n```'
        ... )

        >>> # Parse with partial extraction using ParseConfig
        >>> from pydantic import BaseModel
        >>> from llm_schema_lite import ParseConfig
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        ...     email: str | None = None
        >>> data, metadata = loads(
        ...     '{"name": "Alice", "age": "invalid", "email": "alice@example.com"}',
        ...     parse_config=ParseConfig(partial=True),
        ...     schema=Person
        ... )
    """
    if not text or not text.strip():
        raise ConversionError("Empty or whitespace-only text provided")

    # Create parse_config if not provided
    if parse_config is None:
        parse_config = ParseConfig()

    # If a schema is provided, delegate to schema-aware parsing (via parsers)
    if schema is not None:
        return parse_with_schema(text.strip(), schema, parse_config)

    # Select parser based on mode
    parser: BaseParser
    if mode == "json":
        parser = JSONParser()
    elif mode == "yaml":
        parser = YAMLParser()
    else:
        raise ConversionError(f"Unsupported mode: {mode}. Supported modes: 'json', 'yaml'")

    # Delegate parsing to the selected parser
    result = parser.parse(text.strip(), repair)
    # Return just the dict for backward compatibility
    return result


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

    You can also use validator classes directly for reuse or format-specific control:
        >>> from llm_schema_lite import JSONValidator, YAMLValidator
        >>> json_validator = JSONValidator(User)
        >>> is_valid, errors = json_validator.validate('{"name": "Jane", "age": 25}')

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
    schema_arg = cast(type[Any] | dict[str, Any] | str, schema)
    if mode == "json":
        return JSONValidator(schema_arg).validate(data, return_all_errors=return_all_errors)
    if mode == "yaml":
        return YAMLValidator(schema_arg).validate(data, return_all_errors=return_all_errors)
    # mode == "auto": try JSON first when data looks like JSON, else YAML
    if isinstance(data, str) and data.strip().startswith(("{", "[")):
        return JSONValidator(schema_arg).validate(data, return_all_errors=return_all_errors)
    return YAMLValidator(schema_arg).validate(data, return_all_errors=return_all_errors)


def coerce(
    data: dict[str, Any] | str | list[Any] | int | float | bool | None,
    schema: type[BaseModel] | dict[str, Any] | str,
    config: ParseConfig | None = None,
) -> tuple[dict[str, Any], list[CoercionMetadata]]:
    """
    Coerce data to match schema types.

    This function converts input data to match the expected types defined in a schema.
    It handles various type coercions like string to int, string to bool, etc.

    Args:
        data: Data to coerce (can be dict, list, string, number, boolean, null,
              JSON string, or YAML string)
        schema: Pydantic BaseModel class, JSON schema dict, or JSON schema string
        config: ParseConfig with coercion settings (optional, uses default if None)

    Returns:
        Tuple of (coerced_data, list of CoercionMetadata)

    Raises:
        ConversionError: If coercion fails or schema is invalid

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
    return coerce_to_schema(data, schema, config)
