"""Core API for schema-lite."""

import json
import re
from typing import Any, Literal

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

try:
    import json_repair
except ImportError:
    json_repair = None  # type: ignore[assignment, unused-ignore]

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
from .formatters import BaseFormatter, JSONishFormatter, TypeScriptFormatter, YAMLFormatter


class SchemaLite:
    """
    A simplified schema representation with multiple output formats.

    This class wraps the processed schema and provides convenient methods
    for converting to different formats (JSON, YAML, string, etc.).
    """

    def __init__(
        self,
        processed_data: dict[str, Any],
        formatter: BaseFormatter,
        original_schema: dict[str, Any],
    ):
        """
        Initialize SchemaLite.

        Args:
            processed_data: The processed schema data.
            formatter: The formatter used to process the schema.
            original_schema: The original Pydantic JSON schema.
        """
        self._data = processed_data
        self._formatter = formatter
        self._original_schema = original_schema
        self._string_representation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Get the simplified schema as a dictionary.

        Returns:
            Dictionary representation of the simplified schema.
        """
        return self._data

    def to_json(self, indent: int = 2) -> str:
        """
        Get the simplified schema as JSON string.

        Args:
            indent: Number of spaces for indentation.

        Returns:
            JSON string representation.
        """
        return json.dumps(self._data, indent=indent)

    def to_string(self) -> str:
        """
        Get the simplified schema as a formatted string.

        This uses the formatter's transform_schema method to produce
        a human-readable string representation optimized for LLMs.

        Returns:
            String representation of the schema.
        """
        if self._string_representation is None:
            self._string_representation = self._formatter.transform_schema()
        return self._string_representation

    def to_yaml(self, default_flow_style: bool = False) -> str:
        """
        Get the simplified schema as YAML string.

        Args:
            default_flow_style: Whether to use flow style for YAML.

        Returns:
            YAML string representation.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml

            return str(
                yaml.dump(self._data, default_flow_style=default_flow_style, sort_keys=False)
            )
        except ImportError as e:
            raise ImportError(
                "PyYAML is required for YAML output. Install it with: pip install PyYAML"
            ) from e

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
        try:
            import tiktoken

            enc = tiktoken.get_encoding(encoding)
            return len(enc.encode(self.to_string()))
        except ImportError as e:
            raise ImportError(
                "tiktoken is required for token counting. Install it with: pip install tiktoken"
            ) from e

    def compare_tokens(
        self, original_schema: dict[str, Any] | None = None, encoding: str = "cl100k_base"
    ) -> dict[str, Any]:
        """
        Compare token counts between original and simplified schemas.

        Args:
            original_schema: Original schema dict (uses stored if not provided).
            encoding: Tokenizer encoding to use.

        Returns:
            Dictionary with original, simplified, and reduction metrics.
        """
        try:
            import tiktoken

            enc = tiktoken.get_encoding(encoding)
            schema_to_compare = original_schema or self._original_schema

            original_str = json.dumps(schema_to_compare)
            simplified_str = self.to_string()

            original_tokens = len(enc.encode(original_str))
            simplified_tokens = len(enc.encode(simplified_str))
            reduction_percent = (original_tokens - simplified_tokens) / original_tokens * 100

            return {
                "original_tokens": original_tokens,
                "simplified_tokens": simplified_tokens,
                "tokens_saved": original_tokens - simplified_tokens,
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
        return f"SchemaLite(keys={list(self._data.keys())})"


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
    formatter: JSONishFormatter | TypeScriptFormatter | YAMLFormatter
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

    # Process the schema
    try:
        processed_properties = formatter.process_properties(original_schema.get("properties", {}))
        return SchemaLite(
            processed_data=processed_properties,
            formatter=formatter,
            original_schema=original_schema,
        )
    except Exception as e:
        raise ConversionError(f"Failed to convert schema: {e}") from e


def loads(
    text: str,
    mode: Literal["json", "yaml"] = "json",
    repair: bool = True,
    extract_from_markdown: bool = True,
) -> dict[str, Any]:
    """
    Parse structured text (JSON or YAML) with robust error handling and repair capabilities.

    This function provides a unified interface for parsing JSON and YAML content with
    automatic repair, markdown extraction, and fallback mechanisms.

    Args:
        text: The text content to parse
        mode: The parsing mode - "json" or "yaml"
        repair: Whether to attempt repair for malformed content
        extract_from_markdown: Whether to extract content from markdown code blocks

    Returns:
        Parsed dictionary content

    Raises:
        ConversionError: If parsing fails and repair is disabled or unsuccessful

    Examples:
        >>> # Parse JSON with repair
        >>> data = loads('{"name": "John", "age": 30}', mode="json")

        >>> # Parse YAML with markdown extraction
        >>> data = loads('```yaml\\nname: John\\nage: 30\\n```', mode="yaml")

        >>> # Parse with repair disabled
        >>> data = loads('{"name": "John"}', mode="json", repair=False)
    """
    if not text or not text.strip():
        raise ConversionError("Empty or whitespace-only text provided")

    # Extract from markdown code blocks if requested
    if extract_from_markdown:
        text = _extract_from_markdown(text, mode)

    # Clean and prepare text
    text = text.strip()

    if mode == "json":
        return _parse_json(text, repair, extract_from_markdown)
    elif mode == "yaml":
        return _parse_yaml(text, repair)
    else:
        raise ConversionError(f"Unsupported mode: {mode}. Supported modes: 'json', 'yaml'")


def _extract_from_markdown(text: str, mode: str) -> str:
    """Extract content from markdown code blocks."""
    if mode == "json":
        # Look for ```json code blocks
        match = re.search(r"```json(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    elif mode == "yaml":
        # Look for ```yaml or ```yml code blocks
        match = re.search(r"```ya?ml(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

    return text


def _extract_json_object(text: str) -> str:
    """Extract JSON object from text using brace counting."""
    # Look for JSON object pattern - find the first complete JSON object
    # This handles cases where JSON is embedded in other text
    # Use a simpler approach: find the first { and then find the matching }
    start = text.find("{")
    if start == -1:
        return text

    # Count braces to find the matching closing brace
    brace_count = 0
    for i, char in enumerate(text[start:], start):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[start : i + 1]

    # If no matching brace found, return the original text
    return text


def _parse_json(text: str, repair: bool, extract_from_markdown: bool = True) -> dict[str, Any]:
    """Parse JSON text with optional repair."""
    # Only extract JSON object if markdown extraction was disabled AND
    # the text doesn't look like markdown code blocks
    if not extract_from_markdown and not text.strip().startswith("```"):
        extracted_text = _extract_json_object(text)
    else:
        extracted_text = text

    try:
        # Try standard JSON parsing first
        return json.loads(extracted_text)  # type: ignore[no-any-return]
    except json.JSONDecodeError as e:
        if not repair or json_repair is None:
            raise ConversionError(f"Failed to parse JSON: {extracted_text[:100]}...") from e

        try:
            # Try json_repair for malformed JSON
            repaired = json_repair.repair_json(extracted_text)
            return json.loads(repaired)  # type: ignore[no-any-return]
        except Exception as e:
            raise ConversionError(f"Failed to repair and parse JSON: {e}") from e


def _parse_yaml(text: str, repair: bool) -> dict[str, Any]:
    """Parse YAML text with optional repair."""
    if yaml is None:
        # Fallback to JSON parsing if PyYAML is not available
        try:
            return _parse_json(text, repair)
        except ConversionError as e:
            raise ConversionError("PyYAML not available and JSON fallback failed") from e

    try:
        # Try YAML parsing
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise ConversionError("YAML content did not parse to a dictionary")
        return parsed
    except yaml.YAMLError as e:
        if not repair:
            raise ConversionError(f"Failed to parse YAML: {e}") from e

        # Try to find YAML-like structure if parsing fails
        try:
            # First, try to fix common indentation issues
            lines = text.split("\n")

            # Find the minimum indentation (excluding empty lines)
            min_indent = float("inf")
            for line in lines:
                if line.strip() and not line.strip().startswith("#"):
                    indent = len(line) - len(line.lstrip())
                    if indent > 0:
                        min_indent = min(min_indent, indent)

            # If we found a minimum indentation, normalize it
            if min_indent != float("inf") and min_indent > 0:
                normalized_lines = []
                for line in lines:
                    if line.strip() and not line.strip().startswith("#"):
                        # Remove the common indentation
                        if line.startswith(" " * int(min_indent)):
                            normalized_lines.append(line[int(min_indent) :])
                        else:
                            normalized_lines.append(line)
                    else:
                        normalized_lines.append(line)

                try:
                    normalized_text = "\n".join(normalized_lines)
                    parsed = yaml.safe_load(normalized_text)
                    if isinstance(parsed, dict):
                        return parsed
                except yaml.YAMLError:
                    pass

            # Look for key-value pairs at the start of lines
            yaml_start = None
            for i, line in enumerate(lines):
                if line.strip() and ":" in line and not line.strip().startswith("#"):
                    yaml_start = i
                    break

            if yaml_start is not None:
                cleaned_text = "\n".join(lines[yaml_start:])
                parsed = yaml.safe_load(cleaned_text)
                if isinstance(parsed, dict):
                    return parsed

            # Try to convert simple key-value pairs to proper YAML format
            if ":" in text and not text.strip().startswith("{"):
                try:
                    # Convert simple key: value format to proper YAML
                    lines = text.strip().split("\n")
                    yaml_lines = []
                    for line in lines:
                        if ":" in line and not line.strip().startswith("#"):
                            # Ensure proper indentation
                            if not line.startswith(" "):
                                yaml_lines.append(line)
                            else:
                                yaml_lines.append(line)

                    if yaml_lines:
                        yaml_text = "\n".join(yaml_lines)
                        parsed = yaml.safe_load(yaml_text)
                        if isinstance(parsed, dict):
                            return parsed
                except Exception:
                    pass

            # Final fallback: try JSON parsing
            return _parse_json(text, repair)
        except Exception as fallback_error:
            raise ConversionError(
                f"Failed to parse YAML with repair: {fallback_error}"
            ) from fallback_error


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
