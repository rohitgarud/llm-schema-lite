"""Core functionality for LLM Schema Lite."""

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
from .formatters import JSONishFormatter, TypeScriptFormatter, YAMLFormatter
from .formatters.base import BaseFormatter


class SchemaLite:
    """
    Simplified schema representation with multiple output formats.

    This class provides a unified interface for converting schemas to different
    string representations while maintaining the original data for dictionary access.
    """

    def __init__(
        self,
        formatter: BaseFormatter,
        original_schema: dict[str, Any],
    ):
        """
        Initialize SchemaLite with processed data and formatter.

        Args:
            formatter: The formatter instance for this schema.
            original_schema: The original JSON schema.
        """
        self._data: dict[str, Any] = {}
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

    # Automatically extract content using smart strategies
    extracted_text = _smart_extract_content(text, mode)

    # Clean and prepare text
    extracted_text = extracted_text.strip()

    if mode == "json":
        return _parse_json(extracted_text, repair)
    elif mode == "yaml":
        return _parse_yaml(extracted_text, repair)
    else:
        raise ConversionError(f"Unsupported mode: {mode}. Supported modes: 'json', 'yaml'")


def _smart_extract_content(text: str, mode: str) -> str:
    """
    Intelligently extract structured content from various LLM response formats.

    This function tries multiple extraction strategies in order of preference:
    1. Markdown code blocks (```json, ```yaml, etc.)
    2. Direct structured content detection
    3. Embedded structured content extraction

    Args:
        text: The raw text content
        mode: The parsing mode ("json" or "yaml")

    Returns:
        Extracted structured content
    """
    # Strategy 1: Try markdown code blocks first
    markdown_extracted = _extract_from_markdown(text, mode)
    if markdown_extracted != text:
        return markdown_extracted

    # Strategy 2: Try direct structured content detection
    if mode == "json":
        # Look for JSON objects/arrays directly
        direct_json = _extract_json_content(text)
        if direct_json != text:
            return direct_json
    elif mode == "yaml":
        # Look for YAML content directly
        direct_yaml = _extract_yaml_content(text)
        if direct_yaml != text:
            return direct_yaml

    # Strategy 3: Fallback to original text
    return text


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


def _extract_json_content(text: str) -> str:
    """
    Extract JSON content from text using multiple strategies.

    Tries to find JSON objects or arrays embedded in text.
    """
    # First, check if the text is already valid JSON
    # If so, return it as-is without trying to extract patterns
    try:
        json.loads(text.strip())
        return text
    except json.JSONDecodeError:
        pass

    # Strategy 1: Look for complete JSON objects using brace counting
    json_object = _extract_json_object(text)
    if json_object != text:
        return json_object

    # Strategy 2: Look for JSON arrays
    json_array = _extract_json_array(text)
    if json_array != text:
        return json_array

    # Strategy 3: Look for JSON-like patterns that might be valid
    # Only use this if no complete structures were found
    json_pattern = _extract_json_pattern(text)
    if json_pattern != text:
        return json_pattern

    return text


def _extract_yaml_content(text: str) -> str:
    """
    Extract YAML content from text using multiple strategies.

    Tries to find YAML content embedded in text.
    """
    lines = text.split("\n")

    # Strategy 1: Look for YAML-like structure (key: value patterns)
    yaml_start = None
    yaml_end = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Check if this looks like a YAML key-value pair or list item
        # Must start at beginning of line or with proper indentation
        if (":" in line and not line.strip().startswith("{")) or stripped.startswith("- "):
            # Check if this is a proper YAML key or list item (not part of explanatory text)
            if _is_yaml_key_line(line) or stripped.startswith("- "):
                if yaml_start is None:
                    yaml_start = i
                yaml_end = i
            elif yaml_start is not None and not stripped:
                # Empty line might be part of YAML structure
                continue
            elif yaml_start is not None and not _looks_like_yaml_line(line):
                # This doesn't look like YAML anymore
                break
        elif yaml_start is not None and not stripped:
            # Empty line might be part of YAML structure
            continue
        elif yaml_start is not None and not _looks_like_yaml_line(line):
            # This doesn't look like YAML anymore
            break

    if yaml_start is not None and yaml_end is not None:
        yaml_lines = lines[yaml_start : yaml_end + 1]
        return "\n".join(yaml_lines)

    return text


def _is_yaml_key_line(line: str) -> bool:
    """Check if a line looks like a YAML key line (not explanatory text)."""
    stripped = line.strip()

    # Must have a colon
    if ":" not in stripped:
        return False

    # Must not start with common explanatory words
    # But only reject if it's clearly explanatory text, not a valid YAML key
    explanatory_words = [
        "the",
        "here",
        "this",
        "that",
        "configuration",
        "result",
        "output",
        "input",
        "settings",
        "config",
        "value",
        "content",
    ]

    first_word = stripped.split(":")[0].strip().lower()
    # Only reject if it's clearly explanatory AND doesn't look like a valid key
    if (
        first_word in explanatory_words
        and not first_word.replace("_", "").replace("-", "").isalnum()
    ):
        return False

    # Must look like a proper YAML key (alphanumeric with possible underscores/dashes)
    key_part = stripped.split(":")[0].strip()
    if not key_part.replace("_", "").replace("-", "").isalnum():
        return False

    return True


def _looks_like_yaml_line(line: str) -> bool:
    """Check if a line looks like it could be part of YAML content."""
    stripped = line.strip()
    if not stripped:
        return True  # Empty lines are valid in YAML

    # Check for common YAML patterns
    if ":" in stripped and not stripped.startswith("{"):
        return True

    # Check for list items
    if stripped.startswith("- "):
        return True

    # Check for indented content (might be nested)
    if line.startswith(" ") and any(c.isalnum() for c in stripped):
        return True

    # Check for simple values (not starting with explanatory words)
    if not any(
        stripped.lower().startswith(word + " ")
        for word in ["the", "here", "this", "that", "configuration", "data", "result"]
    ):
        return True

    return False


def _extract_json_array(text: str) -> str:
    """Extract JSON array from text using bracket counting."""
    # Look for JSON array pattern - find the first complete JSON array
    start = text.find("[")
    if start == -1:
        return text

    # Count brackets to find the matching closing bracket
    bracket_count = 0
    for i, char in enumerate(text[start:], start):
        if char == "[":
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1
            if bracket_count == 0:
                return text[start : i + 1]

    # If no matching bracket found, return the original text
    return text


def _extract_json_pattern(text: str) -> str:
    """
    Extract JSON-like patterns that might be valid JSON.

    This is a more aggressive approach that looks for patterns
    that could be JSON even if they're not perfectly formatted.
    Only used when no complete structures are found.
    """
    # Look for patterns like { ... } or [ ... ] that might be JSON
    # But be more selective - prefer larger, more complete structures
    patterns = [
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Objects with nested objects
        r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]",  # Arrays with nested arrays
    ]

    best_match = None
    best_length = 0

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            # Try to parse this as JSON to see if it's valid
            try:
                json.loads(match)
                # Prefer longer matches (more complete structures)
                # Only use this if it's significantly different from the original text
                # and represents a substantial portion of the text
                if len(match) > best_length and len(match) >= len(text) * 0.8:
                    best_match = match
                    best_length = len(match)
            except json.JSONDecodeError:
                continue

    return best_match if best_match else text


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
                extracted = text[start : i + 1]
                # If we extracted the entire text, return it
                # If we extracted a subset, return the subset
                return extracted

    # If no matching brace found, return the original text
    return text


def _parse_json(text: str, repair: bool) -> dict[str, Any]:
    """Parse JSON text with optional repair."""
    try:
        # Try standard JSON parsing first
        return json.loads(text)  # type: ignore[no-any-return]
    except json.JSONDecodeError as e:
        if not repair or json_repair is None:
            raise ConversionError(f"Failed to parse JSON: {text[:100]}...") from e

        try:
            # Try json_repair for malformed JSON
            repaired = json_repair.repair_json(text)
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
