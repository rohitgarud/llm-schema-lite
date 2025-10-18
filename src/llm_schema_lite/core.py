"""Core API for schema-lite."""

import json
from typing import Any, Literal

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None  # type: ignore[assignment, misc]

from .exceptions import ConversionError, UnsupportedModelError
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
    model: type["BaseModel"] | dict[str, Any],
    include_metadata: bool = True,
    format_type: Literal["jsonish", "typescript", "yaml"] = "jsonish",
) -> SchemaLite:
    """
    Convert Pydantic model to simplified schema.

    Args:
        model: Pydantic BaseModel class or JSON schema dict.
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
    else:
        raise UnsupportedModelError(
            f"Unsupported model type: {type(model)}. Expected Pydantic BaseModel or dict."
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
