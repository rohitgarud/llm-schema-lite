"""
schema-lite: Transform verbose Pydantic schemas into LLM-friendly formats

Reduce token usage by 60-85% while preserving essential type information.
"""

__version__ = "0.6.1"

from .coercion import CoercionMetadata, ParseConfig, coerce_to_schema, coerce_value
from .core import (
    SchemaLite,
    coerce,
    loads,
    simplify_schema,
    validate,
)
from .exceptions import (
    ConversionError,
    SchemaLiteError,
    StreamingNotSupportedError,
    UnsupportedModelError,
    ValidationError,
)
from .formatters import (
    BaseFormatter,
    FormatterConfig,
    JSONishFormatter,
    TypeScriptFormatter,
    YAMLFormatter,
)
from .parsers import SchemaParser
from .validators import BaseValidator, JSONValidator, YAMLValidator

__all__ = [
    "__version__",
    # Core API
    "simplify_schema",
    "loads",
    "validate",
    "coerce",
    "SchemaLite",
    # Coercion
    "coerce_to_schema",
    "coerce_value",
    "ParseConfig",
    "CoercionMetadata",
    # Formatters
    "BaseFormatter",
    "FormatterConfig",
    "JSONishFormatter",
    "TypeScriptFormatter",
    "YAMLFormatter",
    # Parsers
    "SchemaParser",
    # Validators
    "BaseValidator",
    "JSONValidator",
    "YAMLValidator",
    # Exceptions
    "SchemaLiteError",
    "UnsupportedModelError",
    "ConversionError",
    "ValidationError",
    "StreamingNotSupportedError",
]
