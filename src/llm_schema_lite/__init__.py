"""
schema-lite: Transform verbose Pydantic schemas into LLM-friendly formats

Reduce token usage by 60-85% while preserving essential type information.
"""

__version__ = "0.6.1"

from .core import SchemaLite, loads, simplify_schema, validate
from .exceptions import (
    ConversionError,
    SchemaLiteError,
    UnsupportedModelError,
    ValidationError,
)
from .formatters import BaseFormatter, JSONishFormatter, TypeScriptFormatter, YAMLFormatter
from .validators import BaseValidator, JSONValidator, YAMLValidator

__all__ = [
    "__version__",
    # Core API
    "simplify_schema",
    "loads",
    "validate",
    "SchemaLite",
    # Formatters
    "BaseFormatter",
    "JSONishFormatter",
    "TypeScriptFormatter",
    "YAMLFormatter",
    # Validators
    "BaseValidator",
    "JSONValidator",
    "YAMLValidator",
    # Exceptions
    "SchemaLiteError",
    "UnsupportedModelError",
    "ConversionError",
    "ValidationError",
]
