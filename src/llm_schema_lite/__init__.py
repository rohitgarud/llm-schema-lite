"""
schema-lite: Transform verbose Pydantic schemas into LLM-friendly formats

Reduce token usage by 60-85% while preserving essential type information.
"""

__version__ = "0.5.0"

from .core import SchemaLite, loads, simplify_schema
from .exceptions import (
    ConversionError,
    SchemaLiteError,
    UnsupportedModelError,
    ValidationError,
)
from .formatters import BaseFormatter, JSONishFormatter, TypeScriptFormatter, YAMLFormatter

__all__ = [
    "__version__",
    # Core API
    "simplify_schema",
    "loads",
    "SchemaLite",
    # Formatters
    "BaseFormatter",
    "JSONishFormatter",
    "TypeScriptFormatter",
    "YAMLFormatter",
    # Exceptions
    "SchemaLiteError",
    "UnsupportedModelError",
    "ConversionError",
    "ValidationError",
]
