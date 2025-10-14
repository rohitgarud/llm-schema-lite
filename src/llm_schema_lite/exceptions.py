"""Custom exceptions for schema-lite."""


class SchemaLiteError(Exception):
    """Base exception for schema-lite errors."""

    pass


class UnsupportedModelError(SchemaLiteError):
    """Raised when a model type is not supported."""

    pass


class ConversionError(SchemaLiteError):
    """Raised when schema conversion fails."""

    pass


class ValidationError(SchemaLiteError):
    """Raised when schema validation fails."""

    pass
