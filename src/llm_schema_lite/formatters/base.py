"""Base formatter abstract class for schema formatters."""

from abc import ABC, abstractmethod
from typing import Any


class BaseFormatter(ABC):
    """
    Abstract base class for schema formatters.

    All schema formatters must inherit from this class and implement
    the required methods.
    """

    def __init__(self, schema: dict[str, Any], include_metadata: bool = True):
        """
        Initialize the formatter.

        Args:
            schema: JSON schema from Pydantic model_json_schema.
            include_metadata: Whether to include metadata in the output.
        """
        self.schema = schema
        self.include_metadata = include_metadata
        self.defs = schema.get("$defs", {})
        self.properties = schema.get("properties", {})

    @abstractmethod
    def transform_schema(self) -> str:
        """
        Transform the schema into the desired format.

        Returns:
            Formatted schema as a string.
        """
        pass

    @abstractmethod
    def process_property(self, property: dict[str, Any]) -> Any:
        """
        Process a single property from the schema.

        Args:
            property: Property definition from the schema.

        Returns:
            Processed property representation.
        """
        pass

    @abstractmethod
    def process_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """
        Process multiple properties from the schema.

        Args:
            properties: Dictionary of property definitions.

        Returns:
            Dictionary of processed properties.
        """
        pass
