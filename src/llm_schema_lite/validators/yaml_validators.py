"""YAML schema validator."""

from typing import Any

from ..exceptions import ConversionError
from ..parsers import YAMLParser
from .base import BaseValidator


class YAMLValidator(BaseValidator):
    """
    Validator for data that is or can be parsed as YAML.

    Uses YAMLParser for string inputs. Supports repair for malformed YAML.
    """

    def __init__(
        self,
        schema: type[Any] | dict[str, Any] | str,
        repair: bool = True,
    ) -> None:
        """
        Initialize the YAML validator.

        Args:
            schema: Pydantic BaseModel class, JSON schema dict, or JSON schema string.
            repair: Whether to attempt repair for malformed YAML. Default is True.
        """
        super().__init__(schema)
        self._repair = repair
        self._parser = YAMLParser()

    def parse_data(
        self,
        data: dict[str, Any] | str | list[Any] | int | float | bool | None,
    ) -> dict[str, Any] | str | list[Any] | int | float | bool | None:
        """
        Parse string data as YAML; on failure treat as plain value.

        Args:
            data: Raw data. If string, parsed as YAML.

        Returns:
            Parsed data for validation, or original data if not a string / parse fails.
        """
        if not isinstance(data, str):
            return data
        try:
            return self._parser.parse(data.strip(), repair=self._repair)
        except ConversionError:
            return data
