"""JSON schema validator."""

from typing import Any

from ..exceptions import ConversionError
from ..parsers import JSONParser
from .base import BaseValidator


class JSONValidator(BaseValidator):
    """
    Validator for data that is or can be parsed as JSON.

    Uses JSONParser for string inputs. Supports repair for malformed JSON.
    """

    def __init__(
        self,
        schema: type[Any] | dict[str, Any] | str,
        repair: bool = True,
    ) -> None:
        """
        Initialize the JSON validator.

        Args:
            schema: Pydantic BaseModel class, JSON schema dict, or JSON schema string.
            repair: Whether to attempt repair for malformed JSON. Default is True.
        """
        super().__init__(schema)
        self._repair = repair
        self._parser = JSONParser()

    def parse_data(
        self,
        data: dict[str, Any] | str | list[Any] | int | float | bool | None,
    ) -> dict[str, Any] | str | list[Any] | int | float | bool | None:
        """
        Parse string data as JSON when it looks like JSON; otherwise return as-is.

        Args:
            data: Raw data. If string starting with '{' or '[', parsed as JSON.

        Returns:
            Parsed data for validation, or original data if not a JSON string / parse fails.
        """
        if not isinstance(data, str):
            return data
        data_str = data.strip()
        if not data_str.startswith(("{", "[")):
            return data
        try:
            return self._parser.parse(data, repair=self._repair)
        except ConversionError:
            return data
