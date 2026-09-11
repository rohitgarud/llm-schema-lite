"""Base parser interface and shared extraction utilities."""

import re
from abc import ABC, abstractmethod
from typing import Any


class BaseParser(ABC):
    """
    Abstract base class for parsers.

    Provides a common interface for parsing text content.
    """

    @abstractmethod
    def parse(self, text: str, repair: bool = True) -> dict[str, Any]:
        """
        Parse text content into a dictionary.

        Args:
            text: The text content to parse
            repair: Whether to attempt repair for malformed content

        Returns:
            Parsed dictionary content

        Raises:
            ConversionError: If parsing fails and repair is disabled or unsuccessful
        """
        pass


def _extract_from_markdown(text: str, mode: str) -> str:
    """Extract content from markdown code blocks.

    The language tag is preferred but optional. Small models routinely fence their
    answer with a bare ``` (or mislabel it, e.g. ```python around JSON), and requiring
    the tag meant the whole reply was parsed instead -- where a single line of prose
    after the closing fence is enough to fail it. The tagged patterns are tried first,
    so a correctly-tagged block is still what wins when one is present.

    Returns ``text`` itself when no fence is found, so callers detect "no block" by
    identity with their input and fall back to their mode-specific extraction.
    """
    tagged = {"json": r"```json(.*?)```", "yaml": r"```ya?ml(.*?)```"}.get(mode)
    patterns = [p for p in (tagged, r"```[^\n`]*\n(.*?)```") if p is not None]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    return text
