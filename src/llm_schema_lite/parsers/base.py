"""Base parser interface and shared extraction utilities."""

import re
from abc import ABC, abstractmethod
from typing import Any


class BaseParser(ABC):
    """
    Abstract base class for parsers.

    Provides a common interface for parsing text content and shared
    extraction utilities for handling various LLM response formats.
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

    def _extract_content(self, text: str, mode: str) -> str:
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
        return _smart_extract_content(text, mode)


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

    # Strategy 2: Return original text (mode-specific extraction handled by subclasses)
    return text


def _extract_from_markdown(text: str, mode: str) -> str:
    """Extract content from markdown code blocks.

    The language tag is preferred but optional. Small models routinely fence their
    answer with a bare ``` (or mislabel it, e.g. ```python around JSON), and requiring
    the tag meant the whole reply was parsed instead -- where a single line of prose
    after the closing fence is enough to fail it. The tagged patterns are tried first,
    so a correctly-tagged block is still what wins when one is present.
    """
    tagged = {"json": r"```json(.*?)```", "yaml": r"```ya?ml(.*?)```"}.get(mode)
    patterns = [p for p in (tagged, r"```[^\n`]*\n(.*?)```") if p is not None]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    return text
