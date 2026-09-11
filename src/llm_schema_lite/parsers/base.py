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


_LEADING_REASONING = re.compile(r"^\s*<(think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)
# A close tag starting a line, with no open tag before it: the chat template supplied the
# open tag, so the reply begins mid-reasoning.
_UNOPENED_REASONING = re.compile(
    r"\A(?:(?!<think).)*?^</think(?:ing)?>", re.DOTALL | re.IGNORECASE | re.MULTILINE
)


def _strip_leading_reasoning(text: str) -> str:
    """Drop a leading <think>/<thinking> block so it is not parsed as the answer.

    Reasoning models (Qwen3, DeepSeek-R1) can reason inline before answering, often
    restating the output format, and that placeholder was extracted first. Only a block
    at the very start is removed: a field value that merely contains the tag is data.
    When the template opened the block, only its close tag is in the reply; that is
    removed with everything before it only at the start of a line, where no JSON string
    can hold it.
    """
    text = _LEADING_REASONING.sub("", text, count=1)
    return _UNOPENED_REASONING.sub("", text, count=1)
