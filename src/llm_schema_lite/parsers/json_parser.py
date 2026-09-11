"""JSON parser implementation."""

import json
import re
from typing import Any

import json_repair

from ..exceptions import ConversionError
from .base import BaseParser, _extract_from_markdown, _strip_leading_reasoning


class JSONParser(BaseParser):
    """
    JSON parser with robust error handling and content extraction.

    Handles various JSON formats including markdown-wrapped,
    embedded JSON, and malformed JSON with optional repair.
    """

    def parse(self, text: str, repair: bool = True) -> dict[str, Any]:
        """
        Parse JSON text with optional repair.

        Args:
            text: The text content to parse
            repair: Whether to attempt repair for malformed content

        Returns:
            Parsed dictionary content

        Raises:
            ConversionError: If parsing fails and repair is disabled or unsuccessful
        """
        # Skip a leading reasoning block, then prefer a markdown code block
        text = _strip_leading_reasoning(text)
        extracted_text = _extract_from_markdown(text, "json")

        # No block found: try JSON-specific extraction
        if extracted_text == text:
            extracted_text = _extract_json_content(extracted_text)

        # Clean and parse
        extracted_text = extracted_text.strip()
        return _parse_json(extracted_text, repair)


def _extract_json_content(text: str) -> str:
    """
    Extract JSON content from text using multiple strategies.

    Tries to find JSON objects or arrays embedded in text.
    """
    # First, check if the text is already valid JSON
    # If so, return it as-is without trying to extract patterns
    try:
        json.loads(text.strip())
        return text
    except json.JSONDecodeError:
        pass

    # Strategy 1: Look for complete JSON objects using brace counting
    json_object = _extract_balanced(text, "{", "}")
    if json_object != text:
        return json_object

    # Strategy 2: Look for JSON arrays
    json_array = _extract_balanced(text, "[", "]")
    if json_array != text:
        return json_array

    # Strategy 3: Look for JSON-like patterns that might be valid
    # Only use this if no complete structures were found
    json_pattern = _extract_json_pattern(text)
    if json_pattern != text:
        return json_pattern

    return text


def _extract_balanced(text: str, open_char: str, close_char: str) -> str:
    """Extract the first balanced open_char..close_char span from text.

    Brackets inside JSON strings are skipped (a `{` in a code-snippet value never
    balanced, so the whole reply went to json_repair: stanfordnlp/dspy#8759).
    """
    start = text.find(open_char)
    if start == -1:
        return text

    depth = 0
    in_string = escaped = False
    for i, char in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # If no matching bracket found, return the original text
    return text


def _extract_json_pattern(text: str) -> str:
    """
    Extract JSON-like patterns that might be valid JSON.

    This is a more aggressive approach that looks for patterns
    that could be JSON even if they're not perfectly formatted.
    Only used when no complete structures are found.
    """
    # Look for patterns like { ... } or [ ... ] that might be JSON
    # But be more selective - prefer larger, more complete structures
    patterns = [
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Objects with nested objects
        r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]",  # Arrays with nested arrays
    ]

    best_match = None
    best_length = 0

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            # Try to parse this as JSON to see if it's valid
            try:
                json.loads(match)
                # Prefer longer matches (more complete structures)
                # Only use this if it's significantly different from the original text
                # and represents a substantial portion of the text
                if len(match) > best_length and len(match) >= len(text) * 0.8:
                    best_match = match
                    best_length = len(match)
            except json.JSONDecodeError:
                continue

    return best_match if best_match else text


def _parse_json(text: str, repair: bool) -> dict[str, Any]:
    """Parse JSON text with optional repair."""
    try:
        # Try standard JSON parsing first
        return json.loads(text)  # type: ignore[no-any-return]
    except json.JSONDecodeError as e:
        if not repair:
            raise ConversionError(f"Failed to parse JSON: {text[:100]}...") from e

        try:
            # Try json_repair for malformed JSON
            repaired = json_repair.repair_json(text)
            return json.loads(repaired)  # type: ignore[no-any-return]
        except Exception as e:
            raise ConversionError(f"Failed to repair and parse JSON: {e}") from e
