"""JSON parser implementation."""

import json
import re
from typing import Any

try:
    import json_repair
except ImportError:
    json_repair = None  # type: ignore[assignment, unused-ignore]

from ..exceptions import ConversionError
from .base import BaseParser, _smart_extract_content


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
        # Extract content using shared strategies
        extracted_text = _smart_extract_content(text, "json")

        # Try JSON-specific extraction if needed
        if extracted_text == text:
            direct_json = _extract_json_content(extracted_text)
            if direct_json != extracted_text:
                extracted_text = direct_json

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
    json_object = _extract_json_object(text)
    if json_object != text:
        return json_object

    # Strategy 2: Look for JSON arrays
    json_array = _extract_json_array(text)
    if json_array != text:
        return json_array

    # Strategy 3: Look for JSON-like patterns that might be valid
    # Only use this if no complete structures were found
    json_pattern = _extract_json_pattern(text)
    if json_pattern != text:
        return json_pattern

    return text


def _extract_json_object(text: str) -> str:
    """Extract JSON object from text using brace counting."""
    # Look for JSON object pattern - find the first complete JSON object
    # This handles cases where JSON is embedded in other text
    # Use a simpler approach: find the first { and then find the matching }
    start = text.find("{")
    if start == -1:
        return text

    # Count braces to find the matching closing brace
    brace_count = 0
    for i, char in enumerate(text[start:], start):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                extracted = text[start : i + 1]
                # If we extracted the entire text, return it
                # If we extracted a subset, return the subset
                return extracted

    # If no matching brace found, return the original text
    return text


def _extract_json_array(text: str) -> str:
    """Extract JSON array from text using bracket counting."""
    # Look for JSON array pattern - find the first complete JSON array
    start = text.find("[")
    if start == -1:
        return text

    # Count brackets to find the matching closing bracket
    bracket_count = 0
    for i, char in enumerate(text[start:], start):
        if char == "[":
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1
            if bracket_count == 0:
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
        if not repair or json_repair is None:
            raise ConversionError(f"Failed to parse JSON: {text[:100]}...") from e

        try:
            # Try json_repair for malformed JSON
            repaired = json_repair.repair_json(text)
            return json.loads(repaired)  # type: ignore[no-any-return]
        except Exception as e:
            raise ConversionError(f"Failed to repair and parse JSON: {e}") from e
