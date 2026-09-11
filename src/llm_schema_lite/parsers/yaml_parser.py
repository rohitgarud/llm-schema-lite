"""YAML parser implementation."""

from typing import Any

import yaml

from ..exceptions import ConversionError
from .base import BaseParser, _extract_from_markdown
from .json_parser import _parse_json


class YAMLParser(BaseParser):
    """
    YAML parser with robust error handling and content extraction.

    Handles various YAML formats including markdown-wrapped,
    embedded YAML, and malformed YAML with optional repair.
    """

    def parse(self, text: str, repair: bool = True) -> dict[str, Any]:
        """
        Parse YAML text with optional repair.

        Args:
            text: The text content to parse
            repair: Whether to attempt repair for malformed content

        Returns:
            Parsed dictionary content

        Raises:
            ConversionError: If parsing fails and repair is disabled or unsuccessful
        """
        # Prefer a markdown code block
        extracted_text = _extract_from_markdown(text, "yaml")

        # No block found: try YAML-specific extraction
        if extracted_text == text:
            extracted_text = _extract_yaml_content(extracted_text)

        # Clean and parse
        extracted_text = extracted_text.strip()
        return _parse_yaml(extracted_text, repair)


def _extract_yaml_content(text: str) -> str:
    """
    Extract YAML content from text using multiple strategies.

    Tries to find YAML content embedded in text.
    """
    lines = text.split("\n")

    # Strategy 1: Look for YAML-like structure (key: value patterns)
    yaml_start = None
    yaml_end = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Check if this looks like a YAML key-value pair or list item
        # Must start at beginning of line or with proper indentation
        if (":" in line and not line.strip().startswith("{")) or stripped.startswith("- "):
            # Check if this is a proper YAML key or list item (not part of explanatory text)
            if _is_yaml_key_line(line) or stripped.startswith("- "):
                if yaml_start is None:
                    yaml_start = i
                yaml_end = i
        elif yaml_start is not None and not _looks_like_yaml_line(line):
            # This doesn't look like YAML anymore
            break

    if yaml_start is not None and yaml_end is not None:
        yaml_lines = lines[yaml_start : yaml_end + 1]
        return "\n".join(yaml_lines)

    return text


def _is_yaml_key_line(line: str) -> bool:
    """Check if a line is a YAML key line: an alphanumeric (plus ``_``/``-``) key before ``:``."""
    key_part = line.strip().split(":")[0].strip()
    return ":" in line and key_part.replace("_", "").replace("-", "").isalnum()


def _looks_like_yaml_line(line: str) -> bool:
    """Check if a non-blank line with no key colon and no list marker could still be YAML."""
    stripped = line.strip()

    # Check for indented content (might be nested)
    if line.startswith(" ") and any(c.isalnum() for c in stripped):
        return True

    # Check for simple values (not starting with explanatory words)
    if not any(
        stripped.lower().startswith(word + " ")
        for word in ["the", "here", "this", "that", "configuration", "data", "result"]
    ):
        return True

    return False


def _parse_yaml(text: str, repair: bool) -> dict[str, Any]:
    """Parse YAML text with optional repair."""
    try:
        # Try YAML parsing
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise ConversionError("YAML content did not parse to a dictionary")
        return parsed
    except yaml.YAMLError as e:
        if not repair:
            raise ConversionError(f"Failed to parse YAML: {e}") from e

        # Try to find YAML-like structure if parsing fails
        try:
            # First, try to fix common indentation issues
            lines = text.split("\n")

            # Find the minimum indentation (excluding empty lines)
            min_indent = float("inf")
            for line in lines:
                if line.strip() and not line.strip().startswith("#"):
                    indent = len(line) - len(line.lstrip())
                    if indent > 0:
                        min_indent = min(min_indent, indent)

            # If we found a minimum indentation, normalize it
            if min_indent != float("inf") and min_indent > 0:
                normalized_lines = []
                for line in lines:
                    if line.strip() and not line.strip().startswith("#"):
                        # Remove the common indentation
                        if line.startswith(" " * int(min_indent)):
                            normalized_lines.append(line[int(min_indent) :])
                        else:
                            normalized_lines.append(line)
                    else:
                        normalized_lines.append(line)

                try:
                    normalized_text = "\n".join(normalized_lines)
                    parsed = yaml.safe_load(normalized_text)
                    if isinstance(parsed, dict):
                        return parsed
                except yaml.YAMLError:
                    pass

            # Look for key-value pairs at the start of lines
            yaml_start = None
            for i, line in enumerate(lines):
                if line.strip() and ":" in line and not line.strip().startswith("#"):
                    yaml_start = i
                    break

            if yaml_start is not None:
                cleaned_text = "\n".join(lines[yaml_start:])
                parsed = yaml.safe_load(cleaned_text)
                if isinstance(parsed, dict):
                    return parsed

            # Final fallback: try JSON parsing
            return _parse_json(text, repair)
        except Exception as fallback_error:
            raise ConversionError(f"Failed to parse YAML with repair: {fallback_error}") from e
