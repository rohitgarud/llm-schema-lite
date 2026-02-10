"""YAML parser implementation."""

from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from ..exceptions import ConversionError
from .base import BaseParser, _smart_extract_content
from .json_parser import _parse_json


class YAMLParser(BaseParser):
    """
    YAML parser with robust error handling and content extraction.

    Handles various YAML formats including markdown-wrapped,
    embedded YAML, and malformed YAML with optional repair.
    Falls back to JSON parsing when PyYAML is not available.
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
        # Extract content using shared strategies
        extracted_text = _smart_extract_content(text, "yaml")

        # Try YAML-specific extraction if needed
        if extracted_text == text:
            direct_yaml = _extract_yaml_content(extracted_text)
            if direct_yaml != extracted_text:
                extracted_text = direct_yaml

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
            elif yaml_start is not None and not stripped:
                # Empty line might be part of YAML structure
                continue
            elif yaml_start is not None and not _looks_like_yaml_line(line):
                # This doesn't look like YAML anymore
                break
        elif yaml_start is not None and not stripped:
            # Empty line might be part of YAML structure
            continue
        elif yaml_start is not None and not _looks_like_yaml_line(line):
            # This doesn't look like YAML anymore
            break

    if yaml_start is not None and yaml_end is not None:
        yaml_lines = lines[yaml_start : yaml_end + 1]
        return "\n".join(yaml_lines)

    return text


def _is_yaml_key_line(line: str) -> bool:
    """Check if a line looks like a YAML key line (not explanatory text)."""
    stripped = line.strip()

    # Must have a colon
    if ":" not in stripped:
        return False

    # Must not start with common explanatory words
    # But only reject if it's clearly explanatory text, not a valid YAML key
    explanatory_words = [
        "the",
        "here",
        "this",
        "that",
        "configuration",
        "result",
        "output",
        "input",
        "settings",
        "config",
        "value",
        "content",
    ]

    first_word = stripped.split(":")[0].strip().lower()
    # Only reject if it's clearly explanatory AND doesn't look like a valid key
    if (
        first_word in explanatory_words
        and not first_word.replace("_", "").replace("-", "").isalnum()
    ):
        return False

    # Must look like a proper YAML key (alphanumeric with possible underscores/dashes)
    key_part = stripped.split(":")[0].strip()
    if not key_part.replace("_", "").replace("-", "").isalnum():
        return False

    return True


def _looks_like_yaml_line(line: str) -> bool:
    """Check if a line looks like it could be part of YAML content."""
    stripped = line.strip()
    if not stripped:
        return True  # Empty lines are valid in YAML

    # Check for common YAML patterns
    if ":" in stripped and not stripped.startswith("{"):
        return True

    # Check for list items
    if stripped.startswith("- "):
        return True

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
    if yaml is None:
        # Fallback to JSON parsing if PyYAML is not available
        try:
            return _parse_json(text, repair)
        except ConversionError as e:
            raise ConversionError("PyYAML not available and JSON fallback failed") from e

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

            # Try to convert simple key-value pairs to proper YAML format
            if ":" in text and not text.strip().startswith("{"):
                try:
                    # Convert simple key: value format to proper YAML
                    lines = text.strip().split("\n")
                    yaml_lines = []
                    for line in lines:
                        if ":" in line and not line.strip().startswith("#"):
                            # Ensure proper indentation
                            if not line.startswith(" "):
                                yaml_lines.append(line)
                            else:
                                yaml_lines.append(line)

                    if yaml_lines:
                        yaml_text = "\n".join(yaml_lines)
                        parsed = yaml.safe_load(yaml_text)
                        if isinstance(parsed, dict):
                            return parsed
                except Exception:
                    pass

            # Final fallback: try JSON parsing
            return _parse_json(text, repair)
        except Exception as fallback_error:
            raise ConversionError(f"Failed to parse YAML with repair: {fallback_error}") from e
