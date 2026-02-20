"""Configuration dataclass for formatters."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class FormatterConfig:
    """
    Configuration class for formatter behavior.

    This class allows customization of formatter output including:
    - Instruction prefix
    - Enum/class hoisting behavior
    - Union separators and markers
    - Indentation
    - Optional/required field markers
    - Description/constraint inclusion toggles

    Attributes:
        prefix: Instruction prefix to prepend to output (default: None)
        hoist_enums: Enum hoisting behavior - True (always), False (never), or "auto" (default)
        hoist_classes: Class/type hoisting behavior - True (always), False (never),
            "auto" (default), or list of specific class names to hoist
        union_separator: Separator for union types (default: " | ")
        indent: Number of spaces for indentation (default: 2)
        optional_marker: Marker for optional fields (default: "" - none)
        required_marker: Marker for required fields (default: "*")
        include_descriptions: Include field descriptions (default: True)
        include_constraints: Include validation constraints like min/max (default: True)
        include_metadata: Include all metadata (default: True) - supersedes
            include_descriptions and include_constraints
    """

    prefix: str | None = None
    hoist_enums: bool | Literal["auto"] = "auto"
    hoist_classes: bool | Literal["auto"] | list[str] = "auto"
    union_separator: str = " | "
    indent: int = 2
    optional_marker: str = ""
    required_marker: str = "*"
    include_descriptions: bool = True
    include_constraints: bool = True
    include_metadata: bool = True
