"""YAML-style formatter for transforming Pydantic schemas."""

import re
from typing import Any

from .base import BaseFormatter


class YAMLFormatter(BaseFormatter):
    """
    Transforms Pydantic schema into YAML-style format.

    This formatter creates a clean YAML-like representation with Python-style
    type hints, optionally including metadata as inline comments.

    Example output:
        name: str  # Product name, minLength: 1
        age: int  # min: 0, max: 120
        tags: list[str]  # Product tags
    """

    REF_PATTERN = re.compile(r"#/\$defs/(.+)$")
    TYPE_MAP = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
        "null": "None",
    }

    METADATA_MAP = {
        "default": lambda v: f"(defaults to {v})",
        "description": lambda v: v,
        "pattern": lambda v: f"pattern: {v}",
        "minimum": lambda v: f"min: {v}",
        "maximum": lambda v: f"max: {v}",
        "minLength": lambda v: f"minLength: {v}",
        "maxLength": lambda v: f"maxLength: {v}",
        "format": lambda v: f"format: {v}",
        "multipleOf": lambda v: f"multipleOf: {v}",
    }

    def __init__(self, schema: dict[str, Any], include_metadata: bool = True):
        """
        Initialize the YAML formatter.

        Args:
            schema: JSON schema from Pydantic model_json_schema.
            include_metadata: Whether to include metadata as inline comments.
        """
        super().__init__(schema, include_metadata)
        self._ref_cache: dict[str, str] = {}

    def add_metadata(self, property_def: dict[str, Any]) -> str:
        """
        Generate metadata comment for a property.

        Args:
            property_def: The property definition containing metadata.

        Returns:
            Metadata comment string or empty string.
        """
        if not self.include_metadata:
            return ""

        metadata = []
        for k, formatter in self.METADATA_MAP.items():
            if k in property_def and not (k == "default" and property_def[k] is None):
                metadata.append(formatter(property_def[k]))  # type: ignore[no-untyped-call]

        return f"  # {', '.join(metadata)}" if metadata else ""

    def process_ref(self, ref: dict[str, Any]) -> str:
        """
        Process a $ref reference to a definition.

        Args:
            ref: Dictionary containing the $ref key.

        Returns:
            YAML-style type representation.
        """
        ref_match = self.REF_PATTERN.search(ref["$ref"])
        if not ref_match:
            return "dict"

        ref_key = ref_match.group(1)
        if ref_key in self._ref_cache:
            return self._ref_cache[ref_key]

        ref_def = self.defs.get(ref_key, {})

        if "enum" in ref_def:
            # Enum type - represent as literal union
            enum_values = " | ".join(
                f'"{v}"' if isinstance(v, str) else str(v) for v in ref_def["enum"]
            )
            result = f"Literal[{enum_values}]"
        elif "properties" in ref_def:
            # Nested object - use the ref name
            result = ref_key
        elif "type" in ref_def:
            result = self.process_type_value(ref_def)
        else:
            result = "dict"

        self._ref_cache[ref_key] = result
        return result

    def process_enum(self, enum_value: dict[str, Any]) -> str:
        """
        Process an enum field.

        Args:
            enum_value: Dictionary containing enum definition.

        Returns:
            YAML-style literal union representation.
        """
        # Safely get enum values
        enum_list = enum_value.get("enum", [])
        if not enum_list:
            return "str"  # Fallback for empty enum

        enum_items = []
        for item in enum_list:
            if isinstance(item, str):
                enum_items.append(f'"{item}"')
            else:
                enum_items.append(str(item))
        return f"Literal[{' | '.join(enum_items)}]"

    def process_type_value(self, type_value: dict[str, Any]) -> str:
        """
        Process a type field.

        Args:
            type_value: Dictionary containing type definition.

        Returns:
            YAML-style type representation.
        """
        type_name = type_value.get("type", "Any")
        type_str = self.TYPE_MAP.get(type_name, type_name)

        if type_str == "list":
            # Handle list types
            if "items" in type_value:
                if "type" in type_value["items"]:
                    items_type = self.process_type_value(type_value["items"])
                    type_str = f"list[{items_type}]"
                elif "$ref" in type_value["items"]:
                    items_type = self.process_ref(type_value["items"])
                    type_str = f"list[{items_type}]"
                elif "enum" in type_value["items"]:
                    items_type = self.process_enum(type_value["items"])
                    type_str = f"list[{items_type}]"
                else:
                    type_str = "list[Any]"
            else:
                type_str = "list[Any]"

        return type_str  # type: ignore[no-any-return]

    def process_anyof(self, anyof: dict[str, Any]) -> str:
        """
        Process an anyOf field (union types).

        Args:
            anyof: Dictionary containing anyOf definition.

        Returns:
            YAML-style union type representation.
        """
        # Safely get anyOf list
        anyof_list = anyof.get("anyOf", [])
        if not anyof_list:
            return "Any"  # Fallback for empty anyOf

        item_types = []
        for item in anyof_list:
            if "enum" in item:
                item_types.append(self.process_enum(item))
            elif "const" in item:
                const_val = item["const"]
                if isinstance(const_val, str):
                    item_types.append(f'"{const_val}"')
                else:
                    item_types.append(str(const_val))
            elif "$ref" in item:
                item_types.append(self.process_ref(item))
            elif "type" in item:
                item_types.append(self.process_type_value(item))
            else:
                # Unknown anyOf item, skip it
                continue

        return " | ".join(item_types) if item_types else "Any"

    def process_property(self, property_def: dict[str, Any]) -> str:
        """
        Process a single property into YAML-style syntax.

        Args:
            property_def: Property definition from the schema.

        Returns:
            YAML-style type representation.
        """
        if "$ref" in property_def:
            return self.process_ref(property_def)
        elif "enum" in property_def:
            return self.process_enum(property_def)
        elif "anyOf" in property_def:
            return self.process_anyof(property_def)
        elif "type" in property_def:
            return self.process_type_value(property_def)
        return "Any"

    def process_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """
        Process all properties.

        Args:
            properties: Dictionary of property definitions.

        Returns:
            Dictionary mapping property names to type info.
        """
        processed = {}
        for name, prop_def in properties.items():
            processed[name] = {
                "type": self.process_property(prop_def),
                "metadata": self.add_metadata(prop_def),
                "original": prop_def,
            }
        return processed

    def transform_schema(self) -> str:
        """
        Transform schema into YAML-style format.

        Returns:
            YAML-style schema definition as a string.
        """
        if not self.properties:
            return "{}"

        all_sections = []

        # Process nested definitions first (from $defs)
        for def_name, def_schema in self.defs.items():
            if "properties" in def_schema:
                nested_lines = []

                # Only add section header if metadata is included
                if self.include_metadata:
                    nested_lines.append(f"# {def_name}")

                nested_props = def_schema["properties"]

                for prop_name, prop_def in nested_props.items():
                    prop_type = self.process_property(prop_def)
                    prop_metadata = self.add_metadata(prop_def)
                    nested_lines.append(f"{def_name}.{prop_name}: {prop_type}{prop_metadata}")

                all_sections.append("\n".join(nested_lines))

        # Process main schema
        lines = []
        processed = self.process_properties(self.properties)

        for name, prop_info in processed.items():
            type_str = prop_info["type"]
            metadata = prop_info["metadata"]

            line = f"{name}: {type_str}{metadata}"
            lines.append(line)

        if all_sections:
            all_sections.append("\n".join(lines))
            return "\n\n".join(all_sections)

        return "\n".join(lines)
