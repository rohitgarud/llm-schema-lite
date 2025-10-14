"""TypeScript interface formatter for transforming Pydantic schemas."""

import re
from typing import Any

from .base import BaseFormatter


class TypeScriptFormatter(BaseFormatter):
    """
    Transforms Pydantic schema into TypeScript interface format.

    This formatter creates TypeScript-style interface definitions,
    optionally including metadata as inline comments.

    Example output:
        interface Schema {
          name: string;  // Product name, minLength: 1
          age: number;  // min: 0, max: 120
          is_active: boolean;  // (defaults to true)
        }
    """

    REF_PATTERN = re.compile(r"#/\$defs/(.+)$")
    TYPE_MAP = {
        "string": "string",
        "integer": "number",
        "number": "number",
        "boolean": "boolean",
        "array": "Array",
        "object": "object",
        "null": "null",
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
        Initialize the TypeScript formatter.

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

        return f"  // {', '.join(metadata)}" if metadata else ""

    def process_ref(self, ref: dict[str, Any]) -> str:
        """
        Process a $ref reference to a definition.

        Args:
            ref: Dictionary containing the $ref key.

        Returns:
            TypeScript type representation.
        """
        ref_match = self.REF_PATTERN.search(ref["$ref"])
        if not ref_match:
            return "object"

        ref_key = ref_match.group(1)
        if ref_key in self._ref_cache:
            return self._ref_cache[ref_key]

        ref_def = self.defs.get(ref_key, {})

        if "enum" in ref_def:
            # Enum type - represent as union of literals
            enum_values = " | ".join(f'"{v}"' for v in ref_def["enum"])
            result = enum_values
        elif "properties" in ref_def:
            # Nested object - for now just use the ref name
            result = ref_key
        elif "type" in ref_def:
            result = self.process_type_value(ref_def)
        else:
            result = "object"

        self._ref_cache[ref_key] = result
        return result

    def process_enum(self, enum_value: dict[str, Any]) -> str:
        """
        Process an enum field.

        Args:
            enum_value: Dictionary containing enum definition.

        Returns:
            TypeScript union type representation.
        """
        # Safely get enum values
        enum_list = enum_value.get("enum", [])
        if not enum_list:
            return "string"  # Fallback for empty enum

        enum_items = []
        for item in enum_list:
            if isinstance(item, str):
                enum_items.append(f'"{item}"')
            else:
                enum_items.append(str(item))
        return " | ".join(enum_items)

    def process_type_value(self, type_value: dict[str, Any]) -> str:
        """
        Process a type field.

        Args:
            type_value: Dictionary containing type definition.

        Returns:
            TypeScript type representation.
        """
        type_name = type_value.get("type", "any")
        type_str = self.TYPE_MAP.get(type_name, type_name)

        if type_str == "Array":
            # Handle array types
            if "items" in type_value:
                if "type" in type_value["items"]:
                    items_type = self.process_type_value(type_value["items"])
                    type_str = f"{items_type}[]"
                elif "$ref" in type_value["items"]:
                    items_type = self.process_ref(type_value["items"])
                    type_str = f"{items_type}[]"
                else:
                    type_str = "any[]"
            else:
                type_str = "any[]"

        return type_str  # type: ignore[no-any-return]

    def process_anyof(self, anyof: dict[str, Any]) -> str:
        """
        Process an anyOf field (union types).

        Args:
            anyof: Dictionary containing anyOf definition.

        Returns:
            TypeScript union type representation.
        """
        # Safely get anyOf list
        anyof_list = anyof.get("anyOf", [])
        if not anyof_list:
            return "any"  # Fallback for empty anyOf

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

        return " | ".join(item_types) if item_types else "any"

    def process_property(self, property_def: dict[str, Any]) -> str:
        """
        Process a single property into TypeScript syntax.

        Args:
            property_def: Property definition from the schema.

        Returns:
            TypeScript type representation.
        """
        if "$ref" in property_def:
            return self.process_ref(property_def)
        elif "enum" in property_def:
            return self.process_enum(property_def)
        elif "anyOf" in property_def:
            return self.process_anyof(property_def)
        elif "type" in property_def:
            return self.process_type_value(property_def)
        return "any"

    def process_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """
        Process all properties.

        Args:
            properties: Dictionary of property definitions.

        Returns:
            Dictionary mapping property names to TypeScript types.
        """
        processed = {}
        for name, prop_def in properties.items():
            processed[name] = {
                "type": self.process_property(prop_def),
                "metadata": self.add_metadata(prop_def),
            }
        return processed

    def transform_schema(self) -> str:
        """
        Transform schema into TypeScript interface syntax.

        Returns:
            TypeScript interface definition as a string.
        """
        if not self.properties:
            return "interface Schema {}"

        # Collect all interface definitions
        all_interfaces = []

        # Process nested definitions first (from $defs)
        for def_name, def_schema in self.defs.items():
            if "properties" in def_schema:
                nested_lines = [f"interface {def_name} {{"]
                nested_props = def_schema["properties"]
                for prop_name, prop_def in nested_props.items():
                    prop_type = self.process_property(prop_def)
                    prop_metadata = self.add_metadata(prop_def)
                    nested_lines.append(f"  {prop_name}: {prop_type};{prop_metadata}")

                nested_lines.append("}")
                all_interfaces.append("\n".join(nested_lines))

        # Process main interface
        lines = ["interface Schema {"]
        processed = self.process_properties(self.properties)
        for name, prop_info in processed.items():
            type_str = prop_info["type"]
            metadata = prop_info["metadata"]
            lines.append(f"  {name}: {type_str};{metadata}")

        lines.append("}")
        all_interfaces.append("\n".join(lines))

        return "\n\n".join(all_interfaces)
