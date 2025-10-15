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

    REF_PATTERN = re.compile(r"#/\$defs/(.+)$", re.IGNORECASE)
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

        # Pre-warm cache for common patterns
        self._warm_cache()

    def _warm_cache(self) -> None:
        """Pre-process common reference patterns."""
        for ref_key, ref_def in self.defs.items():
            # Only cache simple types that are NOT enums
            if (
                "type" in ref_def
                and ref_def["type"] in ["string", "integer", "number", "boolean"]
                and "enum" not in ref_def
            ):
                self._ref_cache[ref_key] = self.TYPE_MAP.get(ref_def["type"], ref_def["type"])

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

        # Pre-filter available metadata keys to avoid unnecessary formatting
        available_metadata = [
            k
            for k in self.METADATA_MAP.keys()
            if k in property_def and not (k == "default" and property_def[k] is None)
        ]
        if not available_metadata:
            return ""

        # Format metadata parts
        metadata_parts = [self.METADATA_MAP[k](property_def[k]) for k in available_metadata]  # type: ignore[no-untyped-call]
        return f"  // {', '.join(metadata_parts)}"

    def process_ref(self, ref: dict[str, Any]) -> str:
        """
        Process a $ref reference to a definition.

        Args:
            ref: Dictionary containing the $ref key.

        Returns:
            TypeScript type representation.
        """
        ref_str = ref.get("$ref", "")
        if not ref_str:
            return "object"

        # Safely extract ref key with null check
        ref_match = self.REF_PATTERN.search(ref_str)
        if not ref_match:
            return "object"

        ref_key = ref_match.group(1)
        if ref_key in self._ref_cache:
            return self._ref_cache[ref_key]

        # Safely get ref definition
        ref_def = self.defs.get(ref_key)
        if not ref_def:
            return "object"

        if "enum" in ref_def:
            # Enum type - represent as union of literals
            enum_values = " | ".join(f'"{v}"' for v in ref_def.get("enum", []))
            result = enum_values if enum_values else "string"
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
            # Safely handle array items
            items = type_value.get("items")
            if not items:
                type_str = "any[]"  # Fallback for array without items
            elif "type" in items:
                items_type = self.process_type_value(items)
                type_str = f"{items_type}[]"
            elif "$ref" in items:
                items_type = self.process_ref(items)
                type_str = f"{items_type}[]"
            elif "anyOf" in items:
                items_type = self.process_anyof(items)
                type_str = f"({items_type})[]"
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

        from io import StringIO

        # Collect all interface definitions
        all_interfaces = []

        # Process nested definitions first (from $defs)
        for def_name, def_schema in self.defs.items():
            if "properties" in def_schema:
                nested_output = StringIO()
                nested_output.write(f"interface {def_name} {{\n")

                nested_props = def_schema["properties"]
                for prop_name, prop_def in nested_props.items():
                    prop_type = self.process_property(prop_def)
                    prop_metadata = self.add_metadata(prop_def)
                    nested_output.write(f"  {prop_name}: {prop_type};{prop_metadata}\n")

                nested_output.write("}")
                all_interfaces.append(nested_output.getvalue())

        # Process main interface
        main_output = StringIO()
        main_output.write("interface Schema {\n")

        processed = self.process_properties(self.properties)
        for name, prop_info in processed.items():
            type_str = prop_info["type"]
            metadata = prop_info["metadata"]
            main_output.write(f"  {name}: {type_str};{metadata}\n")

        main_output.write("}")
        all_interfaces.append(main_output.getvalue())

        return "\n\n".join(all_interfaces)
