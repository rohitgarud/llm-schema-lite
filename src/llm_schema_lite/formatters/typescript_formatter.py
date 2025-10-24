"""TypeScript interface formatter for transforming Pydantic schemas."""

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

    @property
    def TYPE_MAP(self) -> dict[str, str]:
        """Type mapping for TypeScript format."""
        return {
            "string": "string",
            "integer": "number",
            "number": "number",
            "boolean": "boolean",
            "array": "Array",
            "object": "object",
            "null": "null",
        }

    def add_metadata(self, representation: str, value: dict[str, Any]) -> str:
        """
        Add metadata comments to a field representation.

        Args:
            representation: The base field representation.
            value: The field definition containing metadata.

        Returns:
            Field representation with metadata comments.
        """
        if not self.include_metadata:
            return representation

        available_metadata = self.get_available_metadata(value)
        if not available_metadata:
            return representation

        metadata_parts = self.format_metadata_parts(value)
        return f"{representation}  // {', '.join(metadata_parts)}"

    def process_anyof(self, anyof: dict[str, Any]) -> str:
        """
        Process an anyOf field (union types) for TypeScript.

        Args:
            anyof: Dictionary containing anyOf definition.

        Returns:
            Formatted union type representation.
        """
        anyof_list = anyof.get("anyOf", [])
        if not anyof_list:
            return "string"

        item_types = []
        for item in anyof_list:
            if not isinstance(item, dict):
                continue

            if "enum" in item:
                item_types.append(self.process_enum(item))
            elif "const" in item:
                item_types.append(str(item["const"]))
            elif "$ref" in item:
                item_types.append(self.process_ref(item))
            elif "type" in item:
                item_types.append(self.process_type_value(item))

        return " | ".join(item_types) if item_types else "string"

    def process_enum(self, enum_value: dict[str, Any]) -> str:
        """
        Process an enum field for TypeScript.

        Args:
            enum_value: Dictionary containing enum definition.

        Returns:
            Formatted enum representation as TypeScript union literals.
        """
        enum_list = enum_value.get("enum", [])
        if not enum_list:
            return "string"

        # Create TypeScript union of string literals
        enum_literals = [f'"{val}"' for val in enum_list]
        return " | ".join(enum_literals)

    def process_type_value(self, type_value: dict[str, Any]) -> str:
        """
        Process a type field for TypeScript.

        Args:
            type_value: Dictionary containing type definition.

        Returns:
            Formatted type representation.
        """
        type_name = type_value.get("type", "string")

        # Handle array of types (union types like ["string", "null"])
        if isinstance(type_name, list):
            if len(type_name) == 1:
                type_name = type_name[0]
            elif "null" in type_name and len(type_name) == 2:
                # Handle nullable types like ["string", "null"] -> "string | null"
                non_null_type = next(t for t in type_name if t != "null")
                type_str = self.TYPE_MAP.get(non_null_type, non_null_type)
                return f"{type_str} | null"
            else:
                # Multiple non-null types - treat as union
                type_strs = [self.TYPE_MAP.get(t, t) for t in type_name if t != "null"]
                return " | ".join(s for s in type_strs if s is not None)

        # Now type_name is guaranteed to be a string
        type_str = self.TYPE_MAP.get(type_name, type_name)

        if type_str == "Array":
            # Handle array items
            items = type_value.get("items")
            if not items:
                return "Array<any>"
            elif isinstance(items, bool):
                return "Array<any>" if items else "Array<any>"
            elif isinstance(items, dict) and "type" in items:
                items_type = self.process_type_value(items)
                return f"Array<{items_type}>"
            elif isinstance(items, dict) and "$ref" in items:
                items_type = self.process_ref(items)
                return f"Array<{items_type}>"
            elif isinstance(items, dict) and "anyOf" in items:
                items_type = self.process_anyof(items)
                return f"Array<{items_type}>"
            else:
                return "Array<any>"

        return str(type_str)

    def dict_to_string(self, value: Any, indent: int = 1) -> str:
        """
        Convert a dictionary or list to a formatted string representation.

        Args:
            value: The value to convert (dict, list, or primitive).
            indent: Current indentation level.

        Returns:
            Formatted string representation.
        """
        # For TypeScript, we don't need to format nested objects as strings
        # since they're handled by the interface generation
        return str(value)

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
                    nested_output.write(f"  {prop_name}: {prop_type};\n")

                nested_output.write("}")
                all_interfaces.append(nested_output.getvalue())

        # Process main interface
        main_output = StringIO()
        main_output.write("interface Schema {\n")

        processed_properties = self.process_properties(self.properties)
        for name, prop_type in processed_properties.items():
            main_output.write(f"  {name}: {prop_type};\n")

        main_output.write("}")
        all_interfaces.append(main_output.getvalue())

        return "\n\n".join(all_interfaces)
