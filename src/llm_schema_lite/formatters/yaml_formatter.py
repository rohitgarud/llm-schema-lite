"""YAML-style formatter for transforming Pydantic schemas.

This formatter creates a clean YAML-like representation with Python-style
type hints, optionally including metadata as inline comments.
"""

from typing import Any

import yaml

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

    @property
    def TYPE_MAP(self) -> dict[str, str]:
        """Type mapping for YAML format."""
        return {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
            "null": "None",
        }

    @property
    def comment_prefix(self) -> str:
        """Comment prefix for YAML format."""
        return "#"

    def _dump_yaml(self, data: dict[str, Any]) -> str:
        """
        Dump a dictionary to YAML format.

        Args:
            data: Dictionary to serialize to YAML.

        Returns:
            YAML string representation.
        """
        result = yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        return str(result).rstrip()

    def process_anyof(self, anyof: dict[str, Any]) -> str:
        """
        Process an anyOf field (union types) for YAML.

        Args:
            anyof: Dictionary containing anyOf definition.

        Returns:
            Formatted union type representation with | separator.
        """
        anyof_list = anyof.get("anyOf", [])
        if not anyof_list:
            return "str"

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

        return " | ".join(item_types) if item_types else "str"

    def process_enum(self, enum_value: dict[str, Any]) -> str:
        """
        Process an enum field for YAML.

        Args:
            enum_value: Dictionary containing enum definition.

        Returns:
            Formatted enum representation as Literal type.
        """
        enum_list = enum_value.get("enum", [])
        if not enum_list:
            return "str"

        # Create Literal type for Python-style YAML
        enum_literals = [f'"{val}"' for val in enum_list]
        return f"Literal[{', '.join(enum_literals)}]"

    def process_type_value(self, type_value: dict[str, Any]) -> str:
        """
        Process a type field for YAML.

        Args:
            type_value: Dictionary containing type definition.

        Returns:
            Formatted type representation with list[...] and str | None style.
        """
        type_name = type_value.get("type", "string")

        # Handle array of types (union types like ["string", "null"])
        if isinstance(type_name, list):
            if len(type_name) == 1:
                type_name = type_name[0]
            elif "null" in type_name and len(type_name) == 2:
                # Handle nullable types like ["string", "null"] -> "str | None"
                non_null_type = next(t for t in type_name if t != "null")
                type_str = self.TYPE_MAP.get(non_null_type, non_null_type)
                return f"{type_str} | None"
            else:
                # Multiple non-null types - treat as union
                type_strs = [self.TYPE_MAP.get(t, t) for t in type_name if t != "null"]
                return " | ".join(s for s in type_strs if s is not None)

        # Now type_name is guaranteed to be a string
        type_str = self.TYPE_MAP.get(type_name, type_name)

        # Add validation constraints to type description (only if metadata is enabled)
        if self.include_metadata:
            if type_name == "string":
                length_range = self._format_validation_range(
                    type_value, "minLength", "maxLength", " chars"
                )
                if length_range:
                    type_str = f"{type_str} ({length_range})"
            elif type_name in ["number", "integer"]:
                range_info = self._format_validation_range(type_value, "minimum", "maximum")
                if range_info:
                    type_str = f"{type_str} ({range_info})"

        if type_str == "list":
            # Handle list items
            items = type_value.get("items")
            if not items:
                type_str = "list[Any]"
            elif isinstance(items, bool):
                type_str = "list[Any]" if items else "list[Any]"
            elif isinstance(items, dict) and "type" in items:
                items_type = self.process_type_value(items)
                type_str = f"list[{items_type}]"
            elif isinstance(items, dict) and "$ref" in items:
                items_type = self.process_ref(items)
                type_str = f"list[{items_type}]"
            elif isinstance(items, dict) and "anyOf" in items:
                items_type = self.process_anyof(items)
                type_str = f"list[{items_type}]"
            else:
                type_str = "list[Any]"

            # Add array constraints to type description (only if metadata is enabled)
            if self.include_metadata:
                constraints = []
                if type_value.get("uniqueItems"):
                    constraints.append("unique")

                items_range = self._format_validation_range(
                    type_value, "minItems", "maxItems", " items"
                )
                if items_range:
                    constraints.append(f"length: {items_range}")

                if constraints:
                    type_str = f"{type_str} ({', '.join(constraints)})"

                # Add array-specific metadata (contains)
                if "contains" in type_value:
                    type_str += self.process_contains(type_value)

        if type_str == "dict":
            # For empty object types (no properties), return "{}"
            if (
                "properties" not in type_value
                and "patternProperties" not in type_value
                and not isinstance(type_value.get("additionalProperties"), dict)
            ):
                return "{}"
            # For object types with content, return "object" to keep YAML simple
            return "object"

        return str(type_str)

    def add_metadata(self, representation: str, value: dict[str, Any]) -> str:
        """
        Add metadata comments to a field representation.

        Args:
            representation: The base field representation.
            value: The field definition containing metadata.

        Returns:
            Field representation with metadata comments using # prefix.
        """
        if not self.include_metadata:
            return representation

        available_metadata = self.get_available_metadata(value)
        if not available_metadata:
            return representation

        metadata_parts = self.format_metadata_parts(value)
        return f"{representation}  # {', '.join(metadata_parts)}"

    def dict_to_string(self, value: Any, indent: int = 1) -> str:
        """
        Convert a dictionary to YAML-style key: value lines.

        This is used for formatting processed properties dict.

        Args:
            value: The value to convert (dict, list, or primitive).
            indent: Current indentation level (used for recursion depth
                tracking in nested structures).

        Returns:
            Formatted string representation as YAML key: value lines.
        """
        if isinstance(value, dict):
            if not value:  # Empty dict
                return "{}"

            # Format as YAML key: value pairs
            lines = []
            for k, v in value.items():
                lines.append(f"{k}: {v}")
            return "\n".join(lines)
        elif isinstance(value, list):
            # For arrays, show the item type
            if value and isinstance(value[0], dict):
                return f"list[{self.dict_to_string(value[0], indent + 1)}]"
            else:
                return "list"
        else:
            return str(value)

    def transform_schema(self) -> str:
        """
        Transform schema into YAML-style format.

        Returns:
            YAML-style schema definition as a string.
        """
        # First branch: if _processed_data is set, build from cache
        if hasattr(self, "_processed_data") and self._processed_data:
            all_sections = []

            # Process nested definitions first (from $defs) - same as main flow
            for def_name, def_schema in self.defs.items():
                if "properties" in def_schema:
                    nested_props = def_schema["properties"]
                    nested_required = set(def_schema.get("required", []))

                    # Build dict for this $def
                    def_dict = {}
                    for prop_name, prop_def in nested_props.items():
                        # Use process_property to get the type representation
                        prop_type = self.process_property(prop_def)
                        # Format field name with required indicator for nested definitions
                        formatted_prop_name = (
                            f"{prop_name}*" if prop_name in nested_required else prop_name
                        )
                        def_dict[f"{def_name}.{formatted_prop_name}"] = prop_type

                    # Dump to YAML and optionally prepend section header
                    section_str = self._dump_yaml(def_dict)
                    if self.include_metadata:
                        section_str = f"# {def_name}\n{section_str}"

                    all_sections.append(section_str)

            # Build main content from cached processed data
            main_parts = []

            # Add schema info comment if present
            schema_info_comment = self.get_schema_info_comment()
            if schema_info_comment:
                main_parts.append(schema_info_comment)

            # Add required fields comment if there are required fields
            required_comment = self.get_required_fields_comment()
            if required_comment:
                main_parts.append(required_comment)

            # Use cached processed data for main content
            main_parts.append(self._dump_yaml(self._processed_data))

            # Combine nested sections with main content
            if all_sections:
                all_sections.append("\n".join(main_parts))
                return "\n\n".join(all_sections)

            return "\n".join(main_parts)

        # Second branch: no properties - handle schema-level-only cases
        if not self.properties:
            # Handle schema-level features even when there are no properties
            schema_level_features = ""

            if "patternProperties" in self.schema:
                schema_level_features += self.process_pattern_properties(self.schema)

            if "dependencies" in self.schema:
                schema_level_features += self.process_dependencies(self.schema)

            if "if" in self.schema or "then" in self.schema or "else" in self.schema:
                schema_level_features += self.process_conditional(self.schema)

            if "propertyNames" in self.schema:
                schema_level_features += self.process_property_names(self.schema)

            if "unevaluatedProperties" in self.schema:
                schema_level_features += self.process_unevaluated_properties(self.schema)

            # Handle schema with type but no properties
            if "type" in self.schema:
                type_content = self.process_type_value(self.schema)
                # For object type with no properties, return {} instead of "object"
                if type_content == "object" and not schema_level_features:
                    return "{}"
                # Add schema-level features as comments if present
                if schema_level_features and self.include_metadata:
                    return (
                        f"# Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{type_content}"
                    )
                else:
                    return type_content
            elif "oneOf" in self.schema:
                oneof_content = self.process_oneof(self.schema)
                if schema_level_features and self.include_metadata:
                    return (
                        f"# Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{oneof_content}"
                    )
                else:
                    return oneof_content
            elif "anyOf" in self.schema:
                anyof_content = self.process_anyof(self.schema)
                if schema_level_features and self.include_metadata:
                    return (
                        f"# Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{anyof_content}"
                    )
                else:
                    return anyof_content
            elif "allOf" in self.schema:
                allof_content = self.process_allof(self.schema)
                if schema_level_features and self.include_metadata:
                    return (
                        f"# Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{allof_content}"
                    )
                else:
                    return allof_content

            # Return schema-level features as comments if present
            if schema_level_features and self.include_metadata:
                return f"# Schema-level constraints: {schema_level_features.strip()}"
            else:
                return "{}"

        # Third branch: main flow with properties
        all_sections = []

        # Process nested definitions first (from $defs)
        for def_name, def_schema in self.defs.items():
            if "properties" in def_schema:
                nested_props = def_schema["properties"]
                nested_required = set(def_schema.get("required", []))

                # Build dict for this $def
                def_dict = {}
                for prop_name, prop_def in nested_props.items():
                    # Use process_property to get the type representation
                    prop_type = self.process_property(prop_def)
                    # Format field name with required indicator for nested definitions
                    formatted_prop_name = (
                        f"{prop_name}*" if prop_name in nested_required else prop_name
                    )
                    def_dict[f"{def_name}.{formatted_prop_name}"] = prop_type

                # Dump to YAML and optionally prepend section header
                section_str = self._dump_yaml(def_dict)
                if self.include_metadata:
                    section_str = f"# {def_name}\n{section_str}"

                all_sections.append(section_str)

        # Build main content parts
        main_parts = []

        # Add schema info comment if present
        schema_info_comment = self.get_schema_info_comment()
        if schema_info_comment:
            main_parts.append(schema_info_comment)

        # Add required fields comment if there are required fields
        required_comment = self.get_required_fields_comment()
        if required_comment:
            main_parts.append(required_comment)

        # Process properties and cache the result
        processed_properties = self.process_properties(self.properties)

        # Dump processed properties to YAML
        main_parts.append(self._dump_yaml(processed_properties))

        # Set _processed_data for future calls (caching)
        self._processed_data = processed_properties

        # If there are nested sections, combine them
        if all_sections:
            all_sections.append("\n".join(main_parts))
            return "\n\n".join(all_sections)

        return "\n".join(main_parts)
