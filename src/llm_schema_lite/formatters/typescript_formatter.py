"""TypeScript interface formatter for transforming Pydantic schemas."""

from io import StringIO
from typing import Any

from .base import BaseFormatter


class TypeScriptFormatter(BaseFormatter):
    """
    Transforms Pydantic schema into TypeScript interface format.

    This formatter follows Pattern A: it uses base class schema processing
    (via process_schema and related methods) and caches processed data in
    _processed_data within transform_schema for improved performance on
    subsequent calls.

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

    @property
    def comment_prefix(self) -> str:
        """Comment prefix for TypeScript format."""
        return "//"

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

            # Check for properties BEFORE type, because objects with properties have both
            if "properties" in item:
                # Handle object schemas in anyOf - format as TypeScript object type
                processed_props = self.process_properties(item["properties"])
                props_str = ", ".join(f"{k}: {v}" for k, v in processed_props.items())
                item_types.append(f"{{ {props_str} }}")
            elif "enum" in item:
                item_types.append(self.process_enum(item))
            elif "const" in item:
                item_types.append(str(item["const"]))
            elif "$ref" in item:
                item_types.append(self.process_ref(item))
            elif "type" in item:
                item_types.append(self.process_type_value(item))

        # Limit the number of union types to prevent excessive expansion
        # Use same tiers as base formatter
        if self._global_expansion_count > 100:
            max_items = 2  # Very aggressive for deep recursion
        elif self._global_expansion_count > 30:
            max_items = 3  # Aggressive
        elif self._global_expansion_count > 10:
            max_items = 4  # Moderate
        else:
            max_items = 5  # Conservative start

        if len(item_types) > max_items:
            return f"anyOf: {len(item_types)} options"
        else:
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

        # Add validation constraints to type description (only if metadata is enabled)
        if self.include_metadata:
            if type_name == "string":
                length_range = self._format_validation_range(
                    type_value, "minLength", "maxLength", " chars"
                )  # noqa: E501
                if length_range:
                    type_str = f"{type_str} ({length_range})"
            elif type_name in ["number", "integer"]:
                range_info = self._format_validation_range(type_value, "minimum", "maximum")
                if range_info:
                    type_str = f"{type_str} ({range_info})"

        # Handle array type (consolidate both "Array" and "array" cases)
        if type_str == "Array" or type_name == "array":
            items = type_value.get("items")
            if not items:
                array_type = "Array<any>"
            elif isinstance(items, bool):
                array_type = "Array<any>" if items else "Array<any>"
            elif isinstance(items, dict):
                # Handle object items with properties - expand inline
                if "properties" in items:
                    processed_properties = self.process_properties(items["properties"])
                    props_str = ", ".join(f"{k}: {v}" for k, v in processed_properties.items())
                    array_type = f"Array<{{ {props_str} }}>"
                # Handle type in items
                elif "type" in items:
                    items_type = self.process_type_value(items)
                    array_type = f"Array<{items_type}>"
                # Handle $ref in items
                elif "$ref" in items:
                    items_type = self.process_ref(items)
                    array_type = f"Array<{items_type}>"
                # Handle anyOf in items
                elif "anyOf" in items:
                    items_type = self.process_anyof(items)
                    array_type = f"Array<{items_type}>"
                # Handle allOf in items
                elif "allOf" in items:
                    items_type = self.process_allof(items)
                    array_type = f"Array<{items_type}>"
                # Handle oneOf in items
                elif "oneOf" in items:
                    items_type = self.process_oneof(items)
                    array_type = f"Array<{items_type}>"
                else:
                    array_type = "Array<any>"
            else:
                array_type = "Array<any>"

            # Add array constraints (only if metadata is enabled)
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
                    array_type = f"{array_type} ({', '.join(constraints)})"

                # Add contains and unique items metadata
                if "contains" in type_value:
                    array_type += self.process_contains(type_value)
                if "uniqueItems" in type_value:
                    array_type += self.process_unique_items(type_value)

            return array_type

        return str(type_str)

    def dict_to_string(self, value: Any, indent: int = 1) -> str:
        """
        Convert a dictionary or list to a formatted string representation.

        This is used for processed properties dict or nested objects.

        Args:
            value: The value to convert (dict, list, or primitive).
            indent: Current indentation level.

        Returns:
            Formatted string representation.
        """
        if isinstance(value, dict):
            if not value:  # Empty dict
                return "{}"

            # Format as TypeScript object representation (inline style)
            pairs = []
            for k, v in value.items():
                pairs.append(f"'{k}': '{v}'")
            return "{" + ", ".join(pairs) + "}"
        elif isinstance(value, list):
            return "[" + ", ".join(str(v) for v in value) + "]"
        else:
            return str(value)

    def transform_schema(self) -> str:
        """
        Transform schema into TypeScript interface syntax.

        This method implements a three-branch structure:
        1. If _processed_data is cached, build output from cache
        2. If no properties, handle schema-level-only cases
        3. Main flow: process properties, cache result, and return

        Returns:
            TypeScript interface definition as a string.
        """
        # First branch: if _processed_data is set, build from cache
        if hasattr(self, "_processed_data") and self._processed_data:
            all_interfaces = []

            # Process nested definitions first (from $defs) - same as main flow
            for def_name, def_schema in self.defs.items():
                if "properties" in def_schema:
                    nested_output = StringIO()
                    nested_output.write(f"interface {def_name} {{\n")

                    nested_props = def_schema["properties"]
                    nested_required = set(def_schema.get("required", []))

                    for prop_name, prop_def in nested_props.items():
                        prop_type = self.process_property(prop_def)
                        # Format field name with required indicator for nested definitions
                        formatted_prop_name = (
                            f"{prop_name}*" if prop_name in nested_required else prop_name
                        )
                        nested_output.write(f"  {formatted_prop_name}: {prop_type};\n")

                    nested_output.write("}")
                    all_interfaces.append(nested_output.getvalue())

            # Build main content from cached processed data
            main_output = StringIO()

            # Add schema info comment if present
            schema_info_comment = self.get_schema_info_comment()
            if schema_info_comment:
                main_output.write(f"{schema_info_comment}\n")

            # Add required fields comment if there are required fields
            required_comment = self.get_required_fields_comment()
            if required_comment:
                main_output.write(f"{required_comment}\n")

            main_output.write("interface Schema {\n")

            # Use cached processed data
            for name, prop_type in self._processed_data.items():
                main_output.write(f"  {name}: {prop_type};\n")

            main_output.write("}")
            all_interfaces.append(main_output.getvalue())

            return "\n\n".join(all_interfaces)

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
                # For empty object type, return interface instead of type alias
                if self.schema.get("type") == "object" and not schema_level_features:
                    return "interface Schema {}"
                type_content = self.process_type_value(self.schema)
                result = f"type Schema = {type_content};"
                # Add schema-level features as comments if present
                if schema_level_features and self.include_metadata:
                    result = (
                        f"// Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{result}"
                    )
                return result
            elif "oneOf" in self.schema:
                oneof_content = self.process_oneof(self.schema)
                result = f"type Schema = {oneof_content};"
                if schema_level_features and self.include_metadata:
                    result = (
                        f"// Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{result}"
                    )
                return result
            elif "anyOf" in self.schema:
                anyof_content = self.process_anyof(self.schema)
                result = f"type Schema = {anyof_content};"
                if schema_level_features and self.include_metadata:
                    result = (
                        f"// Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{result}"
                    )
                return result
            elif "allOf" in self.schema:
                allof_content = self.process_allof(self.schema)
                result = f"type Schema = {allof_content};"
                if schema_level_features and self.include_metadata:
                    result = (
                        f"// Schema-level constraints: {schema_level_features.strip()}\n"
                        f"{result}"
                    )
                return result
            else:
                # Return schema-level features as comments if present
                if schema_level_features and self.include_metadata:
                    return (
                        f"// Schema-level constraints: {schema_level_features.strip()}\n"
                        "interface Schema {}"
                    )
                return "interface Schema {}"

        # Collect all interface definitions
        all_interfaces = []

        # Process nested definitions first (from $defs)
        for def_name, def_schema in self.defs.items():
            if "properties" in def_schema:
                nested_output = StringIO()
                nested_output.write(f"interface {def_name} {{\n")

                nested_props = def_schema["properties"]
                nested_required = set(def_schema.get("required", []))

                for prop_name, prop_def in nested_props.items():
                    prop_type = self.process_property(prop_def)
                    # Format field name with required indicator for nested definitions
                    formatted_prop_name = (
                        f"{prop_name}*" if prop_name in nested_required else prop_name
                    )
                    nested_output.write(f"  {formatted_prop_name}: {prop_type};\n")

                nested_output.write("}")
                all_interfaces.append(nested_output.getvalue())

        # Process main interface
        main_output = StringIO()

        # Add schema info comment if present
        schema_info_comment = self.get_schema_info_comment()
        if schema_info_comment:
            main_output.write(f"{schema_info_comment}\n")

        # Add required fields comment if there are required fields
        required_comment = self.get_required_fields_comment()
        if required_comment:
            main_output.write(f"{required_comment}\n")

        main_output.write("interface Schema {\n")

        processed_properties = self.process_properties(self.properties)
        for name, prop_type in processed_properties.items():
            # process_properties() already includes metadata via process_property()
            # so we don't need to add it again
            main_output.write(f"  {name}: {prop_type};\n")

        main_output.write("}")
        all_interfaces.append(main_output.getvalue())

        # Set _processed_data for future calls (caching)
        self._processed_data = processed_properties

        return "\n\n".join(all_interfaces)
