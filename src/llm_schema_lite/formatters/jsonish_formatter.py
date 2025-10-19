"""JSONish formatter for transforming Pydantic schemas into BAML-like format."""

from typing import Any

from .base import BaseFormatter


class JSONishFormatter(BaseFormatter):
    """
    Transforms Pydantic schema into a JSONish (BAML-like) representation.

    This formatter creates a JSON-like format (not valid JSON) with inline comments
    for metadata, optimized for LLM consumption with minimal token usage.

    Example output:
        {
         name: string  //description, minLength: 1,
         age: int  //min: 0, max: 120
        }
    """

    @property
    def TYPE_MAP(self) -> dict[str, str]:
        """Type mapping for JSONish format."""
        return {"number": "float", "integer": "int", "boolean": "bool"}

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
        return f"{representation}  //{', '.join(metadata_parts)}"

    @classmethod
    def dict_to_string(cls, value: Any, indent: int = 1) -> str:
        """
        Convert a dictionary or list to a formatted string representation.

        Args:
            value: The value to convert (dict, list, or primitive).
            indent: Current indentation level.

        Returns:
            Formatted string representation.
        """
        from io import StringIO

        def _write_value(val: Any, current_indent: int) -> str:
            """Recursively write value to StringIO buffer."""
            if val is None:
                return "null"
            elif isinstance(val, str):
                return val
            elif isinstance(val, dict):
                if not val:  # Empty dict
                    return "{}"

                output = StringIO()
                output.write("{\n")

                items = []
                for k, v in val.items():
                    item_indent = " " * (current_indent + 1)
                    item_value = _write_value(v, current_indent + 1)
                    items.append(f"{item_indent}{k}: {item_value}")

                # Join items with comma and newline
                output.write(",\n".join(items))
                output.write(f"\n{' ' * current_indent}}}")
                return output.getvalue()

            elif isinstance(val, list):
                if not val:  # Empty list
                    return "[]"

                output = StringIO()
                output.write("[\n")

                items = []
                for v in val:
                    item_indent = " " * (current_indent + 1)
                    item_value = _write_value(v, current_indent + 1)
                    items.append(f"{item_indent}{item_value}")

                # Join items with comma and newline
                output.write(",\n".join(items))
                output.write(f"\n{' ' * current_indent}]")
                return output.getvalue()
            else:
                return str(val)

        return _write_value(value, indent)

    def transform_schema(self) -> str:
        """
        Transform the schema into a simplified string representation.

        Returns:
            Formatted schema as a string.
        """
        result = ""

        # Handle schema-level features first
        if "patternProperties" in self.schema:
            result += self.process_pattern_properties(self.schema)

        if "dependencies" in self.schema:
            result += self.process_dependencies(self.schema)

        if "if" in self.schema or "then" in self.schema or "else" in self.schema:
            result += self.process_conditional(self.schema)

        if "propertyNames" in self.schema:
            result += self.process_property_names(self.schema)

        if "unevaluatedProperties" in self.schema:
            result += self.process_unevaluated_properties(self.schema)

        # Handle main schema content
        if self.properties:
            processed_properties = self.process_properties(self.properties)
            main_content = self.dict_to_string(processed_properties, indent=0)

            # Add schema-level constraints as metadata
            if result:
                # If we have schema-level features, add them as comments
                return f"{main_content}  //{result.strip()}"
            else:
                return main_content
        elif "type" in self.schema:
            type_content = self.process_type_value(self.schema)
            # For object type with no properties, return {} instead of "object"
            if type_content == "object" and not result:
                return "{}"
            # Add schema-level constraints as metadata if present
            if result:
                return f"{type_content}  //{result.strip()}"
            else:
                return type_content

        return result if result else ""
