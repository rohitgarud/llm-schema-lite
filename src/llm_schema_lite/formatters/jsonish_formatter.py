"""JSONish formatter for transforming Pydantic schemas into BAML-like format."""

import re
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

    REF_PATTERN = re.compile(r"#/\$defs/(.+)$")
    TYPE_MAP = {"number": "float", "integer": "int", "boolean": "bool"}

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
        Initialize the JSONish formatter.

        Args:
            schema: JSON schema from Pydantic model_json_schema.
            include_metadata: Whether to include metadata as inline comments.
        """
        super().__init__(schema, include_metadata)
        self._ref_cache: dict[str, str] = {}

    def add_metadata(self, representation: str, value: dict[str, Any]) -> str:
        """
        Add metadata comments to a field representation.

        Args:
            representation: The base field representation.
            value: The field definition containing metadata.

        Returns:
            Field representation with metadata comments.
        """
        metadata = []
        if self.include_metadata:
            for k, formatter in self.METADATA_MAP.items():
                if k in value and not (k == "default" and value[k] is None):
                    metadata.append(formatter(value[k]))  # type: ignore[no-untyped-call]
        return f"{representation}  //{', '.join(metadata)}" if metadata else representation

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

        def join_items(items: list[str], indent: int) -> str:
            join_str = ",\n" + " " * indent
            return " " * indent + join_str.join(items) + "\n" + " " * (indent - 1)

        if value is None:
            return "null"
        elif isinstance(value, str):
            return value
        elif isinstance(value, dict):
            items = []
            for k, v in value.items():
                items.append(f"{k}: {cls.dict_to_string(v, indent + 1)}")
            return "{\n" + join_items(items, indent + 1) + "}"
        elif isinstance(value, list):
            items = []
            for _, v in enumerate(value):
                items.append(f"{cls.dict_to_string(v, indent + 1)}")
            return "[\n" + join_items(items, indent + 1) + "]"
        return str(value)  # Ensure string return type

    def process_ref(self, ref: dict[str, Any]) -> str:
        """
        Process a $ref reference to a definition.

        Args:
            ref: Dictionary containing the $ref key.

        Returns:
            Processed reference representation.
        """
        # Safely extract ref key with null check
        ref_match = self.REF_PATTERN.search(ref.get("$ref", ""))
        if not ref_match:
            return "object"  # Fallback for invalid ref

        ref_key = ref_match.group(1)
        if ref_key in self._ref_cache:
            return self._ref_cache[ref_key]

        # Safely get ref definition
        ref_def = self.defs.get(ref_key)
        if not ref_def:
            return "object"  # Fallback for missing definition

        if "enum" in ref_def:
            # Handle enum definitions
            ref_str = self.process_enum(ref_def)
        elif "properties" in ref_def:
            processed_properties = self.process_properties(ref_def["properties"])
            ref_str = self.dict_to_string(processed_properties, indent=2)
        elif "type" in ref_def:
            ref_str = self.process_type_value(ref_def)
        else:
            ref_str = "object"

        self._ref_cache[ref_key] = ref_str
        return ref_str

    def process_enum(self, enum_value: dict[str, Any]) -> str:
        """
        Process an enum field.

        Args:
            enum_value: Dictionary containing enum definition.

        Returns:
            Formatted enum representation.
        """
        # Safely get enum values
        enum_list = enum_value.get("enum", [])
        if not enum_list:
            return "string"  # Fallback for empty enum

        enum_type = enum_value.get("type", "string")
        type_str = (
            self.TYPE_MAP.get(enum_type, enum_type) if isinstance(enum_type, str) else "string"
        )
        return f"{type_str} //oneOf: {', '.join(str(e) for e in enum_list)}"

    def process_type_value(self, type_value: dict[str, Any]) -> str:
        """
        Process a type field.

        Args:
            type_value: Dictionary containing type definition.

        Returns:
            Formatted type representation.
        """
        # Safely get type with fallback
        type_name = type_value.get("type", "string")
        type_str = self.TYPE_MAP.get(type_name, type_name)

        if type_str == "array":
            # Safely handle array items
            items = type_value.get("items")
            if not items:
                type_str = "array"  # Fallback for array without items
            elif "type" in items:
                items_type = self.process_type_value(items)
                type_str = f"{items_type}[]"
            elif "$ref" in items:
                items_type = self.process_ref(items)
                type_str = f"[{items_type}]"
            elif "anyOf" in items:
                items_type = self.process_anyof(items)
                type_str = f"[{items_type}]"
            else:
                type_str = "array"  # Fallback for unknown array item type

        return type_str  # type: ignore[no-any-return]

    def process_anyof(self, anyof: dict[str, Any]) -> str:
        """
        Process an anyOf field (union types).

        Args:
            anyof: Dictionary containing anyOf definition.

        Returns:
            Formatted union type representation.
        """
        # Safely get anyOf list
        anyof_list = anyof.get("anyOf", [])
        if not anyof_list:
            return "string"  # Fallback for empty anyOf

        item_types = []
        array_found = False
        for item in anyof_list:
            if "enum" in item:
                item_types.append(self.process_enum(item))
            elif "const" in item:
                item_types.append(str(item["const"]))
            elif "$ref" in item:
                item_types.append(self.process_ref(item))
            elif "type" in item:
                if item["type"] == "array":
                    array_found = True

                if array_found and item["type"] == "null":
                    item_types.append("[]")
                else:
                    item_types.append(self.process_type_value(item))
            else:
                # Unknown anyOf item, skip it
                continue

        return " or ".join(item_types) if item_types else "string"

    def process_property(self, _property: dict[str, Any]) -> str:
        """
        Process a single property from the schema.

        Args:
            _property: Property definition from the schema.

        Returns:
            Processed property representation as a string.
        """
        if "$ref" in _property:
            prop_str = self.process_ref(_property)
        elif "enum" in _property:
            prop_str = self.process_enum(_property)
        elif "anyOf" in _property:
            prop_str = self.process_anyof(_property)
        elif "type" in _property:
            prop_str = self.process_type_value(_property)
        else:
            # Fallback for properties without recognizable type
            prop_str = "string"

        return self.add_metadata(prop_str, _property)

    def process_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """
        Process multiple properties from the schema.

        Args:
            properties: Dictionary of property definitions.

        Returns:
            Dictionary of processed properties.
        """
        processed_properties = {}
        for prop_name, value in properties.items():
            processed_properties[prop_name] = self.process_property(value)

        return processed_properties

    def transform_schema(self) -> str:
        """
        Transform the schema into a simplified string representation.

        Returns:
            Formatted schema as a string.
        """
        if self.properties:
            processed_properties = self.process_properties(self.properties)
            return self.dict_to_string(processed_properties, indent=0)
        elif "type" in self.schema:
            return self.process_type_value(self.schema)
        return ""
