"""Base formatter abstract class for schema formatters."""

import re
from abc import ABC, abstractmethod
from typing import Any


class BaseFormatter(ABC):
    """
    Abstract base class for schema formatters.

    All schema formatters must inherit from this class and implement
    the required methods.
    """

    # Common regex pattern for $ref processing
    REF_PATTERN = re.compile(r"#/\$defs/(.+)$", re.IGNORECASE)

    # Common metadata mapping for all formatters
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
        "const": lambda v: f"const: {v}",
        "if": lambda v: f"if: {v}",
        "then": lambda v: f"then: {v}",
        "else": lambda v: f"else: {v}",
        "contains": lambda v: f"contains: {v}",
        "dependencies": lambda v: f"dependencies: {v}",
        "patternProperties": lambda v: f"patternProperties: {v}",
        "propertyNames": lambda v: f"propertyNames: {v}",
        "unevaluatedProperties": lambda v: f"unevaluatedProperties: {v}",
        "minItems": lambda v: f"minItems: {v}",
        "maxItems": lambda v: f"maxItems: {v}",
        "minProperties": lambda v: f"minProperties: {v}",
        "maxProperties": lambda v: f"maxProperties: {v}",
        "exclusiveMinimum": lambda v: f"exclusiveMin: {v}",
        "exclusiveMaximum": lambda v: f"exclusiveMax: {v}",
        "uniqueItems": lambda v: "unique items" if v else "",
        "additionalItems": lambda v: f"additionalItems: {v}" if isinstance(v, dict) else "",
    }

    def __init__(self, schema: dict[str, Any], include_metadata: bool = True):
        """
        Initialize the formatter.

        Args:
            schema: JSON schema from Pydantic model_json_schema.
            include_metadata: Whether to include metadata in the output.
        """
        self.schema = schema
        self.include_metadata = include_metadata
        self.defs = schema.get("$defs", {})
        self.properties = schema.get("properties", {})
        self._ref_cache: dict[str, str] = {}
        self._recursion_depth: dict[str, int] = {}
        self._max_recursion_depth = 10

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

    def get_available_metadata(self, value: dict[str, Any]) -> list[str]:
        """
        Get available metadata keys for a property.

        Args:
            value: The field definition containing metadata.

        Returns:
            List of available metadata keys.
        """
        return [
            k
            for k in self.METADATA_MAP.keys()
            if k in value and not (k == "default" and value[k] is None)
        ]

    def format_metadata_parts(self, value: dict[str, Any]) -> list[str]:
        """
        Format metadata parts for a property.

        Args:
            value: The field definition containing metadata.

        Returns:
            List of formatted metadata strings.
        """
        available_metadata = self.get_available_metadata(value)
        return [self.METADATA_MAP[k](value[k]) for k in available_metadata]  # type: ignore[no-untyped-call]

    @property
    @abstractmethod
    def TYPE_MAP(self) -> dict[str, str]:
        """Type mapping dictionary for the formatter."""
        pass

    def process_ref(self, ref: dict[str, Any]) -> str:
        """
        Process a $ref reference to a definition.

        Args:
            ref: Dictionary containing the $ref key.

        Returns:
            Processed reference representation.
        """
        ref_str: str = ref.get("$ref", "")
        if not ref_str:
            return "object"

        # Safely extract ref key with null check
        ref_match = self.REF_PATTERN.search(ref_str)
        if not ref_match:
            return "object"  # Fallback for invalid ref

        ref_key = ref_match.group(1)
        if ref_key in self._ref_cache:
            return self._ref_cache[ref_key]

        # Check recursion depth
        current_depth = self._recursion_depth.get(ref_key, 0)
        if current_depth >= self._max_recursion_depth:
            return "object"  # Prevent infinite recursion

        # Increment recursion depth
        self._recursion_depth[ref_key] = current_depth + 1

        try:
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
        finally:
            # Decrement recursion depth
            self._recursion_depth[ref_key] = current_depth

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

        # Handle array of types in enum
        if isinstance(enum_type, list):
            if len(enum_type) == 1:
                enum_type = enum_type[0]
                type_str = self.TYPE_MAP.get(enum_type, enum_type)
            elif "null" in enum_type and len(enum_type) == 2:
                # Handle nullable enum types
                non_null_type = next(t for t in enum_type if t != "null")
                type_str = self.TYPE_MAP.get(non_null_type, non_null_type)
            else:
                # Multiple types - use first non-null type
                non_null_types = [t for t in enum_type if t != "null"]
                type_str = self.TYPE_MAP.get(
                    non_null_types[0] if non_null_types else "string", "string"
                )
        else:
            type_str = self.TYPE_MAP.get(enum_type, enum_type)

        # Now type_str is guaranteed to be defined (either from the if/elif branches or the else)
        enum_values = ", ".join(str(e) for e in enum_list)
        return f"{type_str} //oneOf: {enum_values}"

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

        # Handle array of types (union types like ["string", "null"])
        if isinstance(type_name, list):
            if len(type_name) == 1:
                type_name = type_name[0]
            elif "null" in type_name and len(type_name) == 2:
                # Handle nullable types like ["string", "null"] -> "string?"
                non_null_type = next(t for t in type_name if t != "null")
                type_str = self.TYPE_MAP.get(non_null_type, non_null_type)
                return f"{type_str}?"  # Mark as nullable
            else:
                # Multiple non-null types - treat as union
                type_strs = [self.TYPE_MAP.get(t, t) for t in type_name if t != "null"]
                return " or ".join(s for s in type_strs if s is not None)

        # Now type_name is guaranteed to be a string
        type_str = self.TYPE_MAP.get(type_name, type_name)

        if type_str == "array":
            # Safely handle array items
            items = type_value.get("items")
            if not items:
                type_str = "array"  # Fallback for array without items
            elif isinstance(items, bool):
                # Handle boolean items (true means any type, false means no items)
                type_str = "array" if items else "array"
            elif isinstance(items, dict) and "type" in items:
                items_type = self.process_type_value(items)
                type_str = f"{items_type}[]"
            elif isinstance(items, dict) and "$ref" in items:
                items_type = self.process_ref(items)
                type_str = f"[{items_type}]"
            elif isinstance(items, dict) and "anyOf" in items:
                items_type = self.process_anyof(items)
                type_str = f"[{items_type}]"
            else:
                type_str = "array"  # Fallback for unknown array item type

            # Add array-specific metadata (contains, uniqueItems)
            if "contains" in type_value:
                type_str += self.process_contains(type_value)
            if "uniqueItems" in type_value:
                type_str += self.process_unique_items(type_value)

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
            # Skip non-dictionary items (like booleans)
            if not isinstance(item, dict):
                continue

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

    def process_oneof(self, oneof: dict[str, Any]) -> str:
        """Process oneOf (exclusive choice) schemas."""
        oneof_list = oneof.get("oneOf", [])
        if not oneof_list:
            return "string"

        item_types = []
        for item in oneof_list:
            # Skip non-dictionary items (like booleans)
            if not isinstance(item, dict):
                continue

            if "type" in item:
                item_types.append(self.process_type_value(item))
            elif "$ref" in item:
                item_types.append(self.process_ref(item))
            elif "enum" in item:
                item_types.append(self.process_enum(item))

        return f"oneOf: {' | '.join(item_types)}" if item_types else "string"

    def process_allof(self, allof: dict[str, Any]) -> str:
        """Process allOf (intersection) schemas."""
        allof_list = allof.get("allOf", [])
        if not allof_list:
            return "string"

        # For allOf, we process each schema and combine them
        # This is a simplified approach - in practice, allOf is complex
        item_types = []
        for item in allof_list:
            if isinstance(item, dict):
                if "type" in item:
                    item_types.append(self.process_type_value(item))
                elif "$ref" in item:
                    item_types.append(self.process_ref(item))
                elif "properties" in item:
                    # Handle object schemas in allOf
                    processed_props = self.process_properties(item["properties"])
                    item_types.append(self.dict_to_string(processed_props, indent=2))
                else:
                    item_types.append("object")

        if item_types:
            return f"allOf: {' & '.join(item_types)}"
        else:
            return "object"

    def process_not(self, not_schema: dict[str, Any]) -> str:
        """Process not (negation) schemas."""
        not_def = not_schema.get("not", {})
        if not_def:
            return f"not: {self.process_type_value(not_def)}"
        return "string"

    def process_property(self, _property: Any) -> str:
        """
        Process a single property from the schema.

        Args:
            _property: Property definition from the schema.

        Returns:
            Processed property representation as a string.
        """
        # Handle non-dictionary property values (like booleans, strings, numbers)
        if not isinstance(_property, dict):
            if isinstance(_property, bool):
                return "bool"
            elif isinstance(_property, str):
                return "string"
            elif isinstance(_property, int | float):
                return "number"
            else:
                return "any"

        if "$ref" in _property:
            prop_str = self.process_ref(_property)
        elif "enum" in _property:
            prop_str = self.process_enum(_property)
        elif "anyOf" in _property:
            prop_str = self.process_anyof(_property)
        elif "oneOf" in _property:
            prop_str = self.process_oneof(_property)
        elif "allOf" in _property:
            prop_str = self.process_allof(_property)
        elif "not" in _property:
            prop_str = self.process_not(_property)
        elif "type" in _property:
            prop_str = self.process_type_value(_property)
        else:
            # Fallback for properties without recognizable type
            prop_str = "string"

        return self.add_metadata(prop_str, _property)

    @abstractmethod
    def add_metadata(self, representation: str, value: dict[str, Any]) -> str:
        """
        Add metadata comments to a field representation.

        Args:
            representation: The base field representation.
            value: The field definition containing metadata.

        Returns:
            Field representation with metadata comments.
        """
        pass

    def dict_to_string(self, value: Any, indent: int = 1) -> str:
        """
        Convert a dictionary or list to a formatted string representation.

        Args:
            value: The value to convert (dict, list, or primitive).
            indent: Current indentation level.

        Returns:
            Formatted string representation.
        """
        # Default implementation - subclasses can override
        return str(value)

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

    def process_additional_properties(self, schema: dict[str, Any]) -> str:
        """Process additionalProperties constraint."""
        additional_props = schema.get("additionalProperties")
        if additional_props is False:
            return " //no additional properties"
        elif isinstance(additional_props, dict):
            return f" //additional: {self.process_type_value(additional_props)}"
        return ""

    def process_pattern_properties(self, schema: dict[str, Any]) -> str:
        """Process patternProperties constraint."""
        pattern_props = schema.get("patternProperties", {})
        if pattern_props:
            patterns = []
            for pattern, definition in pattern_props.items():
                patterns.append(f"{pattern}: {self.process_type_value(definition)}")
            return f" //patternProperties: {', '.join(patterns)}"
        return ""

    def process_dependencies(self, schema: dict[str, Any]) -> str:
        """Process dependencies constraint."""
        dependencies = schema.get("dependencies", {})
        if dependencies:
            deps = []
            for prop, deps_list in dependencies.items():
                if isinstance(deps_list, list):
                    deps.append(f"{prop} requires {', '.join(deps_list)}")
                else:
                    deps.append(f"{prop} requires {deps_list}")
            return f" //dependencies: {', '.join(deps)}"
        return ""

    def process_conditional(self, schema: dict[str, Any]) -> str:
        """Process if/then/else conditional schemas."""
        if_clause = schema.get("if")
        then_clause = schema.get("then")
        else_clause = schema.get("else")

        if if_clause and then_clause:
            condition = self.process_type_value(if_clause)
            consequence = self.process_type_value(then_clause)
            result = f"if {condition} then {consequence}"

            if else_clause:
                alternative = self.process_type_value(else_clause)
                result += f" else {alternative}"

            return f" //{result}"
        return ""

    def process_contains(self, schema: dict[str, Any]) -> str:
        """Process contains constraint for arrays."""
        contains = schema.get("contains")
        if contains:
            return f" //contains: {self.process_type_value(contains)}"
        return ""

    def process_unique_items(self, schema: dict[str, Any]) -> str:
        """Process uniqueItems constraint."""
        unique = schema.get("uniqueItems")
        if unique:
            return " //unique items"
        return ""

    def process_property_names(self, schema: dict[str, Any]) -> str:
        """Process propertyNames constraint."""
        prop_names = schema.get("propertyNames")
        if prop_names:
            return f" //propertyNames: {self.process_type_value(prop_names)}"
        return ""

    def process_unevaluated_properties(self, schema: dict[str, Any]) -> str:
        """Process unevaluatedProperties constraint."""
        uneval_props = schema.get("unevaluatedProperties")
        if uneval_props is False:
            return " //no unevaluated properties"
        elif isinstance(uneval_props, dict):
            return f" //unevaluated: {self.process_type_value(uneval_props)}"
        return ""

    @abstractmethod
    def transform_schema(self) -> str:
        """
        Transform the schema into the desired format.

        Returns:
            Formatted schema as a string.
        """
        pass
