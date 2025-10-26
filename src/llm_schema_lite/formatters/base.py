"""Base formatter abstract class for schema formatters."""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class BaseFormatter(ABC):
    """
    Abstract base class for schema formatters.

    All schema formatters must inherit from this class and implement
    the required methods.
    """

    # Common regex pattern for $ref processing
    # REF_PATTERN = re.compile(r"#/\$defs/(.+)$", re.IGNORECASE)
    REF_PATTERN = re.compile(r"#/(?:definitions|\$defs)/(.+)$", re.IGNORECASE)

    # Common metadata mapping for all formatters
    METADATA_MAP: dict[str, Callable[[Any], str]] = {
        "default": lambda v: f"(defaults to {v})",
        "description": lambda v: str(v),
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
        self.defs = schema.get("$defs", schema.get("definitions", {}))
        self.properties = schema.get("properties", {})
        self.required_fields = set(schema.get("required", []))
        self._ref_cache: dict[str, str] = {}
        self._recursion_depth: dict[str, int] = {}
        self._max_recursion_depth = 3  # Further reduced to prevent infinite expansion
        self._processed_refs: set[str] = set()  # Track processed refs to prevent cycles
        self._expansion_count: dict[str, int] = {}  # Track expansion count per ref
        self._max_expansions = 3  # Further reduced to prevent excessive expansion
        self._ref_depth_tracker: dict[str, int] = {}  # Track depth per ref path
        self._processed_data: dict[str, Any] | None = None  # Set by core.py for transform_schema()
        self._max_ref_depth = 2  # Maximum depth for $ref resolution

        # Priority 1: Global expansion budget to prevent extreme expansion
        self._global_expansion_budget = 150  # Max total $ref expansions across entire schema
        self._global_expansion_count = 0  # Track total expansions
        self._ref_expansion_path: list[str] = []  # Track current expansion path for cycle detection
        self._expansion_fingerprints: set[str] = set()  # Track expansion patterns to detect cycles

        # Pre-warm cache for common patterns

    def process_schema(self) -> dict[str, Any]:
        """
        Process the schema and return the appropriate data structure.

        This method handles all schema processing logic that was previously in core.py.
        It determines the appropriate processing method based on the schema structure.

        Returns:
            Dictionary containing processed schema data.
        """
        # Handle different top-level schema types
        if "$ref" in self.schema and not self.schema.get("properties"):
            # Handle $ref at top level
            ref_result = self.process_ref(self.schema)
            if ref_result == "object" or not ref_result:
                # Fallback for failed ref resolution
                ref_path = self.schema.get("$ref", "")
                if ref_path:
                    ref_match = self.REF_PATTERN.search(ref_path)
                    if ref_match:
                        ref_key = ref_match.group(1)
                        ref_def = self.defs.get(ref_key)
                        if ref_def and isinstance(ref_def, dict):
                            if "properties" in ref_def:
                                # Process properties from resolved definition
                                processed_props = self.process_properties(ref_def["properties"])
                                return {"schema": self.dict_to_string(processed_props, indent=0)}
                            elif "enum" in ref_def:
                                # Handle enum in resolved definition
                                return {"schema": self.process_enum(ref_def)}
                            elif "oneOf" in ref_def:
                                return {"schema": self.process_oneof(ref_def)}
                            elif "anyOf" in ref_def:
                                return {"schema": self.process_anyof(ref_def)}
                            elif "allOf" in ref_def:
                                return {"schema": self.process_allof(ref_def)}
                            elif "type" in ref_def:
                                return {"schema": self.process_type_value(ref_def)}
                        return {"schema": f"object  {self.comment_prefix}$ref: {ref_path}"}
                    return {"schema": f"object  {self.comment_prefix}$ref resolution failed"}
                return {"schema": "object"}
            else:
                return {"schema": ref_result}
        elif "properties" in self.schema and self.schema.get("properties"):
            # Handle object schemas with properties
            schema_type = self.schema.get("type")
            if schema_type in ("array", "string", "number", "integer", "boolean", "null"):
                # Let type handling take precedence for non-object types
                return {"schema": self.process_type_value(self.schema)}
            else:
                # Check if we have schema-level features that need to be included
                has_schema_features = any(
                    key in self.schema
                    for key in [
                        "dependencies",
                        "if",
                        "then",
                        "else",
                        "patternProperties",
                        "propertyNames",
                        "unevaluatedProperties",
                    ]
                )

                if has_schema_features:
                    # Use transform_schema() to include schema-level features
                    return {"schema": self.transform_schema()}
                else:
                    # Return processed properties dict for to_dict() compatibility
                    return self.process_properties(self.schema.get("properties", {}))
        elif "type" in self.schema:
            # Handle schemas with type but no properties
            if self.schema.get("type") == "object":
                return {"schema": self.transform_schema()}
            else:
                return {"schema": self.process_type_value(self.schema)}
        elif "oneOf" in self.schema:
            return {"schema": self.process_oneof(self.schema)}
        elif "anyOf" in self.schema:
            return {"schema": self.process_anyof(self.schema)}
        elif "allOf" in self.schema:
            return {"schema": self.process_allof(self.schema)}
        else:
            # Fallback for unknown schema types
            return {"schema": "object"}

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

    def _is_problematic_schema(self, schema: dict[str, Any]) -> bool:
        """Detect schemas that are likely to cause issues."""
        # Check for very large schemas
        if len(str(schema)) > 50000:  # Very large schemas
            return True

        # Check for schemas with many definitions
        defs = schema.get("$defs", schema.get("definitions", {}))
        if len(defs) > 100:  # Too many definitions
            return True

        # Check for schemas with deep nesting
        def _check_depth(obj: Any, current_depth: int = 0, max_depth: int = 10) -> bool:
            if current_depth > max_depth:
                return True
            if isinstance(obj, dict):
                for value in obj.values():
                    if _check_depth(value, current_depth + 1, max_depth):
                        return True
            elif isinstance(obj, list):
                for item in obj:
                    if _check_depth(item, current_depth + 1, max_depth):
                        return True
            return False

        return _check_depth(schema)

    def _resolve_nested_definition_path(self, ref_path: str) -> dict[str, Any] | None:
        """
        Priority 2: Resolve nested definition paths like #/definitions/636d/full.

        Args:
            ref_path: The $ref path (e.g., "#/definitions/636d/full")

        Returns:
            The resolved definition or None if not found
        """
        # Remove the leading #/ if present
        if ref_path.startswith("#/"):
            ref_path = ref_path[2:]

        # Split the path into parts
        parts = ref_path.split("/")

        # Start with the root definitions
        current = None
        if parts[0] in ("definitions", "$defs"):
            current = self.defs
            parts = parts[1:]  # Skip the definitions/$defs part
        else:
            return None

        # Navigate through the path
        for part in parts:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None

        return current if isinstance(current, dict) else None

    def get_available_metadata(self, value: dict[str, Any]) -> list[str]:
        """
        Get available metadata keys for a property.

        Args:
            value: The field definition containing metadata.

        Returns:
            List of available metadata keys.
        """
        available = [
            k
            for k in self.METADATA_MAP.keys()
            if k in value and not (k == "default" and value[k] is None)
        ]

        # Also check for underscore-prefixed versions (e.g., _format, _uniqueItems)
        for k in self.METADATA_MAP.keys():
            underscore_key = f"_{k}"
            if underscore_key in value and k not in available:
                available.append(k)  # Add without underscore for processing

        return available

    def format_metadata_parts(self, value: dict[str, Any]) -> list[str]:
        """
        Format metadata parts for a property.

        Args:
            value: The field definition containing metadata.

        Returns:
            List of formatted metadata strings.
        """
        available_metadata = self.get_available_metadata(value)
        formatted_parts = []

        for k in available_metadata:
            # Check both regular and underscore-prefixed keys
            actual_key = k if k in value else f"_{k}"

            if k == "contains":
                formatted_parts.append(f"contains: {self._format_contains(value[actual_key])}")
            elif k == "additionalItems":
                formatted_parts.append(
                    f"additionalItems: {self._format_type_simple(value[actual_key])}"
                )
            elif k == "if" and ("then" in value or "_then" in value):
                # Handle conditional logic as a single unit
                if_schema = value.get("if", {})
                then_schema = value.get("then", {})
                else_schema = value.get("else")
                if else_schema is not None:
                    formatted_parts.append(
                        self._format_conditional(if_schema, then_schema, else_schema)
                    )
                else:
                    formatted_parts.append(self._format_conditional(if_schema, then_schema))
            elif k in ["then", "else"] and ("if" in value or "_if" in value):
                # Skip these as they're handled with "if"
                continue
            elif (
                k in ["uniqueItems", "minItems", "maxItems"]
                and "type" in value
                and value["type"] == "array"
            ):
                # Skip these for arrays as they're integrated into the type description
                continue
            elif k in ["minLength", "maxLength"] and "type" in value and value["type"] == "string":
                # Skip these for strings as they're integrated into the type description
                continue
            elif (
                k in ["minimum", "maximum"]
                and "type" in value
                and value["type"] in ["number", "integer"]
            ):
                # Skip these for numbers as they're integrated into the type description
                continue
            else:
                formatted_parts.append(self.METADATA_MAP[k](value[actual_key]))

        return formatted_parts

    def format_field_name(self, field_name: str) -> str:
        """
        Format field name with required indicator if applicable.

        Args:
            field_name: The name of the field.

        Returns:
            Field name with asterisk if required, otherwise unchanged.
        """
        if field_name in self.required_fields:
            return f"{field_name}*"
        return field_name

    def get_required_fields_comment(self) -> str:
        """
        Get a comment explaining the required field notation.

        Returns:
            Comment string explaining asterisk notation for required fields.
        """
        if not self.required_fields:
            return ""
        return f"{self.comment_prefix} Fields marked with * are required"

    @property
    @abstractmethod
    def TYPE_MAP(self) -> dict[str, str]:
        """Type mapping dictionary for the formatter."""
        pass

    @property
    @abstractmethod
    def comment_prefix(self) -> str:
        """Comment prefix for the formatter (e.g., '//' for JSONish/TypeScript, '#' for YAML)."""
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

        # Priority 1: Check global expansion budget first
        if self._global_expansion_count >= self._global_expansion_budget:
            return "object"  # Hit global budget limit

        # Create expansion fingerprint to detect circular patterns
        expansion_fingerprint = "->".join(self._ref_expansion_path + [ref_key])
        if expansion_fingerprint in self._expansion_fingerprints:
            return "object"  # Detected circular expansion pattern

        # Check if we've already processed this ref in this cycle
        if ref_key in self._processed_refs:
            return "object"  # Prevent circular references

        # Check expansion count
        expansion_count = self._expansion_count.get(ref_key, 0)
        if expansion_count >= self._max_expansions:
            return "object"  # Prevent infinite expansion

        # Check recursion depth
        current_depth = self._recursion_depth.get(ref_key, 0)
        if current_depth >= self._max_recursion_depth:
            return "object"  # Prevent infinite recursion

        # Check ref depth to prevent deep nesting
        ref_depth = self._ref_depth_tracker.get(ref_key, 0)
        if ref_depth >= self._max_ref_depth:
            return "object"  # Prevent deep ref resolution

        # Check cache first
        if ref_key in self._ref_cache:
            return self._ref_cache[ref_key]

        # Mark as being processed
        self._processed_refs.add(ref_key)
        self._recursion_depth[ref_key] = current_depth + 1
        self._expansion_count[ref_key] = expansion_count + 1
        self._ref_depth_tracker[ref_key] = ref_depth + 1

        # Priority 1: Track global expansion and path
        self._global_expansion_count += 1
        self._ref_expansion_path.append(ref_key)
        self._expansion_fingerprints.add(expansion_fingerprint)

        try:
            # Priority 2: Try to resolve nested definition paths first
            ref_def = None
            if "/" in ref_key:
                # This might be a nested path like "636d/full"
                ref_def = self._resolve_nested_definition_path(f"definitions/{ref_key}")
                if not ref_def:
                    ref_def = self._resolve_nested_definition_path(f"$defs/{ref_key}")

            # Fallback to simple lookup
            if not ref_def:
                ref_def = self.defs.get(ref_key)

            if not ref_def:
                return "object"  # Fallback for missing definition

            # Handle different definition types with better structure preservation
            # Prioritize properties when present, as it gives more concrete structure
            if "properties" in ref_def and ref_def["properties"]:
                # Handle object definitions with properties
                processed_properties = self.process_properties(ref_def["properties"])
                ref_str = self.dict_to_string(processed_properties, indent=2)
            elif "enum" in ref_def:
                # Handle enum definitions
                ref_str = self.process_enum(ref_def)
            elif "oneOf" in ref_def:
                # Handle oneOf definitions - preserve structure
                ref_str = self.process_oneof(ref_def)
            elif "anyOf" in ref_def:
                # Handle anyOf definitions - preserve structure
                ref_str = self.process_anyof(ref_def)
            elif "allOf" in ref_def:
                # Handle allOf definitions - preserve structure
                ref_str = self.process_allof(ref_def)
            elif "type" in ref_def:
                # Handle type definitions with constraints
                ref_str = self.process_type_value(ref_def)
            elif "$ref" in ref_def:
                # Handle nested $ref references
                ref_str = self.process_ref(ref_def)
            elif "const" in ref_def:
                # Handle const definitions
                ref_str = str(ref_def["const"])
            elif "pattern" in ref_def:
                # Handle pattern-only definitions (like regex patterns)
                ref_str = f"string (pattern: {ref_def['pattern']})"
            elif "format" in ref_def:
                # Handle format-only definitions
                ref_str = f"string (format: {ref_def['format']})"
            else:
                # For complex definitions that don't match patterns, try to preserve some structure
                if isinstance(ref_def, dict) and len(ref_def) > 0:
                    # Try to extract meaningful information
                    if "description" in ref_def:
                        ref_str = f"object //{ref_def['description']}"
                    elif "title" in ref_def:
                        ref_str = f"object //{ref_def['title']}"
                    else:
                        ref_str = "object"
                else:
                    ref_str = "object"

            # Cache the result
            self._ref_cache[ref_key] = ref_str
            return ref_str
        finally:
            # Clean up tracking
            self._processed_refs.discard(ref_key)
            self._recursion_depth[ref_key] = current_depth
            self._ref_depth_tracker[ref_key] = ref_depth

            # Priority 1: Clean up expansion path
            if self._ref_expansion_path and self._ref_expansion_path[-1] == ref_key:
                self._ref_expansion_path.pop()

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

        # Add validation constraints to type description (only if metadata is enabled)
        if self.include_metadata:
            if type_name == "string":
                constraints = []

                # Add length constraints
                length_range = self._format_validation_range(
                    type_value, "minLength", "maxLength", " chars"
                )
                if length_range:
                    constraints.append(length_range)

                # Add pattern constraints
                if "pattern" in type_value:
                    pattern = type_value["pattern"]
                    # Truncate very long patterns for readability
                    if len(pattern) > 50:
                        pattern = pattern[:47] + "..."
                    constraints.append(f"pattern: {pattern}")

                # Add format constraints
                if "format" in type_value:
                    constraints.append(f"format: {type_value['format']}")

                if constraints:
                    type_str = f"{type_str} ({', '.join(constraints)})"
            elif type_name in ["number", "integer"]:
                range_info = self._format_validation_range(type_value, "minimum", "maximum")
                if range_info:
                    type_str = f"{type_str} ({range_info})"

        if type_str == "array":
            # Safely handle array items
            items = type_value.get("items")
            if not items:
                type_str = "array"  # Fallback for array without items
            elif isinstance(items, bool):
                # Handle boolean items (true means any type, false means no items)
                type_str = "array" if items else "array"
            elif isinstance(items, dict) and "type" in items:
                # For object items, process the full structure
                if items["type"] == "object" and "properties" in items:
                    processed_properties = self.process_properties(items["properties"])
                    items_structure = self.dict_to_string(processed_properties, indent=2)
                    type_str = f"[\n{items_structure}\n]"
                else:
                    items_type = self.process_type_value(items)
                    type_str = f"{items_type}[]"
            elif isinstance(items, dict) and "$ref" in items:
                items_type = self.process_ref(items)
                type_str = f"[{items_type}]"
            elif isinstance(items, dict) and "anyOf" in items:
                items_type = self.process_anyof(items)
                type_str = f"[{items_type}]"
            elif isinstance(items, dict) and "allOf" in items:
                items_type = self.process_allof(items)
                type_str = f"[{items_type}]"
            elif isinstance(items, dict) and "oneOf" in items:
                items_type = self.process_oneof(items)
                type_str = f"[{items_type}]"
            else:
                type_str = "array"  # Fallback for unknown array item type

            # Add array-specific constraints
            constraints = []
            if "uniqueItems" in type_value and type_value["uniqueItems"]:
                constraints.append("unique")

            items_range = self._format_validation_range(
                type_value, "minItems", "maxItems", " items"
            )
            if items_range:
                constraints.append(f"length: {items_range}")

            if constraints:
                type_str = f"{type_str} ({', '.join(constraints)})"

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
            elif "properties" in item:
                # Handle object schemas in anyOf
                processed_props = self.process_properties(item["properties"])
                items_structure = self.dict_to_string(processed_props, indent=2)
                item_types.append(f"{{\n{items_structure}\n}}")
            else:
                # Unknown anyOf item, skip it
                continue

        # Limit the number of union types to prevent excessive expansion
        # Be very aggressive to prevent recursive anyOf explosion
        if self._global_expansion_count > 100:
            max_items = 2  # Very aggressive for deep recursion
        elif self._global_expansion_count > 30:
            max_items = 3  # Aggressive (lowered threshold from 50)
        elif self._global_expansion_count > 10:
            max_items = 4  # Moderate (new tier)
        else:
            max_items = 5  # Conservative start (reduced from 6)

        if len(item_types) > max_items:
            return f"anyOf: {len(item_types)} options"
        else:
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

            if "allOf" in item:
                # Handle allOf inside oneOf
                item_types.append(self.process_allof(item))
            elif "anyOf" in item:
                # Handle anyOf inside oneOf
                item_types.append(self.process_anyof(item))
            elif "enum" in item:
                # Check enum before type to preserve enum constraints
                item_types.append(self.process_enum(item))
            elif "type" in item:
                item_types.append(self.process_type_value(item))
            elif "$ref" in item:
                item_types.append(self.process_ref(item))
            elif "properties" in item:
                # Handle object schemas in oneOf
                processed_props = self.process_properties(item["properties"])
                items_structure = self.dict_to_string(processed_props, indent=2)
                item_types.append(f"{{\n{items_structure}\n}}")
            elif "const" in item:
                item_types.append(str(item["const"]))

        # Preserve oneOf structure but limit to reasonable number of options
        # Be very aggressive to prevent recursive oneOf explosion
        if self._global_expansion_count > 100:
            max_items = 3  # Very aggressive for deep recursion
        elif self._global_expansion_count > 30:
            max_items = 4  # Aggressive (lowered threshold from 50)
        elif self._global_expansion_count > 10:
            max_items = 5  # Moderate (new tier)
        else:
            max_items = 6  # Conservative start (reduced from 8)

        if len(item_types) > max_items:
            return f"oneOf: {len(item_types)} options"
        elif item_types:
            return f"oneOf: {' | '.join(item_types)}"
        else:
            return "string"

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
                if "type" in item and "properties" in item:
                    # Handle object schemas with properties in allOf
                    processed_props = self.process_properties(item["properties"])
                    items_structure = self.dict_to_string(processed_props, indent=2)
                    item_types.append(f"{{\n{items_structure}\n}}")
                elif "type" in item:
                    item_types.append(self.process_type_value(item))
                elif "$ref" in item:
                    item_types.append(self.process_ref(item))
                elif "properties" in item:
                    # Handle object schemas in allOf
                    processed_props = self.process_properties(item["properties"])
                    item_types.append(self.dict_to_string(processed_props, indent=2))
                elif "description" in item and not item.get("type"):
                    # Skip items that only have description
                    continue
                else:
                    item_types.append("object")

        # Limit allOf combinations to prevent excessive expansion
        if len(item_types) > 3:  # Reduced from 5 to prevent over-expansion
            return f"allOf: {len(item_types)} schemas"
        elif item_types:
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
            # Check if this is an object with nested properties that should be expanded
            if (
                _property.get("type") == "object"
                and "properties" in _property
                and _property["properties"]
            ):
                # Expand nested object properties
                nested_props = self.process_properties(_property["properties"])
                prop_str = self.dict_to_string(nested_props, indent=1)
            # Check if this is an object with patternProperties
            elif _property.get("type") == "object" and "patternProperties" in _property:
                # Process patternProperties and show the structure
                pattern_props = _property["patternProperties"]
                pattern_results = []
                for pattern, pattern_def in list(pattern_props.items())[:2]:  # Limit to 2
                    if isinstance(pattern_def, dict):
                        if "$ref" in pattern_def:
                            pattern_type = self.process_ref(pattern_def)
                        elif "properties" in pattern_def:
                            nested_props = self.process_properties(pattern_def["properties"])
                            pattern_type = self.dict_to_string(nested_props, indent=1)
                        elif "type" in pattern_def:
                            pattern_type = self.process_type_value(pattern_def)
                        else:
                            pattern_type = "object"
                    else:
                        pattern_type = str(pattern_def)
                    pattern_results.append(f"[{pattern}]: {pattern_type}")
                prop_str = f"object  //pattern: {', '.join(pattern_results)}"
            else:
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
            Dictionary of processed properties with required field names formatted.
        """
        processed_properties = {}
        for prop_name, value in properties.items():
            # Format field name with required indicator
            formatted_name = self.format_field_name(prop_name)
            processed_properties[formatted_name] = self.process_property(value)
        return processed_properties

    def process_additional_properties(self, schema: dict[str, Any]) -> str:
        """Process additionalProperties constraint."""
        additional_props = schema.get("additionalProperties")
        if additional_props is False:
            return " //no additional properties"
        elif isinstance(additional_props, dict) and additional_props:
            # Only process if dict is non-empty
            return f" //additional: {self.process_type_value(additional_props)}"
        return ""

    def process_pattern_properties(self, schema: dict[str, Any]) -> str:
        """Process patternProperties constraint."""
        pattern_props = schema.get("patternProperties", {})
        if pattern_props:
            patterns = []
            for pattern, definition in pattern_props.items():
                # Process pattern definition properly (handle $ref, properties, etc.)
                if isinstance(definition, dict):
                    if "$ref" in definition:
                        pattern_type = self.process_ref(definition)
                    elif "properties" in definition:
                        # Expand properties for pattern
                        nested_props = self.process_properties(definition["properties"])
                        pattern_type = self.dict_to_string(nested_props, indent=1)
                    elif "type" in definition:
                        pattern_type = self.process_type_value(definition)
                    else:
                        pattern_type = "object"
                else:
                    pattern_type = str(definition)
                patterns.append(f"{pattern}: {pattern_type}")
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
            return f" //contains: {self._format_contains(contains)}"
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

    def _format_contains(self, contains_schema: Any) -> str:
        """Format contains constraint in user-friendly way."""
        if isinstance(contains_schema, dict):
            if "enum" in contains_schema:
                return f"string ({', '.join(contains_schema['enum'])})"
            elif "type" in contains_schema:
                return str(contains_schema["type"])
            else:
                return str(contains_schema)
        return str(contains_schema)

    def _format_type_simple(self, schema: Any) -> str:
        """Extract simple type from schema."""
        if isinstance(schema, dict):
            return str(schema.get("type", "any"))
        return str(schema)

    def _format_validation_range(
        self, schema: dict[str, Any], min_key: str, max_key: str, unit: str = ""
    ) -> str:
        """Format validation range constraints."""
        min_val = schema.get(min_key)
        max_val = schema.get(max_key)

        if min_val is not None and max_val is not None:
            return f"{min_val}-{max_val}{unit}"
        elif min_val is not None:
            return f"≥{min_val}{unit}"
        elif max_val is not None:
            return f"≤{max_val}{unit}"
        return ""

    def _format_conditional(
        self,
        if_schema: dict[str, Any],
        then_schema: dict[str, Any],
        else_schema: dict[str, Any] | None = None,
    ) -> str:
        """Format conditional logic in user-friendly way."""
        if_desc = self._describe_condition(if_schema)
        then_desc = self._describe_schema(then_schema)

        if else_schema:
            else_desc = self._describe_schema(else_schema)
            return f"if {if_desc} then {then_desc} else {else_desc}"
        else:
            return f"if {if_desc} then {then_desc}"

    def _describe_condition(self, condition: dict[str, Any]) -> str:
        """Describe a condition in user-friendly way."""
        if "properties" in condition:
            props = condition["properties"]
            if len(props) == 1:
                prop_name, prop_schema = next(iter(props.items()))
                if "minimum" in prop_schema:
                    return f"{prop_name} ≥ {prop_schema['minimum']}"
                elif "maximum" in prop_schema:
                    return f"{prop_name} ≤ {prop_schema['maximum']}"
                elif "pattern" in prop_schema:
                    return f"{prop_name} matches {prop_schema['pattern']}"
            return f"condition on {', '.join(props.keys())}"
        return "condition"

    def _describe_schema(self, schema: dict[str, Any]) -> str:
        """Describe a schema in user-friendly way."""
        if "required" in schema:
            return f"requires {', '.join(schema['required'])}"
        elif "properties" in schema:
            return f"object with {', '.join(schema['properties'].keys())}"
        return "schema"

    @abstractmethod
    def transform_schema(self) -> str:
        """
        Transform the schema into the desired format.

        Returns:
            Formatted schema as a string.
        """
        pass
